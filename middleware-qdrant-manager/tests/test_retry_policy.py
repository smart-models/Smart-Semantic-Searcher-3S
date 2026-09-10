"""The retry policy honors a server's Retry-After and jitters its own backoff.

Every test here patches ``asyncio.sleep`` inside the retry module and asserts
on the recorded delays, so the suite never waits in real time for a policy it
is only inspecting.
"""

import random
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi import HTTPException

from qdrant_middleware.clients import retry as retry_module
from qdrant_middleware.clients.embedder import EmbedderClient
from qdrant_middleware.clients.retry import (
    RETRY_AFTER_CAP_SECONDS,
    backoff_delay,
    parse_retry_after,
    retry_async,
)
from qdrant_middleware.core.config import Settings


@pytest.fixture
def recorded_sleeps(monkeypatch):
    """Replace the retry module's sleep with a recorder and return the list."""
    delays: list[float] = []

    async def _sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(retry_module.asyncio, "sleep", _sleep)
    return delays


async def _embedder_client(handler, **overrides):
    """Build an EmbedderClient whose transport is a MockTransport.

    Mirrors tests/test_rerank_capacity_error.py: EmbedderClient builds its own
    httpx.AsyncClient from Settings, so the established pattern is to close
    that one and swap in a MockTransport-backed client.
    """
    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        SMART_EMBEDDER_URL="http://embedder:8010",
        SMART_EMBEDDER_CLIENT_TOKEN="t",
        RETRY_MAX_ATTEMPTS=3,
        RETRY_BASE_DELAY=0.1,
        **overrides,
    )
    client = EmbedderClient(settings)
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url="http://embedder:8010", transport=httpx.MockTransport(handler)
    )
    return client


# --- parse_retry_after -------------------------------------------------------


def test_delta_seconds_form_is_read_as_seconds():
    assert parse_retry_after("2") == 2.0


def test_http_date_form_is_read_against_a_reference_instant():
    now = datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)
    later = now + timedelta(seconds=3)
    header = later.strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert parse_retry_after(header, now=now) == pytest.approx(3.0)


def test_an_http_date_already_in_the_past_asks_for_no_wait():
    now = datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)
    earlier = now - timedelta(seconds=30)
    header = earlier.strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert parse_retry_after(header, now=now) == 0.0


def test_a_missing_header_asks_for_nothing():
    assert parse_retry_after(None) is None


@pytest.mark.parametrize("raw", ["", "soon", "-5", "2.5", "NaN", "not a date"])
def test_an_unusable_value_falls_back_to_computed_backoff(raw):
    assert parse_retry_after(raw) is None


def test_a_long_wait_is_capped():
    # Smart Embedder sends Retry-After: 60 with its rate-limit 429. Honoring it
    # literally would outlive MIDDLEWARE_REQUEST_TIMEOUT and turn an actionable
    # 502 into a 504.
    assert parse_retry_after("60") == RETRY_AFTER_CAP_SECONDS


def test_the_vram_pressure_hint_survives_the_cap_unchanged():
    # The case the cap must not distort: 503 rerank_vram_pressure.
    assert parse_retry_after("2") == 2.0


# --- backoff_delay -----------------------------------------------------------


def test_backoff_grows_exponentially_and_stays_inside_the_jitter_band():
    random.seed(1234)
    for attempt in range(4):
        undelayed = 0.1 * (2**attempt)
        for _ in range(50):
            delay = backoff_delay(attempt, 0.1)
            assert 0.5 * undelayed <= delay <= undelayed


def test_a_zero_base_delay_still_means_no_wait():
    # Several suites set RETRY_BASE_DELAY=0 to keep tests instant; jitter must
    # not resurrect a wait there.
    assert backoff_delay(3, 0.0) == 0.0


def test_jitter_desynchronizes_callers_that_failed_together():
    random.seed(99)
    delays = {backoff_delay(0, 0.1) for _ in range(20)}
    assert len(delays) > 1


# --- retry_async -------------------------------------------------------------


class _Boom(Exception):
    """A transient failure carrying an optional server-suggested wait."""

    def __init__(self, retry_after: float | None = None) -> None:
        super().__init__("boom")
        self.suggested = retry_after


async def test_a_suggested_wait_replaces_the_computed_backoff(recorded_sleeps):
    with pytest.raises(_Boom):
        await retry_async(
            _always_fail,
            attempts=3,
            base_delay=0.1,
            should_retry=lambda exc: True,
            retry_after=lambda exc: 2.0,
        )
    assert recorded_sleeps == [2.0, 2.0]


async def test_without_a_suggestion_the_jittered_backoff_is_used(recorded_sleeps):
    random.seed(7)
    with pytest.raises(_Boom):
        await retry_async(
            _always_fail,
            attempts=3,
            base_delay=0.1,
            should_retry=lambda exc: True,
            retry_after=lambda exc: None,
        )
    assert len(recorded_sleeps) == 2
    assert 0.05 <= recorded_sleeps[0] <= 0.1
    assert 0.10 <= recorded_sleeps[1] <= 0.2


async def test_an_extractor_is_optional(recorded_sleeps):
    random.seed(7)
    with pytest.raises(_Boom):
        await retry_async(
            _always_fail,
            attempts=2,
            base_delay=0.1,
            should_retry=lambda exc: True,
        )
    assert len(recorded_sleeps) == 1


async def test_a_non_retryable_error_never_sleeps(recorded_sleeps):
    with pytest.raises(_Boom):
        await retry_async(
            _always_fail,
            attempts=3,
            base_delay=0.1,
            should_retry=lambda exc: False,
            retry_after=lambda exc: 2.0,
        )
    assert recorded_sleeps == []


async def _always_fail():
    raise _Boom()


# --- through the Smart Embedder client ---------------------------------------


async def test_vram_pressure_is_retried_after_the_wait_the_server_asked_for(
    recorded_sleeps,
):
    calls: list[httpx.Request] = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            503,
            json={"detail": {"error": "rerank_vram_pressure"}},
            headers={"Retry-After": "2"},
        )

    embedder = await _embedder_client(handler)
    with pytest.raises(HTTPException):
        await embedder.rerank("q", ["p"])
    assert len(calls) == 3
    # Not the 0.1/0.2 the computed backoff would have produced.
    assert recorded_sleeps == [2.0, 2.0]


async def test_a_rate_limited_embedder_is_retried_at_the_capped_wait(recorded_sleeps):
    def handler(request):
        return httpx.Response(
            429, json={"detail": "slow down"}, headers={"Retry-After": "60"}
        )

    embedder = await _embedder_client(handler)
    with pytest.raises(HTTPException):
        await embedder.rerank("q", ["p"])
    assert recorded_sleeps == [RETRY_AFTER_CAP_SECONDS, RETRY_AFTER_CAP_SECONDS]


async def test_a_response_without_the_header_keeps_the_computed_backoff(
    recorded_sleeps,
):
    random.seed(3)

    def handler(request):
        return httpx.Response(503, json={"detail": "unavailable"})

    embedder = await _embedder_client(handler)
    with pytest.raises(HTTPException):
        await embedder.rerank("q", ["p"])
    assert len(recorded_sleeps) == 2
    assert all(delay <= 0.2 for delay in recorded_sleeps)


async def test_a_capacity_error_is_still_never_retried(recorded_sleeps):
    calls: list[httpx.Request] = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            413,
            json={"detail": {"error": "rerank_capacity_exceeded"}},
            headers={"Retry-After": "2"},
        )

    embedder = await _embedder_client(handler)
    with pytest.raises(HTTPException) as exc:
        await embedder.rerank("q", ["p"])
    # 413 is outside RETRYABLE_HTTP_STATUS: a Retry-After on it changes nothing.
    assert len(calls) == 1
    assert recorded_sleeps == []
    assert exc.value.status_code == 413

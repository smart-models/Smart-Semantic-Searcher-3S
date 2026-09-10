"""Outbound HTTP behaviour: auth header, path joining, retry, error raising."""

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import pytest

from smart_qdrant_mcp.client import (
    RETRY_AFTER_CAP_SECONDS,
    MiddlewareClient,
    _retry_after_delay,
)
from smart_qdrant_mcp.config import load_settings
from smart_qdrant_mcp.errors import MiddlewareError


async def test_sends_bearer_token(client, routes):
    routes.add("GET", "/collections", body={"collections": []})
    await client.request("GET", "/collections")
    assert routes.calls[-1].headers["authorization"] == "Bearer middleware-secret"


async def test_joins_path_onto_base_url(client, routes):
    routes.add("GET", "/collections/docs", body={"ok": True})
    await client.request("GET", "/collections/docs")
    assert str(routes.calls[-1].url) == "http://middleware:6336/collections/docs"


async def test_returns_decoded_json(client, routes):
    routes.add("POST", "/points/count", body={"count": 7})
    assert await client.request("POST", "/points/count", json={}) == {"count": 7}


async def test_omits_none_params(client, routes):
    routes.add("GET", "/collections/docs", body={"ok": True})
    await client.request("GET", "/collections/docs", params={"validate": None})
    assert routes.calls[-1].url.params.get("validate") is None


async def test_raises_middleware_error_with_status_and_detail(client, routes):
    routes.add("GET", "/collections/gone", status=404, body={"detail": "not found"})
    with pytest.raises(MiddlewareError) as excinfo:
        await client.request("GET", "/collections/gone")
    assert excinfo.value.status == 404
    assert excinfo.value.detail == "not found"


async def test_does_not_retry_client_errors(client, routes):
    routes.add("POST", "/points", status=400, body={"detail": "bad"})
    with pytest.raises(MiddlewareError):
        await client.request("POST", "/points", json={})
    assert len(routes.calls) == 1


async def test_retries_server_errors_then_succeeds(client, routes):
    routes.add("GET", "/collections", status=503, body={"detail": "starting"})
    routes.add("GET", "/collections", body={"collections": []})
    assert await client.request("GET", "/collections") == {"collections": []}
    assert len(routes.calls) == 2


async def test_does_not_retry_mutating_post_by_default(client, routes):
    routes.add("POST", "/collections/docs/reindex", status=503, body={})
    routes.add("POST", "/collections/docs/reindex", body={"ok": True})

    with pytest.raises(MiddlewareError):
        await client.request("POST", "/collections/docs/reindex", json={})

    assert len(routes.calls) == 1


async def test_read_only_post_can_opt_into_retries(client, routes):
    routes.add("POST", "/points/count", status=503, body={})
    routes.add("POST", "/points/count", body={"count": 1})

    result = await client.request("POST", "/points/count", json={}, retryable=True)

    assert result == {"count": 1}
    assert len(routes.calls) == 2


async def test_gives_up_after_configured_attempts(settings, routes):
    for _ in range(5):
        routes.add("GET", "/collections", status=502, body={"detail": "bad gateway"})
    api = MiddlewareClient(settings, transport=routes.transport())
    with pytest.raises(MiddlewareError):
        await api.request("GET", "/collections")
    await api.aclose()
    assert len(routes.calls) == settings.retry_attempts


async def test_retries_429(client, routes):
    routes.add("GET", "/collections", status=429, body={"detail": "slow down"})
    routes.add("GET", "/collections", body={"collections": []})
    await client.request("GET", "/collections")
    assert len(routes.calls) == 2


async def test_text_returns_raw_body(client, routes):
    routes.add("GET", "/metrics", body="qdrant_up 1")
    assert await client.text("/metrics") == "qdrant_up 1"


async def test_transport_errors_propagate_after_retries(settings):
    attempts = {"n": 0}

    def boom(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        raise httpx.ConnectError("refused", request=request)

    api = MiddlewareClient(settings, transport=httpx.MockTransport(boom))
    with pytest.raises(httpx.ConnectError):
        await api.request("GET", "/collections")
    await api.aclose()
    assert attempts["n"] == settings.retry_attempts


async def test_mutating_transport_error_is_not_retried(settings):
    attempts = {"n": 0}

    def boom(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        raise httpx.ConnectError("response lost", request=request)

    api = MiddlewareClient(settings, transport=httpx.MockTransport(boom))
    with pytest.raises(httpx.ConnectError):
        await api.request("POST", "/collections/docs/reindex", json={})
    await api.aclose()

    assert attempts["n"] == 1


def test_retry_after_accepts_rfc_http_date_and_caps_it():
    now = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)
    retry_at = now + timedelta(minutes=2)
    response = httpx.Response(
        429, headers={"Retry-After": format_datetime(retry_at, usegmt=True)}
    )

    assert _retry_after_delay(response, now=now) == RETRY_AFTER_CAP_SECONDS


@pytest.mark.parametrize("value", ["1.5", "-1", "NaN", "not-a-date"])
def test_retry_after_rejects_non_rfc_values(value):
    response = httpx.Response(429, headers={"Retry-After": value})
    assert _retry_after_delay(response) is None


async def test_route_queue_is_consumed_not_replayed(client, routes):
    routes.add("GET", "/y", body={"n": 1})
    routes.add("GET", "/y", body={"n": 2})
    assert await client.request("GET", "/y") == {"n": 1}
    assert await client.request("GET", "/y") == {"n": 2}
    with pytest.raises(MiddlewareError) as excinfo:
        await client.request("GET", "/y")
    assert excinfo.value.status == 404
    assert "exhaust" in excinfo.value.detail.lower()


async def test_single_queued_response_is_returned_once(client, routes):
    routes.add("GET", "/z", body={"ok": True})
    assert await client.request("GET", "/z") == {"ok": True}


async def test_omits_authorization_header_when_client_token_is_blank(routes):
    """A blank MIDDLEWARE_CLIENT_TOKEN must send no Authorization header at all.

    ``Bearer `` with an empty token is not a legal HTTP header value -- the
    trailing space makes h11 raise ``LocalProtocolError`` before the request
    leaves the process. Since a blank client token is a supported
    configuration (the middleware may itself serve unauthenticated), sending
    the header unconditionally breaks every middleware call in that setup.
    """
    settings = load_settings(
        _env_file=None,
        MCP_API_TOKENS="tok-alice:alice",
        MCP_MIDDLEWARE_URL="http://middleware:6336",
        MIDDLEWARE_CLIENT_TOKEN="",
    )
    api = MiddlewareClient(settings, transport=routes.transport())
    try:
        routes.add("GET", "/collections", body={"collections": []})
        await api.request("GET", "/collections")
    finally:
        await api.aclose()
    assert "authorization" not in routes.calls[-1].headers

"""The only place that speaks HTTP to the middleware.

The retry policy mirrors ``qdrant_middleware/clients/retry.py``'s trigger set —
transient 5xx, 429, timeouts and connection errors — but is reimplemented here
because this service must not import the middleware package at runtime, and the
wait between attempts is hardened two ways the middleware's is not: a 429's
``Retry-After`` header (delta-seconds form, capped) overrides the computed
backoff, and the backoff itself carries jitter so concurrent callers that
failed together do not retry in lockstep.
"""

from __future__ import annotations

import asyncio
import math
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx

from smart_qdrant_mcp.config import Settings
from smart_qdrant_mcp.errors import MiddlewareError

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
SAFE_RETRY_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

# A 429's Retry-After is honored only up to this many seconds: without a cap,
# a broken or hostile server could park every caller indefinitely.
RETRY_AFTER_CAP_SECONDS = 30.0


def _retry_after_delay(
    response: httpx.Response, *, now: datetime | None = None
) -> float | None:
    """Return the wait a response's ``Retry-After`` header requests, if usable.

    Both RFC 9110 forms are honored: non-negative integer delta-seconds and an
    HTTP-date. Invalid or non-finite values fall back to computed backoff. The
    resulting delay is capped at ``RETRY_AFTER_CAP_SECONDS``.
    """
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    value = raw.strip()
    if value.isascii() and value.isdigit():
        seconds = float(value)
    else:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        reference = now or datetime.now(timezone.utc)
        seconds = max(0.0, (retry_at - reference).total_seconds())
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return min(seconds, RETRY_AFTER_CAP_SECONDS)


def _backoff_delay(attempt: int, base: float) -> float:
    """Return the exponential backoff for ``attempt``, jittered to [0.5x, 1x].

    The uniform jitter desynchronizes concurrent callers so they do not stampede
    the middleware in lockstep (thundering herd); the lower bound keeps the wait
    near the pre-jitter policy rather than ever dropping to zero. ``random`` is
    used at module level, so a test can fix the sequence with ``random.seed``.
    """
    return base * (2 ** (attempt - 1)) * random.uniform(0.5, 1.0)


class MiddlewareClient:
    """An async client for the middleware REST API."""

    def __init__(
        self, settings: Settings, transport: httpx.BaseTransport | None = None
    ) -> None:
        """Open a pooled client bound to the middleware base URL and token.

        A blank ``MIDDLEWARE_CLIENT_TOKEN`` sends no Authorization header at
        all, rather than an empty one: the middleware may itself be serving
        unauthenticated, and ``Bearer `` is an illegal header value that h11
        refuses to serialize, failing every request before it is sent.

        Args:
            settings: Source of the base URL, request timeout and bearer token
                sent on every request.
            transport: Optional httpx transport override, for tests that serve
                the middleware API without a network round trip.
        """
        self._settings = settings
        headers = (
            {"Authorization": f"Bearer {settings.middleware_client_token}"}
            if settings.middleware_client_token
            else {}
        )
        self._client = httpx.AsyncClient(
            base_url=settings.middleware_url,
            timeout=settings.http_timeout,
            headers=headers,
            transport=transport,
        )

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        await self._client.aclose()

    async def _send(
        self,
        method: str,
        path: str,
        *,
        json: dict | None,
        params: dict | None,
        retryable: bool,
    ) -> httpx.Response:
        """Send one request, retrying transient failures with backoff."""
        clean_params = (
            {k: v for k, v in params.items() if v is not None} if params else None
        )
        attempts = self._settings.retry_attempts if retryable else 1
        last_response: httpx.Response | None = None
        for attempt in range(1, attempts + 1):
            delay: float | None = None
            try:
                response = await self._client.request(
                    method, path, json=json, params=clean_params
                )
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt == attempts:
                    raise
            else:
                if response.status_code not in RETRY_STATUSES or attempt == attempts:
                    return response
                last_response = response
                if response.status_code == 429:
                    delay = _retry_after_delay(response)
            if delay is None:
                delay = _backoff_delay(attempt, self._settings.retry_backoff)
            await asyncio.sleep(delay)
        # Unreachable in practice: the loop returns or raises on its last attempt.
        assert last_response is not None
        return last_response

    def _raise_for_status(self, response: httpx.Response, path: str) -> None:
        """Convert a non-success response into a MiddlewareError."""
        if response.is_success:
            return
        try:
            payload = response.json()
        except ValueError:
            payload = response.text
        detail = (
            payload.get("detail", payload) if isinstance(payload, dict) else payload
        )
        raise MiddlewareError(status=response.status_code, detail=detail, path=path)

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        params: dict | None = None,
        retryable: bool | None = None,
        accepted_statuses: frozenset[int] = frozenset(),
    ) -> object:
        """Call the middleware and return the decoded JSON body."""
        should_retry = (
            method.upper() in SAFE_RETRY_METHODS if retryable is None else retryable
        )
        response = await self._send(
            method,
            path,
            json=json,
            params=params,
            retryable=should_retry,
        )
        if response.status_code not in accepted_statuses:
            self._raise_for_status(response, path)
        if not response.content:
            return {}
        return response.json()

    async def text(self, path: str) -> str:
        """Call the middleware and return the raw response body as text."""
        response = await self._send("GET", path, json=None, params=None, retryable=True)
        self._raise_for_status(response, path)
        return response.text

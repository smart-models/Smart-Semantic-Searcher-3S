"""Tests for the request rate limit boundary."""

from types import SimpleNamespace

import pytest

from qdrant_middleware.middleware.rate_limit import (
    is_rate_limit_exempt,
    rate_limit_middleware,
)


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/assets/app.js",
        "/assets/fonts/barlow.woff2",
        "/docs",
        "/docs/oauth2-redirect",
        "/health",
        "/livez",
        "/metrics",
        "/openapi.json",
        "/readyz",
        "/redoc",
        "/stats",
    ],
)
def test_passive_resources_are_rate_limit_exempt(path):
    assert is_rate_limit_exempt(path)


@pytest.mark.parametrize(
    "path",
    [
        "/collections",
        "/collections/example",
        "/query",
        "/assets",
        "/health/details",
    ],
)
def test_operational_api_paths_remain_rate_limited(path):
    assert not is_rate_limit_exempt(path)


@pytest.mark.asyncio
async def test_exempt_request_does_not_consult_the_limiter():
    request = SimpleNamespace(url=SimpleNamespace(path="/assets/app.js"))
    downstream_response = object()

    async def call_next(_request):
        return downstream_response

    assert await rate_limit_middleware(request, call_next) is downstream_response


@pytest.mark.asyncio
async def test_protected_request_is_rejected_when_bucket_is_empty(monkeypatch):
    class EmptyLimiter:
        """Limiter stand-in with no tokens left, rejecting every request."""

        async def check_rate_limit(self, client_ip):
            assert client_ip == "127.0.0.1"
            return False

    runtime = SimpleNamespace(
        rate_limiter=EmptyLimiter(),
        stats=SimpleNamespace(rejected_requests=0),
    )
    app = SimpleNamespace(
        state=SimpleNamespace(
            runtime=runtime,
            settings=SimpleNamespace(rate_limit_requests_per_minute=300),
        )
    )
    request = SimpleNamespace(
        app=app,
        client=SimpleNamespace(host="127.0.0.1"),
        url=SimpleNamespace(path="/collections"),
    )
    rejections = []
    monkeypatch.setattr(
        "qdrant_middleware.middleware.rate_limit.record_rejection",
        rejections.append,
    )

    async def call_next(_request):
        pytest.fail("a rejected request must not reach the route")

    response = await rate_limit_middleware(request, call_next)

    assert response.status_code == 429
    assert response.headers["retry-after"] == "60"
    assert runtime.stats.rejected_requests == 1
    assert rejections == ["rate_limit"]

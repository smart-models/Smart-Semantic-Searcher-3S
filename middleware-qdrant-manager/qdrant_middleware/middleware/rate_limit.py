"""Request rate limiting middleware."""

from fastapi import Request
from fastapi.responses import JSONResponse

from qdrant_middleware.middleware.metrics import record_rejection

RATE_LIMIT_EXEMPT_PATHS = frozenset(
    {
        "/",
        "/docs",
        "/health",
        "/livez",
        "/metrics",
        "/openapi.json",
        "/readyz",
        "/redoc",
        "/stats",
    }
)
RATE_LIMIT_EXEMPT_PREFIXES = ("/assets/", "/docs/", "/redoc/")


def is_rate_limit_exempt(path: str) -> bool:
    """Return whether *path* is a passive UI, docs, or monitoring resource."""
    return path in RATE_LIMIT_EXEMPT_PATHS or path.startswith(
        RATE_LIMIT_EXEMPT_PREFIXES
    )


async def rate_limit_middleware(request: Request, call_next):
    """Apply per-client token-bucket limits to protected request paths."""
    if is_rate_limit_exempt(request.url.path):
        return await call_next(request)

    runtime = request.app.state.runtime
    settings = request.app.state.settings
    client_ip = request.client.host if request.client else "unknown"
    if not await runtime.rate_limiter.check_rate_limit(client_ip):
        runtime.stats.rejected_requests += 1
        record_rejection("rate_limit")
        return JSONResponse(
            status_code=429,
            content={
                "detail": (
                    "Rate limit exceeded. "
                    f"Max {settings.rate_limit_requests_per_minute} req/min."
                ),
                "retry_after": 60,
            },
            headers={"Retry-After": "60"},
        )
    return await call_next(request)

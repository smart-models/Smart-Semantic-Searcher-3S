"""Middleware that rejects new requests during graceful shutdown."""

from fastapi import Request
from fastapi.responses import JSONResponse

from qdrant_middleware.middleware.metrics import record_rejection


async def shutdown_middleware(request: Request, call_next):
    """Return 503 after shutdown begins, otherwise continue the chain."""
    if request.app.state.runtime.is_shutting_down:
        request.app.state.runtime.stats.rejected_requests += 1
        record_rejection("shutting_down")
        return JSONResponse(
            status_code=503, content={"detail": "Server is shutting down"}
        )
    return await call_next(request)

"""Prometheus metrics and request timeout instrumentation."""

import asyncio
import time

from fastapi import Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram, Info
from starlette.status import HTTP_504_GATEWAY_TIMEOUT

requests_total = Counter(
    "qdrant_manager_requests_total",
    "Total Qdrant manager requests",
    ["endpoint", "status"],
)
requests_rejected = Counter(
    "qdrant_manager_requests_rejected_total",
    "Total rejected Qdrant manager requests",
    ["reason"],
)
request_latency = Histogram(
    "qdrant_manager_request_duration_seconds",
    "Qdrant manager request latency",
    ["endpoint"],
)
qdrant_up = Gauge("qdrant_manager_vectorstore_up", "Whether Qdrant is reachable")
server_info = Info("qdrant_manager_info", "Qdrant manager information")


def record_rejection(reason: str) -> None:
    """Increment the rejection counter for a normalized reason."""
    requests_rejected.labels(reason=reason).inc()


def metric_endpoint(request: Request) -> str:
    """Return a stable route label for metrics cardinality control."""
    route = request.scope.get("route")
    route_path = getattr(route, "path", None)
    if route_path:
        return route_path
    settings = request.app.state.settings
    if request.url.path in settings.public_auth_paths:
        return request.url.path
    return "unmatched"


class TimeoutMetricsMiddleware:
    """Own and cancel the ASGI handler before emitting a timeout response."""

    def __init__(self, app):
        """Retain the next ASGI application in the middleware stack."""
        self.app = app

    async def __call__(self, scope, receive, send):
        """Time response headers and await request-owned cancellation cleanup."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope, receive=receive)
        settings = request.app.state.settings
        started_at = time.perf_counter()
        headers_started = asyncio.Event()
        recorded = False
        handler = None
        headers = None

        def record(status=None):
            """Count a response once, retaining header-time latency semantics."""
            nonlocal recorded
            if recorded:
                return
            recorded = True
            endpoint = metric_endpoint(request)
            if status is not None:
                requests_total.labels(endpoint=endpoint, status=str(status)).inc()
            request.app.state.runtime.stats.total_requests += 1
            request_latency.labels(endpoint=endpoint).observe(
                time.perf_counter() - started_at
            )

        async def observed_send(message):
            """Record the first response start without buffering its body."""
            if message["type"] == "http.response.start":
                headers_started.set()
                record(message["status"])
            await send(message)

        try:
            if request.url.path in settings.timeout_exempt_paths:
                await self.app(scope, receive, observed_send)
                return
            handler = asyncio.create_task(self.app(scope, receive, observed_send))
            headers = asyncio.create_task(headers_started.wait())
            done, _ = await asyncio.wait(
                {handler, headers},
                timeout=settings.request_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done and not headers_started.is_set():
                handler.cancel()
                # Observe cleanup errors too; never leave a detached mutation.
                await asyncio.gather(handler, return_exceptions=True)
                response = JSONResponse(
                    {"detail": "Request timed out."},
                    status_code=HTTP_504_GATEWAY_TIMEOUT,
                )
                await response(scope, receive, observed_send)
            else:
                await handler
        except Exception:
            record(500)
            raise
        finally:
            pending = [task for task in (headers, handler) if task is not None]
            for task in pending:
                if not task.done():
                    task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            record()

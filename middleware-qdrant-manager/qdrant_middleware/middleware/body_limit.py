"""ASGI middleware for enforcing streamed request body size limits."""

import json
from collections.abc import Awaitable, Callable
from typing import Any

from qdrant_middleware.middleware.metrics import record_rejection

Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]


class BodySizeLimitMiddleware:
    """Reject HTTP requests once streamed body bytes exceed ``max_bytes``."""

    def __init__(self, app, max_bytes: int) -> None:
        """Wrap ``app`` and reject bodies larger than ``max_bytes``."""
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(
        self, scope: dict[str, Any], receive: Receive, send: Send
    ) -> None:
        """Enforce the body size limit on HTTP scopes, passing others through."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        total = 0
        rejected = False

        async def limited_receive() -> dict[str, Any]:
            nonlocal total, rejected
            message = await receive()
            if message["type"] != "http.request" or rejected:
                return message
            body = message.get("body", b"")
            total += len(body)
            if total <= self.max_bytes:
                return message
            rejected = True
            app = scope.get("app")
            if app is not None:
                app.state.runtime.stats.rejected_requests += 1
            record_rejection("request_body_too_large")
            return {"type": "http.disconnect"}

        await self.app(
            scope, limited_receive, self._rejecting_send(send, lambda: rejected)
        )

    def _rejecting_send(self, send: Send, is_rejected: Callable[[], bool]) -> Send:
        """Wrap ``send`` to replace the app response with a 413 once rejected."""

        async def wrapped_send(message: dict[str, Any]) -> None:
            if not is_rejected():
                await send(message)
                return
            if message["type"] != "http.response.start":
                return
            content = json.dumps(
                {"detail": f"Request body too large. Max {self.max_bytes} bytes."}
            ).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 413,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": content})

        return wrapped_send

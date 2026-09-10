"""Tests for the streamed request body size limit middleware."""

import pytest

from qdrant_middleware.middleware.body_limit import BodySizeLimitMiddleware


@pytest.mark.asyncio
async def test_body_limit_counts_streamed_body_without_content_length():
    async def app(scope, receive, send):
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                break
            if message["type"] == "http.request" and not message.get(
                "more_body", False
            ):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = BodySizeLimitMiddleware(app, max_bytes=3)
    scope = {"type": "http", "method": "POST", "path": "/points", "headers": []}
    messages = iter(
        [
            {"type": "http.request", "body": b"ab", "more_body": True},
            {"type": "http.request", "body": b"cd", "more_body": False},
        ]
    )
    sent = []

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    await middleware(scope, receive, send)

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413

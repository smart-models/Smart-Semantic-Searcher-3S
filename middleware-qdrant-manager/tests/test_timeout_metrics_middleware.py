"""Actual ASGI cancellation and response-start timeout regressions."""

import asyncio

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from qdrant_middleware.middleware.metrics import TimeoutMetricsMiddleware

from qdrant_middleware.core.config import Settings
from qdrant_middleware.main import create_app


def _app():
    """Build a locally isolated app with a short test-only deadline."""
    settings = Settings(
        MIDDLEWARE_API_TOKENS="", MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True
    ).model_copy(update={"request_timeout": 0.2})
    return create_app(settings)


async def _request(app, send):
    """Drive ASGI without a response-buffering HTTP client."""
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await asyncio.Event().wait()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/held",
        "raw_path": b"/held",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("127.0.0.1", 1),
        "server": ("test", 80),
    }
    await app(scope, receive, send)


async def test_timeout_cancels_handler_and_releases_lease_before_504():
    app = _app()
    release = asyncio.Event()
    started = asyncio.Event()
    cancelled = asyncio.Event()
    writes = []
    statuses = []
    coordinator = app.state.runtime.mutations

    @app.get("/held")
    async def held():
        async with coordinator.lease("owned"):
            try:
                await release.wait()
                writes.append("late write")
                return {"ok": True}
            except asyncio.CancelledError:
                cancelled.set()
                raise

    async def send(message):
        if message["type"] == "http.response.start":
            statuses.append(message["status"])
            started.set()

    task = asyncio.create_task(_request(app, send))
    try:
        await asyncio.wait_for(started.wait(), 1)
        assert statuses == [504]
        assert cancelled.is_set()
        assert coordinator._entries == {}
        release.set()
        await asyncio.wait_for(task, 1)
        assert writes == []
        assert app.state.runtime.stats.total_requests == 1
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("exempt", [False, True])
async def test_streaming_body_has_no_new_deadline_after_headers(exempt):
    state_owner = _app()
    app = FastAPI()
    app.state.settings = state_owner.state.settings.model_copy(
        update={"request_timeout": 0.03}
    )
    app.state.runtime = state_owner.state.runtime
    app.add_middleware(TimeoutMetricsMiddleware)
    path = "/points/export" if exempt else "/held"

    @app.get(path)
    async def held():
        if exempt:
            await asyncio.sleep(0.06)

        async def body():
            yield b"first"
            await asyncio.sleep(0.06)
            yield b"last"

        return StreamingResponse(body())

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(path)
    assert response.status_code == 200
    assert response.content == b"firstlast"
    assert app.state.runtime.stats.total_requests == 1


async def test_client_cancellation_does_not_leave_a_handler_or_lease():
    app = _app()
    entered = asyncio.Event()
    exited = asyncio.Event()

    @app.get("/held")
    async def held():
        async with app.state.runtime.mutations.lease("owned"):
            try:
                entered.set()
                await asyncio.Event().wait()
            finally:
                exited.set()

    async def send(message):
        pytest.fail(f"Canceled request must not emit a response: {message['type']}")

    task = asyncio.create_task(_request(app, send))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert exited.is_set()
    assert app.state.runtime.mutations._entries == {}


async def test_handler_timeout_exception_is_not_the_response_deadline():
    app = _app()

    @app.get("/held")
    async def held():
        raise TimeoutError("internal operation failure")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.get("/held")
    assert response.status_code == 500
    assert app.state.runtime.stats.total_requests == 1

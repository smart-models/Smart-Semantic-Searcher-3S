"""Shared fixtures: a fake middleware built on httpx.MockTransport."""

from __future__ import annotations

import json as jsonlib

import httpx
import pytest

from smart_qdrant_mcp.client import MiddlewareClient
from smart_qdrant_mcp.config import Settings, load_settings

ENV = {
    "MCP_API_TOKENS": "tok-alice:alice",
    "MCP_MIDDLEWARE_URL": "http://middleware:6336",
    "MIDDLEWARE_CLIENT_TOKEN": "middleware-secret",
    # Keeps the retry tests from sleeping in real time between attempts.
    "MCP_RETRY_BACKOFF_SECONDS": "0.001",
}


class FakeMiddleware:
    """Record requests and answer them from a registered route table."""

    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []
        self._routes: dict[tuple[str, str], list[tuple[int, object]]] = {}
        # Total responses ever queued per route, kept even after the queue
        # empties, so an exhausted-queue error can report how many there were.
        self._queued_total: dict[tuple[str, str], int] = {}

    def add(self, method: str, path: str, status: int = 200, body: object = None):
        """Queue one response for (method, path). Calls consume the queue in order."""
        key = (method.upper(), path)
        self._routes.setdefault(key, []).append((status, body))
        self._queued_total[key] = self._queued_total.get(key, 0) + 1
        return self

    def handler(self, request: httpx.Request) -> httpx.Response:
        """MockTransport entry point.

        Every call pops the next queued response for its route, so a route's
        queue is genuinely consumed rather than replayed. A route that was
        never registered gets a 404 "no route" response; a route whose queue
        has run out gets a distinct 404 "queue exhausted" response, so a test
        that over-calls a route fails loudly instead of silently replaying
        the last queued response forever.
        """
        self.calls.append(request)
        key = (request.method, request.url.path)
        if key not in self._routes:
            return httpx.Response(404, json={"detail": f"no route {key[1]}"})
        queued = self._routes[key]
        if not queued:
            total = self._queued_total[key]
            detail = f"queue exhausted for {key[0]} {key[1]} after {total} response(s)"
            return httpx.Response(404, json={"detail": detail})
        status, body = queued.pop(0)
        if isinstance(body, str):
            return httpx.Response(status, text=body)
        return httpx.Response(status, json=body)

    def last_body(self) -> dict:
        """Decode the JSON body of the most recent request."""
        return jsonlib.loads(self.calls[-1].content or b"{}")

    def transport(self) -> httpx.MockTransport:
        """Build a MockTransport bound to this fake's handler."""
        return httpx.MockTransport(self.handler)


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's exported shell variables out of every test.

    Two sources can leak real values into a test: the process environment and
    the repository-root .env file. This fixture removes the first; the second
    is disabled per call by passing ``_env_file=None`` to ``load_settings``,
    because pydantic-settings reads the dotenv file directly and clearing
    os.environ does not affect it. Both are needed.
    """
    for name in (
        "MCP_API_TOKENS",
        "MCP_AUTH_MODE",
        "MCP_OAUTH_ISSUER_URL",
        "MCP_OAUTH_RESOURCE_SERVER_URL",
        "MCP_OAUTH_INTROSPECTION_URL",
        "MCP_OAUTH_CLIENT_ID",
        "MCP_OAUTH_CLIENT_SECRET",
        "MCP_OAUTH_AUDIENCE",
        "MCP_OAUTH_REQUIRED_SCOPES",
        "MCP_BIND_HOST",
        "MCP_LISTEN_HOST",
        "MCP_PORT",
        "MCP_MIDDLEWARE_URL",
        "MCP_ALLOWED_HOSTS",
        "MCP_ALLOWED_ORIGINS",
        "MCP_HTTP_TIMEOUT",
        "MCP_RETRY_ATTEMPTS",
        "MCP_RETRY_BACKOFF_SECONDS",
        "MIDDLEWARE_CLIENT_TOKEN",
        "SMART_EMBEDDER_CLIENT_TOKEN",
        "MCP_ALLOW_UNAUTHENTICATED_LOCAL",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def settings() -> Settings:
    """Build isolated Settings from the fixed test ENV, bypassing the repo .env."""
    # _env_file=None keeps pydantic-settings from reading the repo .env.
    return load_settings(_env_file=None, **ENV)


@pytest.fixture
def routes() -> FakeMiddleware:
    """A fresh fake middleware route table for one test."""
    return FakeMiddleware()


@pytest.fixture
async def client(settings, routes):
    """A MiddlewareClient wired to the fake middleware's MockTransport."""
    api = MiddlewareClient(settings, transport=routes.transport())
    yield api
    await api.aclose()


@pytest.fixture
async def mcp_server(settings, routes):
    """Build the real server wired to the fake middleware."""
    from smart_qdrant_mcp.server import build_server

    server, ctx = build_server(settings, transport=routes.transport())
    yield server
    await ctx.aclose()


@pytest.fixture
async def mcp_client(mcp_server):
    """An in-memory MCP client connected to the server under test.

    Entry/exit are managed manually rather than via ``async with ... yield``
    because ``Client.__aexit__`` closes an internal anyio ``TaskGroup``, and
    pytest-asyncio's async-generator-fixture teardown always resumes the
    generator in a *different* top-level asyncio Task than the one that ran
    setup (verified against the pinned ``pytest-asyncio==1.4.0`` and
    ``anyio==4.14.2``: `_wrap_asyncgen_fixture` creates one task for the
    ``__anext__()`` before the yield and a second, separate task for the
    ``__anext__()`` after it). anyio's ``CancelScope.__exit__`` requires the
    exiting task to match the entering task and raises ``RuntimeError:
    Attempted to exit cancel scope in a different task than it was entered
    in`` otherwise — unconditionally, for every test, regardless of what a
    tool does. The transport here is in-memory (``httpx.MockTransport``, no
    real socket or file descriptor), so this teardown-only failure is safe to
    swallow rather than propagate as a spurious test error.
    """
    from mcp.client import Client

    client_cm = Client(mcp_server)
    client = await client_cm.__aenter__()
    try:
        yield client
    finally:
        try:
            await client_cm.__aexit__(None, None, None)
        except RuntimeError as exc:
            if "different task" not in str(exc):
                raise


class CallText(str):
    """A tool call's joined text content, carrying the call's ``is_error`` flag.

    Subclassing ``str`` keeps every existing ``out == "..."`` / ``"..." in
    out`` assertion working unchanged across the suite's 250+ call sites;
    a failure-path test additionally asserts ``out.is_error is True``
    without needing a second fixture or a return-type change threaded
    through every one of them.
    """

    is_error: bool


@pytest.fixture
def call(mcp_client):
    """Call a tool and return its text content as a single ``CallText``.

    Every tool declares exactly one input argument named ``params`` (a
    Pydantic model); the SDK's generated argument schema therefore nests the
    tool's real fields one level under a ``params`` key on the wire
    (verified against the installed ``mcp==2.0.0``: a function parameter
    typed as a ``BaseModel`` is not flattened into the top-level arguments
    the way the v1 FastMCP reference docs describe). This is the one place
    that nesting is applied, so every test can keep writing
    ``call("tool_name", field=value)`` against the model's own fields.
    """

    async def _call(tool: str, **arguments: object) -> CallText:
        result = await mcp_client.call_tool(tool, {"params": arguments})
        text = "\n".join(
            block.text for block in result.content if getattr(block, "text", None)
        )
        out = CallText(text)
        out.is_error = bool(result.is_error)
        return out

    return _call

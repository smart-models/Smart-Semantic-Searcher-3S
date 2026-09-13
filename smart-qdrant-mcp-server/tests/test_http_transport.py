"""Real ASGI transport: bearer auth, Host allowlist, and one full tools/call.

The in-memory ``Client(mcp)`` fixture used everywhere else in this suite
(``mcp_client`` in ``conftest.py``) skips transport and authorization
entirely -- ``mcp.shared.memory``'s in-process session never runs
``AuthenticationMiddleware``, ``RequireAuthMiddleware``, or
``TransportSecurityMiddleware`` at all. That gap is exactly how a
``421 Invalid Host header`` reached production and was only caught by hand.

These tests instead drive the real ASGI app ``mcp.streamable_http_app(...)``
returns, through ``httpx.ASGITransport``, so the whole auth/transport-security
stack actually runs on every request -- still with no Docker and no live
middleware: the middleware stays faked with ``httpx.MockTransport``, exactly
as the rest of the suite fakes it.
"""

from __future__ import annotations

import json

import httpx
import pytest
from mcp_types import (
    CLIENT_CAPABILITIES_META_KEY,
    CLIENT_INFO_META_KEY,
    PROTOCOL_VERSION_META_KEY,
    SERVER_INFO_META_KEY,
)
from mcp.server.transport_security import TransportSecuritySettings

from smart_qdrant_mcp import __version__
from smart_qdrant_mcp.config import load_settings
from smart_qdrant_mcp.server import build_server

JSON_HEADERS = {
    "content-type": "application/json",
    "accept": "application/json, text/event-stream",
}
MODERN_VERSION = "2026-07-28"


def _modern_request(
    request_id: int,
    method: str,
    params: dict | None = None,
) -> tuple[dict, dict]:
    """Build one stateless 2026-07-28 HTTP exchange."""
    request_params = dict(params or {})
    request_params["_meta"] = {
        PROTOCOL_VERSION_META_KEY: MODERN_VERSION,
        CLIENT_INFO_META_KEY: {"name": "transport-test", "version": "1.0.0"},
        CLIENT_CAPABILITIES_META_KEY: {},
    }
    headers = {
        **JSON_HEADERS,
        "authorization": "Bearer tok-alice",
        "mcp-protocol-version": MODERN_VERSION,
        "mcp-method": method,
    }
    if method == "tools/call":
        headers["mcp-name"] = str(request_params["name"])
    return (
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": request_params,
        },
        headers,
    )


def _sse_json(body: str) -> dict:
    r"""Pull the one JSON-RPC payload out of an SSE-formatted response body.

        The session manager's default (``json_response=False``, matching
        ``create_app()``) replies over ``text/event-stream``, not plain JSON --
        a body shaped like ``event: message
    data: {...}

    ``.
    """
    for line in body.splitlines():
        if line.startswith("data: "):
            return json.loads(line.removeprefix("data: "))
    raise AssertionError(f"no SSE data line in response body: {body!r}")


@pytest.fixture
async def http_app(settings, routes):
    """Build the real ASGI app the container serves, wired to a fake middleware.

    Entering ``app.router.lifespan_context(app)`` is required: it starts the
    ``StreamableHTTPSessionManager``'s background task group, without which
    every request raises ``RuntimeError: Task group is not initialized``
    (verified against the installed SDK -- ``streamable_http_manager.py``).
    ``httpx.ASGITransport`` never sends ASGI lifespan events on its own, so
    this fixture drives them explicitly, the way ``TestClient`` would.

    Entry/exit are managed manually rather than ``async with ... yield``, for
    the same reason ``conftest.py``'s ``mcp_client`` fixture does: a
    pytest-asyncio async-generator fixture resumes teardown in a *different*
    top-level Task than the one that ran setup, and anyio's
    ``CancelScope.__exit__`` (entered here via the session manager's own
    task group) requires the exiting task to match the entering one. That
    mismatch is a teardown-only artifact of this in-process test harness,
    not a real failure, so it is swallowed the same narrow way.
    """
    mcp, ctx = build_server(settings, transport=routes.transport())
    app = mcp.streamable_http_app(
        transport_security=TransportSecuritySettings(
            allowed_hosts=settings.allowed_hosts,
            allowed_origins=settings.allowed_origins,
        )
    )
    lifespan_cm = app.router.lifespan_context(app)
    await lifespan_cm.__aenter__()
    try:
        yield app
    finally:
        try:
            await lifespan_cm.__aexit__(None, None, None)
        except RuntimeError as exc:
            if "different task" not in str(exc):
                raise
        # Idempotent (`httpx.AsyncClient.aclose` no-ops once already
        # CLOSED): the build_server-owned lifespan already closed this once
        # a session connected and disconnected; a request rejected before
        # any session exists (401/421 below) never reaches that path at all.
        await ctx.client.aclose()


@pytest.fixture
async def http_client(http_app, settings):
    transport = httpx.ASGITransport(app=http_app)
    base_url = f"http://{settings.bind_host}:{settings.port}"
    async with httpx.AsyncClient(transport=transport, base_url=base_url) as client:
        yield client


async def test_missing_bearer_token_is_refused_with_401(http_client):
    response = await http_client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        headers=JSON_HEADERS,
    )
    assert response.status_code == 401
    assert response.json()["error"] == "invalid_token"


async def test_wrong_bearer_token_is_refused_with_401(http_client):
    response = await http_client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        headers={**JSON_HEADERS, "authorization": "Bearer not-a-real-token"},
    )
    assert response.status_code == 401


async def test_host_outside_the_allowlist_is_refused_with_421(http_client):
    """A valid token is still required to reach Host validation at all.

    ``RequireAuthMiddleware`` wraps the endpoint that runs
    ``TransportSecurityMiddleware``'s Host check (verified: the auth
    middleware is Starlette-level, applied to every route, while the Host
    check runs inside the streamable-HTTP session's own request handling) --
    an unauthenticated request is refused with 401 before the Host header is
    ever inspected, so this exercises the 421 path specifically with a token
    that would otherwise succeed.
    """
    response = await http_client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        headers={
            **JSON_HEADERS,
            "authorization": "Bearer tok-alice",
            "host": "evil.example.com",
        },
    )
    assert response.status_code == 421
    assert "Invalid Host header" in response.text


async def test_oauth_discovery_document_is_not_mounted(http_client):
    """The protected-resource discovery route stays unmounted.

    ``resource_server_url`` is left unset (see ``auth.py``); this locks in
    that the well-known route it would otherwise mount stays absent, rather
    than silently reappearing on a future change.
    """
    response = await http_client.get("/.well-known/oauth-protected-resource")
    assert response.status_code == 404


async def test_oauth_mode_publishes_rfc9728_metadata_and_challenge(routes):
    oauth_settings = load_settings(
        _env_file=None,
        MCP_API_TOKENS="",
        MCP_AUTH_MODE="oauth",
        MCP_OAUTH_ISSUER_URL="https://auth.example.com",
        MCP_OAUTH_RESOURCE_SERVER_URL="http://127.0.0.1:6337/mcp",
        MCP_OAUTH_INTROSPECTION_URL="https://auth.example.com/introspect",
        MCP_OAUTH_CLIENT_ID="smart-qdrant-mcp",
        MCP_OAUTH_CLIENT_SECRET="oauth-secret",
        MCP_OAUTH_REQUIRED_SCOPES="qdrant:read",
        MCP_MIDDLEWARE_URL="http://middleware:6336",
        MIDDLEWARE_CLIENT_TOKEN="middleware-secret",
    )
    auth_transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"active": False})
    )
    mcp, _ctx = build_server(
        oauth_settings,
        transport=routes.transport(),
        auth_transport=auth_transport,
    )
    app = mcp.streamable_http_app(
        transport_security=TransportSecuritySettings(
            allowed_hosts=oauth_settings.allowed_hosts,
            allowed_origins=oauth_settings.allowed_origins,
        )
    )
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1:6337"
        ) as client:
            metadata = await client.get("/.well-known/oauth-protected-resource/mcp")
            unauthorized = await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "server/discover",
                    "params": {},
                },
                headers=JSON_HEADERS,
            )

    assert metadata.status_code == 200
    document = metadata.json()
    assert document["resource"] == "http://127.0.0.1:6337/mcp"
    assert document["authorization_servers"] == ["https://auth.example.com/"]
    assert document["scopes_supported"] == ["qdrant:read"]
    assert unauthorized.status_code == 401
    assert "resource_metadata=" in unauthorized.headers["www-authenticate"]


async def test_valid_token_tools_call_returns_the_result(http_client, routes):
    routes.add("GET", "/collections", body={"collections": ["docs", "notes"]})
    auth = {**JSON_HEADERS, "authorization": "Bearer tok-alice"}

    init = await http_client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "0.0.1"},
            },
        },
        headers=auth,
    )
    assert init.status_code == 200
    session_id = init.headers["mcp-session-id"]
    init_result = _sse_json(init.text)["result"]
    assert init_result["serverInfo"]["name"] == "smart_qdrant_mcp"

    session_headers = {**auth, "mcp-session-id": session_id}
    notified = await http_client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        headers=session_headers,
    )
    assert notified.status_code == 202

    call = await http_client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "qdrant_list_collections",
                "arguments": {"params": {}},
            },
        },
        headers=session_headers,
    )
    assert call.status_code == 200
    result = _sse_json(call.text)["result"]
    assert result["isError"] is False
    text = result["content"][0]["text"]
    assert "docs" in text
    assert "notes" in text


async def test_modern_discover_is_stateless_and_reports_app_version(http_client):
    body, headers = _modern_request(1, "server/discover")
    response = await http_client.post("/mcp", json=body, headers=headers)

    assert response.status_code == 200
    assert "mcp-session-id" not in response.headers
    result = response.json()["result"]
    server_info = result["_meta"][SERVER_INFO_META_KEY]
    assert server_info["name"] == "smart_qdrant_mcp"
    assert server_info["version"] == __version__
    assert MODERN_VERSION in result["supportedVersions"]


async def test_modern_resources_list_needs_no_initialize_or_session(http_client):
    body, headers = _modern_request(2, "resources/list")
    response = await http_client.post("/mcp", json=body, headers=headers)

    assert response.status_code == 200
    assert "mcp-session-id" not in response.headers
    resources = response.json()["result"]["resources"]
    assert len(resources) == 24
    assert all(resource.get("name") for resource in resources)


async def test_modern_tool_call_returns_text_and_structured_content(
    http_client, routes
):
    routes.add("GET", "/collections", body={"collections": ["docs", "notes"]})
    body, headers = _modern_request(
        3,
        "tools/call",
        {
            "name": "qdrant_list_collections",
            "arguments": {"params": {}},
        },
    )
    response = await http_client.post("/mcp", json=body, headers=headers)

    assert response.status_code == 200
    assert "mcp-session-id" not in response.headers
    result = response.json()["result"]
    assert result["isError"] is False
    assert "docs" in result["content"][0]["text"]
    assert result["structuredContent"]["items"] == ["docs", "notes"]
    assert result["structuredContent"]["total"] == 2


async def test_open_server_serves_without_an_authorization_header(routes):
    """MCP_ALLOW_UNAUTHENTICATED_LOCAL=true must actually serve, not 401.

    The whole ASGI stack runs here, so this is the test that would have caught
    an empty ``StaticTokenVerifier`` being installed for an empty ACL: that
    rejects a request with no ``Authorization`` header exactly as it rejects a
    wrong token.
    """
    open_settings = load_settings(
        _env_file=None,
        MCP_API_TOKENS="",
        MCP_ALLOW_UNAUTHENTICATED_LOCAL=True,
        MCP_MIDDLEWARE_URL="http://middleware:6336",
        MIDDLEWARE_CLIENT_TOKEN="middleware-secret",
    )
    mcp, ctx = build_server(open_settings, transport=routes.transport())
    app = mcp.streamable_http_app(
        transport_security=TransportSecuritySettings(
            allowed_hosts=open_settings.allowed_hosts,
            allowed_origins=open_settings.allowed_origins,
        )
    )
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1:6337"
        ) as client:
            body, headers = _modern_request(1, "initialize")
            headers.pop("authorization")
            response = await client.post("/mcp", json=body, headers=headers)
    await ctx.client.aclose()
    assert response.status_code != 401

"""Static and OAuth bearer-token verification."""

import logging
import time

import httpx
import pytest

from smart_qdrant_mcp.auth import (
    OAuthIntrospectionTokenVerifier,
    StaticTokenVerifier,
    build_auth_settings,
    build_token_verifier,
)
from smart_qdrant_mcp.config import load_settings

ENV = {
    "MCP_API_TOKENS": "tok-alice:alice,tok-bob:bob",
    "MCP_MIDDLEWARE_URL": "http://middleware:6336",
    "MIDDLEWARE_CLIENT_TOKEN": "middleware-secret",
}

OAUTH_ENV = {
    **ENV,
    "MCP_API_TOKENS": "",
    "MCP_AUTH_MODE": "oauth",
    "MCP_OAUTH_ISSUER_URL": "https://auth.example.com",
    "MCP_OAUTH_RESOURCE_SERVER_URL": "https://mcp.example.com/mcp",
    "MCP_OAUTH_INTROSPECTION_URL": "https://auth.example.com/introspect",
    "MCP_OAUTH_CLIENT_ID": "smart-qdrant-mcp",
    "MCP_OAUTH_CLIENT_SECRET": "oauth-secret",
    "MCP_OAUTH_AUDIENCE": "https://mcp.example.com/mcp",
    "MCP_OAUTH_REQUIRED_SCOPES": "qdrant:read,qdrant:write",
}


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    """Strip real MCP_*/middleware env vars so the repo-root .env never leaks in."""
    for var in (
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
        "MIDDLEWARE_CLIENT_TOKEN",
        "SMART_EMBEDDER_CLIENT_TOKEN",
        "MCP_ALLOW_UNAUTHENTICATED_LOCAL",
    ):
        monkeypatch.delenv(var, raising=False)


async def test_valid_token_resolves_to_label():
    verifier = StaticTokenVerifier({"tok-alice": "alice"})
    access = await verifier.verify_token("tok-alice")
    assert access is not None
    assert access.client_id == "alice"


async def test_unknown_token_returns_none():
    verifier = StaticTokenVerifier({"tok-alice": "alice"})
    assert await verifier.verify_token("nope") is None


def test_static_auth_uses_a_bracketed_ipv6_issuer():
    """A loopback IPv6 bind produces a syntactically valid issuer URL."""
    settings = load_settings(**ENV, MCP_BIND_HOST="::1", _env_file=None)

    auth = build_auth_settings(settings)
    assert str(auth.issuer_url).startswith("http://[::1]:6337")


async def test_empty_token_returns_none():
    verifier = StaticTokenVerifier({"tok-alice": "alice"})
    assert await verifier.verify_token("") is None


@pytest.mark.parametrize("token", ["TOK-ALICE", " tok-alice", "tok-alice "])
async def test_token_comparison_is_exact(token):
    verifier = StaticTokenVerifier({"tok-alice": "alice"})
    assert await verifier.verify_token(token) is None


def test_static_auth_does_not_advertise_oauth_resource_metadata():
    settings = load_settings(**ENV, _env_file=None)
    auth = build_auth_settings(settings)
    assert auth.resource_server_url is None
    assert auth.required_scopes == []


def test_oauth_settings_publish_real_resource_metadata():
    settings = load_settings(**OAUTH_ENV, _env_file=None)
    auth = build_auth_settings(settings)
    assert auth is not None
    assert str(auth.issuer_url).rstrip("/") == "https://auth.example.com"
    assert str(auth.resource_server_url) == "https://mcp.example.com/mcp"
    assert auth.required_scopes == ["qdrant:read", "qdrant:write"]


async def test_oauth_introspection_accepts_valid_token():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"].startswith("Basic ")
        assert b"token=opaque-token" in request.content
        return httpx.Response(
            200,
            json={
                "active": True,
                "iss": "https://auth.example.com/",
                "aud": ["https://mcp.example.com/mcp"],
                "client_id": "agent-42",
                "sub": "user-7",
                "scope": "qdrant:read qdrant:write",
                "exp": int(time.time()) + 60,
            },
        )

    settings = load_settings(**OAUTH_ENV, _env_file=None)
    verifier = OAuthIntrospectionTokenVerifier(
        settings, transport=httpx.MockTransport(handler)
    )
    try:
        access = await verifier.verify_token("opaque-token")
    finally:
        await verifier.aclose()

    assert access is not None
    assert access.client_id == "agent-42"
    assert access.subject == "user-7"
    assert access.scopes == ["qdrant:read", "qdrant:write"]
    assert access.resource == "https://mcp.example.com/mcp"


@pytest.mark.parametrize(
    "response",
    [
        {"active": False},
        {
            "active": True,
            "aud": "wrong-audience",
            "client_id": "agent-42",
        },
        {
            "active": True,
            "aud": "https://mcp.example.com/mcp",
            "client_id": "agent-42",
            "exp": 0,
        },
        {
            "active": True,
            "aud": "https://mcp.example.com/mcp",
            "client_id": "agent-42",
            "exp": "not-a-timestamp",
        },
    ],
)
async def test_oauth_introspection_fails_closed(response):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=response))
    settings = load_settings(**OAUTH_ENV, _env_file=None)
    verifier = OAuthIntrospectionTokenVerifier(settings, transport=transport)
    try:
        assert await verifier.verify_token("opaque-token") is None
    finally:
        await verifier.aclose()


# --- Final review finding 4: this is the only audit trail the service has
# (no OAuth, no per-request access log) -- README promises the label "is
# logged," so a successful verification must actually log it, and neither
# path may ever log the plaintext token. ---


async def test_successful_verification_logs_only_the_label(caplog):
    verifier = StaticTokenVerifier({"tok-alice": "alice"})
    with caplog.at_level(logging.INFO, logger="smart_qdrant_mcp.auth"):
        await verifier.verify_token("tok-alice")
    assert "alice" in caplog.text
    assert "tok-alice" not in caplog.text


async def test_failed_verification_does_not_log_the_presented_token(caplog):
    verifier = StaticTokenVerifier({"tok-alice": "alice"})
    with caplog.at_level(logging.INFO, logger="smart_qdrant_mcp.auth"):
        await verifier.verify_token("tok-super-secret-guess")
    assert "tok-super-secret-guess" not in caplog.text
    assert "failed" in caplog.text.lower()


@pytest.mark.parametrize("token", ["tokén-alice", "tok-aliceÿ", "🔑"])
async def test_non_ascii_token_is_rejected_not_raised(token):
    """A non-ASCII bearer value must fail authentication, not the request.

    HTTP headers reach the verifier latin-1 decoded, so any byte >= 0x80 in an
    Authorization header arrives here as a non-ASCII str — from an
    unauthenticated caller. hmac.compare_digest refuses those outright, so
    without a guard this path raises TypeError instead of returning None.
    """
    verifier = StaticTokenVerifier({"tok-alice": "alice"})
    assert await verifier.verify_token(token) is None


async def test_non_ascii_configured_token_still_matches_itself():
    """A non-ASCII token in MCP_API_TOKENS must still authenticate its owner.

    The guard must reject the comparison, not the token: an operator who
    configured a non-ASCII secret keeps working.
    """
    verifier = StaticTokenVerifier({"tokén-alice": "alice"})
    access = await verifier.verify_token("tokén-alice")
    assert access is not None
    assert access.client_id == "alice"
    assert await verifier.verify_token("tokén-bob") is None


async def test_oauth_non_ascii_audience_is_rejected_not_raised():
    """A non-ASCII `aud` claim must fail closed like any other mismatch."""

    def handler(request: httpx.Request) -> httpx.Response:
        # No `iss`: the issuer check accepts an absent claim, so the audience
        # comparison below is the branch under test rather than an early exit.
        return httpx.Response(
            200,
            json={
                "active": True,
                "aud": "https://mcp.example.com/mcp/café",
                "client_id": "smart-qdrant-mcp",
                "sub": "user-1",
            },
        )

    settings = load_settings(**OAUTH_ENV, _env_file=None)
    verifier = OAuthIntrospectionTokenVerifier(
        settings, transport=httpx.MockTransport(handler)
    )
    try:
        assert await verifier.verify_token("opaque-token") is None
    finally:
        await verifier.aclose()


# --- A deliberately open configuration installs no verifier at all:
# StaticTokenVerifier({}) would answer 401 to every request, including one
# with no Authorization header, which is the opposite of "auth disabled". ---


def _open_settings():
    """Build settings for an empty inbound list behind the explicit flag."""
    return load_settings(
        **{**ENV, "MCP_API_TOKENS": ""},
        MCP_ALLOW_UNAUTHENTICATED_LOCAL=True,
        _env_file=None,
    )


def test_open_configuration_installs_no_verifier():
    settings = _open_settings()
    assert build_token_verifier(settings) is None
    assert build_auth_settings(settings) is None


def test_configured_tokens_still_install_a_verifier():
    settings = load_settings(**ENV, _env_file=None)
    assert build_token_verifier(settings) is not None
    assert build_auth_settings(settings) is not None

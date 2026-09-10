"""Settings parsing and startup validation."""

import pytest

from smart_qdrant_mcp.config import (
    ConfigError,
    load_settings,
    parse_api_tokens,
    reject_removed_env_names,
)

# Every construction goes through load_settings, never Settings(...) directly:
# pydantic wraps any exception raised inside a validator in a ValidationError, so
# ConfigError reaches a caller only through the loader that unwraps it.

BASE_ENV = {
    "MCP_API_TOKENS": "tok-alice:alice,tok-bob:bob",
    "MCP_MIDDLEWARE_URL": "http://middleware:6336",
    "MIDDLEWARE_CLIENT_TOKEN": "middleware-secret",
}

OAUTH_ENV = {
    **BASE_ENV,
    "MCP_API_TOKENS": "",
    "MCP_AUTH_MODE": "oauth",
    "MCP_OAUTH_ISSUER_URL": "https://auth.example.com",
    "MCP_OAUTH_RESOURCE_SERVER_URL": "https://mcp.example.com/mcp",
    "MCP_OAUTH_INTROSPECTION_URL": "https://auth.example.com/introspect",
    "MCP_OAUTH_CLIENT_ID": "smart-qdrant-mcp",
    "MCP_OAUTH_CLIENT_SECRET": "oauth-secret",
    "MCP_OAUTH_REQUIRED_SCOPES": "qdrant:read, qdrant:write",
}


def test_parse_api_tokens_maps_token_to_label():
    assert parse_api_tokens("tok-a:alice,tok-b:bob") == {
        "tok-a": "alice",
        "tok-b": "bob",
    }


def test_parse_api_tokens_rejects_missing_label():
    with pytest.raises(ConfigError, match="token:label"):
        parse_api_tokens("tok-a,tok-b:bob")


def test_parse_api_tokens_rejects_duplicate_token():
    with pytest.raises(ConfigError, match="duplicate"):
        parse_api_tokens("tok-a:alice,tok-a:bob")


def test_settings_defaults_to_loopback_bind():
    settings = load_settings(**BASE_ENV, _env_file=None)
    assert settings.bind_host == "127.0.0.1"
    assert settings.port == 6337


def test_settings_rejects_empty_token_list():
    env = {**BASE_ENV, "MCP_API_TOKENS": ""}
    with pytest.raises(ConfigError, match="MCP_API_TOKENS"):
        load_settings(**env, _env_file=None)


def test_oauth_mode_does_not_require_static_tokens():
    settings = load_settings(**OAUTH_ENV, _env_file=None)
    assert settings.api_tokens == {}
    assert settings.oauth_expected_audience == "https://mcp.example.com/mcp"
    assert settings.oauth_required_scopes == ["qdrant:read", "qdrant:write"]


def test_oauth_mode_requires_complete_provider_configuration():
    env = {**OAUTH_ENV, "MCP_OAUTH_CLIENT_SECRET": ""}
    with pytest.raises(ConfigError, match="MCP_OAUTH_CLIENT_SECRET"):
        load_settings(**env, _env_file=None)


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("MCP_OAUTH_ISSUER_URL", "http://auth.example.com"),
        ("MCP_OAUTH_RESOURCE_SERVER_URL", "http://mcp.example.com/mcp"),
        ("MCP_OAUTH_INTROSPECTION_URL", "http://auth.example.com/introspect"),
    ],
)
def test_oauth_rejects_plain_http_away_from_loopback(variable, value):
    env = {**OAUTH_ENV, variable: value}

    with pytest.raises(ConfigError, match=rf"{variable}.*https"):
        load_settings(**env, _env_file=None)


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "127.10.20.30", "[::1]"])
def test_oauth_accepts_http_for_explicit_loopback_development(host):
    env = {
        **OAUTH_ENV,
        "MCP_OAUTH_ISSUER_URL": f"http://{host}",
        "MCP_OAUTH_RESOURCE_SERVER_URL": f"http://{host}/mcp",
        "MCP_OAUTH_INTROSPECTION_URL": f"http://{host}/introspect",
    }

    settings = load_settings(**env, _env_file=None)

    assert settings.oauth_introspection_url.scheme == "http"


def test_settings_rejects_token_reused_from_middleware():
    env = {**BASE_ENV, "MCP_API_TOKENS": "middleware-secret:alice"}
    with pytest.raises(ConfigError, match="distinct"):
        load_settings(**env, _env_file=None)


def test_settings_rejects_token_reused_from_embedder():
    env = {
        **BASE_ENV,
        "SMART_EMBEDDER_CLIENT_TOKEN": "embedder-secret",
        "MCP_API_TOKENS": "embedder-secret:alice",
    }
    with pytest.raises(ConfigError, match="distinct"):
        load_settings(**env, _env_file=None)


def test_allowed_hosts_always_include_bind_host():
    settings = load_settings(
        **BASE_ENV, MCP_ALLOWED_HOSTS="vpn-box.local", _env_file=None
    )
    assert "127.0.0.1" in settings.allowed_hosts
    assert "vpn-box.local" in settings.allowed_hosts


def test_allowed_hosts_cover_the_port_bearing_header_a_client_really_sends():
    """A live client sent Host: 127.0.0.1:6337 and the server answered 421.

    The SDK matches the Host header exactly or against a name:* wildcard, so
    publishing only the bare name rejects every real browser and MCP client.
    """
    settings = load_settings(
        **BASE_ENV, MCP_ALLOWED_HOSTS="vpn-box.local", _env_file=None
    )
    for name in ("127.0.0.1", "localhost", "vpn-box.local"):
        assert name in settings.allowed_hosts
        assert f"{name}:*" in settings.allowed_hosts


def test_allowed_hosts_do_not_double_wildcard_an_entry():
    settings = load_settings(**BASE_ENV, MCP_ALLOWED_HOSTS="host:*", _env_file=None)
    assert "host:*" in settings.allowed_hosts
    assert "host:*:*" not in settings.allowed_hosts


# --- Final review finding 5: MCP_LISTEN_HOST is the address uvicorn actually
# binds to, independent of MCP_BIND_HOST (the published/policed address,
# which never reaches a socket). ---


def test_listen_host_defaults_to_loopback():
    settings = load_settings(**BASE_ENV, _env_file=None)
    assert settings.listen_host == "127.0.0.1"


def test_listen_host_accepts_a_wildcard_for_container_use():
    # Unlike MCP_BIND_HOST, a wildcard here is not refused: the smart-qdrant-mcp
    # Compose service sets exactly this so uvicorn binds every interface
    # inside the container's own private network namespace.
    settings = load_settings(**BASE_ENV, MCP_LISTEN_HOST="0.0.0.0", _env_file=None)
    assert settings.listen_host == "0.0.0.0"


def test_listen_host_rejects_a_blank_value():
    with pytest.raises(ConfigError, match="MCP_LISTEN_HOST"):
        load_settings(**BASE_ENV, MCP_LISTEN_HOST="   ", _env_file=None)


# --- Multi-token contract: label charset, reserved '@' prefix, and error text
# that names the variable it was given without ever echoing a token. ---


def test_parse_api_tokens_accepts_multiple_entries_and_strips_spaces():
    assert parse_api_tokens("t1:alice, t2:bob") == {"t1": "alice", "t2": "bob"}


def test_parse_api_tokens_skips_empty_entries():
    assert parse_api_tokens("t1:alice,,t2:bob") == {"t1": "alice", "t2": "bob"}


def test_parse_api_tokens_rejects_entry_without_separator():
    with pytest.raises(ConfigError, match="is not token:label"):
        parse_api_tokens("t1alice")


def test_parse_api_tokens_rejects_empty_halves():
    with pytest.raises(ConfigError, match="is not token:label"):
        parse_api_tokens(":alice")
    with pytest.raises(ConfigError, match="is not token:label"):
        parse_api_tokens("t1:")


def test_parse_api_tokens_rejects_duplicate_token_pair():
    with pytest.raises(ConfigError, match="duplicate token"):
        parse_api_tokens("t1:alice,t1:bob")


def test_parse_api_tokens_rejects_label_containing_colon():
    with pytest.raises(ConfigError, match=r"\[A-Za-z0-9\._-\]\+"):
        parse_api_tokens("my:secret:alice")


def test_parse_api_tokens_rejects_reserved_at_prefix():
    with pytest.raises(ConfigError, match="reserved"):
        parse_api_tokens("t1:@mcp")


def test_parse_api_tokens_error_names_the_variable_it_was_given():
    with pytest.raises(ConfigError, match="MIDDLEWARE_API_TOKENS"):
        parse_api_tokens("broken", variable="MIDDLEWARE_API_TOKENS")


def test_parse_api_tokens_never_echoes_the_token_for_a_label_error():
    with pytest.raises(ConfigError) as excinfo:
        parse_api_tokens("s3cret:bad label")
    assert "s3cret" not in str(excinfo.value)


# --- Optional authentication: an empty list is legitimate only behind the
# per-service flag, and only on a loopback publish address. ---


def _open_env(**overrides):
    """Build a BASE_ENV variant, defaulting to an empty inbound list."""
    env = {**BASE_ENV, "MCP_API_TOKENS": ""}
    env.update(overrides)
    return env


def test_empty_list_without_flag_is_refused():
    with pytest.raises(ConfigError, match="MCP_ALLOW_UNAUTHENTICATED_LOCAL"):
        load_settings(**_open_env(), _env_file=None)


def test_empty_list_with_flag_boots():
    settings = load_settings(
        **_open_env(MCP_ALLOW_UNAUTHENTICATED_LOCAL=True), _env_file=None
    )
    assert settings.api_tokens == {}


def test_flag_with_public_bind_host_is_refused():
    with pytest.raises(ConfigError, match="MCP_BIND_HOST"):
        load_settings(
            **_open_env(
                MCP_ALLOW_UNAUTHENTICATED_LOCAL=True,
                MCP_BIND_HOST="192.168.1.10",
            ),
            _env_file=None,
        )


@pytest.mark.parametrize(
    "host", ["10.1.2.3", "172.16.0.1", "172.31.255.254", "192.168.1.10"]
)
def test_authenticated_mcp_accepts_rfc1918_publish_addresses(host):
    settings = load_settings(**BASE_ENV, MCP_BIND_HOST=host, _env_file=None)

    assert settings.bind_host == host


@pytest.mark.parametrize(
    "host",
    [
        "8.8.8.8",
        "192.0.2.1",
        "169.254.1.1",
        "224.0.0.1",
        "example.com",
        "::ffff:10.0.0.1",
        "fd00::1",
    ],
)
def test_authenticated_mcp_rejects_non_private_publish_addresses(host):
    with pytest.raises(ConfigError, match="loopback or private RFC 1918"):
        load_settings(**BASE_ENV, MCP_BIND_HOST=host, _env_file=None)


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "::", "[::]", "::0", "::ffff:0.0.0.0", " 0.0.0.0 ", "0.0.0.0 "],
)
def test_wildcard_bind_is_refused_in_every_spelling(host):
    # The unspecified address must be refused semantically, not by matching the
    # two canonical strings: bracketed, compressed, IPv4-mapped and whitespace-
    # padded spellings all name the same "bind every interface" intent.
    with pytest.raises(ConfigError, match="must not bind all interfaces"):
        load_settings(**BASE_ENV, MCP_BIND_HOST=host, _env_file=None)


def test_loopback_bind_is_not_treated_as_a_wildcard():
    settings = load_settings(**BASE_ENV, MCP_BIND_HOST="127.0.0.1", _env_file=None)
    assert settings.bind_host == "127.0.0.1"


def test_ipv6_loopback_is_allowed_and_published_with_brackets():
    """IPv6 loopback has a valid URL and Host-header representation."""
    settings = load_settings(**BASE_ENV, MCP_BIND_HOST="::1", _env_file=None)

    assert settings.bind_host == "::1"
    assert "[::1]" in settings.allowed_hosts
    assert "[::1]:*" in settings.allowed_hosts


def test_flag_is_ignored_in_oauth_mode():
    settings = load_settings(**OAUTH_ENV, _env_file=None)
    assert settings.auth_mode == "oauth"
    assert settings.allow_unauthenticated_local is False


def test_missing_middleware_client_token_is_accepted():
    """The middleware may itself be open; the preflight enforces the pair."""
    env = {**BASE_ENV, "MIDDLEWARE_CLIENT_TOKEN": ""}
    settings = load_settings(**env, _env_file=None)
    assert settings.middleware_client_token == ""


def test_removed_name_in_the_process_environment_is_refused(monkeypatch):
    monkeypatch.setenv("MIDDLEWARE_API_TOKEN", "stale")
    with pytest.raises(ConfigError, match="MIDDLEWARE_CLIENT_TOKEN"):
        reject_removed_env_names(None)


def test_removed_name_in_the_env_file_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("ALLOW_UNAUTHENTICATED_LOCAL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nexport ALLOW_UNAUTHENTICATED_LOCAL=true\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="SMART_EMBEDDER_ALLOW_UNAUTHENTICATED_LOCAL"):
        reject_removed_env_names(env_file)


def test_env_file_scan_ignores_commented_out_names(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("#MIDDLEWARE_API_TOKEN=old\n", encoding="utf-8")
    reject_removed_env_names(env_file)


def test_env_file_scan_ignores_an_empty_assignment(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("MIDDLEWARE_API_TOKEN=\n", encoding="utf-8")
    reject_removed_env_names(env_file)

"""Tests for the cross-service token preflight."""

from scripts.verify_token_config import check_token_config

BASE = {
    "MIDDLEWARE_API_TOKENS": "mw1:alice",
    "SMART_EMBEDDER_API_TOKENS": "emb1:ops",
    "MCP_API_TOKENS": "mcp1:agent",
    "MIDDLEWARE_CLIENT_TOKEN": "mw-service",
    "SMART_EMBEDDER_CLIENT_TOKEN": "emb-service",
    "MCP_SERVER_ENABLED": "true",
    "EMBEDDER_ONLY": "false",
    "MCP_BIND_HOST": "127.0.0.1",
}


def _values(**overrides):
    """Build a resolved-environment mapping from the valid baseline."""
    values = dict(BASE)
    values.update(overrides)
    return values


def test_a_valid_configuration_produces_no_errors():
    assert check_token_config(_values()) == []


def test_the_same_token_in_two_inbound_lists_is_an_overlap():
    errors = check_token_config(_values(MCP_API_TOKENS="mw1:agent"))
    assert any("overlap" in error for error in errors)


def test_a_client_token_colliding_with_another_services_list_is_an_overlap():
    errors = check_token_config(_values(MCP_API_TOKENS="emb-service:agent"))
    assert any("overlap" in error for error in errors)


def test_missing_middleware_client_token_is_an_error_when_mcp_is_enabled():
    errors = check_token_config(_values(MIDDLEWARE_CLIENT_TOKEN=""))
    assert any("MIDDLEWARE_CLIENT_TOKEN" in error for error in errors)


def test_missing_middleware_client_token_is_ignored_when_mcp_is_disabled():
    assert (
        check_token_config(
            _values(MIDDLEWARE_CLIENT_TOKEN="", MCP_SERVER_ENABLED="false")
        )
        == []
    )


def test_missing_embedder_client_token_is_ignored_in_embedder_only_mode():
    assert (
        check_token_config(
            _values(
                SMART_EMBEDDER_CLIENT_TOKEN="",
                EMBEDDER_ONLY="true",
                MIDDLEWARE_CLIENT_TOKEN="",
                MCP_SERVER_ENABLED="false",
            )
        )
        == []
    )


def test_unauthenticated_mcp_on_a_public_bind_host_is_an_error():
    errors = check_token_config(
        _values(
            MCP_API_TOKENS="",
            MCP_ALLOW_UNAUTHENTICATED_LOCAL="true",
            MCP_BIND_HOST="192.168.1.10",
        )
    )
    assert any("MCP_BIND_HOST" in error for error in errors)


def test_unauthenticated_mcp_on_loopback_is_accepted():
    assert (
        check_token_config(
            _values(MCP_API_TOKENS="", MCP_ALLOW_UNAUTHENTICATED_LOCAL="true")
        )
        == []
    )


def test_empty_middleware_tokens_without_opt_out_is_an_error():
    errors = check_token_config(
        _values(MIDDLEWARE_API_TOKENS="", MIDDLEWARE_CLIENT_TOKEN="")
    )
    assert any(
        "MIDDLEWARE_API_TOKENS" in error
        and "MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL" in error
        for error in errors
    )


def test_empty_middleware_tokens_with_opt_out_is_accepted():
    assert (
        check_token_config(
            _values(
                MIDDLEWARE_API_TOKENS="",
                MIDDLEWARE_CLIENT_TOKEN="",
                MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL="true",
            )
        )
        == []
    )


def test_empty_middleware_tokens_is_ignored_in_embedder_only_mode():
    assert (
        check_token_config(
            _values(
                MIDDLEWARE_API_TOKENS="",
                MIDDLEWARE_CLIENT_TOKEN="",
                SMART_EMBEDDER_CLIENT_TOKEN="",
                EMBEDDER_ONLY="true",
                MCP_SERVER_ENABLED="false",
            )
        )
        == []
    )


def test_a_middleware_client_token_alone_satisfies_the_boot_check():
    # The middleware auto-accepts MIDDLEWARE_CLIENT_TOKEN under a reserved
    # label, so an empty MIDDLEWARE_API_TOKENS with a client token present is
    # a valid caller-only lockdown that boots without the opt-out flag.
    assert (
        check_token_config(
            _values(MIDDLEWARE_API_TOKENS="", MIDDLEWARE_CLIENT_TOKEN="mw-service")
        )
        == []
    )


def test_an_embedder_client_token_alone_satisfies_the_boot_check():
    # The embedder auto-accepts SMART_EMBEDDER_CLIENT_TOKEN, so an empty
    # SMART_EMBEDDER_API_TOKENS with a client token present boots fine.
    assert (
        check_token_config(
            _values(
                SMART_EMBEDDER_API_TOKENS="",
                SMART_EMBEDDER_CLIENT_TOKEN="emb-service",
            )
        )
        == []
    )


def test_empty_embedder_tokens_without_opt_out_is_an_error():
    errors = check_token_config(
        _values(SMART_EMBEDDER_API_TOKENS="", SMART_EMBEDDER_CLIENT_TOKEN="")
    )
    assert any(
        "SMART_EMBEDDER_API_TOKENS" in error
        and "SMART_EMBEDDER_ALLOW_UNAUTHENTICATED_LOCAL" in error
        for error in errors
    )


def test_empty_embedder_tokens_applies_even_in_embedder_only_mode():
    errors = check_token_config(
        _values(
            SMART_EMBEDDER_API_TOKENS="",
            MIDDLEWARE_CLIENT_TOKEN="",
            SMART_EMBEDDER_CLIENT_TOKEN="",
            EMBEDDER_ONLY="true",
            MCP_SERVER_ENABLED="false",
        )
    )
    assert any("SMART_EMBEDDER_API_TOKENS" in error for error in errors)


def test_empty_mcp_tokens_without_opt_out_is_an_error():
    errors = check_token_config(_values(MCP_API_TOKENS=""))
    assert any(
        "MCP_API_TOKENS" in error and "MCP_ALLOW_UNAUTHENTICATED_LOCAL" in error
        for error in errors
    )


def test_empty_mcp_tokens_is_ignored_when_mcp_is_disabled():
    errors = check_token_config(_values(MCP_API_TOKENS="", MCP_SERVER_ENABLED="false"))
    assert all("MCP_API_TOKENS" not in error for error in errors)


def test_empty_mcp_tokens_is_ignored_under_oauth_mode():
    assert check_token_config(_values(MCP_API_TOKENS="", MCP_AUTH_MODE="oauth")) == []


def test_uppercase_oauth_mode_does_not_skip_the_empty_list_check():
    # MCP_AUTH_MODE is a case-sensitive Literal["static", "oauth"] in the
    # service; "OAUTH" is not the valid oauth mode, so the empty-list rule still
    # applies and the preflight must flag it rather than silently passing.
    errors = check_token_config(_values(MCP_API_TOKENS="", MCP_AUTH_MODE="OAUTH"))
    assert any("MCP_API_TOKENS" in error for error in errors)


def test_a_removed_variable_name_is_an_error():
    errors = check_token_config(_values(MIDDLEWARE_API_TOKEN="stale"))
    assert any("MIDDLEWARE_API_TOKEN" in error for error in errors)


def test_a_malformed_entry_is_reported_without_echoing_the_token():
    errors = check_token_config(_values(MCP_API_TOKENS="s3cret"))
    assert errors
    assert all("s3cret" not in error for error in errors)


def test_an_overlap_message_never_reports_the_shared_value():
    errors = check_token_config(_values(MCP_API_TOKENS="mw1:agent"))
    assert all("mw1" not in error for error in errors)


def test_cli_rejects_a_line_without_a_separator(monkeypatch, capsys):
    import io

    from scripts.verify_token_config import main

    monkeypatch.setattr("sys.stdin", io.StringIO("MIDDLEWARE_API_TOKENS\n"))
    assert main() == 2
    assert "NAME=VALUE" in capsys.readouterr().err

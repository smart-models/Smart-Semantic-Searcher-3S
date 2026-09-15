"""Tests for Settings environment parsing and cross-field validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from qdrant_middleware.core.config import Settings, reject_removed_env_names


# Real environment variables must win over .env values in container/CI deployments.
def test_settings_env_wins_over_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    env_file = tmp_path / ".env"
    env_file.write_text("QDRANT_URL=http://from-dotenv:6333\n", encoding="utf-8")
    monkeypatch.setenv("QDRANT_URL", "http://from-env:6333")

    settings = Settings(_env_file=env_file)

    assert str(settings.qdrant_url) == "http://from-env:6333/"


def test_settings_reads_smart_embedder_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("SMART_EMBEDDER_CLIENT_TOKEN", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SMART_EMBEDDER_CLIENT_TOKEN=embedder-token\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.smart_embedder_client_token_value == "embedder-token"


def test_settings_reads_namespaced_middleware_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    namespaced_names = (
        "MIDDLEWARE_PORT",
        "MIDDLEWARE_API_TOKENS",
        "MIDDLEWARE_REQUEST_TIMEOUT",
        "MIDDLEWARE_RATE_LIMIT_REQUESTS_PER_MINUTE",
        "MIDDLEWARE_RATE_LIMIT_BURST_SIZE",
    )
    for name in namespaced_names:
        monkeypatch.delenv(name, raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                "MIDDLEWARE_PORT=7444",
                "MIDDLEWARE_API_TOKENS=middleware-token:alice",
                "MIDDLEWARE_REQUEST_TIMEOUT=45",
                "MIDDLEWARE_RATE_LIMIT_REQUESTS_PER_MINUTE=450",
                "MIDDLEWARE_RATE_LIMIT_BURST_SIZE=75",
            )
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.api_port == 7444
    assert settings.api_tokens == {"middleware-token": "alice"}
    assert settings.request_timeout == 45
    assert settings.rate_limit_requests_per_minute == 450
    assert settings.rate_limit_burst_size == 75


def test_settings_rejects_invalid_fusion():
    with pytest.raises(ValidationError):
        Settings(QUERY_DEFAULT_FUSION="invalid")


def test_settings_validates_limit_invariants():
    with pytest.raises(ValidationError):
        Settings(QUERY_DEFAULT_LIMIT=100, QUERY_MAX_LIMIT=10)


def test_settings_rejects_candidate_limits_below_result_limits():
    with pytest.raises(ValidationError, match="QUERY_DEFAULT_CANDIDATE_LIMIT"):
        Settings(QUERY_DEFAULT_LIMIT=20, QUERY_DEFAULT_CANDIDATE_LIMIT=10)

    with pytest.raises(ValidationError, match="QUERY_MAX_CANDIDATE_LIMIT"):
        Settings(QUERY_MAX_LIMIT=300, QUERY_MAX_CANDIDATE_LIMIT=200)


def test_settings_builds_public_paths_from_swagger_exposure():
    settings = Settings(EXPOSE_MIDDLEWARE_SWAGGER=False)

    assert "/docs" not in settings.public_auth_paths
    assert "/openapi.json" not in settings.public_auth_paths


def test_exposure_flags_default_to_true():
    settings = Settings()

    assert settings.expose_frontend is True
    assert settings.expose_middleware_swagger is True


@pytest.mark.parametrize("value", ["true", "TRUE", "yes", "1", "on"])
def test_exposure_flags_parse_true_values(value: str):
    settings = Settings(
        EXPOSE_FRONTEND=value,
        EXPOSE_MIDDLEWARE_SWAGGER=value,
    )

    assert settings.expose_frontend is True
    assert settings.expose_middleware_swagger is True


@pytest.mark.parametrize("value", ["false", "FALSE", "no", "0", "off"])
def test_exposure_flags_parse_false_values(value: str):
    settings = Settings(
        EXPOSE_FRONTEND=value,
        EXPOSE_MIDDLEWARE_SWAGGER=value,
    )

    assert settings.expose_frontend is False
    assert settings.expose_middleware_swagger is False


def test_query_filter_allowlist_defaults_to_empty():
    settings = Settings()

    assert settings.query_filter_field_allowlist_raw == ""
    assert settings.query_filter_field_allowlist == frozenset()


def test_query_filter_allowlist_parses_comma_separated_fields():
    settings = Settings(QUERY_FILTER_FIELD_ALLOWLIST="source_id, type, Header 1")

    assert settings.query_filter_field_allowlist == frozenset(
        {"source_id", "type", "Header 1"}
    )


def test_query_filter_limits_are_configurable():
    settings = Settings(QUERY_FILTER_MAX_DEPTH=10, QUERY_FILTER_MAX_CONDITIONS=25)

    assert settings.query_filter_max_depth == 10
    assert settings.query_filter_max_conditions == 25


def test_balanced_optimizer_defaults_align_memmap_and_indexing_thresholds():
    settings = Settings()

    assert settings.indexing_threshold == 20_000
    assert settings.memmap_threshold == settings.indexing_threshold


def test_export_settings_have_documented_defaults():
    settings = Settings()

    assert settings.export_data_root.name == "exports"
    assert settings.export_data_root.parent == settings.ingest_data_root.parent
    assert settings.export_max_points == 50_000


def test_export_settings_read_environment(monkeypatch):
    monkeypatch.setenv("EXPORT_DATA_ROOT", "/srv/dumps")
    monkeypatch.setenv("EXPORT_MAX_POINTS", "1234")

    settings = Settings()

    assert settings.export_data_root == Path("/srv/dumps")
    assert settings.export_max_points == 1234


def test_export_route_is_exempt_from_the_request_timeout():
    # A whole-collection walk is many sequential scroll pages; the 30s default
    # would 504 it. The ingest routes are exempt for the same reason.
    settings = Settings()

    assert "/points/export" in settings.timeout_exempt_paths
    assert "/points/ingest" in settings.timeout_exempt_paths


# --- Multi-token inbound ACL: operator entries plus the one credential the
# MCP server presents, which the middleware auto-accepts under '@mcp'. ---


def _settings(**overrides):
    """Build Settings from a minimal valid token configuration."""
    values = {"MIDDLEWARE_API_TOKENS": "t1:alice"}
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_effective_acl_maps_each_token_to_its_label():
    assert _settings(MIDDLEWARE_API_TOKENS="t1:alice,t2:bob").api_tokens == {
        "t1": "alice",
        "t2": "bob",
    }


def test_service_token_is_accepted_under_the_reserved_label():
    settings = _settings(SERVICE_TOKEN="mcp-secret")
    assert settings.api_tokens == {"t1": "alice", "mcp-secret": "@mcp"}


def test_service_token_duplicating_an_operator_entry_is_refused():
    with pytest.raises(ValueError, match="SERVICE_TOKEN"):
        _settings(MIDDLEWARE_API_TOKENS="t1:alice", SERVICE_TOKEN="t1")


def test_empty_token_list_without_flag_is_refused():
    with pytest.raises(ValueError, match="MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL"):
        _settings(MIDDLEWARE_API_TOKENS="")


def test_bind_host_defaults_to_loopback():
    assert _settings().bind_host == "127.0.0.1"


@pytest.mark.parametrize(
    "host", ["127.0.0.1", "::1", "10.10.141.10", "172.16.0.5", "192.168.1.10"]
)
def test_bind_host_accepts_loopback_or_a_private_ipv4(host):
    assert _settings(MIDDLEWARE_BIND_HOST=host).bind_host == host


@pytest.mark.parametrize(
    "host", ["0.0.0.0", "::", "8.8.8.8", "169.254.1.1", "vpn-box.local", ""]
)
def test_bind_host_refuses_wildcard_public_link_local_or_hostname(host):
    with pytest.raises(ValueError, match="MIDDLEWARE_BIND_HOST"):
        _settings(MIDDLEWARE_BIND_HOST=host)


def test_unauthenticated_middleware_on_a_private_bind_host_is_refused():
    with pytest.raises(ValueError, match="MIDDLEWARE_BIND_HOST"):
        _settings(
            MIDDLEWARE_API_TOKENS="",
            MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
            MIDDLEWARE_BIND_HOST="10.10.141.10",
        )


def test_invalid_token_configuration_hides_the_supplied_value():
    """Configuration errors must not render a bearer token in startup output."""
    sentinel = "audit-synthetic-secret-123456"
    with pytest.raises(ValidationError) as caught:
        _settings(MIDDLEWARE_API_TOKENS=sentinel)

    rendered = str(caught.value)
    assert sentinel not in rendered
    assert "input_value" not in rendered
    assert "MIDDLEWARE_API_TOKENS" in rendered


def test_empty_token_list_with_flag_disables_auth():
    settings = _settings(
        MIDDLEWARE_API_TOKENS="", MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True
    )
    assert settings.api_tokens == {}


def test_empty_token_list_with_flag_is_allowed_on_a_public_bind_host():
    """Under Compose QDRANT_API_HOST is 0.0.0.0, inside a private namespace.

    The old "a non-loopback bind requires a token" invariant made the opt-out
    unreachable in the only deployment that ships; the published port stays
    bound to 127.0.0.1 on the host either way.
    """
    settings = _settings(
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        QDRANT_API_HOST="0.0.0.0",
    )
    assert settings.api_tokens == {}


def test_service_token_alone_satisfies_the_non_empty_requirement():
    settings = _settings(MIDDLEWARE_API_TOKENS="", SERVICE_TOKEN="mcp-secret")
    assert settings.api_tokens == {"mcp-secret": "@mcp"}


def test_removed_env_name_is_refused(monkeypatch):
    monkeypatch.setenv("MIDDLEWARE_API_TOKEN", "stale")
    with pytest.raises(ValueError, match="MIDDLEWARE_API_TOKENS"):
        reject_removed_env_names(None)

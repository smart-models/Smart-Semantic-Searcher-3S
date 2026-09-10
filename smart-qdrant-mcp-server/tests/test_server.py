"""Server assembly: registration, naming, and annotation completeness."""

import re
from pathlib import Path

import pytest

from smart_qdrant_mcp import __version__
from smart_qdrant_mcp.tool_map import TOOL_ROUTES

REQUIRED_ANNOTATIONS = {
    "title",
    "readOnlyHint",
    "destructiveHint",
    "idempotentHint",
    "openWorldHint",
}


async def test_server_is_named_smart_qdrant_mcp(mcp_server):
    assert mcp_server.name == "smart_qdrant_mcp"


async def test_server_reports_its_application_version(mcp_server):
    assert mcp_server.version == __version__
    assert mcp_server.version != "unknown"


def test_application_version_matches_container_metadata_default():
    dockerfile = Path(__file__).resolve().parents[2] / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")
    match = re.search(r"^ARG SMART_QDRANT_MCP_VERSION=(\S+)$", text, re.MULTILINE)
    assert match is not None
    assert match.group(1) == __version__


async def test_server_has_instructions(mcp_server):
    """The server must expose instructions so clients know how to use it.

    ``MCPServer`` accepts an ``instructions`` string that surfaces in the
    ``initialize`` handshake. Without it, an agent connecting to the server
    gets no guidance on response_format, confirm tokens, pagination, or the
    workflow prompts — it would have to discover all of that by reading tool
    descriptions one by one.
    """
    assert mcp_server.instructions, "server must declare instructions"
    assert "response_format" in mcp_server.instructions
    assert "confirm" in mcp_server.instructions
    assert "pagination" in mcp_server.instructions.lower()


async def test_every_declared_tool_is_registered(mcp_client):
    listed = {tool.name for tool in (await mcp_client.list_tools()).tools}
    assert listed == set(TOOL_ROUTES)


async def test_every_tool_has_a_description(mcp_client):
    for tool in (await mcp_client.list_tools()).tools:
        assert tool.description and len(tool.description) > 80, tool.name


async def test_every_tool_declares_all_annotations(mcp_client):
    for tool in (await mcp_client.list_tools()).tools:
        present = {
            key
            for key in REQUIRED_ANNOTATIONS
            if getattr(tool.annotations, _attr(key), None) is not None
        }
        assert present == REQUIRED_ANNOTATIONS, tool.name


async def test_every_tool_advertises_structured_object_output(mcp_client):
    for tool in (await mcp_client.list_tools()).tools:
        assert tool.output_schema is not None, tool.name
        assert tool.output_schema["type"] == "object", tool.name


async def test_open_world_hint_is_always_true(mcp_client):
    for tool in (await mcp_client.list_tools()).tools:
        assert getattr(tool.annotations, _attr("openWorldHint")) is True, tool.name


@pytest.mark.parametrize(
    "name",
    [
        "qdrant_delete_collection",
        "qdrant_create_collection",
        "qdrant_delete_points",
        "qdrant_clear_payload",
        "qdrant_delete_payload_index",
    ],
)
async def test_destructive_tools_are_marked_destructive(mcp_client, name):
    tools = {tool.name: tool for tool in (await mcp_client.list_tools()).tools}
    annotations = tools[name].annotations
    assert getattr(annotations, _attr("destructiveHint")) is True


@pytest.mark.parametrize(
    "name",
    [
        "qdrant_update_schema_toggles",
        "qdrant_create_payload_index",
        "qdrant_start_reindex",
        "qdrant_control_reindex",
        "qdrant_upsert_points",
        "qdrant_delete_payload_keys",
    ],
)
async def test_worst_case_destructive_paths_are_annotated(mcp_client, name):
    tools = {tool.name: tool for tool in (await mcp_client.list_tools()).tools}
    assert tools[name].annotations.destructive_hint is True


def test_app_is_cached_after_first_access(monkeypatch, tmp_path):
    """A second access to ``app`` must return the same object, not rebuild it.

    ``create_app()`` builds a new ``MCPServer``, a new ``MiddlewareClient``,
    and a new (unclosed) ``httpx.AsyncClient`` connection pool. Without
    caching, every attribute access would repeat all of that.
    """
    monkeypatch.setenv("MCP_API_TOKENS", "t:alice")
    monkeypatch.setenv("MIDDLEWARE_CLIENT_TOKEN", "m")
    import smart_qdrant_mcp.config as config_module
    import smart_qdrant_mcp.server as server_module

    # get_settings() scans the repository .env for removed variable names, so
    # point it at an empty file: this test must not depend on whether the
    # developer's own .env has been migrated yet.
    monkeypatch.setattr(config_module, "REPO_ENV_FILE", tmp_path / ".env")
    server_module.__dict__.pop("app", None)
    config_module.get_settings.cache_clear()
    try:
        assert server_module.app is server_module.app
    finally:
        server_module.__dict__.pop("app", None)
        config_module.get_settings.cache_clear()


def test_app_access_raises_config_error_without_tokens(monkeypatch):
    """A missing MCP_API_TOKENS must still surface as ConfigError on access.

    The caching in ``__getattr__`` must not swallow or cache a configuration
    failure: every access with a still-broken environment raises again.
    """
    # Set empty rather than delete: this path goes through get_settings(),
    # which reads the repository .env, so deleting the variable only falls
    # through to whatever the developer has configured there. An explicit
    # empty value is a broken configuration only when the unauthenticated
    # flag is also off; set both to keep the test hermetic regardless of the
    # local .env.
    monkeypatch.setenv("MCP_API_TOKENS", "")
    monkeypatch.setenv("MCP_ALLOW_UNAUTHENTICATED_LOCAL", "false")
    import smart_qdrant_mcp.config as config_module
    import smart_qdrant_mcp.server as server_module

    server_module.__dict__.pop("app", None)
    config_module.get_settings.cache_clear()
    try:
        with pytest.raises(config_module.ConfigError):
            _ = server_module.app
        # Confirmed uncached: a second access with the same broken
        # environment raises again rather than returning a stale value.
        with pytest.raises(config_module.ConfigError):
            _ = server_module.app
    finally:
        server_module.__dict__.pop("app", None)
        config_module.get_settings.cache_clear()


async def test_instructions_name_the_inline_ingest_ceiling(mcp_server):
    """The 4 MiB transport cap has no tool-level error to announce it.

    It is enforced before any tool runs, so a caller that exceeds it gets a
    transport rejection and nothing else. The handshake is therefore the
    only place an agent can learn the limit before hitting it, along with
    where to read what to do instead.
    """
    assert "4 MiB" in mcp_server.instructions
    assert "docs://workflows/ingest" in mcp_server.instructions


def test_transport_body_size_default_matches_the_documented_4_mib():
    """The SDK's default request body size stays the documented 4 MiB.

    The "4 MiB" figure appears as prose in five documents, but
    ``create_app()`` never passes ``max_request_body_size`` to
    ``streamable_http_app()`` — the number is whatever the ``mcp`` SDK's own
    default happens to be. Pinning that default here means a dependency bump
    that silently changes it fails this test instead of quietly falsifying
    five documents while every other test still passes.
    """
    from mcp.server.streamable_http_manager import DEFAULT_MAX_REQUEST_BODY_SIZE

    assert DEFAULT_MAX_REQUEST_BODY_SIZE == 4 * 1024 * 1024


def _attr(key: str) -> str:
    """Map a camelCase annotation key to the SDK model's snake_case attribute.

    `mcp_types.ToolAnnotations` (verified against the installed ``mcp==2.0.0``)
    exposes Python attributes in snake_case (``read_only_hint``,
    ``open_world_hint``, ...) and only carries the camelCase spelling
    (``readOnlyHint``, ``openWorldHint``, ...) as a pydantic serialization
    alias, so plain attribute access with the camelCase name raises
    AttributeError. This mirrors that mapping for test-side lookups.
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower()

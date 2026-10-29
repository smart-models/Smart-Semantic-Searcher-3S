"""Every tool call logs one line: its name, ok|error and duration, nothing else.

The lines are read from the package's real stream handler, not ``caplog``:
``caplog`` listens on the root logger, which ``propagate=False`` keeps these
records away from, and it would pass even if the real handler were missing.
Uvicorn's own access line is covered here too: it stays, without its query
string.
"""

import io
import logging
import re

import smart_qdrant_mcp.config as config_module
from smart_qdrant_mcp.server import configure_package_logging, create_app
from tests.test_tools_query import RESPONSE

SECRET_QUERY = "salary of the chief executive"


async def _tool_lines(*calls) -> list[str]:
    """Await ``calls`` and return the ``tool ...`` lines the real handler wrote."""
    configure_package_logging()
    handler = logging.getLogger("smart_qdrant_mcp").handlers[0]
    buffer = io.StringIO()
    previous = handler.setStream(buffer)
    try:
        for pending in calls:
            await pending
    finally:
        handler.setStream(previous)
    output = buffer.getvalue()
    assert SECRET_QUERY not in output
    assert "hr-private" not in output
    return [line for line in output.splitlines() if "tool " in line]


async def test_a_successful_call_logs_ok_and_no_arguments(call, routes):
    routes.add("POST", "/query/search", body=RESPONSE)
    lines = await _tool_lines(
        call("qdrant_search", query=SECRET_QUERY, collection_name="hr-private")
    )
    assert len(lines) == 1
    assert re.fullmatch(r"INFO: +tool 'qdrant_search' ok \d+\.\ds", lines[0])


async def test_an_error_result_and_a_raised_error_both_log_error(
    call, mcp_client, routes
):
    routes.add("POST", "/query/search", status=400, body={"detail": "bad"})
    lines = await _tool_lines(
        # The tool reports the upstream 400 as a result with is_error=True.
        call("qdrant_search", query=SECRET_QUERY, collection_name="hr-private"),
        # The SDK's argument validation raises instead of returning a result.
        mcp_client.call_tool(
            "qdrant_search", {"params": {"query": SECRET_QUERY, "limit": "many"}}
        ),
    )
    assert len(lines) == 2
    for line in lines:
        assert re.fullmatch(r"INFO: +tool 'qdrant_search' error \d+\.\ds", line)


async def test_a_client_chosen_name_stays_on_one_line(mcp_client):
    lines = await _tool_lines(mcp_client.call_tool("no_such\nINFO: forged", {}))
    assert len(lines) == 1
    assert re.fullmatch(
        r"INFO: +tool 'no_such\\nINFO: forged' error \d+\.\ds", lines[0]
    )


def _access_line(path_with_query: str) -> str:
    """Log one access line exactly as uvicorn does; return what a handler got."""
    access = logging.getLogger("uvicorn.access")
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setLevel(logging.INFO)
    previous_level = access.level
    access.addHandler(handler)
    access.setLevel(logging.INFO)
    try:
        access.info(
            '%s - "%s %s HTTP/%s" %d',
            "127.0.0.1:50000",
            "POST",
            path_with_query,
            "1.1",
            401,
        )
    finally:
        access.removeHandler(handler)
        access.setLevel(previous_level)
    return buffer.getvalue()


def test_the_access_log_keeps_its_line_but_drops_the_query_string(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("MCP_API_TOKENS", "t:alice")
    monkeypatch.setenv("MIDDLEWARE_CLIENT_TOKEN", "m")
    # get_settings() scans the repository .env; keep the developer's out.
    monkeypatch.setattr(config_module, "REPO_ENV_FILE", tmp_path / ".env")
    config_module.get_settings.cache_clear()
    try:
        create_app()
    finally:
        config_module.get_settings.cache_clear()
    line = _access_line("/mcp?trace=qs-value")
    assert '"POST /mcp HTTP/1.1" 401' in line
    assert "qs-value" not in line

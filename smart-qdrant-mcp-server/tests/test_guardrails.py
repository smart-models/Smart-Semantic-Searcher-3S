"""Confirmation tokens and the delete dry run."""

import pytest

from smart_qdrant_mcp.context import ToolContext
from smart_qdrant_mcp.errors import MiddlewareError, ToolError
from smart_qdrant_mcp.guardrails import (
    count_matching,
    refuse_if,
    require_confirm,
    require_selector_args,
)
from smart_qdrant_mcp.tool_map import RESOURCE_ONLY_ROUTES, TOOL_ROUTES


def test_refuse_if_true_raises_with_the_message():
    with pytest.raises(ToolError, match="^boom$"):
        refuse_if(True, "boom")


def test_refuse_if_false_does_nothing():
    refuse_if(False, "boom")


def test_require_selector_args_ids_with_no_ids_raises():
    with pytest.raises(ToolError, match="non-empty ids list"):
        require_selector_args("ids", None, None)


def test_require_selector_args_ids_with_ids_passes():
    require_selector_args("ids", ["p1"], None)


def test_require_selector_args_filter_with_no_filters_raises():
    with pytest.raises(ToolError, match="non-empty filters object"):
        require_selector_args("filter", None, None)


def test_require_selector_args_filter_with_empty_dict_raises():
    with pytest.raises(ToolError, match="non-empty filters object"):
        require_selector_args("filter", None, {})


def test_require_selector_args_filter_with_filters_passes():
    require_selector_args("filter", None, {"must": []})


def test_matching_confirm_passes():
    require_confirm("docs", "docs")


@pytest.mark.parametrize("value", [None, "", "Docs", "doc", " docs"])
def test_wrong_confirm_raises_with_the_expected_value(value):
    with pytest.raises(ToolError, match="docs"):
        require_confirm(value, "docs")


def test_empty_string_confirm_against_an_empty_target_is_refused():
    """A falsy confirm is refused outright, before comparison.

    Without this, a caller whose target itself normalises to "" (e.g. an
    unset value coerced to an empty string) could satisfy `confirm == target`
    with an equally empty confirm and never actually confirm anything.
    """
    with pytest.raises(ToolError):
        require_confirm("", "")


def test_none_confirm_against_an_empty_target_is_refused():
    with pytest.raises(ToolError):
        require_confirm(None, "")


def test_confirm_error_names_the_parameter():
    with pytest.raises(ToolError, match="confirm"):
        require_confirm(None, "docs")


def test_confirm_error_default_noun_is_collection_name():
    """The default confirm noun stays "collection name".

    Every existing caller (collections.py, payload_indexes.py, reindex.py)
    confirms against a collection name and relies on the default wording;
    changing it would silently reword four call sites.
    """
    with pytest.raises(ToolError, match="collection name"):
        require_confirm(None, "docs")


def test_confirm_error_uses_the_supplied_noun():
    with pytest.raises(ToolError, match="source_id") as exc_info:
        require_confirm(None, "manual", noun="source_id")
    assert "collection name" not in str(exc_info.value)


async def test_count_matching_calls_the_count_endpoint(settings, routes):
    from smart_qdrant_mcp.client import MiddlewareClient

    routes.add("POST", "/points/count", body={"count": 12})
    api = MiddlewareClient(settings, transport=routes.transport())
    ctx = ToolContext(client=api, settings=settings)
    assert await count_matching(ctx, "docs", {"must": []}) == 12
    assert routes.last_body() == {"collection_name": "docs", "filters": {"must": []}}
    await api.aclose()


async def test_count_matching_returns_zero_for_a_well_formed_zero_count(
    settings, routes, client
):
    """A genuine zero count is not confused with a malformed-response failure."""
    routes.add("POST", "/points/count", body={"count": 0})
    ctx = ToolContext(client=client, settings=settings)
    assert await count_matching(ctx, "docs", {"must": []}) == 0


async def test_count_matching_raises_on_a_list_body(settings, routes, client):
    routes.add("POST", "/points/count", body=[1, 2, 3])
    ctx = ToolContext(client=client, settings=settings)
    with pytest.raises(ToolError):
        await count_matching(ctx, "docs", {"must": []})


async def test_count_matching_raises_when_count_key_is_missing(
    settings, routes, client
):
    routes.add("POST", "/points/count", body={"total": 12})
    ctx = ToolContext(client=client, settings=settings)
    with pytest.raises(ToolError):
        await count_matching(ctx, "docs", {"must": []})


@pytest.mark.parametrize("bad_count", ["not-a-number", None])
async def test_count_matching_raises_on_a_non_numeric_count(
    settings, routes, client, bad_count
):
    routes.add("POST", "/points/count", body={"count": bad_count})
    ctx = ToolContext(client=client, settings=settings)
    with pytest.raises(ToolError):
        await count_matching(ctx, "docs", {"must": []})


async def test_count_matching_raises_on_a_numeric_string_count(
    settings, routes, client
):
    """A stringly-typed count ("12") is rejected rather than coerced.

    The /points/count contract returns a JSON integer; a string in that field
    means the response shape has already drifted from what this dry run
    trusts before a mass delete, so it is treated the same as any other
    malformed body rather than silently parsed.
    """
    routes.add("POST", "/points/count", body={"count": "12"})
    ctx = ToolContext(client=client, settings=settings)
    with pytest.raises(ToolError):
        await count_matching(ctx, "docs", {"must": []})


async def test_count_matching_propagates_middleware_error_on_503(
    settings, routes, client
):
    """A persistent 503 surfaces as MiddlewareError after every retry.

    A 503 is retried by the client (default retry_attempts=3), so the fake
    must queue one response per attempt, or the queue-exhausted 404 would be
    returned instead of the 503 this test means to exercise.
    """
    for _ in range(settings.retry_attempts):
        routes.add("POST", "/points/count", status=503, body={"detail": "not ready"})
    ctx = ToolContext(client=client, settings=settings)
    with pytest.raises(MiddlewareError):
        await count_matching(ctx, "docs", {"must": []})


def test_every_tool_route_entry_is_a_method_path_pair():
    for name, routes_for_tool in TOOL_ROUTES.items():
        assert name.startswith("qdrant_"), name
        for method, path in routes_for_tool:
            assert method in {"GET", "POST", "PUT", "PATCH", "DELETE"}
            assert path.startswith("/")


def test_resource_only_routes_are_disjoint_from_tool_routes():
    covered = {pair for pairs in TOOL_ROUTES.values() for pair in pairs}
    assert covered.isdisjoint(RESOURCE_ONLY_ROUTES)


def test_tool_routes_cover_exactly_twenty_eight_tools():
    assert len(TOOL_ROUTES) == 28


def test_tool_routes_cover_forty_three_operations():
    covered = {pair for pairs in TOOL_ROUTES.values() for pair in pairs}
    assert len(covered) + len(RESOURCE_ONLY_ROUTES) == 43

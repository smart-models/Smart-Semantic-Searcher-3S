"""Upsert and delete, including the delete-by-filter dry run."""

import json

import pytest

ONE = {"ok": True, "point_id": "p1", "collection_name": "docs", "skipped": False}
BATCH = {"ok": True, "upserted": 2, "collection_name": "docs", "skipped": 0}
# Both delete endpoints now carry `deleted`: the single-point DeleteResponse
# reports 1 or 0 (an absent id is an idempotent no-op, not a 404), matching
# BulkDeleteResponse's count of ids that actually existed.
DELETED_ONE = {"ok": True, "deleted": 1, "collection_name": "docs"}
DELETED_ONE_MISSING = {"ok": True, "deleted": 0, "collection_name": "docs"}
DELETED_MANY = {"ok": True, "deleted": 2, "collection_name": "docs"}


def point(pid, text="hello"):
    return {"point_id": pid, "text": text}


async def test_single_point_uses_the_flat_endpoint(call, routes):
    routes.add("POST", "/points", body=ONE)
    await call("qdrant_upsert_points", points=[point("p1")], collection_name="docs")
    assert routes.calls[-1].url.path == "/points"
    body = routes.last_body()
    assert body["point_id"] == "p1"
    assert body["text"] == "hello"
    assert "points" not in body


async def test_multiple_points_use_the_batch_endpoint(call, routes):
    routes.add("POST", "/points/batch", body=BATCH)
    await call("qdrant_upsert_points", points=[point("p1"), point("p2")])
    assert routes.calls[-1].url.path == "/points/batch"
    assert len(routes.last_body()["points"]) == 2


async def test_update_mode_is_forwarded_in_both_shapes(call, routes):
    routes.add("POST", "/points", body=ONE)
    await call("qdrant_upsert_points", points=[point("p1")], update_mode="insert_only")
    assert routes.last_body()["update_mode"] == "insert_only"


async def test_duplicate_point_ids_are_rejected_before_http(call, routes):
    # This used to be a Pydantic model_validator (reject_duplicate_point_ids);
    # it is now a ToolError raised from the tool body, so it must return the
    # same clean "Error: ..." sentence describe_error produces for every
    # other guardrail, not the SDK's own validation-error envelope.
    out = await call("qdrant_upsert_points", points=[point("p1"), point("p1")])
    assert routes.calls == []
    assert out.is_error is True
    assert out.startswith("Error:")
    assert "points contains duplicate point_id values: p1" in out
    assert "validation error" not in out
    assert "pydantic.dev" not in out


async def test_blank_text_is_rejected_before_http(call, routes):
    out = await call("qdrant_upsert_points", points=[point("p1", "   ")])
    assert routes.calls == []
    assert out.is_error is True
    assert "Error" in out or "validation" in out.lower()


async def test_single_point_body_omits_unset_collection_name(call, routes):
    routes.add("POST", "/points", body=ONE)
    await call("qdrant_upsert_points", points=[point("p1")])
    assert routes.last_body() == {
        "point_id": "p1",
        "text": "hello",
        "payload": {},
        "update_mode": "upsert",
    }


async def test_batch_body_is_exact_and_carries_the_collection_name(call, routes):
    routes.add("POST", "/points/batch", body=BATCH)
    await call(
        "qdrant_upsert_points",
        points=[point("p1"), point("p2")],
        collection_name="docs",
    )
    assert routes.last_body() == {
        "points": [
            {"point_id": "p1", "text": "hello", "payload": {}},
            {"point_id": "p2", "text": "hello", "payload": {}},
        ],
        "update_mode": "upsert",
        "collection_name": "docs",
    }


async def test_upsert_renders_the_skip_count_in_markdown(call, routes):
    routes.add(
        "POST",
        "/points",
        body={"ok": True, "point_id": "p1", "collection_name": "docs", "skipped": True},
    )
    out = await call(
        "qdrant_upsert_points", points=[point("p1")], update_mode="insert_only"
    )
    assert "Skipped**: 1" in out
    assert "Upserted**: 0" in out


async def test_delete_one_id_uses_the_path_endpoint(call, routes):
    routes.add("DELETE", "/points/p1", body=DELETED_ONE)
    await call(
        "qdrant_delete_points",
        selector="ids",
        ids=["p1"],
        collection_name="docs",
        confirm="docs",
    )
    assert routes.calls[-1].url.path == "/points/p1"
    assert routes.calls[-1].url.params["collection_name"] == "docs"


async def test_delete_one_id_reports_a_count_of_one(call, routes):
    routes.add("DELETE", "/points/p1", body=DELETED_ONE)
    payload = json.loads(
        await call(
            "qdrant_delete_points",
            selector="ids",
            ids=["p1"],
            collection_name="docs",
            confirm="docs",
            response_format="json",
        )
    )
    assert payload["deleted"] == 1


async def test_delete_one_absent_id_reports_zero_not_one(call, routes):
    """A single-id delete that hit nothing must not report a phantom deletion.

    The tool routes one id to the single-point endpoint and many ids to the
    bulk one; both now report how many ids actually existed, so the same
    absent id gives the same answer either way instead of depending on how
    many ids happened to be in the list.
    """
    routes.add("DELETE", "/points/p1", body=DELETED_ONE_MISSING)
    payload = json.loads(
        await call(
            "qdrant_delete_points",
            selector="ids",
            ids=["p1"],
            collection_name="docs",
            confirm="docs",
            response_format="json",
        )
    )
    assert payload["deleted"] == 0


async def test_delete_many_ids_uses_the_bulk_endpoint(call, routes):
    routes.add("DELETE", "/points", body=DELETED_MANY)
    await call(
        "qdrant_delete_points",
        selector="ids",
        ids=["p1", "p2"],
        collection_name="docs",
        confirm="docs",
    )
    assert routes.calls[-1].url.path == "/points"
    assert routes.last_body()["ids"] == ["p1", "p2"]


async def test_delete_many_ids_body_is_exact(call, routes):
    routes.add("DELETE", "/points", body=DELETED_MANY)
    await call(
        "qdrant_delete_points",
        selector="ids",
        ids=["p1", "p2"],
        collection_name="docs",
        confirm="docs",
    )
    assert routes.last_body() == {"collection_name": "docs", "ids": ["p1", "p2"]}


async def test_delete_by_filter_without_confirm_runs_a_dry_run(call, routes):
    routes.add(
        "POST",
        "/points/count",
        body={"ok": True, "collection_name": "docs", "count": 9, "exact": True},
    )
    out = await call(
        "qdrant_delete_points",
        selector="filter",
        filters={"must": [{"key": "lang", "match": {"value": "it"}}]},
        collection_name="docs",
    )
    assert routes.calls[-1].url.path == "/points/count"
    assert out.is_error is False
    assert "9" in out
    assert "confirm=docs" in out.replace('"', "").replace("'", "")


async def test_delete_by_filter_with_confirm_deletes(call, routes):
    routes.add(
        "POST",
        "/points/delete-by-filter",
        body={"ok": True, "deleted": 9, "collection_name": "docs"},
    )
    out = await call(
        "qdrant_delete_points",
        selector="filter",
        filters={"must": []},
        collection_name="docs",
        confirm="docs",
    )
    assert routes.calls[-1].url.path == "/points/delete-by-filter"
    assert "9" in out


async def test_delete_by_filter_body_is_exact(call, routes):
    routes.add(
        "POST",
        "/points/delete-by-filter",
        body={"ok": True, "deleted": 9, "collection_name": "docs"},
    )
    await call(
        "qdrant_delete_points",
        selector="filter",
        filters={"must": []},
        collection_name="docs",
        confirm="docs",
    )
    assert routes.last_body() == {"collection_name": "docs", "filters": {"must": []}}


async def test_delete_by_ids_without_confirm_is_refused_without_dry_run(call, routes):
    out = await call(
        "qdrant_delete_points", selector="ids", ids=["p1"], collection_name="docs"
    )
    assert routes.calls == []
    assert out.is_error is True
    assert "confirm" in out


@pytest.mark.parametrize("bad_confirm", [" docs", "docs ", "Docs"])
async def test_delete_confirm_must_match_exactly(call, routes, bad_confirm):
    out = await call(
        "qdrant_delete_points",
        selector="ids",
        ids=["p1"],
        collection_name="docs",
        confirm=bad_confirm,
    )
    assert routes.calls == []
    assert out.is_error is True
    assert "confirm" in out


async def test_selector_ids_requires_ids(call, routes):
    # This used to be a Pydantic model_validator (selector_matches_its_argument);
    # it is now guardrails.require_selector_args, raised as a ToolError from
    # the tool body, so it must return the same clean "Error: ..." sentence
    # describe_error produces, not the SDK's own validation-error envelope.
    out = await call(
        "qdrant_delete_points", selector="ids", collection_name="docs", confirm="docs"
    )
    assert routes.calls == []
    assert out.is_error is True
    assert out.startswith("Error:")
    assert "selector='ids' requires a non-empty ids list" in out
    assert "validation error" not in out
    assert "pydantic.dev" not in out


async def test_selector_filter_requires_non_empty_filters(call, routes):
    out = await call(
        "qdrant_delete_points",
        selector="filter",
        filters={},
        collection_name="docs",
        confirm="docs",
    )
    assert routes.calls == []
    assert out.is_error is True
    assert out.startswith("Error:")
    assert (
        "selector='filter' requires a non-empty filters object; an empty "
        "filter would match every point"
    ) in out
    assert "validation error" not in out
    assert "pydantic.dev" not in out


async def test_delete_by_filter_reports_the_upstream_count(call, routes):
    routes.add(
        "POST",
        "/points/delete-by-filter",
        body={"ok": True, "deleted": 9, "collection_name": "docs"},
    )
    payload = json.loads(
        await call(
            "qdrant_delete_points",
            selector="filter",
            filters={"must": []},
            collection_name="docs",
            confirm="docs",
            response_format="json",
        )
    )
    assert payload["deleted"] == 9


async def test_delete_renders_the_collection_and_selector_in_markdown(call, routes):
    routes.add("DELETE", "/points/p1", body=DELETED_ONE)
    out = await call(
        "qdrant_delete_points",
        selector="ids",
        ids=["p1"],
        collection_name="docs",
        confirm="docs",
    )
    assert "Collection**: docs" in out
    assert "Selector**: ids" in out


async def test_blank_id_is_refused_before_any_call(call, routes):
    """A blank id must not reach the middleware as an empty path segment.

    `/points/` (single-id path built from "") is a 307 redirect to `/points`,
    the BULK delete route. httpx does not follow redirects today, so the call
    fails confusingly rather than dangerously — but the id is meaningless
    either way, and one flipped client flag would turn it into a mass delete.
    """
    result = await call(
        "qdrant_delete_points",
        selector="ids",
        ids=[""],
        collection_name="docs",
        confirm="docs",
    )
    assert result.is_error
    assert "blank" in result.lower()
    assert routes.calls == []


async def test_blank_id_among_valid_ones_is_refused(call, routes):
    """One bad entry refuses the whole call rather than deleting the rest."""
    result = await call(
        "qdrant_delete_points",
        selector="ids",
        ids=["p1", "  ", "p2"],
        collection_name="docs",
        confirm="docs",
    )
    assert result.is_error
    assert routes.calls == []


async def test_dry_run_says_the_count_is_a_snapshot(call, routes):
    """The count and the delete are two calls: the set can move between them.

    An agent that reports "this deletes 9 points" to a user is making a promise
    the second call does not keep, so the dry-run text has to qualify it.
    """
    routes.add(
        "POST",
        "/points/count",
        body={"ok": True, "collection_name": "docs", "count": 9, "exact": True},
    )
    out = await call(
        "qdrant_delete_points",
        selector="filter",
        filters={"must": [{"key": "lang", "match": {"value": "it"}}]},
        collection_name="docs",
    )
    assert "now" in out.lower() or "snapshot" in out.lower()

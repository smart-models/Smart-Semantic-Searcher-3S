"""Reindex preflight, start, status and control."""

import json

import pytest

from smart_qdrant_mcp.tools import reindex

PREFLIGHT = {
    "collection_name": "docs",
    "can_start": True,
    "qdrant_version": "1.12.0",
    "total_points": 500,
    "eligible_points": 498,
    "invalid_points": 2,
    "invalid_point_ids": ["p9", "p10"],
    "payload_field": "text",
    "blockers": [],
}
# ReindexStatusResponse (schemas/reindex.py:58-66) has no `ok` field — only
# ReindexActionResponse does. PREFLIGHT above intentionally omits
# target_contract / source_vectors: MockTransport returns the dict verbatim and
# the tool reads none of them. The nested `job` object below uses the real
# ReindexJobState field names (schemas/embedding_contract.py:86-111): `status`
# (not `state`) and `processed_points`/`total_points` (not `processed`/
# `total`) — inventing fields that don't exist on the real response would
# make this test pass against a fake that a live server could never produce.
STATUS = {
    "collection_name": "docs",
    "job": {
        "job_id": "job-1",
        "status": "running",
        "processed_points": 120,
        "total_points": 500,
    },
    "task_running": True,
    "metadata_invalid": False,
}
ACTION = {**STATUS, "ok": True}


async def test_preflight_reports_eligibility(call, routes):
    routes.add("POST", "/collections/docs/reindex/preflight", body=PREFLIGHT)
    out = await call("qdrant_reindex_preflight", collection_name="docs")
    assert "498" in out and "500" in out


async def test_preflight_surfaces_blockers(call, routes):
    body = {**PREFLIGHT, "can_start": False, "blockers": ["a job is already running"]}
    routes.add("POST", "/collections/docs/reindex/preflight", body=body)
    out = await call("qdrant_reindex_preflight", collection_name="docs")
    assert "already running" in out


async def test_preflight_lists_invalid_point_ids_in_json(call, routes):
    routes.add("POST", "/collections/docs/reindex/preflight", body=PREFLIGHT)
    payload = json.loads(
        await call(
            "qdrant_reindex_preflight", collection_name="docs", response_format="json"
        )
    )
    assert payload["invalid_point_ids"] == ["p9", "p10"]


async def test_start_requires_confirm(call, routes):
    out = await call("qdrant_start_reindex", collection_name="docs")
    assert routes.calls == []
    assert out.is_error is True
    assert "confirm" in out


async def test_start_rejects_raw_confirmation_variants_before_http(call, routes):
    for confirm in [" docs ", " docs", "docs ", "\tdocs\n", "Docs", "other"]:
        out = await call(
            "qdrant_start_reindex", collection_name="docs", confirm=confirm
        )
        assert out.is_error is True
        assert routes.calls == []


async def test_start_forwards_confirm_collection(call, routes):
    routes.add("POST", "/collections/docs/reindex", body=ACTION)
    out = await call("qdrant_start_reindex", collection_name="docs", confirm="docs")
    assert routes.last_body() == {"confirm_collection": "docs"}
    assert out.is_error is False
    # Proves _action_rows actually renders the response rather than
    # silently producing an empty/placeholder body: the "Ok" line and the
    # job fields (job_id, state, processed/total) must all be present.
    assert "**Ok**: True" in out
    assert "**Job ID**: job-1" in out
    assert "**State**: running" in out
    assert "**Processed**: 120 of 500" in out


async def test_start_json_returns_the_action_payload(call, routes):
    # response_format='json' must return the raw upstream payload untouched:
    # the whole ACTION dict, not a rendered subset of it.
    routes.add("POST", "/collections/docs/reindex", body=ACTION)
    payload = json.loads(
        await call(
            "qdrant_start_reindex",
            collection_name="docs",
            confirm="docs",
            response_format="json",
        )
    )
    assert payload == ACTION


async def test_status_reports_progress(call, routes):
    routes.add("GET", "/collections/docs/reindex", body=STATUS)
    out = await call("qdrant_get_reindex_status", collection_name="docs")
    assert "running" in out
    assert "120" in out


async def test_status_handles_absent_job(call, routes):
    routes.add(
        "GET",
        "/collections/docs/reindex",
        body={
            "collection_name": "docs",
            "job": None,
            "task_running": False,
            "metadata_invalid": False,
        },
    )
    out = await call("qdrant_get_reindex_status", collection_name="docs")
    assert "no reindex job" in out.lower()


@pytest.fixture
def waits(monkeypatch):
    """Record each wait between status reads instead of sleeping through it."""
    recorded: list[float] = []

    async def no_sleep(seconds):
        recorded.append(seconds)

    monkeypatch.setattr(reindex, "sleep", no_sleep)
    return recorded


def _status(state: str, running: bool, processed: int = 120) -> dict:
    job = {**STATUS["job"], "status": state, "processed_points": processed}
    return {**STATUS, "job": job, "task_running": running}


async def test_status_wait_returns_once_a_pause_takes_effect(call, routes, waits):
    """One call does the waiting: no client-side polling loop for a pause."""
    for body in (
        _status("paused", True),
        _status("paused", True, 128),
        _status("paused", False, 128),
    ):
        routes.add("GET", "/collections/docs/reindex", body=body)

    out = await call(
        "qdrant_get_reindex_status", collection_name="docs", wait_seconds=30
    )

    assert len(routes.calls) == 3
    assert waits == [reindex.STATUS_POLL_SECONDS] * 2
    assert "**Running**: False" in out
    assert "128 of 500" in out


async def test_status_wait_ignores_progress_within_the_same_state(call, routes, waits):
    """Processed counts move every batch; only a state change ends the wait."""
    for processed in (120, 184, 248):
        routes.add(
            "GET", "/collections/docs/reindex", body=_status("running", True, processed)
        )

    out = await call(
        "qdrant_get_reindex_status", collection_name="docs", wait_seconds=4
    )

    assert len(routes.calls) == 3
    assert "248 of 500" in out


async def test_status_wait_answers_at_once_when_no_task_runs(call, routes, waits):
    """A job no worker is running cannot change on its own: waiting is wasted."""
    routes.add("GET", "/collections/docs/reindex", body=_status("paused", False))

    await call("qdrant_get_reindex_status", collection_name="docs", wait_seconds=30)

    assert len(routes.calls) == 1
    assert waits == []


async def test_status_rejects_a_wait_longer_than_thirty_seconds(call, routes):
    out = await call(
        "qdrant_get_reindex_status", collection_name="docs", wait_seconds=31
    )

    assert "Error" in out or "validation" in out.lower()
    assert routes.calls == []


@pytest.mark.parametrize("action", ["pause", "resume", "cancel"])
async def test_control_hits_the_matching_subpath(call, routes, action):
    routes.add("POST", f"/collections/docs/reindex/{action}", body=ACTION)
    out = await call("qdrant_control_reindex", collection_name="docs", action=action)
    assert routes.calls[-1].url.path == f"/collections/docs/reindex/{action}"
    # Same _action_rows renderer as qdrant_start_reindex: prove it actually
    # renders content for the control path too, not just the start path.
    assert "**Ok**: True" in out
    assert "**Job ID**: job-1" in out
    assert "**Processed**: 120 of 500" in out


async def test_control_json_returns_the_action_payload(call, routes):
    # Same raw-payload passthrough as qdrant_start_reindex in json mode.
    routes.add("POST", "/collections/docs/reindex/pause", body=ACTION)
    payload = json.loads(
        await call(
            "qdrant_control_reindex",
            collection_name="docs",
            action="pause",
            response_format="json",
        )
    )
    assert payload == ACTION


async def test_control_rejects_unknown_action(call, routes):
    out = await call("qdrant_control_reindex", collection_name="docs", action="restart")
    assert routes.calls == []
    assert out.is_error is True
    assert "Error" in out or "validation" in out.lower()


async def test_status_url_encodes_a_slash_in_collection_name(call, routes):
    # `.url.path` decodes "%2F" back to "/", so only `raw_path` — the bytes
    # actually sent on the wire — proves quote(..., safe="") ran on
    # collection_name: with it, the name stays one path segment ("a%2Fb");
    # without it, "/" would split the URL and this route would never match
    # as one collection name.
    routes.add("GET", "/collections/a/b/reindex", body=STATUS)
    await call("qdrant_get_reindex_status", collection_name="a/b")
    raw_path = routes.calls[-1].url.raw_path.split(b"?")[0]
    assert raw_path == b"/collections/a%2Fb/reindex"


async def test_status_url_encodes_a_percent_sign_in_collection_name(call, routes):
    routes.add("GET", "/collections/50%/reindex", body=STATUS)
    await call("qdrant_get_reindex_status", collection_name="50%")
    raw_path = routes.calls[-1].url.raw_path.split(b"?")[0]
    assert raw_path == b"/collections/50%25/reindex"


async def test_control_reports_metadata_invalid(call, routes):
    routes.add(
        "POST",
        "/collections/docs/reindex/cancel",
        body={
            "ok": True,
            "collection_name": "docs",
            "job": None,
            "task_running": False,
            "metadata_invalid": True,
        },
    )
    out = await call("qdrant_control_reindex", collection_name="docs", action="cancel")
    assert "metadata" in out.lower()

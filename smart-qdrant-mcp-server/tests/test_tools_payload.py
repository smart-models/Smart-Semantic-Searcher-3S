"""Payload writes: per-id fan-out, by-filter single call, and the replace refusal."""

import json

import pytest

UPDATED = {"ok": True, "point_id": "p1", "collection_name": "docs", "updated": True}
MATCHED = {"ok": True, "matched": 4, "collection_name": "docs"}


async def test_merge_by_single_id_uses_patch(call, routes):
    routes.add("PATCH", "/points/p1/payload", body=UPDATED)
    await call(
        "qdrant_set_payload",
        selector="ids",
        ids=["p1"],
        payload={"lang": "it"},
        collection_name="docs",
    )
    assert routes.calls[-1].method == "PATCH"
    assert routes.calls[-1].url.path == "/points/p1/payload"


async def test_replace_by_single_id_uses_put(call, routes):
    routes.add("PUT", "/points/p1/payload", body=UPDATED)
    await call(
        "qdrant_set_payload",
        selector="ids",
        ids=["p1"],
        mode="replace",
        payload={"lang": "it"},
        collection_name="docs",
        confirm="docs",
    )
    assert routes.calls[-1].method == "PUT"


# --- Final review finding 1: mode="replace" destroys every unprotected
# field on the addressed points, the same as qdrant_clear_payload, so it is
# gated the same way. ---


async def test_replace_requires_confirm(call, routes):
    out = await call(
        "qdrant_set_payload",
        selector="ids",
        ids=["p1"],
        mode="replace",
        payload={"lang": "it"},
        collection_name="docs",
    )
    assert routes.calls == []
    assert out.is_error is True
    assert "confirm" in out


async def test_replace_confirm_must_match_exactly(call, routes):
    out = await call(
        "qdrant_set_payload",
        selector="ids",
        ids=["p1"],
        mode="replace",
        payload={"lang": "it"},
        collection_name="docs",
        confirm="not-docs",
    )
    assert routes.calls == []
    assert out.is_error is True
    assert "confirm" in out


async def test_replace_requires_an_explicit_collection_name(call, routes):
    # This used to be a Pydantic model_validator
    # (replace_requires_an_explicit_collection_name); it is now a ToolError
    # raised from the tool body, so it must return the same clean
    # "Error: ..." sentence describe_error produces for every other
    # guardrail, not the SDK's own validation-error envelope.
    out = await call(
        "qdrant_set_payload",
        selector="ids",
        ids=["p1"],
        mode="replace",
        payload={"lang": "it"},
        confirm="docs",
    )
    assert routes.calls == []
    assert out.is_error is True
    assert out.startswith("Error:")
    assert (
        "mode='replace' requires an explicit collection_name so that "
        "confirm can be checked against it"
    ) in out
    assert "validation error" not in out
    assert "pydantic.dev" not in out


async def test_merge_needs_no_confirm_and_no_collection_name(call, routes):
    # mode="merge" (the default) must keep working exactly as before: no
    # confirm, no required collection_name.
    routes.add("PATCH", "/points/p1/payload", body=UPDATED)
    out = await call(
        "qdrant_set_payload",
        selector="ids",
        ids=["p1"],
        payload={"lang": "it"},
    )
    assert routes.calls
    assert not out.startswith("Error:")
    assert out.is_error is False


async def test_replace_by_filter_is_refused_before_http(call, routes):
    # This refusal is a ToolError raised from the tool body (not a Pydantic
    # model_validator), so it must come back as the same clean "Error: ..."
    # sentence every other guardrail in this service produces -- not the
    # SDK's own validation-error envelope ("Error executing tool ...: 1
    # validation error ...", with a pydantic.dev URL).
    out = await call(
        "qdrant_set_payload",
        selector="filter",
        mode="replace",
        filters={"must": []},
        payload={"lang": "it"},
        collection_name="docs",
    )
    assert routes.calls == []
    assert out.is_error is True
    assert out.startswith("Error:")
    assert "validation error" not in out
    assert "merge" in out
    assert "qdrant_scroll_points" in out


async def test_merge_by_filter_uses_the_collection_endpoint(call, routes):
    routes.add("PATCH", "/points/payload", body=MATCHED)
    await call(
        "qdrant_set_payload",
        selector="filter",
        filters={"must": []},
        payload={"lang": "it"},
        collection_name="docs",
    )
    assert routes.calls[-1].url.path == "/points/payload"
    assert routes.last_body()["filters"] == {"must": []}


async def test_many_ids_fan_out_to_one_call_each(call, routes):
    routes.add("PATCH", "/points/p1/payload", body=UPDATED)
    routes.add("PATCH", "/points/p2/payload", body=UPDATED)
    await call(
        "qdrant_set_payload",
        selector="ids",
        ids=["p1", "p2"],
        payload={"lang": "it"},
        collection_name="docs",
    )
    assert [c.url.path for c in routes.calls] == [
        "/points/p1/payload",
        "/points/p2/payload",
    ]


async def test_fan_out_reports_partial_failure_per_id(call, routes):
    routes.add("PATCH", "/points/p1/payload", body=UPDATED)
    routes.add(
        "PATCH", "/points/p2/payload", status=404, body={"detail": "point not found"}
    )
    payload = json.loads(
        await call(
            "qdrant_set_payload",
            selector="ids",
            ids=["p1", "p2"],
            payload={"lang": "it"},
            collection_name="docs",
            response_format="json",
        )
    )
    assert payload["updated"] == ["p1"]
    assert payload["failed"][0]["point_id"] == "p2"
    assert "not found" in payload["failed"][0]["reason"]


async def test_fan_out_failure_reason_is_redacted(call, routes):
    # Final review finding 2: _fan_out used to return str(exc.detail)
    # verbatim -- the one place in this service where upstream internals
    # (a filesystem path, a host:port) reached the client unredacted.
    routes.add("PATCH", "/points/p1/payload", body=UPDATED)
    routes.add(
        "PATCH",
        "/points/p2/payload",
        status=404,
        body={
            "detail": (
                "point not found; see /app/qdrant_middleware/foo.py at 10.0.4.22:6333"
            )
        },
    )
    payload = json.loads(
        await call(
            "qdrant_set_payload",
            selector="ids",
            ids=["p1", "p2"],
            payload={"lang": "it"},
            collection_name="docs",
            response_format="json",
        )
    )
    reason = payload["failed"][0]["reason"]
    assert "/app/" not in reason
    assert "10.0.4.22:6333" not in reason
    assert "not found" in reason


async def test_fan_out_continues_after_a_failure(call, routes):
    routes.add("PATCH", "/points/p1/payload", status=404, body={"detail": "gone"})
    routes.add("PATCH", "/points/p2/payload", body=UPDATED)
    payload = json.loads(
        await call(
            "qdrant_set_payload",
            selector="ids",
            ids=["p1", "p2"],
            payload={"lang": "it"},
            collection_name="docs",
            response_format="json",
        )
    )
    assert payload["updated"] == ["p2"]
    assert len(payload["failed"]) == 1


async def test_markdown_fan_out_states_non_atomicity(call, routes):
    routes.add("PATCH", "/points/p1/payload", body=UPDATED)
    routes.add("PATCH", "/points/p2/payload", status=404, body={"detail": "gone"})
    out = await call(
        "qdrant_set_payload",
        selector="ids",
        ids=["p1", "p2"],
        payload={"lang": "it"},
        collection_name="docs",
    )
    assert "not atomic" in out.lower()


async def test_empty_payload_is_rejected(call, routes):
    out = await call(
        "qdrant_set_payload",
        selector="ids",
        ids=["p1"],
        payload={},
        collection_name="docs",
    )
    assert routes.calls == []
    assert out.is_error is True
    assert "payload" in out


async def test_delete_keys_by_id_posts_the_key_list(call, routes):
    routes.add("POST", "/points/p1/payload/delete-keys", body=UPDATED)
    await call(
        "qdrant_delete_payload_keys",
        selector="ids",
        ids=["p1"],
        keys=["lang"],
        collection_name="docs",
    )
    assert routes.last_body()["keys"] == ["lang"]


async def test_delete_keys_by_filter_posts_filter_and_keys(call, routes):
    routes.add("POST", "/points/payload/delete-keys", body=MATCHED)
    await call(
        "qdrant_delete_payload_keys",
        selector="filter",
        filters={"must": []},
        keys=["lang"],
        collection_name="docs",
    )
    body = routes.last_body()
    assert body["keys"] == ["lang"] and body["filters"] == {"must": []}


async def test_delete_keys_rejects_blank_keys(call, routes):
    # strip_and_reject_blank_keys used to strip AND reject in one Pydantic
    # field_validator; the rejection half is now a ToolError raised from the
    # tool body (the stripping half stays on the model), so it must return
    # the same clean "Error: ..." sentence describe_error produces, not the
    # SDK's own validation-error envelope.
    out = await call(
        "qdrant_delete_payload_keys",
        selector="ids",
        ids=["p1"],
        keys=["  "],
        collection_name="docs",
    )
    assert routes.calls == []
    assert out.is_error is True
    assert out.startswith("Error:")
    assert "payload keys cannot be blank" in out
    assert "validation error" not in out
    assert "pydantic.dev" not in out


async def test_delete_keys_selector_ids_requires_ids(call, routes):
    out = await call(
        "qdrant_delete_payload_keys",
        selector="ids",
        keys=["lang"],
        collection_name="docs",
    )
    assert routes.calls == []
    assert out.is_error is True
    assert out.startswith("Error:")
    assert "selector='ids' requires a non-empty ids list" in out
    assert "validation error" not in out
    assert "pydantic.dev" not in out


async def test_delete_keys_selector_filter_requires_non_empty_filters(call, routes):
    out = await call(
        "qdrant_delete_payload_keys",
        selector="filter",
        filters={},
        keys=["lang"],
        collection_name="docs",
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


async def test_clear_payload_selector_ids_requires_ids(call, routes):
    out = await call(
        "qdrant_clear_payload", selector="ids", collection_name="docs", confirm="docs"
    )
    assert routes.calls == []
    assert out.is_error is True
    assert out.startswith("Error:")
    assert "selector='ids' requires a non-empty ids list" in out
    assert "validation error" not in out
    assert "pydantic.dev" not in out


async def test_clear_payload_selector_filter_requires_non_empty_filters(call, routes):
    out = await call(
        "qdrant_clear_payload",
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


async def test_clear_payload_requires_confirm(call, routes):
    out = await call(
        "qdrant_clear_payload", selector="ids", ids=["p1"], collection_name="docs"
    )
    assert routes.calls == []
    assert out.is_error is True
    assert "confirm" in out


async def test_clear_payload_by_filter_with_confirm(call, routes):
    routes.add("POST", "/points/payload/clear", body=MATCHED)
    out = await call(
        "qdrant_clear_payload",
        selector="filter",
        filters={"must": []},
        collection_name="docs",
        confirm="docs",
    )
    assert routes.calls[-1].url.path == "/points/payload/clear"
    assert "4" in out


async def test_clear_payload_confirm_gate_makes_zero_http_calls_by_filter(call, routes):
    out = await call(
        "qdrant_clear_payload",
        selector="filter",
        filters={"must": []},
        collection_name="docs",
    )
    assert routes.calls == []
    assert out.is_error is True
    assert "confirm" in out


async def test_clear_payload_confirm_must_match_exactly(call, routes):
    out = await call(
        "qdrant_clear_payload",
        selector="ids",
        ids=["p1"],
        collection_name="docs",
        confirm=" docs",
    )
    assert routes.calls == []
    assert out.is_error is True
    assert "confirm" in out


async def test_clear_payload_empty_confirm_and_no_collection_name_is_refused(
    call, routes
):
    # collection_name is a required field on ClearPayloadInput, so this is
    # refused by Pydantic before the confirm gate is ever reached -- but it
    # must still make zero HTTP calls either way.
    await call("qdrant_clear_payload", selector="ids", ids=["p1"], confirm="")
    assert routes.calls == []


async def test_clear_payload_empty_confirm_with_explicit_collection_name_is_refused(
    call, routes
):
    # Guards against the `target or ""` shape: an explicit collection_name
    # plus confirm="" must not satisfy the gate by both sides normalising to
    # the same empty string.
    out = await call(
        "qdrant_clear_payload",
        selector="ids",
        ids=["p1"],
        collection_name="docs",
        confirm="",
    )
    assert routes.calls == []
    assert out.is_error is True
    assert "confirm" in out


async def test_clear_payload_requires_an_explicit_collection_name(call, routes):
    out = await call("qdrant_clear_payload", selector="ids", ids=["p1"], confirm="docs")
    assert routes.calls == []
    assert out.is_error is True
    assert "collection_name" in out


async def test_clear_payload_fan_out_with_confirm(call, routes):
    cleared = {"ok": True, "point_id": "p1", "collection_name": "docs"}
    routes.add("POST", "/points/p1/payload/clear", body=cleared)
    await call(
        "qdrant_clear_payload",
        selector="ids",
        ids=["p1"],
        collection_name="docs",
        confirm="docs",
    )
    assert routes.calls[-1].url.path == "/points/p1/payload/clear"
    assert routes.last_body() == {"collection_name": "docs"}


async def test_selector_ids_requires_ids(call, routes):
    # This used to be a Pydantic model_validator (selector_matches_its_argument);
    # it is now guardrails.require_selector_args, raised as a ToolError from
    # the tool body, so it must return the same clean "Error: ..." sentence
    # describe_error produces, not the SDK's own validation-error envelope.
    out = await call(
        "qdrant_set_payload",
        selector="ids",
        payload={"lang": "it"},
        collection_name="docs",
    )
    assert routes.calls == []
    assert out.is_error is True
    assert out.startswith("Error:")
    assert "selector='ids' requires a non-empty ids list" in out
    assert "validation error" not in out
    assert "pydantic.dev" not in out


async def test_selector_filter_requires_non_empty_filters(call, routes):
    out = await call(
        "qdrant_set_payload",
        selector="filter",
        filters={},
        payload={"lang": "it"},
        collection_name="docs",
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


async def test_set_payload_ids_body_omits_unset_collection_name(call, routes):
    routes.add("PATCH", "/points/p1/payload", body=UPDATED)
    await call("qdrant_set_payload", selector="ids", ids=["p1"], payload={"lang": "it"})
    assert routes.last_body() == {"payload": {"lang": "it"}}


async def test_set_payload_filter_body_is_exact(call, routes):
    routes.add("PATCH", "/points/payload", body=MATCHED)
    await call(
        "qdrant_set_payload",
        selector="filter",
        filters={"must": []},
        payload={"lang": "it"},
        collection_name="docs",
    )
    assert routes.last_body() == {
        "collection_name": "docs",
        "filters": {"must": []},
        "payload": {"lang": "it"},
    }


async def test_set_payload_filter_renders_matched_count_in_markdown(call, routes):
    routes.add("PATCH", "/points/payload", body=MATCHED)
    out = await call(
        "qdrant_set_payload",
        selector="filter",
        filters={"must": []},
        payload={"lang": "it"},
        collection_name="docs",
    )
    assert "**Matched**: 4" in out
    assert "**Collection**: docs" in out


async def test_set_payload_ids_url_encodes_a_slash_in_the_point_id(call, routes):
    # `.url.path` decodes "%2F" back to "/", so only `raw_path` — the bytes
    # actually sent on the wire — proves quote(..., safe="") ran on the
    # point id before it was spliced into the path.
    routes.add("PATCH", "/points/a/b/payload", body=UPDATED)
    await call(
        "qdrant_set_payload",
        selector="ids",
        ids=["a/b"],
        payload={"lang": "it"},
        collection_name="docs",
    )
    raw_path = routes.calls[-1].url.raw_path.split(b"?")[0]
    assert raw_path == b"/points/a%2Fb/payload"


async def test_delete_keys_ids_url_encodes_a_percent_in_the_point_id(call, routes):
    routes.add("POST", "/points/50%/payload/delete-keys", body=UPDATED)
    await call(
        "qdrant_delete_payload_keys",
        selector="ids",
        ids=["50%"],
        keys=["lang"],
        collection_name="docs",
    )
    raw_path = routes.calls[-1].url.raw_path.split(b"?")[0]
    assert raw_path == b"/points/50%25/payload/delete-keys"


async def test_delete_keys_filter_body_is_exact(call, routes):
    routes.add("POST", "/points/payload/delete-keys", body=MATCHED)
    await call(
        "qdrant_delete_payload_keys",
        selector="filter",
        filters={"must": []},
        keys=["lang"],
        collection_name="docs",
    )
    assert routes.last_body() == {
        "collection_name": "docs",
        "filters": {"must": []},
        "keys": ["lang"],
    }


async def test_delete_keys_filter_renders_matched_count_in_markdown(call, routes):
    routes.add("POST", "/points/payload/delete-keys", body=MATCHED)
    out = await call(
        "qdrant_delete_payload_keys",
        selector="filter",
        filters={"must": []},
        keys=["lang"],
        collection_name="docs",
    )
    assert "**Matched**: 4" in out


async def test_delete_keys_fan_out_reports_partial_failure(call, routes):
    routes.add("POST", "/points/p1/payload/delete-keys", body=UPDATED)
    routes.add(
        "POST",
        "/points/p2/payload/delete-keys",
        status=404,
        body={"detail": "point not found"},
    )
    payload = json.loads(
        await call(
            "qdrant_delete_payload_keys",
            selector="ids",
            ids=["p1", "p2"],
            keys=["lang"],
            collection_name="docs",
            response_format="json",
        )
    )
    assert payload["updated"] == ["p1"]
    assert payload["failed"][0]["point_id"] == "p2"


async def test_clear_payload_fan_out_reports_partial_failure(call, routes):
    routes.add("POST", "/points/p1/payload/clear", body=UPDATED)
    routes.add("POST", "/points/p2/payload/clear", status=404, body={"detail": "gone"})
    payload = json.loads(
        await call(
            "qdrant_clear_payload",
            selector="ids",
            ids=["p1", "p2"],
            collection_name="docs",
            confirm="docs",
            response_format="json",
        )
    )
    assert payload["updated"] == ["p1"]
    assert payload["failed"][0]["point_id"] == "p2"


async def test_clear_payload_ids_markdown_states_non_atomicity_on_failure(call, routes):
    routes.add("POST", "/points/p1/payload/clear", body=UPDATED)
    routes.add("POST", "/points/p2/payload/clear", status=404, body={"detail": "gone"})
    out = await call(
        "qdrant_clear_payload",
        selector="ids",
        ids=["p1", "p2"],
        collection_name="docs",
        confirm="docs",
    )
    assert "not atomic" in out.lower()


@pytest.mark.parametrize(
    "tool,extra",
    [
        ("qdrant_set_payload", {"payload": {"lang": "it"}}),
        ("qdrant_delete_payload_keys", {"keys": ["lang"]}),
        ("qdrant_clear_payload", {"confirm": "docs"}),
    ],
)
async def test_ids_selector_is_capped_at_what_a_serial_fan_out_can_finish(
    call, routes, tool, extra
):
    """These tools issue one request per id, serially.

    The middleware has no bulk-by-ids payload endpoint, so a large ids list is
    N sequential round trips inside a single tool call — which no client
    timeout survives, and a cancellation mid-way discards the per-id report of
    what was already written. The cap has to reflect that, and point at the
    selector that does the same work in one request.
    """
    out = await call(
        tool,
        selector="ids",
        ids=[f"p{i}" for i in range(1001)],
        collection_name="docs",
        **extra,
    )
    assert out.is_error is True
    assert routes.calls == []
    assert "filter" in out.lower()

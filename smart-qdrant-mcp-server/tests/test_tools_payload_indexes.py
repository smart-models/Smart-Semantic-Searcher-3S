"""Payload key discovery and index lifecycle."""

import json

import pytest


@pytest.mark.parametrize("field", ["a.b", "items[].label"])
async def test_create_index_preserves_traversal_and_literal_pair(call, routes, field):
    names = [field, json.dumps(field)]
    routes.add(
        "POST",
        "/collections/docs/payload-indexes",
        body={"ok": True, "created": names, "failed": []},
    )
    result = await call(
        "qdrant_create_payload_index",
        collection_name="docs",
        indexes=[{"field_name": name, "field_schema": "keyword"} for name in names],
    )
    assert result.is_error is False
    assert [spec["field_name"] for spec in routes.last_body()["indexes"]] == names


@pytest.mark.parametrize("field", ["lang", "order items"])
async def test_create_index_rejects_equivalent_quoted_names(call, routes, field):
    result = await call(
        "qdrant_create_payload_index",
        collection_name="docs",
        indexes=[
            {"field_name": field, "field_schema": "keyword"},
            {"field_name": json.dumps(field), "field_schema": "keyword"},
        ],
    )
    assert result.is_error is True
    assert routes.calls == []


async def test_list_keys_sends_default_sample_size(call, routes):
    routes.add(
        "GET",
        "/collections/docs/payload-keys",
        body={"collection_name": "docs", "keys": ["text", "lang"]},
    )
    await call("qdrant_list_payload_keys", collection_name="docs")
    assert routes.calls[-1].url.params["sample_size"] == "100"


async def test_list_keys_forwards_custom_sample_size(call, routes):
    routes.add(
        "GET",
        "/collections/docs/payload-keys",
        body={"collection_name": "docs", "keys": []},
    )
    await call("qdrant_list_payload_keys", collection_name="docs", sample_size=500)
    assert routes.calls[-1].url.params["sample_size"] == "500"


async def test_list_keys_rejects_out_of_range_sample_size(call, routes):
    out = await call(
        "qdrant_list_payload_keys", collection_name="docs", sample_size=5000
    )
    assert routes.calls == []
    assert out.is_error is True
    assert "Error" in out or "validation" in out.lower()


async def test_list_keys_renders_markdown_with_keys(call, routes):
    routes.add(
        "GET",
        "/collections/docs/payload-keys",
        body={
            "collection_name": "docs",
            "keys": ["text", "lang"],
            "count": 2,
            "sampled_points": 100,
        },
    )
    out = await call("qdrant_list_payload_keys", collection_name="docs")
    assert "**Collection**: docs" in out
    assert "**Sampled points**: 100" in out
    assert "**Key count**: 2" in out
    assert "- text" in out
    assert "- lang" in out


async def test_list_keys_renders_markdown_when_empty(call, routes):
    routes.add(
        "GET",
        "/collections/docs/payload-keys",
        body={"collection_name": "docs", "keys": [], "count": 0, "sampled_points": 50},
    )
    out = await call("qdrant_list_payload_keys", collection_name="docs")
    assert "**Collection**: docs" in out
    assert "**Sampled points**: 50" in out
    assert "**Key count**: 0" in out
    assert "**Keys**: —" in out
    assert "## Keys" not in out


async def test_list_indexes_renders_field_and_type(call, routes):
    routes.add(
        "GET",
        "/collections/docs/payload-indexes",
        body={
            "collection_name": "docs",
            "indexes": [
                {
                    "field_name": "lang",
                    "field_schema": "keyword",
                    "phrase_matching": None,
                }
            ],
        },
    )
    out = await call("qdrant_list_payload_indexes", collection_name="docs")
    assert "lang" in out and "keyword" in out


async def test_create_index_posts_the_index_list(call, routes):
    routes.add(
        "POST",
        "/collections/docs/payload-indexes",
        body={
            "ok": True,
            "collection_name": "docs",
            "created": ["lang"],
            "updated": [],
            "skipped": [],
            "failed": [],
        },
    )
    await call(
        "qdrant_create_payload_index",
        collection_name="docs",
        indexes=[{"field_name": "lang", "field_schema": "keyword"}],
    )
    assert routes.last_body() == {
        "indexes": [
            {"field_name": "lang", "field_schema": "keyword", "phrase_matching": False}
        ]
    }


async def test_create_index_reports_each_outcome_bucket(call, routes):
    routes.add(
        "POST",
        "/collections/docs/payload-indexes",
        body={
            "ok": True,
            "collection_name": "docs",
            "created": ["a"],
            "updated": ["b"],
            "skipped": ["c"],
            "failed": ["d"],
        },
    )
    payload = json.loads(
        await call(
            "qdrant_create_payload_index",
            collection_name="docs",
            indexes=[{"field_name": "a", "field_schema": "keyword"}],
            response_format="json",
        )
    )
    assert payload["created"] == ["a"]
    assert payload["updated"] == ["b"]
    assert payload["skipped"] == ["c"]
    assert payload["failed"] == ["d"]


async def test_create_index_renders_markdown_across_all_buckets(call, routes):
    routes.add(
        "POST",
        "/collections/docs/payload-indexes",
        body={
            "ok": True,
            "collection_name": "docs",
            "created": ["a"],
            "updated": ["b"],
            "skipped": ["c"],
            "failed": ["d"],
        },
    )
    out = await call(
        "qdrant_create_payload_index",
        collection_name="docs",
        indexes=[{"field_name": "a", "field_schema": "keyword"}],
    )
    assert "**Collection**: docs" in out
    assert "**Created**: a" in out
    assert "**Updated**: b" in out
    assert "**Skipped**: c" in out
    assert "**Failed**: d" in out
    assert "1 index failed to create: d" in out


async def test_create_index_all_failed_leads_with_a_failure_summary(call, routes):
    routes.add(
        "POST",
        "/collections/docs/payload-indexes",
        body={
            "ok": True,
            "collection_name": "docs",
            "created": [],
            "updated": [],
            "skipped": [],
            "failed": ["a", "b"],
        },
    )
    out = await call(
        "qdrant_create_payload_index",
        collection_name="docs",
        indexes=[
            {"field_name": "a", "field_schema": "keyword"},
            {"field_name": "b", "field_schema": "keyword"},
        ],
    )
    # The failure summary must lead the output, ahead of every per-bucket
    # bullet, so a skim of the top of the response cannot read as an
    # ordinary success.
    lines = out.strip().splitlines()
    assert lines[0] == "# Payload indexes created"
    assert lines[1] == "**2 indexes failed to create: a, b**"
    assert "**Created**: —" in out
    assert "**Failed**: a, b" in out


async def test_create_index_rejects_phrase_matching_on_keyword(call, routes):
    # This used to be a Pydantic model_validator on the nested PayloadIndexSpec
    # model (phrase_matching_requires_text); it is now a ToolError raised
    # from the qdrant_create_payload_index tool body (looping over the
    # submitted specs), so it must return the same clean "Error: ..."
    # sentence describe_error produces, not the SDK's own validation-error
    # envelope.
    out = await call(
        "qdrant_create_payload_index",
        collection_name="docs",
        indexes=[
            {"field_name": "lang", "field_schema": "keyword", "phrase_matching": True}
        ],
    )
    assert routes.calls == []
    assert out.is_error is True
    assert out.startswith("Error:")
    assert "phrase_matching requires field_schema 'text'" in out
    assert "validation error" not in out
    assert "pydantic.dev" not in out


async def test_create_index_rejects_duplicate_fields(call, routes):
    # This used to be a Pydantic field_validator (reject_duplicate_fields);
    # it is now a ToolError raised from the tool body, so it must return the
    # same clean "Error: ..." sentence describe_error produces, not the
    # SDK's own validation-error envelope.
    out = await call(
        "qdrant_create_payload_index",
        collection_name="docs",
        indexes=[
            {"field_name": "lang", "field_schema": "keyword"},
            {"field_name": "lang", "field_schema": "text"},
        ],
    )
    assert routes.calls == []
    assert out.is_error is True
    assert out.startswith("Error:")
    assert "duplicate payload index field: lang" in out
    assert "validation error" not in out
    assert "pydantic.dev" not in out


async def test_delete_index_requires_confirm(call, routes):
    out = await call(
        "qdrant_delete_payload_index", collection_name="docs", field_name="lang"
    )
    assert routes.calls == []
    assert out.is_error is True
    assert "confirm" in out


async def test_delete_index_calls_the_field_path(call, routes):
    routes.add(
        "DELETE",
        "/collections/docs/payload-indexes/lang",
        body={"ok": True, "collection_name": "docs", "deleted": True},
    )
    await call(
        "qdrant_delete_payload_index",
        collection_name="docs",
        field_name="lang",
        confirm="docs",
    )
    assert routes.calls[-1].method == "DELETE"


async def test_delete_index_renders_the_outcome_in_markdown(call, routes):
    routes.add(
        "DELETE",
        "/collections/docs/payload-indexes/lang",
        body={
            "ok": True,
            "collection_name": "docs",
            "field_name": "lang",
            "deleted": True,
        },
    )
    out = await call(
        "qdrant_delete_payload_index",
        collection_name="docs",
        field_name="lang",
        confirm="docs",
    )
    assert "**Deleted**: True" in out
    assert "**Collection**: docs" in out
    assert "**Field**: lang" in out


async def test_delete_index_url_encodes_a_slash_in_the_field_name(call, routes):
    # `.url.path` decodes "%2F" back to "/", so only `raw_path` — the bytes
    # actually sent on the wire — proves quote(..., safe="") ran on
    # field_name: with it, the name stays one path segment ("a%2Fb"); without
    # it, "/" would split the URL and this route would never match as one
    # field name.
    routes.add(
        "DELETE",
        "/collections/docs/payload-indexes/a/b",
        body={"ok": True, "collection_name": "docs", "deleted": True},
    )
    await call(
        "qdrant_delete_payload_index",
        collection_name="docs",
        field_name="a/b",
        confirm="docs",
    )
    raw_path = routes.calls[-1].url.raw_path.split(b"?")[0]
    assert raw_path == b"/collections/docs/payload-indexes/a%2Fb"


async def test_delete_index_confirm_gate_makes_zero_http_calls(call, routes):
    routes.add(
        "DELETE",
        "/collections/docs/payload-indexes/lang",
        body={"ok": True, "collection_name": "docs", "deleted": True},
    )
    out = await call(
        "qdrant_delete_payload_index",
        collection_name="docs",
        field_name="lang",
        confirm="Docs",
    )
    assert routes.calls == []
    assert out.is_error is True
    assert "Error" in out

"""Collection listing, reporting and verification."""

import json

# Field names and types copied from CollectionReport
# (middleware-qdrant-manager/qdrant_middleware/schemas/collections.py:65-84).
# Note: quantization is a STRING describing the configured quantization, not a
# boolean toggle, and there is no "vectors" or "toggles" grouping.
REPORT = {
    "collection_name": "docs",
    "exists": True,
    "schema_ok": True,
    "errors": [],
    "findings": [],
    "suggested_fixes": [],
    "status": "green",
    "points_count": 120,
    "vectors_count": 120,
    "dense": {"size": 1024, "distance": "Cosine", "on_disk": False},
    "sparse": {"idf": True, "on_disk": False},
    "quantization": "scalar",
    "shards": 1,
    "replicas": 1,
    "on_disk_payload": True,
    "embedding_compatibility": None,
}


async def test_list_collections_renders_names(call, routes):
    routes.add("GET", "/collections", body={"collections": ["docs", "notes"]})
    out = await call("qdrant_list_collections")
    assert "docs" in out and "notes" in out
    assert out.is_error is False


async def test_list_collections_json_is_paginated(call, routes):
    routes.add("GET", "/collections", body={"collections": ["docs"]})
    payload = json.loads(await call("qdrant_list_collections", response_format="json"))
    assert payload["count"] == 1
    assert payload["items"] == ["docs"]
    assert payload["has_more"] is False


async def test_list_collections_reports_none_found(call, routes):
    routes.add("GET", "/collections", body={"collections": []})
    assert "No collections" in await call("qdrant_list_collections")


async def test_get_collection_sends_validate_true_by_default(call, routes):
    routes.add("GET", "/collections/docs", body=REPORT)
    await call("qdrant_get_collection", collection_name="docs")
    assert routes.calls[-1].url.params["validate"] == "true"


async def test_get_collection_can_skip_validation(call, routes):
    routes.add("GET", "/collections/docs", body=REPORT)
    await call("qdrant_get_collection", collection_name="docs", validate=False)
    assert routes.calls[-1].url.params["validate"] == "false"


async def test_get_collection_json_returns_the_report(call, routes):
    routes.add("GET", "/collections/docs", body=REPORT)
    payload = json.loads(
        await call(
            "qdrant_get_collection", collection_name="docs", response_format="json"
        )
    )
    assert payload["points_count"] == 120


async def test_get_collection_missing_is_actionable(call, routes):
    routes.add("GET", "/collections/gone", status=404, body={"detail": "not found"})
    out = await call("qdrant_get_collection", collection_name="gone")
    assert out.is_error is True
    assert "qdrant_list_collections" in out


async def test_get_collection_url_encodes_a_space_in_the_name(call, routes):
    # httpx re-encodes a raw space and decodes it back on `.url.path`
    # regardless of whether the tool calls quote() first, so this case alone
    # cannot prove encoding happened. Kept for coverage, not as the proof —
    # see the slash/percent tests below for cases that actually distinguish.
    routes.add("GET", "/collections/my docs", body=REPORT)
    await call("qdrant_get_collection", collection_name="my docs")
    assert routes.calls[-1].url.path == "/collections/my docs"


async def test_get_collection_url_encodes_a_slash_in_the_name(call, routes):
    # `.url.path` decodes "%2F" back to "/", so an unencoded and an encoded
    # slash both dispatch to the same fake route. Only `raw_path` — the bytes
    # actually sent on the wire — shows whether quote(..., safe="") ran: with
    # it, the name stays a single path segment ("my%2Fdoc"); without it, "/"
    # would split the URL into two segments and never reach this route as
    # one name at all.
    routes.add("GET", "/collections/my/doc", body=REPORT)
    await call("qdrant_get_collection", collection_name="my/doc")
    raw_path = routes.calls[-1].url.raw_path.split(b"?")[0]
    assert raw_path == b"/collections/my%2Fdoc"


async def test_get_collection_url_encodes_a_percent_in_the_name(call, routes):
    # Same reasoning for a literal "%": encoded it is safely "%25doc" on the
    # wire; unencoded it would reach the middleware as a raw, unescaped "%".
    routes.add("GET", "/collections/my%doc", body=REPORT)
    await call("qdrant_get_collection", collection_name="my%doc")
    raw_path = routes.calls[-1].url.raw_path.split(b"?")[0]
    assert raw_path == b"/collections/my%25doc"


async def test_get_collection_reports_unchecked_schema(call, routes):
    routes.add("GET", "/collections/docs", body={**REPORT, "schema_ok": None})
    out = await call("qdrant_get_collection", collection_name="docs", validate=False)
    assert "not checked" in out


# Field names as the renderer reads them from EmbeddingCompatibility
# (middleware-qdrant-manager/qdrant_middleware/schemas/embedding_contract.py:
# 121-129): the status verdict, the stored/active contract profiles, and the
# message explaining a mismatch — the same fields the frontend's collection
# report surfaces.
COMPATIBILITY = {
    "status": "incompatible",
    "stored": {"profile": "basic"},
    "active": {"profile": "advanced"},
    "message": "dense model bge-m3 differs from the active embedder",
}


async def test_get_collection_renders_embedding_compatibility(call, routes):
    body = {**REPORT, "embedding_compatibility": COMPATIBILITY}
    routes.add("GET", "/collections/docs", body=body)
    out = await call("qdrant_get_collection", collection_name="docs")
    assert "**Embedding compatibility**: incompatible" in out
    assert "**Stored profile**: basic" in out
    assert "**Active profile**: advanced" in out
    assert "**Compatibility detail**: dense model bge-m3 differs" in out


async def test_get_collection_renders_a_compatible_verdict(call, routes):
    compatible = {
        **COMPATIBILITY,
        "status": "compatible",
        "active": {"profile": "basic"},
        "message": "collection matches the active embedding contract",
    }
    body = {**REPORT, "embedding_compatibility": compatible}
    routes.add("GET", "/collections/docs", body=body)
    out = await call("qdrant_get_collection", collection_name="docs")
    assert "**Embedding compatibility**: compatible" in out
    assert "**Stored profile**: basic" in out


async def test_get_collection_renders_a_missing_compatibility_as_a_dash(call, routes):
    # REPORT's embedding_compatibility is None (validate=false or an
    # unreachable embedder): the verdict line must still render, as an em
    # dash, and none of the nested-field lines should appear.
    routes.add("GET", "/collections/docs", body=REPORT)
    out = await call("qdrant_get_collection", collection_name="docs")
    assert "**Embedding compatibility**: —" in out
    assert "**Stored profile**" not in out
    assert "**Active profile**" not in out
    assert "**Compatibility detail**" not in out


async def test_verify_posts_expected_toggles(call, routes):
    routes.add("POST", "/collections/verify", body=REPORT)
    await call(
        "qdrant_verify_collection",
        collection_name="docs",
        quantization=True,
        sparse_idf=False,
    )
    assert routes.last_body() == {
        "collection_name": "docs",
        "quantization": True,
        "sparse_idf": False,
    }


async def test_verify_surfaces_schema_errors(call, routes):
    body = {**REPORT, "errors": ["dense vector size mismatch: 1024 != 768"]}
    routes.add("POST", "/collections/verify", body=body)
    out = await call("qdrant_verify_collection", collection_name="docs")
    assert "1024 != 768" in out


async def test_verify_reports_missing_collection_without_error(call, routes):
    # verify_collection_endpoint answers 200 even when the collection is
    # missing: absence shows up as exists=false / schema_ok=false with the
    # reason in `errors`, never as a 404. A tool that mistranslated this into
    # an "Error: ..." string would give a false negative to anything
    # scanning for that prefix to detect failure.
    body = {
        **REPORT,
        "exists": False,
        "schema_ok": False,
        "errors": ["collection 'gone' not found"],
    }
    routes.add("POST", "/collections/verify", body=body)
    out = await call("qdrant_verify_collection", collection_name="gone")
    assert "collection 'gone' not found" in out
    assert "Error:" not in out
    assert out.is_error is False

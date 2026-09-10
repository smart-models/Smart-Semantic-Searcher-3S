"""Collection creation, toggle updates, deletion, and their confirm gates."""

import pytest

OK = {
    "ok": True,
    "created": True,
    "collection_name": "docs",
    "report": {"collection_name": "docs", "exists": True, "errors": []},
}
TOGGLES = {
    "dense_on_disk": False,
    "quantization": True,
    "sparse_idf": True,
    "sparse_on_disk": False,
}


async def test_response_format_literal_violation_uses_the_framework_envelope(
    call, routes
):
    # Pins the boundary deliberately: response_format enum membership is
    # shape validation, not a cross-field business rule, so it stays a
    # Pydantic check and the SDK's own validation-error envelope (with the
    # pydantic.dev URL) is the expected -- not accidental -- result here,
    # unlike every refusal converted to a ToolError elsewhere in this suite.
    out = await call(
        "qdrant_create_collection",
        collection_name="docs",
        response_format="bogus",
        **TOGGLES,
    )
    assert routes.calls == []
    assert out.is_error is True
    assert "1 validation error" in out
    assert "pydantic.dev" in out


async def test_create_posts_all_required_toggles(call, routes):
    routes.add("POST", "/collections", body=OK)
    await call("qdrant_create_collection", collection_name="docs", **TOGGLES)
    assert routes.last_body() == {
        "collection_name": "docs",
        "recreate": False,
        **TOGGLES,
    }


async def test_create_without_recreate_needs_no_confirm(call, routes):
    routes.add("POST", "/collections", body=OK)
    out = await call("qdrant_create_collection", collection_name="docs", **TOGGLES)
    assert "Error" not in out
    assert out.is_error is False


async def test_recreate_without_confirm_is_refused_before_http(call, routes):
    out = await call(
        "qdrant_create_collection", collection_name="docs", recreate=True, **TOGGLES
    )
    assert routes.calls == []
    assert out.is_error is True
    assert "confirm" in out and "docs" in out


async def test_recreate_without_an_explicit_collection_name_is_refused_before_http(
    call, routes
):
    # This used to be a Pydantic model_validator (recreate_requires_an_explicit_name);
    # it is now a ToolError raised from the tool body, so it must return the
    # same clean "Error: ..." sentence describe_error produces for every
    # other guardrail, not the SDK's own validation-error envelope.
    out = await call(
        "qdrant_create_collection", recreate=True, confirm="docs", **TOGGLES
    )
    assert routes.calls == []
    assert out.is_error is True
    assert out.startswith("Error:")
    assert (
        "recreate=true requires an explicit collection_name so that confirm "
        "can be checked against it"
    ) in out
    assert "validation error" not in out
    assert "pydantic.dev" not in out


async def test_recreate_with_wrong_confirm_is_refused(call, routes):
    out = await call(
        "qdrant_create_collection",
        collection_name="docs",
        recreate=True,
        confirm="Docs",
        **TOGGLES,
    )
    assert routes.calls == []
    assert out.is_error is True
    assert "Error" in out


async def test_recreate_forwards_confirm_as_confirm_recreate(call, routes):
    routes.add("POST", "/collections", body=OK)
    await call(
        "qdrant_create_collection",
        collection_name="docs",
        recreate=True,
        confirm="docs",
        **TOGGLES,
    )
    body = routes.last_body()
    assert body["confirm_recreate"] == "docs"
    assert "confirm" not in body


@pytest.mark.parametrize("bad_confirm", [" docs", "docs ", "Docs"])
async def test_recreate_confirm_is_compared_exactly(call, routes, bad_confirm):
    # str_strip_whitespace=True on the model must not launder a padded or
    # differently-cased confirm token into a match: require_confirm's
    # contract is an exact comparison against the value the caller typed.
    out = await call(
        "qdrant_create_collection",
        collection_name="docs",
        recreate=True,
        confirm=bad_confirm,
        **TOGGLES,
    )
    assert routes.calls == []
    assert out.is_error is True
    assert "Error" in out


async def test_schema_toggles_uses_patch(call, routes):
    routes.add(
        "PATCH", "/collections/docs/schema-toggles", body={**OK, "created": False}
    )
    await call(
        "qdrant_update_schema_toggles", collection_name="docs", quantization=True
    )
    assert routes.calls[-1].method == "PATCH"
    assert routes.last_body() == {"quantization": True}


async def test_schema_toggles_reports_upstream_mismatch(call, routes):
    detail = {"message": "Collection update did not match requested schema toggles."}
    routes.add(
        "PATCH", "/collections/docs/schema-toggles", status=409, body={"detail": detail}
    )
    out = await call(
        "qdrant_update_schema_toggles", collection_name="docs", quantization=True
    )
    assert out.is_error is True
    assert "conflict" in out.lower()


async def test_schema_toggles_without_any_toggle_is_refused_before_http(call, routes):
    # This used to be a Pydantic model_validator (require_at_least_one_toggle);
    # it is now a ToolError raised from the tool body, so it must return the
    # same clean "Error: ..." sentence describe_error produces for every
    # other guardrail, not the SDK's own validation-error envelope.
    out = await call("qdrant_update_schema_toggles", collection_name="docs")
    assert routes.calls == []
    assert out.is_error is True
    assert out.startswith("Error:")
    assert "at least one schema toggle must be provided" in out
    assert "validation error" not in out
    assert "pydantic.dev" not in out


async def test_delete_without_confirm_is_refused_before_http(call, routes):
    out = await call("qdrant_delete_collection", collection_name="docs")
    assert routes.calls == []
    assert out.is_error is True
    assert "confirm" in out


async def test_confirm_over_255_chars_fails_validation_before_http(call, routes):
    # ExactString caps confirmation tokens at 255 chars, mirroring the
    # middleware's own confirm_delete Query(max_length=255). An over-long
    # token is a shape violation, not a business rule, so it stays a
    # Pydantic check: the SDK's validation-error envelope is the expected
    # result, and the require_confirm guardrail never runs — no HTTP call.
    out = await call(
        "qdrant_delete_collection", collection_name="docs", confirm="d" * 256
    )
    assert routes.calls == []
    assert out.is_error is True
    assert "1 validation error" in out


async def test_confirm_at_the_255_char_cap_reaches_the_guardrail(call, routes):
    # Boundary check: exactly 255 chars is valid shape, so it passes
    # Pydantic and fails only the exact-match guardrail (a ToolError) —
    # proving the cap is 255, not 256.
    out = await call(
        "qdrant_delete_collection", collection_name="docs", confirm="d" * 255
    )
    assert routes.calls == []
    assert out.is_error is True
    assert out.startswith("Error:")
    assert "validation error" not in out


@pytest.mark.parametrize("bad_confirm", [" docs", "docs ", "Docs"])
async def test_delete_confirm_is_compared_exactly(call, routes, bad_confirm):
    out = await call(
        "qdrant_delete_collection", collection_name="docs", confirm=bad_confirm
    )
    assert routes.calls == []
    assert out.is_error is True
    assert "Error" in out


async def test_delete_forwards_confirm_as_query_param(call, routes):
    routes.add(
        "DELETE",
        "/collections/docs",
        body={"ok": True, "deleted": True, "collection_name": "docs"},
    )
    await call("qdrant_delete_collection", collection_name="docs", confirm="docs")
    assert routes.calls[-1].url.params["confirm_delete"] == "docs"


async def test_delete_reports_the_outcome(call, routes):
    routes.add(
        "DELETE",
        "/collections/docs",
        body={"ok": True, "deleted": True, "collection_name": "docs"},
    )
    out = await call("qdrant_delete_collection", collection_name="docs", confirm="docs")
    assert "Deleted" in out
    assert out.is_error is False

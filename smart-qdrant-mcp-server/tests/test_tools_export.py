"""Whole-collection export: the request it sends and what it renders back."""

from smart_qdrant_mcp.tools.export import ExportCollectionInput

RESULT = {
    "ok": True,
    "collection_name": "docs",
    "n_chunks": 42,
    "path": "/app/middleware-qdrant-manager/exports/docs_export_x.json",
    "bytes": 1234,
}


async def test_export_always_asks_for_the_file_destination(call, routes):
    routes.add("POST", "/points/export", body=RESULT)

    await call("qdrant_export_collection")

    assert routes.last_body() == {"destination": "file"}


async def test_export_forwards_an_explicit_collection_name(call, routes):
    routes.add("POST", "/points/export", body=RESULT)

    await call("qdrant_export_collection", collection_name="other")

    assert routes.last_body()["collection_name"] == "other"


async def test_export_renders_the_path_and_the_count(call, routes):
    routes.add("POST", "/points/export", body=RESULT)

    result = await call("qdrant_export_collection")

    assert "42" in result
    assert "docs_export_x.json" in result


async def test_export_input_model_has_no_destination_field():
    # Exposing destination="body" to an agent would put the whole collection
    # in its context window, which is what the endpoint exists to avoid.
    assert "destination" not in ExportCollectionInput.model_fields


async def test_export_reports_a_cap_rejection_as_an_error(call, routes):
    routes.add(
        "POST",
        "/points/export",
        status=413,
        body={
            "detail": (
                "collection 'docs' holds 90000 points, above EXPORT_MAX_POINTS=50000"
            )
        },
    )

    result = await call("qdrant_export_collection")

    assert "EXPORT_MAX_POINTS" in result


async def test_export_of_an_empty_collection_reports_no_path(call, routes):
    routes.add(
        "POST",
        "/points/export",
        body={
            "ok": True,
            "collection_name": "docs",
            "n_chunks": 0,
            "path": None,
            "bytes": 2,
        },
    )

    result = await call("qdrant_export_collection")

    assert "0" in result

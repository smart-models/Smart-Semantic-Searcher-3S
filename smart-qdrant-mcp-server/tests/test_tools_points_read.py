"""Point retrieval, scrolling, counting and faceting."""

import json

from smart_qdrant_mcp.tools.points import ScrollPointsInput

RETRIEVED = {
    "collection_name": "docs",
    "returned": 1,
    "results": [{"id": "p1", "payload": {"text": "a"}}],
}
SCROLLED = {
    "collection_name": "docs",
    "returned": 2,
    "total": 7,
    "next_page_offset": "cursor-2",
    "results": [{"id": "p1", "payload": {}}, {"id": "p2", "payload": {}}],
}


async def test_read_points_uses_the_batch_endpoint(call, routes):
    routes.add("POST", "/points/retrieve", body=RETRIEVED)
    await call("qdrant_read_points", ids=["p1"], collection_name="docs")
    assert routes.calls[-1].url.path == "/points/retrieve"
    assert routes.last_body() == {
        "ids": ["p1"],
        "collection_name": "docs",
        "with_vectors": False,
    }


async def test_read_points_reports_missing_ids_separately(call, routes):
    routes.add("POST", "/points/retrieve", body=RETRIEVED)
    payload = json.loads(
        await call("qdrant_read_points", ids=["p1", "p2"], response_format="json")
    )
    assert [hit["id"] for hit in payload["found"]] == ["p1"]
    assert payload["not_found"] == ["p2"]


async def test_read_points_markdown_names_missing_ids(call, routes):
    routes.add("POST", "/points/retrieve", body=RETRIEVED)
    out = await call("qdrant_read_points", ids=["p1", "p2"])
    assert "p2" in out
    assert "not found" in out.lower()


async def test_read_points_forwards_with_vectors(call, routes):
    routes.add("POST", "/points/retrieve", body=RETRIEVED)
    await call("qdrant_read_points", ids=["p1"], with_vectors=True)
    assert routes.last_body() == {"ids": ["p1"], "with_vectors": True}


async def test_read_points_rejects_more_than_a_thousand_ids(call, routes):
    out = await call("qdrant_read_points", ids=[str(n) for n in range(1001)])
    assert routes.calls == []
    assert out.is_error is True
    assert "Error" in out or "validation" in out.lower()


async def test_scroll_defaults_to_limit_twenty(call, routes):
    routes.add("POST", "/points/scroll", body=SCROLLED)
    await call("qdrant_scroll_points")
    assert routes.last_body() == {
        "limit": 20,
        "with_payload": True,
        "include_total": True,
    }


async def test_scroll_returns_the_upstream_cursor(call, routes):
    routes.add("POST", "/points/scroll", body=SCROLLED)
    payload = json.loads(await call("qdrant_scroll_points", response_format="json"))
    assert payload["total"] == 7
    assert payload["has_more"] is True
    assert payload["next_offset"] == "cursor-2"


async def test_scroll_terminal_cursor_overrides_total_and_offset(call, routes):
    routes.add(
        "POST",
        "/points/scroll",
        body={**SCROLLED, "total": 100, "next_page_offset": None},
    )
    out = await call("qdrant_scroll_points", offset="last-uuid", response_format="json")
    payload = json.loads(out)
    assert payload["has_more"] is False
    assert payload["next_offset"] is None


async def test_scroll_zero_is_a_valid_cursor(call, routes):
    routes.add("POST", "/points/scroll", body={**SCROLLED, "next_page_offset": 0})
    payload = json.loads(await call("qdrant_scroll_points", response_format="json"))
    assert payload["has_more"] is True
    assert payload["next_offset"] == 0


async def test_scroll_accepts_a_cursor_offset(call, routes):
    routes.add("POST", "/points/scroll", body=SCROLLED)
    await call("qdrant_scroll_points", offset="cursor-2")
    assert routes.last_body() == {
        "limit": 20,
        "offset": "cursor-2",
        "with_payload": True,
        "include_total": True,
    }


async def test_scroll_forwards_payload_selection(call, routes):
    routes.add("POST", "/points/scroll", body=SCROLLED)
    await call("qdrant_scroll_points", with_payload=["text"])
    assert routes.last_body() == {
        "limit": 20,
        "with_payload": ["text"],
        "include_total": True,
    }


async def test_scroll_markdown_shows_the_cursor_for_the_next_page(call, routes):
    routes.add("POST", "/points/scroll", body=SCROLLED)
    out = await call("qdrant_scroll_points")
    assert "- **More pages**: yes — pass `cursor-2` as offset" in out


async def test_scroll_markdown_marks_the_last_page_as_complete(call, routes):
    routes.add(
        "POST",
        "/points/scroll",
        body={**SCROLLED, "total": 2, "next_page_offset": None},
    )
    out = await call("qdrant_scroll_points")
    assert "- **More pages**: no — the result set is complete" in out


def test_scroll_limit_description_documents_the_divergent_default():
    """The scroll limit's description explains its deliberate drift.

    The middleware's own ScrollPointsRequest defaults limit to 10 (middleware
    points.py:322); this server deliberately overrides that with 20, and the
    field description must say so — otherwise the drift looks accidental to
    an agent reading the schema.
    """
    schema = ScrollPointsInput.model_json_schema()
    description = schema["properties"]["limit"]["description"]
    assert "deliberate default" in description
    assert "middleware" in description
    assert "10" in description


async def test_count_returns_the_number(call, routes):
    routes.add(
        "POST",
        "/points/count",
        body={"ok": True, "collection_name": "docs", "count": 41, "exact": True},
    )
    out = await call("qdrant_count_points", filters={"must": []})
    assert routes.last_body() == {"filters": {"must": []}, "exact": True}
    assert "41" in out


async def test_facet_returns_values_with_counts(call, routes):
    routes.add(
        "POST",
        "/points/facet",
        body={
            "ok": True,
            "collection_name": "docs",
            "key": "lang",
            "hits": [{"value": "it", "count": 30}, {"value": "en", "count": 11}],
        },
    )
    out = await call("qdrant_facet_points", key="lang")
    assert routes.last_body() == {"key": "lang", "limit": 20, "exact": True}
    assert "it" in out and "30" in out


async def test_facet_requires_a_non_blank_key(call, routes):
    out = await call("qdrant_facet_points", key="   ")
    assert routes.calls == []
    assert out.is_error is True
    assert "Error" in out or "validation" in out.lower()

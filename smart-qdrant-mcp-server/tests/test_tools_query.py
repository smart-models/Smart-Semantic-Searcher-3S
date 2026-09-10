"""qdrant_search forwards query options and paginates its results."""

import json

from mcp.client import Client

from smart_qdrant_mcp.config import load_settings
from smart_qdrant_mcp.server import build_server
from tests.conftest import ENV

# Every field SearchInput can send upstream, excluding `query` (always sent)
# and `rerank_normalize` (the one field with a non-None default, so it is
# always sent too). Used to assert "all optional fields absent" without
# spot-checking a handful.
OPTIONAL_BODY_FIELDS = (
    "collection_name",
    "limit",
    "candidate_limit",
    "fusion",
    "rrf_k",
    "rrf_weights",
    "rerank",
    "rerank_top_k",
    "filters",
    "with_payload",
    "offset",
    "score_threshold",
    "hnsw_ef",
    "exact",
    "rescore",
    "oversampling",
    "timeout",
)


async def _call_with_settings(routes, overrides, tool, **arguments):
    """Call a tool against a server built with custom settings.

    The shared `settings` fixture pins every env var (including
    MCP_DEFAULT_LIMIT) to a fixed value, which is exactly what the
    "the server must not consult its own default limit" tests need to
    control directly.
    """
    settings = load_settings(_env_file=None, **{**ENV, **overrides})
    server, ctx = build_server(settings, transport=routes.transport())
    try:
        async with Client(server) as client:
            result = await client.call_tool(tool, {"params": arguments})
            return "\n".join(
                block.text for block in result.content if getattr(block, "text", None)
            )
    finally:
        await ctx.client.aclose()


RESPONSE = {
    "query": "vector db",
    "collection_name": "docs",
    "fusion": "rrf",
    "reranked": True,
    "candidates": 40,
    "returned": 2,
    "skipped_without_text": 0,
    "offset": 0,
    "score_threshold": None,
    "results": [
        {"id": "p1", "score": 0.9, "rerank_score": 0.81, "payload": {"text": "a"}},
        {"id": "p2", "score": 0.7, "rerank_score": 0.44, "payload": {"text": "b"}},
    ],
}


async def test_posts_to_query_search(call, routes):
    routes.add("POST", "/query/search", body=RESPONSE)
    await call("qdrant_search", query="vector db", collection_name="docs")
    assert routes.calls[-1].url.path == "/query/search"
    assert routes.last_body()["query"] == "vector db"


async def test_omits_unset_options_so_middleware_defaults_apply(call, routes):
    routes.add("POST", "/query/search", body=RESPONSE)
    await call("qdrant_search", query="vector db")
    body = routes.last_body()
    assert body == {"query": "vector db", "rerank_normalize": True}
    for field in OPTIONAL_BODY_FIELDS:
        assert field not in body


async def test_forwards_every_supplied_option(call, routes):
    routes.add("POST", "/query/search", body=RESPONSE)
    await call(
        "qdrant_search",
        query="vector db",
        collection_name="docs",
        limit=5,
        candidate_limit=50,
        fusion="dbsf",
        rerank=True,
        rerank_top_k=10,
        rerank_normalize=False,
        filters={"must": [{"key": "lang", "match": {"value": "it"}}]},
        with_payload=["text", "lang"],
        offset=10,
        score_threshold=0.2,
        hnsw_ef=128,
        exact=False,
        rescore=True,
        oversampling=2.0,
        timeout=30,
    )
    body = routes.last_body()
    assert body == {
        "query": "vector db",
        "collection_name": "docs",
        "limit": 5,
        "candidate_limit": 50,
        "fusion": "dbsf",
        "rerank": True,
        "rerank_top_k": 10,
        "rerank_normalize": False,
        "filters": {"must": [{"key": "lang", "match": {"value": "it"}}]},
        "with_payload": ["text", "lang"],
        "offset": 10,
        "score_threshold": 0.2,
        "hnsw_ef": 128,
        "exact": False,
        "rescore": True,
        "oversampling": 2.0,
        "timeout": 30,
    }


async def test_json_result_carries_the_pagination_envelope(call, routes):
    routes.add("POST", "/query/search", body=RESPONSE)
    payload = json.loads(
        await call("qdrant_search", query="vector db", limit=2, response_format="json")
    )
    assert payload["count"] == 2
    assert payload["offset"] == 0
    assert payload["has_more"] is True
    assert payload["next_offset"] == 2
    assert payload["items"][0]["id"] == "p1"


async def test_omitted_limit_with_a_nonempty_page_always_reports_more(routes):
    """A non-empty page is reported conservatively as "there may be more".

    The server cannot know the middleware's own page size. The value it must
    not consult, MCP_DEFAULT_LIMIT, is set to a distinctive number here to
    prove it never leaks into the envelope.
    """
    routes.add("POST", "/query/search", body=RESPONSE)
    out = await _call_with_settings(
        routes,
        {"MCP_DEFAULT_LIMIT": "37"},
        "qdrant_search",
        query="vector db",
        response_format="json",
    )
    payload = json.loads(out)
    assert payload["has_more"] is True
    assert payload["next_offset"] == payload["offset"] + payload["count"]
    assert "37" not in out


async def test_omitted_limit_with_zero_hits_reports_the_end(routes):
    routes.add("POST", "/query/search", body={**RESPONSE, "returned": 0, "results": []})
    out = await _call_with_settings(
        routes,
        {"MCP_DEFAULT_LIMIT": "37"},
        "qdrant_search",
        query="nothing matches this",
        response_format="json",
    )
    payload = json.loads(out)
    assert payload["has_more"] is False
    assert payload["next_offset"] is None


async def test_explicit_limit_with_a_short_page_reports_the_end(call, routes):
    routes.add(
        "POST",
        "/query/search",
        body={**RESPONSE, "returned": 1, "results": RESPONSE["results"][:1]},
    )
    payload = json.loads(
        await call("qdrant_search", query="vector db", limit=5, response_format="json")
    )
    assert payload["has_more"] is False
    assert payload["next_offset"] is None


async def test_json_result_keeps_the_ranking_metadata(call, routes):
    routes.add("POST", "/query/search", body=RESPONSE)
    payload = json.loads(
        await call("qdrant_search", query="vector db", response_format="json")
    )
    assert payload["fusion"] == "rrf"
    assert payload["reranked"] is True
    assert payload["candidates"] == 40


async def test_markdown_lists_hits_with_scores(call, routes):
    routes.add("POST", "/query/search", body=RESPONSE)
    out = await call("qdrant_search", query="vector db")
    assert "p1" in out
    assert "0.9" in out
    assert "0.81" in out
    assert out.is_error is False


async def test_markdown_shows_the_search_diagnostics(call, routes):
    routes.add("POST", "/query/search", body=RESPONSE)
    out = await call("qdrant_search", query="vector db", limit=2)
    assert "- **Candidates**: 40" in out
    assert "- **Skipped without text**: 0" in out
    assert "- **Score threshold**: —" in out


async def test_markdown_shows_the_next_offset_when_more_pages_remain(call, routes):
    routes.add("POST", "/query/search", body=RESPONSE)
    out = await call("qdrant_search", query="vector db", limit=2)
    assert "- **More pages**: yes — pass `2` as offset to fetch the next page" in out


async def test_markdown_marks_a_short_page_as_the_end(call, routes):
    routes.add(
        "POST",
        "/query/search",
        body={**RESPONSE, "returned": 1, "results": RESPONSE["results"][:1]},
    )
    out = await call("qdrant_search", query="vector db", limit=5)
    assert "- **More pages**: no — the result set is complete" in out


async def test_empty_results_are_reported_plainly(call, routes):
    routes.add("POST", "/query/search", body={**RESPONSE, "returned": 0, "results": []})
    out = await call("qdrant_search", query="nothing matches this")
    assert "No results" in out


async def test_upstream_validation_error_is_actionable(call, routes):
    detail = [{"loc": ["body", "limit"], "msg": "Input should be less than 200"}]
    routes.add("POST", "/query/search", status=400, body={"detail": detail})
    out = await call("qdrant_search", query="x", limit=5)
    assert out.is_error is True
    assert out.startswith("Error: invalid request")
    assert "limit" in out


async def test_blank_query_is_rejected_before_any_http_call(call, routes):
    out = await call("qdrant_search", query="   ")
    assert routes.calls == []
    assert out.is_error is True
    assert "Error" in out or "validation" in out.lower()

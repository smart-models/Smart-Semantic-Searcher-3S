"""Tests for build_qdrant_filter validation and limits."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from qdrant_client import models

from qdrant_middleware.clients.embedder import RerankResult
from qdrant_middleware.core.config import Settings
from qdrant_middleware.services.filter_builder import build_qdrant_filter
from qdrant_middleware.services.query_executor import (
    QueryHit,
    QueryRequest,
    _build_search_params,
    _rerank_hits,
    run_hybrid_query,
)


@pytest.mark.parametrize("rescore", [True, False])
def test_quantized_search_preserves_explicit_rescore_setting(rescore):
    """Quantized searches send both true and false rescore choices to Qdrant."""
    params = _build_search_params(
        True,
        Settings(QDRANT_COLLECTION="docs", QUERY_QUANTIZATION_RESCORE=rescore),
    )

    assert params is not None
    assert params.quantization is not None
    assert params.quantization.rescore is rescore


def test_legacy_scalar_filter_builds_match_value():
    flt = build_qdrant_filter({"type": "paragraph"})

    assert isinstance(flt, models.Filter)
    assert len(flt.must) == 1
    condition = flt.must[0]
    assert condition.key == "type"
    assert condition.match.value == "paragraph"


def test_legacy_list_filter_builds_match_any():
    flt = build_qdrant_filter({"type": ["paragraph", "table"]})

    condition = flt.must[0]
    assert condition.key == "type"
    assert condition.match.any == ["paragraph", "table"]


@pytest.mark.parametrize(
    "value",
    [["paragraph", 1], [True], [{}]],
)
def test_legacy_list_filter_rejects_invalid_match_any_values(value):
    with pytest.raises(ValueError, match="legacy filter 'type'"):
        build_qdrant_filter({"type": value})


def test_legacy_space_field_is_quoted_for_qdrant():
    flt = build_qdrant_filter({"Header 1": "Intro"})

    assert flt.must[0].key == '"Header 1"'
    assert flt.must[0].match.value == "Intro"


@pytest.mark.parametrize(
    "key", ["a.b", "items[].label", '"a.b"', '"items[].label"', '"order items"[].label']
)
@pytest.mark.parametrize("legacy", [True, False])
def test_filter_preserves_traversal_and_explicit_literals(key, legacy):
    """Path syntax must survive conversion to the actual Qdrant condition."""
    filters = {key: "x"} if legacy else {"must": [{"key": key, "is_empty": True}]}
    assert build_qdrant_filter(filters).must[0].key == key


def test_query_request_accepts_custom_filter_before_runtime_validation():
    request = QueryRequest(query="abc", filters={"custom": "x"})

    assert request.filters == {"custom": "x"}


def test_apply_defaults_rejects_custom_filter_when_runtime_allowlist_is_set():
    request = QueryRequest(query="abc", filters={"custom": "x"})
    settings = Settings(
        QDRANT_COLLECTION="docs",
        QUERY_FILTER_FIELD_ALLOWLIST="source_id,type",
    )

    with pytest.raises(ValueError, match="not allowed: custom"):
        request.apply_defaults(settings)


def test_apply_defaults_allows_custom_filter_when_runtime_allowlist_is_empty():
    request = QueryRequest(query="abc", filters={"custom": "x"})
    settings = Settings(QDRANT_COLLECTION="docs")

    resolved = request.apply_defaults(settings)

    assert resolved.filters == {"custom": "x"}


def test_apply_defaults_preserves_legacy_filter():
    request = QueryRequest(query="abc", filters={"type": "table"})
    resolved = request.apply_defaults(Settings(QDRANT_COLLECTION="docs"))

    assert resolved.filters == {"type": "table"}


def test_advanced_match_value_filter():
    flt = build_qdrant_filter(
        {"must": [{"key": "type", "match": {"value": "paragraph"}}]}
    )

    assert flt.must[0].key == "type"
    assert flt.must[0].match.value == "paragraph"


def test_advanced_match_any_filter():
    flt = build_qdrant_filter(
        {"must": [{"key": "type", "match": {"any": ["paragraph", "table"]}}]}
    )

    assert flt.must[0].match.any == ["paragraph", "table"]


def test_advanced_match_except_filter():
    flt = build_qdrant_filter(
        {"must": [{"key": "type", "match": {"except": ["header"]}}]}
    )

    assert flt.must[0].match.except_ == ["header"]


def test_advanced_match_text_filter():
    flt = build_qdrant_filter(
        {"must": [{"key": "description", "match": {"text": "laptop portable"}}]}
    )

    assert flt.must[0].match.text == "laptop portable"


def test_advanced_match_text_any_filter():
    flt = build_qdrant_filter(
        {"must": [{"key": "description", "match": {"text_any": "good cheap"}}]}
    )

    assert flt.must[0].match.text_any == "good cheap"


def test_advanced_match_phrase_filter():
    flt = build_qdrant_filter(
        {
            "must": [
                {"key": "description", "match": {"phrase": "supply chain disruption"}}
            ]
        }
    )

    assert flt.must[0].match.phrase == "supply chain disruption"


def test_advanced_numeric_range_filter():
    flt = build_qdrant_filter(
        {"must": [{"key": "token_count", "range": {"gte": 100, "lt": 500}}]}
    )

    assert flt.must[0].range.gte == 100
    assert flt.must[0].range.lt == 500


def test_advanced_datetime_range_filter_maps_iso_strings_to_datetime_range():
    flt = build_qdrant_filter(
        {"must": [{"key": "created_at", "range": {"gte": "2024-01-01T00:00:00Z"}}]}
    )

    assert isinstance(flt.must[0].range, models.DatetimeRange)
    assert flt.must[0].range.gte.isoformat() == "2024-01-01T00:00:00+00:00"


def test_advanced_logical_should_and_must_not_filters():
    flt = build_qdrant_filter(
        {
            "must": [{"key": "type", "match": {"value": "paragraph"}}],
            "should": [{"key": "source_id", "match": {"value": "doc-1"}}],
            "must_not": [{"key": "status", "match": {"value": "deleted"}}],
        }
    )

    assert flt.must[0].match.value == "paragraph"
    assert flt.should[0].match.value == "doc-1"
    assert flt.must_not[0].match.value == "deleted"


def test_min_should_filter():
    flt = build_qdrant_filter(
        {
            "min_should": {
                "conditions": [
                    {"key": "type", "match": {"value": "paragraph"}},
                    {"key": "type", "match": {"value": "table"}},
                ],
                "min_count": 1,
            }
        }
    )

    assert flt.min_should.min_count == 1
    assert len(flt.min_should.conditions) == 2


def test_has_id_filter():
    ids = [
        "00000000-0000-4000-8000-000000000001",
        "00000000-0000-4000-8000-000000000002",
    ]
    flt = build_qdrant_filter({"must": [{"has_id": ids}]})

    assert flt.must[0].has_id == ids


def test_has_id_converts_json_safe_uint64_string():
    flt = build_qdrant_filter({"must": [{"has_id": ["18446744073709551615"]}]})

    assert flt.must[0].has_id == [18446744073709551615]


def test_has_vector_filter():
    flt = build_qdrant_filter({"must": [{"has_vector": "dense"}]})

    assert flt.must[0].has_vector == "dense"


def test_recursive_filter_condition():
    flt = build_qdrant_filter(
        {
            "must_not": [
                {
                    "must": [
                        {"key": "city", "match": {"value": "London"}},
                        {"key": "color", "match": {"value": "red"}},
                    ]
                }
            ]
        }
    )

    assert isinstance(flt.must_not[0], models.Filter)
    assert len(flt.must_not[0].must) == 2


def test_recursive_filter_inside_min_should():
    flt = build_qdrant_filter(
        {
            "min_should": {
                "conditions": [
                    {
                        "should": [
                            {"key": "type", "match": {"value": "paragraph"}},
                            {"key": "type", "match": {"value": "table"}},
                        ]
                    }
                ],
                "min_count": 1,
            }
        }
    )

    assert isinstance(flt.min_should.conditions[0], models.Filter)
    assert len(flt.min_should.conditions[0].should) == 2


def test_nested_filter():
    flt = build_qdrant_filter(
        {
            "must": [
                {
                    "nested": {
                        "key": "diet",
                        "filter": {
                            "must": [
                                {"key": "food", "match": {"value": "meat"}},
                                {"key": "likes", "match": {"value": True}},
                            ]
                        },
                    }
                }
            ]
        }
    )

    nested = flt.must[0].nested
    assert nested.key == "diet"
    assert nested.filter.must[0].key == "food"
    assert nested.filter.must[1].match.value is True


def test_nested_filter_allowlist_uses_full_indexed_paths():
    settings = Settings(
        QDRANT_COLLECTION="docs",
        QUERY_FILTER_FIELD_ALLOWLIST="items[].label,items[].qty",
    )
    flt = build_qdrant_filter(
        {
            "must": [
                {
                    "nested": {
                        "key": "items",
                        "filter": {
                            "must": [
                                {"key": "label", "match": {"value": "a"}},
                                {"key": "qty", "range": {"gte": 10}},
                            ]
                        },
                    }
                }
            ]
        },
        settings=settings,
    )

    assert flt.must[0].nested.filter.must[0].key == "label"


def test_nested_filter_allowlist_rejects_unlisted_full_path():
    settings = Settings(
        QDRANT_COLLECTION="docs",
        QUERY_FILTER_FIELD_ALLOWLIST="items[].label",
    )
    filters = {
        "must": [
            {
                "nested": {
                    "key": "items",
                    "filter": {"must": [{"key": "qty", "range": {"gte": 10}}]},
                }
            }
        ]
    }

    with pytest.raises(ValueError, match=r"not allowed: items\[\]\.qty"):
        build_qdrant_filter(filters, settings=settings)


def test_values_count_filter():
    flt = build_qdrant_filter({"must": [{"key": "tags", "values_count": {"gte": 2}}]})

    assert flt.must[0].values_count.gte == 2


def test_is_empty_filter():
    flt = build_qdrant_filter({"must": [{"key": "notes", "is_empty": True}]})

    assert flt.must[0].is_empty is True


def test_is_null_filter():
    flt = build_qdrant_filter({"must": [{"key": "optional_field", "is_null": True}]})

    assert flt.must[0].is_null is True


def test_geo_radius_filter():
    flt = build_qdrant_filter(
        {
            "must": [
                {
                    "key": "location",
                    "geo_radius": {
                        "center": {"lat": 45.0, "lon": 9.0},
                        "radius": 1000,
                    },
                }
            ]
        }
    )

    assert flt.must[0].geo_radius.center.lat == 45.0
    assert flt.must[0].geo_radius.center.lon == 9.0
    assert flt.must[0].geo_radius.radius == 1000


def test_geo_bounding_box_filter():
    flt = build_qdrant_filter(
        {
            "must": [
                {
                    "key": "location",
                    "geo_bounding_box": {
                        "top_left": {"lat": 46.0, "lon": 8.0},
                        "bottom_right": {"lat": 44.0, "lon": 10.0},
                    },
                }
            ]
        }
    )

    assert flt.must[0].geo_bounding_box.top_left.lat == 46.0
    assert flt.must[0].geo_bounding_box.bottom_right.lon == 10.0


@pytest.mark.parametrize(
    "operator, value",
    [
        ("range", {"gte": float("inf")}),
        (
            "geo_radius",
            {
                "center": {"lat": 45.0, "lon": float("nan")},
                "radius": 100,
            },
        ),
    ],
)
def test_non_finite_numeric_filter_values_are_rejected(operator, value):
    with pytest.raises(ValueError):
        build_qdrant_filter({"must": [{"key": "location", operator: value}]})


def test_geo_polygon_filter():
    flt = build_qdrant_filter(
        {
            "must": [
                {
                    "key": "location",
                    "geo_polygon": {
                        "exterior": {
                            "points": [
                                {"lat": 45.0, "lon": 9.0},
                                {"lat": 45.1, "lon": 9.0},
                                {"lat": 45.0, "lon": 9.1},
                                {"lat": 45.0, "lon": 9.0},
                            ]
                        }
                    },
                }
            ]
        }
    )

    assert len(flt.must[0].geo_polygon.exterior.points) == 4


@pytest.mark.parametrize(
    "filters, message",
    [
        ({"must": [], "type": "paragraph"}, "cannot mix logical keys"),
        ({"must": {"key": "type"}}, "filters.must must be a list"),
        ({"must": [{"key": "type"}]}, "exactly one supported operator"),
        (
            {"must": [{"key": "type", "match": {"value": "x", "any": ["y"]}}]},
            "match must contain exactly one operator",
        ),
        (
            {"must": [{"key": "type", "unknown": True}]},
            "unsupported field condition keys",
        ),
        (
            {"min_should": {"conditions": [], "min_count": 1}},
            "conditions must be a non-empty list",
        ),
        (
            {
                "min_should": {
                    "conditions": [{"key": "type", "match": {"value": "paragraph"}}],
                    "min_count": 2,
                }
            },
            "min_should.min_count cannot exceed condition count",
        ),
        (
            {
                "must": [
                    {
                        "key": "created_at",
                        "range": {"gte": "2024-01-01T00:00:00Z", "lt": 500},
                    }
                ]
            },
            "range bounds must be all numeric or all ISO datetime strings",
        ),
    ],
)
def test_invalid_advanced_filters_raise_value_error(filters, message):
    with pytest.raises(ValueError, match=message):
        build_qdrant_filter(filters)


def test_filter_depth_limit_is_enforced():
    filters = {
        "must": [
            {
                "nested": {
                    "key": "items",
                    "filter": {
                        "must": [
                            {
                                "nested": {
                                    "key": "inner",
                                    "filter": {
                                        "must": [
                                            {"key": "type", "match": {"value": "x"}}
                                        ]
                                    },
                                }
                            }
                        ]
                    },
                }
            }
        ]
    }
    settings = Settings(QDRANT_COLLECTION="docs", QUERY_FILTER_MAX_DEPTH=2)

    with pytest.raises(ValueError, match="maximum depth"):
        build_qdrant_filter(filters, settings=settings)


def test_filter_condition_limit_is_enforced():
    filters = {
        "must": [
            {"key": "type", "match": {"value": "a"}},
            {"key": "source_id", "match": {"value": "b"}},
        ]
    }
    settings = Settings(QDRANT_COLLECTION="docs", QUERY_FILTER_MAX_CONDITIONS=1)

    with pytest.raises(ValueError, match="maximum condition count"):
        build_qdrant_filter(filters, settings=settings)


@pytest.mark.parametrize(
    "filters, message",
    [
        (
            {"must": [{"has_id": [1], "typo": True}]},
            "unsupported condition keys",
        ),
        (
            {"must": [{"has_vector": "dense", "typo": True}]},
            "unsupported condition keys",
        ),
        (
            {"min_should": {"conditions": [{"has_id": [1]}], "min_count": True}},
            "positive integer",
        ),
        (
            {"must": [{"key": "tags", "values_count": {"gte": None}}]},
            "non-null",
        ),
        (
            {"must": [{"key": "tags", "values_count": {"gte": True}}]},
            "non-negative integers",
        ),
        (
            {"must": [{"key": "tags", "values_count": {"gte": 1.5}}]},
            "non-negative integers",
        ),
        (
            {"must": [{"key": "location", "geo_radius": "bad"}]},
            "geo_radius must be an object",
        ),
        (
            {
                "must": [
                    {
                        "key": "location",
                        "geo_radius": {
                            "center": {"lat": 91, "lon": 9},
                            "radius": 100,
                        },
                    }
                ]
            },
            "latitude",
        ),
        (
            {
                "must": [
                    {
                        "key": "location",
                        "geo_radius": {
                            "center": {"lat": 45, "lon": 9},
                            "radius": 0,
                        },
                    }
                ]
            },
            "positive",
        ),
    ],
)
def test_malformed_filters_raise_stable_value_error(filters, message):
    with pytest.raises(ValueError, match=message):
        build_qdrant_filter(filters)


# ── run_hybrid_query error mapping ──────────────────────────────────────────


class _FakePagingClient:
    """Return a deliberately unsorted fused pool and capture Qdrant arguments."""

    def __init__(self) -> None:
        self.query_kwargs = None

    async def get_collection(self, collection_name):
        raise RuntimeError("no collection info in this fake")

    async def query_points(self, **kwargs):
        self.query_kwargs = kwargs
        return SimpleNamespace(
            points=[
                SimpleNamespace(id="10", score=0.5, payload={}),
                SimpleNamespace(id="2", score=0.7, payload={}),
                SimpleNamespace(id="1", score=0.5, payload={}),
                SimpleNamespace(id="3", score=0.7, payload={}),
            ]
        )


@pytest.mark.asyncio
async def test_run_hybrid_query_stably_sorts_full_pool_and_ignores_offset():
    """``run_hybrid_query`` no longer applies ``offset`` itself.

    Regression test for the pagination/rerank overlap bug: offset must be
    applied exactly once, by the caller, after any reranking has reordered
    the whole pool — not baked into this function's own slice. Passing
    ``result_limit`` deep enough to cover the page returns the full
    stably-sorted pool from position 0; the caller slices its own page out
    of that.
    """
    settings = Settings(
        QDRANT_COLLECTION="docs",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )
    request = QueryRequest(
        query="hybrid test",
        collection_name="docs",
        limit=2,
        candidate_limit=4,
        offset=1,
        rerank=False,
    ).apply_defaults(settings)
    embedding = SimpleNamespace(
        dense=[0.1, 0.2], sparse=models.SparseVector(indices=[0], values=[1.0])
    )
    client = _FakePagingClient()

    # offset(1) + limit(2) = the depth this page needs, from position 0.
    hits = await run_hybrid_query(
        request,
        embedding,
        client,
        settings,
        result_limit=3,
    )

    assert client.query_kwargs["limit"] == 4
    assert client.query_kwargs["offset"] is None
    # Full stable pool truncated to depth 3 from position 0 — no offset
    # applied here; the caller (query.py's _build_response) slices [1:3].
    assert [hit.id for hit in hits] == ["2", "3", "1"]


class _FakeRerankEmbedder:
    """Stub embedder: reranks passages by a fixed score keyed on their text."""

    def __init__(self, scores_by_text: dict[str, float]):
        self._scores = scores_by_text

    async def rerank(self, query, passages, normalize=False):
        # The real Smart Embedder /rerank endpoint returns results sorted by
        # score descending (each still carrying its original input index) —
        # _rerank_hits trusts that order rather than re-sorting itself, so
        # this fake must match that contract.
        scored = [(i, self._scores[p]) for i, p in enumerate(passages)]
        scored.sort(key=lambda pair: -pair[1])
        return [RerankResult(index=i, score=score) for i, score in scored]


@pytest.mark.asyncio
async def test_reranked_pagination_does_not_overlap_across_pages():
    """Regression test for the rerank+offset pagination bug.

    Before the fix, ``offset`` was applied *before* reranking, so each page
    reranked a different, overlapping window of the fused pool
    independently — the same id could come back on two adjacent pages (or a
    point could be skipped by both). The fix reranks the whole pool once and
    applies offset exactly once, afterward, so adjacent pages must now tile
    the same single reranked order with zero overlap.
    """
    settings = Settings(
        QDRANT_COLLECTION="docs",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )
    # 8 fused hits, already in fused order (ids "0".."7"). The reranker's
    # scores deliberately scramble that order so paging has something real
    # to get wrong if offset leaks into the rerank window.
    pool = [
        QueryHit(id=str(i), score=1.0 - i * 0.01, payload={"text": f"t{i}"})
        for i in range(8)
    ]
    rerank_scores = {
        "t0": 0.9,
        "t1": 0.1,
        "t2": 0.8,
        "t3": 0.7,
        "t4": 0.6,
        "t5": 0.5,
        "t6": 0.4,
        "t7": 0.3,
    }
    embedder = _FakeRerankEmbedder(rerank_scores)

    def request_for(offset):
        return QueryRequest(
            query="q",
            limit=3,
            offset=offset,
            candidate_limit=11,
            rerank=True,
            rerank_top_k=8,
        ).apply_defaults(settings)

    # Both pages rerank the identical full pool (position 0) — exactly what
    # run_hybrid_query now returns for any offset, per its updated contract.
    page1 = (await _rerank_hits(request_for(0), pool, embedder, settings)).results
    page2 = (await _rerank_hits(request_for(3), pool, embedder, settings)).results

    page1_ids = [hit.id for hit in page1]
    page2_ids = [hit.id for hit in page2]

    assert not set(page1_ids) & set(page2_ids), (
        f"pages overlap: {page1_ids} vs {page2_ids}"
    )
    # The global rerank order by score is t0,t2,t3,t4,t5,t6,t7,t1 ->
    # ids 0,2,3,4,5,6,7,1. Page 1 (offset 0) = first 3, page 2 (offset 3) =
    # next 3 of that same single order.
    assert page1_ids == ["0", "2", "3"]
    assert page2_ids == ["4", "5", "6"]


class _FakeUnexpectedResponse(Exception):
    """Stand-in for qdrant_client's UnexpectedResponse (status_code + content)."""

    def __init__(self, status_code, content: bytes):
        self.status_code = status_code
        self.content = content
        super().__init__(f"Unexpected Response: {status_code}")


class _FakeQueryClient:
    """AsyncQdrantClient stand-in whose query_points raises a canned exception."""

    def __init__(self, query_points_exc):
        self._exc = query_points_exc

    async def get_collection(self, collection_name):
        # Falls back to settings.quantization_enabled — no fake info needed.
        raise RuntimeError("no collection info in this fake")

    async def query_points(self, **kwargs):
        raise self._exc


def _hybrid_query_request(settings, filters=None):
    request = QueryRequest(query="hybrid test", filters=filters)
    return request.apply_defaults(settings)


@pytest.mark.asyncio
async def test_run_hybrid_query_maps_qdrant_400_to_400_with_real_reason():
    """Regression: a filter on an unindexed field used to surface as a raw 502.

    Qdrant's own rejection ("Index required but not found for ...") is a
    client-input problem, not a Qdrant outage — see run_hybrid_query's
    _qdrant_query_failure.
    """
    settings = Settings(
        QDRANT_COLLECTION="docs",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )
    request = _hybrid_query_request(settings, filters={"not_indexed_field": "x"})
    embedding = SimpleNamespace(
        dense=[0.1, 0.2], sparse=models.SparseVector(indices=[0], values=[1.0])
    )
    exc = _FakeUnexpectedResponse(
        400,
        b'{"status":{"error":"Bad request: Index required but not found for '
        b'\\"not_indexed_field\\""}}',
    )
    client = _FakeQueryClient(exc)

    with pytest.raises(HTTPException) as exc_info:
        await run_hybrid_query(request, embedding, client, settings)

    assert exc_info.value.status_code == 400
    assert "Index required but not found" in exc_info.value.detail
    assert "not_indexed_field" in exc_info.value.detail


@pytest.mark.asyncio
async def test_run_hybrid_query_keeps_502_for_a_real_qdrant_failure():
    settings = Settings(
        QDRANT_COLLECTION="docs",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )
    request = _hybrid_query_request(settings)
    embedding = SimpleNamespace(
        dense=[0.1, 0.2], sparse=models.SparseVector(indices=[0], values=[1.0])
    )
    client = _FakeQueryClient(ConnectionError("connection refused"))

    with pytest.raises(HTTPException) as exc_info:
        await run_hybrid_query(request, embedding, client, settings)

    assert exc_info.value.status_code == 502

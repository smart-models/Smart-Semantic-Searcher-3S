"""Ingest refuses a chunk count it cannot hold in memory, on both paths.

Ingest embeds every chunk before its first write, so one request keeps every
dense and sparse vector resident at once. ``INGEST_MAX_CHUNKS`` is the bound
that peak actually scales with; without it the container is OOM-killed and the
caller sees a dropped connection instead of an error.
"""

import json

import pytest
from fastapi import HTTPException

from qdrant_middleware.api.routers.points import (
    _load_and_validate_chunks,
    _validate_chunks,
)
from qdrant_middleware.core.config import Settings


def _settings(**overrides) -> Settings:
    return Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        **overrides,
    )


def _chunks(count: int) -> list[dict]:
    return [{"id": i, "text": f"chunk {i}"} for i in range(count)]


def test_the_shipped_default_matches_the_sibling_batch_endpoint():
    # /points/batch already caps its point list at max_length=10000; ingest
    # holds strictly more per item (payload plus both vectors), so it must not
    # allow more.
    assert _settings().ingest_max_chunks == 10_000


def test_a_count_at_the_limit_is_accepted():
    settings = _settings(INGEST_MAX_CHUNKS=5)
    assert len(_validate_chunks(_chunks(5), settings)) == 5


def test_a_count_over_the_limit_is_refused_with_413():
    settings = _settings(INGEST_MAX_CHUNKS=5)
    with pytest.raises(HTTPException) as exc:
        _validate_chunks(_chunks(6), settings)
    assert exc.value.status_code == 413


def test_the_refusal_names_both_the_count_and_the_limit():
    settings = _settings(INGEST_MAX_CHUNKS=5)
    with pytest.raises(HTTPException) as exc:
        _validate_chunks(_chunks(6), settings)
    detail = exc.value.detail
    assert "6" in detail
    assert "5" in detail
    assert "INGEST_MAX_CHUNKS" in detail


def test_the_count_is_checked_before_the_per_chunk_structure():
    # An oversized batch must be refused without walking every chunk: the walk
    # is what the limit exists to avoid paying for.
    settings = _settings(INGEST_MAX_CHUNKS=2)
    malformed = [{"id": 0, "text": "ok"}, {"id": 1, "text": "ok"}, "not an object"]
    with pytest.raises(HTTPException) as exc:
        _validate_chunks(malformed, settings)
    assert exc.value.status_code == 413


def test_a_non_list_is_still_a_400_not_a_413():
    settings = _settings(INGEST_MAX_CHUNKS=2)
    with pytest.raises(HTTPException) as exc:
        _validate_chunks({"chunks": []}, settings)
    assert exc.value.status_code == 400


@pytest.mark.parametrize(
    "chunk",
    [
        {"id": True, "text": "body"},
        {"id": 1, "text": None},
        {"id": 1, "text": 123},
        {"id": 1, "text": []},
        {"id": 1, "text": "   "},
    ],
)
def test_chunk_structure_requires_a_real_integer_and_text(chunk):
    """Booleans and coerced values cannot become chunk identities or content."""
    with pytest.raises(HTTPException) as exc:
        _validate_chunks([chunk], _settings())
    assert exc.value.status_code == 400


def test_the_file_path_enforces_the_same_limit(tmp_path):
    settings = _settings(INGEST_MAX_CHUNKS=3)
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps({"chunks": _chunks(4)}), encoding="utf-8")
    with pytest.raises(HTTPException) as exc:
        _load_and_validate_chunks(path, settings)
    assert exc.value.status_code == 413


def test_the_file_path_accepts_a_count_within_the_limit(tmp_path):
    settings = _settings(INGEST_MAX_CHUNKS=3)
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps({"chunks": _chunks(3)}), encoding="utf-8")
    assert len(_load_and_validate_chunks(path, settings)) == 3


def test_the_limit_is_operator_tunable_upward():
    # A single static schema cap would silently veto this.
    settings = _settings(INGEST_MAX_CHUNKS=50_000)
    assert len(_validate_chunks(_chunks(20_000), settings)) == 20_000


def test_the_inline_schema_carries_no_competing_cap():
    from qdrant_middleware.api.routers.points import IngestChunksRequest

    field = IngestChunksRequest.model_fields["chunks"]
    maxima = [
        getattr(meta, "max_length", None)
        for meta in field.metadata
        if getattr(meta, "max_length", None) is not None
    ]
    assert maxima == []


def test_the_file_byte_limit_no_longer_promises_more_than_memory_allows():
    # 500 MB of JSON parses to gigabytes of Python objects on its own, before
    # a single vector exists, against a 1 GiB container.
    assert _settings().ingest_max_file_bytes == 67_108_864

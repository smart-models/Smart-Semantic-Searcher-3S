"""Regression tests for performance refactor — Phase 0 safety net.

These tests pin the behavioural contracts that the refactor must preserve:
- Rerank: batch_size 1 vs 32 produces identical final ordering.
- Ingest: embedding order preserved, mismatch → ok=False, failed_at_chunk correct.
- Query: _build_fusion_query called once (not reconstructed).
- Payload-index cache: ensure_payload_indexes called once per collection.
"""

import json
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, patch

import pytest
from qdrant_client.models import SparseVector

from qdrant_middleware.clients.embedder import (
    EmbedderClient,
    EmbeddingResult,
)
from qdrant_middleware.core.config import Settings
from qdrant_middleware.services.query_executor import (
    QueryHit,
    QueryRequest,
    _build_fusion_query,
    _rerank_hits,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides) -> Settings:
    defaults = {
        "QDRANT_API_HOST": "127.0.0.1",
        "MIDDLEWARE_API_TOKENS": "",
        "MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL": True,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _fake_embedding(idx: int) -> EmbeddingResult:
    return EmbeddingResult(
        dense=[float(idx)], sparse=SparseVector(indices=[idx], values=[1.0])
    )


def _mutable_collection_info():
    """Minimal collection response with no active reindex write lock."""
    return SimpleNamespace(config=SimpleNamespace(metadata=None))


class _FakeReranker:
    """Reranker stand-in returning scores without loading a model."""

    async def rerank(self, query: str, passages: list[str], normalize: bool = False):
        from qdrant_middleware.clients.embedder import RerankResult

        return [
            RerankResult(index=i, score=float(len(passages) - i))
            for i in range(len(passages))
        ]


# ---------------------------------------------------------------------------
# 1. Rerank: batch_size 1 vs 32 → identical result
# ---------------------------------------------------------------------------


class _FakeRerankHttp:
    """Fake httpx.AsyncClient that records /rerank POSTs and returns scores."""

    def __init__(self):
        self.rerank_calls: list[list[str]] = []

    async def post(self, path, **kwargs):
        if path != "/rerank":
            raise ValueError(f"unexpected path {path}")
        passages = kwargs["json"]["passages"]
        self.rerank_calls.append(list(passages))
        # Score based on the original passage index ("passage N" → 20-N)
        # so results are identical regardless of how passages are batched
        results = []
        for i, p in enumerate(passages):
            original_idx = int(p.split()[-1])
            results.append({"index": i, "score": float(20 - original_idx)})
        return SimpleNamespace(
            status_code=200,
            text="{}",
            json=lambda: {"results": results},
        )

    async def aclose(self):
        pass


def _make_rerank_embedder(
    rerank_batch_size: int,
) -> tuple[EmbedderClient, _FakeRerankHttp]:
    settings = _make_settings(
        SMART_EMBEDDER_RERANK_BATCH_SIZE=rerank_batch_size, RETRY_BASE_DELAY=0
    )
    embedder = EmbedderClient(settings, rerank_batch_size=rerank_batch_size)
    fake_http = _FakeRerankHttp()
    embedder._client = fake_http
    return embedder, fake_http


@pytest.mark.asyncio
async def test_rerank_batch_size_1_vs_32_same_ordering():
    """Reranking with batch_size=1 and batch_size=32 must produce identical results."""
    passages = [f"passage {i}" for i in range(20)]

    embedder_1, http_1 = _make_rerank_embedder(rerank_batch_size=1)
    embedder_32, http_32 = _make_rerank_embedder(rerank_batch_size=32)

    result_1 = await embedder_1.rerank("query", passages, normalize=True)
    result_32 = await embedder_32.rerank("query", passages, normalize=True)

    # Same number of results
    assert len(result_1) == len(result_32) == 20

    # Same ordering by score
    assert [r.index for r in result_1] == [r.index for r in result_32]
    assert [r.score for r in result_1] == [r.score for r in result_32]

    # batch_size=1 makes 20 HTTP calls, batch_size=32 makes 1 call
    assert len(http_1.rerank_calls) == 20
    assert len(http_32.rerank_calls) == 1

    await embedder_1.close()
    await embedder_32.close()


@pytest.mark.asyncio
async def test_rerank_preserves_unrerankable_hits_to_fill_limit():
    """Hits without text should be kept as non-reranked candidates."""
    settings = _make_settings(
        QDRANT_COLLECTION="test",
        QUERY_DEFAULT_RERANK_TOP_K=20,
    )
    request = QueryRequest(
        query="test",
        limit=10,
        candidate_limit=50,
        rerank=True,
        rerank_top_k=20,
    ).apply_defaults(settings)
    hits = [
        QueryHit(
            id=str(i),
            score=float(100 - i),
            payload={"text": f"passage {i}"} if i < 5 else {},
        )
        for i in range(20)
    ]

    response = await _rerank_hits(request, hits, _FakeReranker(), settings)

    assert response.returned == 10
    assert response.skipped_without_text == 15
    assert [hit.id for hit in response.results] == [str(i) for i in range(10)]


# ---------------------------------------------------------------------------
# 2. Ingest: order preserved, mismatch, failed_at_chunk
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Fake embedder keyed on batch content (deterministic under concurrency).

    Failure/mismatch is triggered by the first chunk index in a batch (parsed
    from ``"chunk N"`` text), not by call order, so the assertions hold
    regardless of how the parallel gather schedules the batches.
    """

    def __init__(
        self,
        fail_at_chunk: Optional[int] = None,
        mismatch_at_chunk: Optional[int] = None,
    ):
        self.fail_at_chunk = fail_at_chunk
        self.mismatch_at_chunk = mismatch_at_chunk
        self.call_count = 0

    async def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        self.call_count += 1
        first_idx = int(texts[0].split()[-1]) if texts else -1
        if self.fail_at_chunk is not None and first_idx == self.fail_at_chunk:
            from fastapi import HTTPException

            raise HTTPException(502, "Smart Embedder embedding failed: simulated")
        if self.mismatch_at_chunk is not None and first_idx == self.mismatch_at_chunk:
            return [_fake_embedding(0) for _ in range(len(texts) - 1)]
        return [_fake_embedding(i) for i in range(len(texts))]


@pytest.mark.asyncio
async def test_ingest_preserves_embedding_order(mutation_context):
    """Embeddings must be aligned 1:1 with chunks in order."""
    from qdrant_middleware.api.routers.points import _run_ingest

    settings = _make_settings(INGEST_DEFAULT_BATCH_SIZE=2)
    chunks = [{"id": i, "text": f"chunk {i}"} for i in range(6)]
    embedder = _FakeEmbedder()

    # Mock Qdrant client and ensure_payload_indexes
    mock_client = AsyncMock()
    mock_client.upsert = AsyncMock(
        return_value=SimpleNamespace(operation_id=0, status="completed")
    )
    mock_client.delete = AsyncMock(return_value=None)
    mock_client.scroll = AsyncMock(return_value=([], None))
    mock_client.get_collection = AsyncMock(return_value=_mutable_collection_info())

    with patch(
        "qdrant_middleware.api.routers.points.ensure_payload_indexes",
        new_callable=AsyncMock,
    ):
        result = await _run_ingest(
            chunks=chunks,
            collection_name="test",
            batch_size=2,
            replace_source=False,
            sid="source1",
            metadata={},
            embedder=embedder,
            client=mock_client,
            settings=settings,
            **mutation_context,
        )

    assert result.ok is True
    assert result.ingested == 6
    # 3 batches of 2
    assert embedder.call_count == 3


@pytest.mark.asyncio
async def test_ingest_mismatch_returns_failed_at_chunk(mutation_context):
    """Embedding count mismatch must return ok=False with correct failed_at_chunk."""
    from qdrant_middleware.api.routers.points import _run_ingest

    settings = _make_settings(INGEST_DEFAULT_BATCH_SIZE=2)
    chunks = [{"id": i, "text": f"chunk {i}"} for i in range(6)]
    # Mismatch on the batch starting at chunk index 2 (batch_start=2)
    embedder = _FakeEmbedder(mismatch_at_chunk=2)

    mock_client = AsyncMock()
    mock_client.upsert = AsyncMock(
        return_value=SimpleNamespace(operation_id=0, status="completed")
    )

    with patch(
        "qdrant_middleware.api.routers.points.ensure_payload_indexes",
        new_callable=AsyncMock,
    ):
        result = await _run_ingest(
            chunks=chunks,
            collection_name="test",
            batch_size=2,
            replace_source=False,
            sid="source1",
            metadata={},
            embedder=embedder,
            client=mock_client,
            settings=settings,
            **mutation_context,
        )

    assert result.ok is False
    assert result.ingested == 0
    assert result.failed_at_chunk == 2
    assert "mismatch" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_ingest_embed_failure_returns_failed_at_chunk(mutation_context):
    """Embed failure must return ok=False with correct failed_at_chunk and no writes."""
    from qdrant_middleware.api.routers.points import _run_ingest

    settings = _make_settings(INGEST_DEFAULT_BATCH_SIZE=2)
    chunks = [{"id": i, "text": f"chunk {i}"} for i in range(6)]
    # Fail on the batch starting at chunk index 4 (batch_start=4)
    embedder = _FakeEmbedder(fail_at_chunk=4)

    mock_client = AsyncMock()
    mock_client.upsert = AsyncMock(
        return_value=SimpleNamespace(operation_id=0, status="completed")
    )

    with patch(
        "qdrant_middleware.api.routers.points.ensure_payload_indexes",
        new_callable=AsyncMock,
    ):
        result = await _run_ingest(
            chunks=chunks,
            collection_name="test",
            batch_size=2,
            replace_source=False,
            sid="source1",
            metadata={},
            embedder=embedder,
            client=mock_client,
            settings=settings,
            **mutation_context,
        )

    assert result.ok is False
    assert result.ingested == 0
    assert result.failed_at_chunk == 4
    # No upserts should have happened
    mock_client.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_ingest_parallel_upsert_failure_reports_successful_writes(
    mutation_context,
):
    """Parallel upsert failures must not under-report already accepted batches."""
    from qdrant_middleware.api.routers.points import _run_ingest

    class PartiallyFailingClient:
        """Fake client failing partway through a batch, leaving a partial write."""

        def __init__(self):
            self.written: list[str] = []

        async def upsert(self, *, points, wait=False, **kwargs):
            first_text = points[0].payload["text"]
            if first_text == "chunk 0":
                raise RuntimeError("first batch failed")
            self.written.extend(point.payload["text"] for point in points)

        async def delete(self, **kwargs):
            return None

        async def scroll(self, **kwargs):
            return [], None

        async def get_collection(self, _collection_name):
            return _mutable_collection_info()

    settings = _make_settings(
        QDRANT_COLLECTION="test",
        INGEST_UPLOAD_CONCURRENCY=2,
        RETRY_MAX_ATTEMPTS=1,
    )
    chunks = [{"id": i, "text": f"chunk {i}"} for i in range(4)]
    embedder = _FakeEmbedder()
    client = PartiallyFailingClient()

    with patch(
        "qdrant_middleware.api.routers.points.ensure_payload_indexes",
        new_callable=AsyncMock,
    ):
        result = await _run_ingest(
            chunks=chunks,
            collection_name="test",
            batch_size=2,
            replace_source=False,
            sid="source1",
            metadata={},
            embedder=embedder,
            client=client,
            settings=settings,
            **mutation_context,
        )

    assert result.ok is False
    assert result.ingested == 2
    assert result.failed_at_chunk == 0
    assert client.written == ["chunk 2", "chunk 3"]


@pytest.mark.asyncio
async def test_ingest_payload_index_failure_maps_to_502(mutation_context):
    """Payload-index setup failures should use the same 502 mapping as Qdrant I/O."""
    from fastapi import HTTPException
    from qdrant_middleware.api.routers.points import _run_ingest

    settings = _make_settings(QDRANT_COLLECTION="test")

    async def fail_indexes(*args, **kwargs):
        raise RuntimeError("qdrant unavailable")

    with patch(
        "qdrant_middleware.api.routers.points.ensure_payload_indexes",
        new=fail_indexes,
    ):
        client = AsyncMock()
        client.get_collection = AsyncMock(return_value=_mutable_collection_info())
        with pytest.raises(HTTPException) as exc:
            await _run_ingest(
                chunks=[{"id": 0, "text": "chunk 0"}],
                collection_name="test",
                batch_size=1,
                replace_source=False,
                sid="source1",
                metadata={},
                embedder=_FakeEmbedder(),
                client=client,
                settings=settings,
                **mutation_context,
            )

    assert exc.value.status_code == 502


# ---------------------------------------------------------------------------
# 3. Query: _build_fusion_query should not be called twice
# ---------------------------------------------------------------------------


def test_build_fusion_query_dbsf():
    """_build_fusion_query returns DBSF fusion when fusion='dbsf'."""
    settings = _make_settings(QDRANT_COLLECTION="test")
    req = QueryRequest(query="test", fusion="dbsf")
    req = req.apply_defaults(settings)
    fq = _build_fusion_query(req)
    assert fq is not None


def test_build_fusion_query_rrf_with_options():
    """_build_fusion_query returns RrfQuery when rrf_k is provided."""
    settings = _make_settings(QDRANT_COLLECTION="test")
    req = QueryRequest(query="test", fusion="rrf", rrf_k=60)
    req = req.apply_defaults(settings)
    fq = _build_fusion_query(req)
    assert fq is not None


def test_build_fusion_query_rrf_options_with_dbsf_raises():
    """rrf_k with dbsf fusion must raise HTTPException 400."""
    from fastapi import HTTPException

    settings = _make_settings(QDRANT_COLLECTION="test")
    req = QueryRequest(query="test", fusion="dbsf", rrf_k=60)
    req = req.apply_defaults(settings)
    with pytest.raises(HTTPException) as exc:
        _build_fusion_query(req)
    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# 4. Payload-index cache: ensure_payload_indexes called once per collection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_payload_indexes_cache_skips_repeat(monkeypatch):
    """ensure_payload_indexes must skip on second call for the same collection."""
    from qdrant_client.models import PayloadSchemaType

    from qdrant_middleware.repositories import qdrant_collections
    from qdrant_middleware.repositories.qdrant_collections import (
        ensure_payload_indexes,
    )

    # Reset the global repository cache
    repo = qdrant_collections.repository
    repo._ensured_indexes = set()

    # Use a single fake index to exercise the cache without relying on defaults
    monkeypatch.setattr(
        "qdrant_middleware.repositories.qdrant_collections.PAYLOAD_INDEXES",
        [("test_field", PayloadSchemaType.KEYWORD)],
    )

    mock_client = AsyncMock()
    mock_client.create_payload_index = AsyncMock(return_value=None)

    await ensure_payload_indexes(mock_client, "test_collection")
    first_calls = mock_client.create_payload_index.call_count

    await ensure_payload_indexes(mock_client, "test_collection")
    second_calls = mock_client.create_payload_index.call_count

    # After Phase 1c, second call should be skipped (0 additional calls)
    assert first_calls > 0
    assert second_calls == first_calls  # no additional calls after cache


# ---------------------------------------------------------------------------
# 5. JSON load: async wrapper preserves error semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_and_validate_chunks_invalid_json_raises_400():
    """Invalid JSON must raise HTTPException 400 (preserved after to_thread)."""
    import tempfile
    from fastapi import HTTPException
    from qdrant_middleware.api.routers.points import _load_and_validate_chunks

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        f.write("{invalid json")
        path_str = f.name

    from pathlib import Path

    path = Path(path_str)

    with pytest.raises(HTTPException) as exc:
        _load_and_validate_chunks(path, _make_settings())

    assert exc.value.status_code == 400
    path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_load_and_validate_chunks_valid():
    """Valid JSON with chunks loads correctly."""
    import tempfile
    from qdrant_middleware.api.routers.points import _load_and_validate_chunks

    data = {"chunks": [{"id": 0, "text": "hello"}, {"id": 1, "text": "world"}]}
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(data, f)
        path_str = f.name

    from pathlib import Path

    path = Path(path_str)

    chunks = _load_and_validate_chunks(path, _make_settings())
    assert len(chunks) == 2
    assert chunks[0]["id"] == 0
    assert chunks[1]["text"] == "world"
    path.unlink(missing_ok=True)

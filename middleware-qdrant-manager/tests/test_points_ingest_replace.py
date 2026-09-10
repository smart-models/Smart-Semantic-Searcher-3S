"""Tests for ingest replace_source pruning and durability behavior."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from qdrant_client import models

from qdrant_middleware.api.dependencies import get_embedder_client, get_qdrant_client
from qdrant_middleware.clients.embedder import EmbeddingResult
from qdrant_middleware.core.config import Settings
from qdrant_middleware.main import create_app


class FakeEmbedder:
    """Embedder stand-in returning fixed vectors without a model."""

    max_batch_size = 64

    async def embed(self, texts):
        sparse = models.SparseVector(indices=[0], values=[1.0])
        return [EmbeddingResult(dense=[0.1] * 4, sparse=sparse) for _ in texts]


class FakeQdrant:
    """Fake that mimics a strict-mode collection with no payload indexes."""

    def __init__(self):
        self.payload_schema = {}
        self.created_indexes = []
        self.scroll_calls = []
        self.deleted = []

    async def get_collection(self, collection_name):
        return SimpleNamespace(
            payload_schema=dict(self.payload_schema),
            config=SimpleNamespace(),
        )

    async def create_payload_index(self, **kwargs):
        self.created_indexes.append(kwargs)
        self.payload_schema[kwargs["field_name"]] = kwargs["field_schema"]

    async def scroll(self, **kwargs):
        self.scroll_calls.append(kwargs)
        return [], None

    async def upsert(self, **kwargs):
        return None

    async def delete(self, **kwargs):
        self.deleted.append(kwargs)


def _make_app(fake):
    app = create_app(
        Settings(
            QDRANT_COLLECTION="docs",
            MIDDLEWARE_API_TOKENS="",
            MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        )
    )
    app.dependency_overrides[get_qdrant_client] = lambda: fake
    app.dependency_overrides[get_embedder_client] = lambda: FakeEmbedder()
    return app


def _make_client(fake):
    return TestClient(_make_app(fake))


def _ingest_body(replace_source):
    return {
        "chunks": [{"id": 1, "text": "alpha"}],
        "source_id": "src_a",
        "collection_name": "docs",
        "replace_source": replace_source,
    }


def test_replace_source_ensures_source_id_keyword_index():
    fake = FakeQdrant()
    response = _make_client(fake).post("/points/ingest-chunks", json=_ingest_body(True))

    assert response.status_code == 200
    assert response.json()["ok"] is True
    created = [
        call for call in fake.created_indexes if call["field_name"] == "source_id"
    ]
    assert len(created) == 1
    assert created[0]["field_schema"] == models.PayloadSchemaType.KEYWORD
    # The prune scroll must run after the index is guaranteed to exist.
    assert fake.scroll_calls, "replace_source must scroll for stale points"


def test_replace_source_skips_index_creation_when_already_present():
    fake = FakeQdrant()
    fake.payload_schema["source_id"] = models.PayloadSchemaType.KEYWORD

    response = _make_client(fake).post("/points/ingest-chunks", json=_ingest_body(True))

    assert response.status_code == 200
    assert fake.created_indexes == []


def test_single_upsert_rejects_whitespace_only_text():
    fake = FakeQdrant()
    response = _make_client(fake).post(
        "/points",
        json={"point_id": "2002", "text": "   ", "collection_name": "docs"},
    )

    assert response.status_code == 400
    assert "text must be a non-empty string" in response.text


def test_batch_upsert_rejects_whitespace_only_text():
    fake = FakeQdrant()
    response = _make_client(fake).post(
        "/points/batch",
        json={
            "collection_name": "docs",
            "points": [{"point_id": "2001", "text": "   "}],
        },
    )

    assert response.status_code == 400
    assert "text must be a non-empty string" in response.text


def test_plain_ingest_does_not_touch_payload_indexes():
    fake = FakeQdrant()
    response = _make_client(fake).post(
        "/points/ingest-chunks", json=_ingest_body(False)
    )

    assert response.status_code == 200
    assert fake.created_indexes == []
    assert fake.scroll_calls == []


async def test_http_source_replacements_share_snapshot_and_prune_exclusion(
    observe_mutation_attempt,
):
    """A later replacement snapshots only after the earlier one has committed."""

    class PausedStore(FakeQdrant):
        """Store actual points and pause one upload after its source snapshot."""

        def __init__(self):
            super().__init__()
            self.points = {}
            self.pause = False
            self.upload_entered = asyncio.Event()
            self.upload_release = asyncio.Event()

        async def scroll(self, **kwargs):
            self.scroll_calls.append(kwargs)
            return list(self.points.values()), None

        async def upsert(self, **kwargs):
            if self.pause:
                self.pause = False
                self.upload_entered.set()
                await self.upload_release.wait()
            self.points.update((point.id, point) for point in kwargs["points"])

        async def delete(self, **kwargs):
            for point_id in kwargs["points_selector"].points:
                self.points.pop(point_id, None)

    store = PausedStore()
    app = _make_app(store)
    tasks = []
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        seed = _ingest_body(False)
        seed["chunks"].append({"id": 2, "text": "second"})
        assert (
            await client.post("/points/ingest-chunks", json=seed)
        ).status_code == 200
        assert len(store.points) == 2
        store.pause = True
        try:
            tasks.append(
                asyncio.create_task(
                    client.post("/points/ingest-chunks", json=_ingest_body(True))
                )
            )
            await asyncio.wait_for(store.upload_entered.wait(), 2)
            attempted = observe_mutation_attempt(app.state.runtime.mutations)
            second = _ingest_body(True)
            second["chunks"] = [{"id": 2, "text": "last version"}]
            tasks.append(
                asyncio.create_task(client.post("/points/ingest-chunks", json=second))
            )
            await asyncio.wait_for(attempted.wait(), 2)
            assert len(store.scroll_calls) == 1
            store.upload_release.set()
            responses = await asyncio.wait_for(asyncio.gather(*tasks), 2)
            assert all(response.status_code == 200 for response in responses)
            assert all(response.json()["ok"] for response in responses)
            assert [point.payload for point in store.points.values()] == [
                {"id": 2, "text": "last version", "source_id": "src_a"}
            ]
        finally:
            store.upload_release.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


async def test_mutation_locks_are_isolated_between_apps():
    first, second = _make_app(FakeQdrant()), _make_app(FakeQdrant())
    async with first.state.runtime.mutations.lease("docs"):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=second), base_url="http://test"
        ) as client:
            response = await asyncio.wait_for(
                client.post("/points/ingest-chunks", json=_ingest_body(False)), 2
            )
    assert response.status_code == 200
    assert response.json()["ok"] is True


@pytest.mark.asyncio
async def test_ingest_batches_confirm_each_batch_once_with_wait_true():
    """Every first upload is also its one durable shard-local confirmation."""
    from qdrant_middleware.api.routers.points import (
        _IngestContext,
        _upsert_ingest_batches,
    )

    sent = []

    class RecordingQdrant:
        async def upsert(self, *, collection_name, points, wait):
            sent.append((wait, tuple(p.id for p in points)))

    def _point(point_id):
        return models.PointStruct(
            id=point_id,
            vector={
                "dense": [0.1] * 4,
                "sparse": models.SparseVector(indices=[0], values=[1.0]),
            },
        )

    batches = [
        (0, [_point("a"), _point("b")]),
        (2, [_point("c")]),
        (3, [_point("d"), _point("e"), _point("f")]),
    ]
    context = _IngestContext(
        collection_name="docs",
        batch_size=2,
        replace_source=True,
        sid="src_a",
        source_id_key="source_id",
        source_id_value="src_a",
        chunk_id_key="id",
        metadata={},
    )
    settings = Settings(
        QDRANT_COLLECTION="docs",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )

    ingested, failure = await _upsert_ingest_batches(
        batches, RecordingQdrant(), context, settings
    )

    assert ingested == 6
    assert failure is None
    # Every batch is uploaded once with its own durability confirmation.
    assert all(wait is True for wait, _ in sent)
    assert sorted(ids for _, ids in sent) == [
        ("a", "b"),
        ("c",),
        ("d", "e", "f"),
    ]

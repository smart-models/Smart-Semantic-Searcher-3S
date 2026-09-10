"""Tests for the configurable chunk-ordinal payload key on ingest.

A chunk's ``id`` is the chunker's per-source ordinal, not the Qdrant point id
(that is ``UUID5(source_id:id)``). It is ordinary payload whose name the caller
owns via ``chunk_id_key``, so renaming it must *relocate* the value rather than
copy it — a rename that left ``id`` behind would store the ordinal twice.
"""

from types import SimpleNamespace

from fastapi.testclient import TestClient
from qdrant_client import models

from qdrant_middleware.api.dependencies import get_embedder_client, get_qdrant_client
from qdrant_middleware.clients.embedder import EmbeddingResult
from qdrant_middleware.core.config import Settings
from qdrant_middleware.main import create_app
from qdrant_middleware.services.collection_schema import make_point_id


class FakeEmbedder:
    """Embedder stand-in returning fixed vectors without a model."""

    max_batch_size = 64

    async def embed(self, texts):
        sparse = models.SparseVector(indices=[0], values=[1.0])
        return [EmbeddingResult(dense=[0.1] * 4, sparse=sparse) for _ in texts]


class FakeQdrant:
    """Fake Qdrant client recording the ingest writes and deletes."""

    def __init__(self):
        self.payload_schema = {}
        self.upserted = []

    async def get_collection(self, collection_name):
        return SimpleNamespace(
            payload_schema=dict(self.payload_schema), config=SimpleNamespace()
        )

    async def create_payload_index(self, **kwargs):
        self.payload_schema[kwargs["field_name"]] = kwargs["field_schema"]

    async def scroll(self, **kwargs):
        return [], None

    async def upsert(self, **kwargs):
        self.upserted.extend(kwargs["points"])

    async def delete(self, **kwargs):
        pass


def _client(fake):
    app = create_app(
        Settings(
            QDRANT_COLLECTION="docs",
            MIDDLEWARE_API_TOKENS="",
            MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        )
    )
    app.dependency_overrides[get_qdrant_client] = lambda: fake
    app.dependency_overrides[get_embedder_client] = lambda: FakeEmbedder()
    return TestClient(app)


def _body(**extra):
    body = {
        "chunks": [{"id": 7, "text": "alpha"}],
        "source_id": "doc_a",
        "collection_name": "docs",
    }
    body.update(extra)
    return body


def test_default_stores_the_ordinal_under_id():
    fake = FakeQdrant()
    resp = _client(fake).post("/points/ingest-chunks", json=_body())
    assert resp.status_code == 200 and resp.json()["ok"] is True
    assert fake.upserted[0].payload["id"] == 7


def test_custom_key_relocates_the_ordinal_leaving_no_duplicate():
    fake = FakeQdrant()
    resp = _client(fake).post(
        "/points/ingest-chunks", json=_body(chunk_id_key="id_chunk")
    )
    assert resp.status_code == 200 and resp.json()["ok"] is True
    payload = fake.upserted[0].payload
    assert payload["id_chunk"] == 7
    assert "id" not in payload


def test_renaming_the_ordinal_does_not_move_the_point_id():
    # The point id derives from the chunk's `id` value, never from its key, so
    # a rename must not silently repoint the write.
    fake = FakeQdrant()
    resp = _client(fake).post(
        "/points/ingest-chunks", json=_body(chunk_id_key="ordinal")
    )
    assert resp.status_code == 200 and resp.json()["ok"] is True
    assert fake.upserted[0].id == make_point_id("doc_a", 7)


def test_metadata_cannot_shadow_the_ordinal():
    fake = FakeQdrant()
    resp = _client(fake).post(
        "/points/ingest-chunks",
        json=_body(chunk_id_key="ordinal", metadata={"ordinal": "hijacked"}),
    )
    assert resp.status_code == 200 and resp.json()["ok"] is True
    assert fake.upserted[0].payload["ordinal"] == 7


# The validators below raise pydantic ValueErrors, which the app's global
# RequestValidationError handler remaps to HTTP 400 (not 422).
def test_blank_chunk_id_key_is_rejected():
    fake = FakeQdrant()
    resp = _client(fake).post("/points/ingest-chunks", json=_body(chunk_id_key="  "))
    assert resp.status_code == 400
    assert fake.upserted == []


def test_clobbering_chunk_id_key_is_rejected():
    for reserved in ("text", "split_metadata"):
        fake = FakeQdrant()
        resp = _client(fake).post(
            "/points/ingest-chunks", json=_body(chunk_id_key=reserved)
        )
        assert resp.status_code == 400, reserved
        assert fake.upserted == []


def test_chunk_id_key_colliding_with_source_id_key_is_rejected():
    fake = FakeQdrant()
    resp = _client(fake).post(
        "/points/ingest-chunks",
        json=_body(chunk_id_key="ref", source_id_key="ref"),
    )
    assert resp.status_code == 400
    assert fake.upserted == []


def test_default_chunk_id_key_is_not_rejected():
    resp = _client(FakeQdrant()).post(
        "/points/ingest-chunks", json=_body(chunk_id_key="id")
    )
    assert resp.status_code == 200 and resp.json()["ok"] is True

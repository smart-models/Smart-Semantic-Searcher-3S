"""A search tells the embedder its text is a query; nothing else does.

Smart Embedder embeds a query with the dense model's query prompt where the
active profile has one. The middleware must say ``query`` for the text of a
search and stay silent everywhere it embeds stored text (ingest, upsert,
reindex), or documents would be embedded as queries.
"""

import json
import re
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from qdrant_client.models import SparseVector

import qdrant_middleware.api.routers.query as query_router
from qdrant_middleware.api.dependencies import get_embedder_client, get_qdrant_client
from qdrant_middleware.clients.embedder import EmbedderClient, EmbeddingResult
from qdrant_middleware.core.config import Settings
from qdrant_middleware.main import create_app

ROOT = Path(query_router.__file__).parents[3]
EMBEDDINGS = {
    "data": [
        {"embeddings": {"dense": [0.5], "sparse": {"indices": [3], "values": [1.0]}}}
    ]
}


async def _sent_body(**embed_kwargs) -> dict:
    """The JSON body one ``embed`` call posts to ``/embeddings/``."""
    sent = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=EMBEDDINGS)

    client = object.__new__(EmbedderClient)
    client.max_batch_size = 8
    client.embed_concurrency = 1
    client.retry_max_attempts = 1
    client.retry_base_delay = 0
    async with httpx.AsyncClient(
        base_url="http://embedder", transport=httpx.MockTransport(respond)
    ) as transport:
        client._client = transport
        await client.embed(["testo"], **embed_kwargs)
    (body,) = sent
    return body


async def test_query_embedding_request_says_it_is_a_query():
    body = await _sent_body(input_type="query")
    assert body["input_type"] == "query"
    assert body["sentences"] == ["testo"]


async def test_document_embedding_request_keeps_its_shape():
    # No ``input_type``: the embedder's default is ``document``, and a body an
    # older embedder already accepts keeps ingest and reindex working with it.
    assert "input_type" not in await _sent_body()


class RecordingEmbedder:
    """Embedder stand-in recording how each embedding was asked for."""

    def __init__(self):
        self.calls = []

    async def embed(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return [
            EmbeddingResult(dense=[0.1], sparse=SparseVector(indices=[0], values=[1.0]))
        ]


def test_search_embeds_its_text_as_a_query(monkeypatch):
    app = create_app(
        Settings(
            QDRANT_COLLECTION="docs",
            MIDDLEWARE_API_TOKENS="",
            MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        )
    )
    embedder = RecordingEmbedder()

    async def no_hits(*args, **kwargs):
        return []

    monkeypatch.setattr(query_router, "run_hybrid_query", no_hits)
    app.dependency_overrides[get_embedder_client] = lambda: embedder
    app.dependency_overrides[get_qdrant_client] = lambda: object()

    response = TestClient(app).post("/query/search", json={"query": "che dice?"})

    assert response.status_code == 200
    assert embedder.calls == [(["che dice?"], {"input_type": "query"})]


def test_only_the_search_route_embeds_a_query():
    asking = [
        path.relative_to(ROOT).as_posix()
        for path in sorted((ROOT / "qdrant_middleware").rglob("*.py"))
        if re.search(r"input_type\s*=\s*[\"']query[\"']", path.read_text("utf-8"))
    ]
    assert asking == ["qdrant_middleware/api/routers/query.py"]

"""Regression tests: runtime Settings must control request defaults."""

import pytest
from fastapi.testclient import TestClient
from qdrant_client.models import SparseVector

from qdrant_middleware.api.dependencies import get_embedder_client, get_qdrant_client
from qdrant_middleware.core.config import Settings
from qdrant_middleware.main import create_app
from qdrant_middleware.clients.embedder import EmbeddingResult
from qdrant_middleware.services.query_executor import QueryRequest
import qdrant_middleware.api.routers.query as query_router


class FakeEmbedder:
    """Embedder stand-in returning fixed vectors without a model."""

    async def embed(self, texts: list[str], input_type: str = "document"):
        assert texts == ["test"]
        return [
            EmbeddingResult(dense=[0.1], sparse=SparseVector(indices=[0], values=[1.0]))
        ]


def test_query_request_apply_defaults_uses_runtime_settings():
    settings = Settings(
        QDRANT_COLLECTION="custom_collection",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )
    req = QueryRequest(query="abc")
    resolved = req.apply_defaults(settings)

    assert resolved.collection_name == "custom_collection"
    assert resolved.limit == settings.query_default_limit
    assert resolved.fusion == settings.query_default_fusion


def test_query_request_apply_defaults_preserves_explicit_values():
    settings = Settings(
        QDRANT_COLLECTION="custom_collection",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )
    req = QueryRequest(query="abc", collection_name="other", limit=5)
    resolved = req.apply_defaults(settings)

    assert resolved.collection_name == "other"
    assert resolved.limit == 5


def test_query_request_rejects_explicit_rerank_top_k_above_runtime_maximum():
    settings = Settings(
        QDRANT_COLLECTION="custom_collection",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        QUERY_DEFAULT_RERANK_TOP_K=5,
        QUERY_MAX_RERANK_TOP_K=5,
    )
    req = QueryRequest(query="abc", candidate_limit=10, rerank_top_k=6)

    with pytest.raises(ValueError, match="rerank_top_k must be between 1 and 5"):
        req.apply_defaults(settings)


def test_app_factory_settings_propagate_to_query_request_defaults(monkeypatch):
    settings = Settings(
        QDRANT_COLLECTION="from_app_factory",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )
    app = create_app(settings)
    captured = {}

    async def fake_run_hybrid_query(
        request,
        embedding,
        client,
        runtime_settings,
        result_limit=None,
        fusion_query=None,
        vector_names=None,
    ):
        captured["request"] = request
        captured["settings"] = runtime_settings
        captured["result_limit"] = result_limit
        captured["vector_names"] = vector_names
        return []

    monkeypatch.setattr(query_router, "run_hybrid_query", fake_run_hybrid_query)
    app.dependency_overrides[get_embedder_client] = lambda: FakeEmbedder()
    app.dependency_overrides[get_qdrant_client] = lambda: object()

    response = TestClient(app).post("/query/search", json={"query": "test"})

    assert response.status_code == 200
    assert captured["request"].collection_name == "from_app_factory"
    assert captured["settings"] is app.state.settings
    assert captured["result_limit"] == settings.query_default_rerank_top_k
    assert captured["vector_names"].dense == settings.dense_vector_name
    assert captured["vector_names"].sparse == settings.sparse_vector_name


def test_query_search_invalid_filter_returns_422():
    settings = Settings(
        QDRANT_COLLECTION="from_app_factory",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )
    app = create_app(settings)
    app.dependency_overrides[get_embedder_client] = lambda: FakeEmbedder()
    app.dependency_overrides[get_qdrant_client] = lambda: object()

    response = TestClient(app).post(
        "/query/search",
        json={
            "query": "test",
            "filters": {
                "must": [{"key": "type", "match": {"value": "x", "any": ["y"]}}]
            },
        },
    )

    assert response.status_code == 422
    assert "match must contain exactly one operator" in response.text

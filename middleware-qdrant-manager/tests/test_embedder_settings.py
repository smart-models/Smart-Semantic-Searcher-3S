"""Tests for EmbedderClient batching and settings-derived configuration."""

import asyncio
from types import SimpleNamespace

import httpx

from qdrant_middleware.api.dependencies import get_embedder_client
from qdrant_middleware.core.config import Settings
from qdrant_middleware.main import create_app


class FakeAsyncClient:
    """Stand-in httpx async client recording whether it was closed."""

    def __init__(self, *, base_url, headers, timeout):
        self.base_url = str(base_url)
        self.headers = headers
        self.timeout = timeout
        self.closed = False

    async def aclose(self):
        self.closed = True


def test_app_embedder_uses_runtime_settings(monkeypatch):
    captured = {}

    def fake_async_client(**kwargs):
        captured.update(kwargs)
        return FakeAsyncClient(**kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake_async_client)
    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        SMART_EMBEDDER_URL="http://runtime-embedder:9999",
        SMART_EMBEDDER_CLIENT_TOKEN="runtime-token",
        SMART_EMBEDDER_TIMEOUT=7,
        SMART_EMBEDDER_MAX_BATCH_SIZE=5,
        SMART_EMBEDDER_RERANK_BATCH_SIZE=3,
        SMART_EMBEDDER_EMBED_CONCURRENCY=2,
    )
    app = create_app(settings)
    request = SimpleNamespace(app=app)

    embedder = get_embedder_client(request)

    assert captured["base_url"] == "http://runtime-embedder:9999"
    assert captured["headers"] == {"Authorization": "Bearer runtime-token"}
    assert captured["timeout"] == 7
    assert embedder.max_batch_size == 5
    assert embedder.rerank_batch_size == 3
    assert embedder.embed_concurrency == 2

    asyncio.run(embedder.close())

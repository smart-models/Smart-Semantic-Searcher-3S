"""Tests for the Qdrant client factory and lifespan open/close handling."""

import pytest

from qdrant_middleware.core.config import Settings
from qdrant_middleware.services.qdrant_client import create_qdrant_client


class FakeAsyncQdrantClient:
    """Stand-in async Qdrant client recording whether it was closed."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False

    async def close(self):
        self.closed = True


def test_create_qdrant_client_uses_settings(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return FakeAsyncQdrantClient(**kwargs)

    monkeypatch.setattr(
        "qdrant_middleware.services.qdrant_client.AsyncQdrantClient",
        fake_client,
    )
    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        QDRANT_URL="http://qdrant.local:6333",
        QDRANT_API_KEY="secret",
        QDRANT_TIMEOUT=12,
    )

    client = create_qdrant_client(settings)

    assert isinstance(client, FakeAsyncQdrantClient)
    assert captured == {
        "url": "http://qdrant.local:6333",
        "api_key": "secret",
        "timeout": 12,
    }

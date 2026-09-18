"""Tests for the Qdrant client factory and lifespan open/close handling."""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from qdrant_middleware.core import lifespan as lifecycle

from qdrant_middleware.core.config import Settings
from qdrant_middleware.services.qdrant_client import create_qdrant_client


class FakeAsyncQdrantClient:
    """Stand-in async Qdrant client recording whether it was closed."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False

    async def close(self):
        self.closed = True


@pytest.mark.parametrize(
    "failure", ["startup", "shutdown", "embedder-close", "limiter-stop", "manager-stop"]
)
async def test_lifespan_failure_still_closes_every_resource(monkeypatch, failure):
    qdrant = SimpleNamespace(close=AsyncMock())
    embedder = SimpleNamespace(close=AsyncMock())
    probe = SimpleNamespace(probe_once=AsyncMock(), start=Mock(), stop=AsyncMock())
    if failure == "startup":
        probe.probe_once.side_effect = RuntimeError("probe failed")
    elif failure == "shutdown":
        probe.stop.side_effect = RuntimeError("probe failed")
    limiter = SimpleNamespace(start_cleanup_task=Mock(), stop_cleanup_task=AsyncMock())
    manager = SimpleNamespace(stop=AsyncMock())
    if failure == "embedder-close":
        embedder.close.side_effect = RuntimeError("probe failed")
    elif failure == "limiter-stop":
        limiter.stop_cleanup_task.side_effect = RuntimeError("probe failed")
    elif failure == "manager-stop":
        manager.stop.side_effect = RuntimeError("probe failed")
    state = SimpleNamespace(
        runtime=SimpleNamespace(rate_limiter=limiter),
        settings=SimpleNamespace(
            api_tokens={},
            collection_name="docs",
            health_probe_interval=1,
            smart_embedder_url="http://embedder",
            embedder_probe_timeout=1,
            smart_embedder_client_token_value="",
        ),
        vector_reindex_manager=manager,
    )
    monkeypatch.setattr(lifecycle, "create_qdrant_client", lambda _: qdrant)
    monkeypatch.setattr(lifecycle, "EmbedderClient", lambda _: embedder)
    monkeypatch.setattr(lifecycle, "HealthProbe", lambda *a, **k: probe)
    monkeypatch.setattr(lifecycle, "_align_existing_strict_mode", AsyncMock())
    with pytest.raises(RuntimeError, match="probe failed"):
        async with lifecycle.lifespan(SimpleNamespace(state=state)):
            pass
    qdrant.close.assert_awaited_once()
    embedder.close.assert_awaited_once()
    limiter.stop_cleanup_task.assert_awaited_once()
    manager.stop.assert_awaited_once()
    assert state.qdrant_client is None
    assert state.embedder is None
    assert state.runtime.is_shutting_down


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

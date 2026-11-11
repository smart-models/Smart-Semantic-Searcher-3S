"""Tests for EmbedderClient batching and settings-derived configuration."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from qdrant_middleware.clients.embedder import EmbedderClient
from fastapi import HTTPException

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


async def test_failed_batch_cancels_and_drains_siblings():
    client = object.__new__(EmbedderClient)
    client.max_batch_size = 1
    client.embed_concurrency = 2
    entered = asyncio.Event()
    finished = asyncio.Event()

    async def batch(texts, input_type="document"):
        if texts == ["fail"]:
            await entered.wait()
            raise ValueError("batch failed")
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    client._embed_batch = batch
    with pytest.raises(ValueError, match="batch failed"):
        await asyncio.wait_for(client.embed(["fail", "pending"]), 2)
    assert finished.is_set()


@pytest.mark.parametrize("cancel_again", [False, True])
async def test_caller_cancellation_drains_running_and_queued_batches(cancel_again):
    client = object.__new__(EmbedderClient)
    client.max_batch_size = 1
    client.embed_concurrency = 2
    entered = asyncio.Event()
    cleanup_entered = asyncio.Event()
    release_cleanup = asyncio.Event()
    started = []
    finished = []

    async def batch(texts, input_type="document"):
        started.extend(texts)
        if len(started) == 2:
            entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_entered.set()
            await release_cleanup.wait()
            finished.extend(texts)

    client._embed_batch = batch
    task = asyncio.create_task(client.embed(["first", "second", "queued"]))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.wait_for(cleanup_entered.wait(), 2)
        if cancel_again:
            task.cancel()
        assert not task.done()
        assert started == ["first", "second"]
    finally:
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
    assert sorted(finished) == ["first", "second"]


async def test_late_cancellation_during_drain_preserves_original_batch_error():
    client = object.__new__(EmbedderClient)
    client.max_batch_size = 1
    client.embed_concurrency = 2
    entered = asyncio.Event()
    cleanup_entered = asyncio.Event()
    release_cleanup = asyncio.Event()
    finished = asyncio.Event()
    original_error = ValueError("batch failed")

    async def batch(texts, input_type="document"):
        if texts == ["fail"]:
            await entered.wait()
            raise original_error
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_entered.set()
            await release_cleanup.wait()
            finished.set()

    client._embed_batch = batch
    task = asyncio.create_task(client.embed(["fail", "pending"]))
    try:
        await asyncio.wait_for(cleanup_entered.wait(), 2)
        task.cancel()
        assert not task.done()
    finally:
        release_cleanup.set()
        with pytest.raises(ValueError, match="batch failed") as error:
            await asyncio.wait_for(task, 2)
    assert error.value is original_error
    assert finished.is_set()


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"results": None},
        {"results": []},
        {"results": [None]},
        {"results": [{"index": True, "score": 0.5}]},
        {"results": [{"index": 1, "score": 0.5}]},
        {"results": [{"index": 0, "score": "0.5"}]},
        {"results": [{"index": 0, "score": 10**400}]},
    ],
)
async def test_malformed_rerank_results_are_reported_as_bad_gateway(payload):
    client = object.__new__(EmbedderClient)
    client.rerank_batch_size = 2
    client.retry_max_attempts = 1
    client.retry_base_delay = 0
    async with httpx.AsyncClient(
        base_url="http://embedder",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)),
    ) as transport:
        client._client = transport
        with pytest.raises(HTTPException) as error:
            await client.rerank("query", ["text"])
    assert error.value.status_code == 502
    assert error.value.detail == "Smart Embedder returned invalid rerank results"


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"data": []},
        {"data": [{"embeddings": {}}]},
        {
            "data": [
                {
                    "embeddings": {
                        "dense": [10**400],
                        "sparse": {"indices": [], "values": []},
                    }
                }
            ]
        },
    ],
)
async def test_malformed_embeddings_are_reported_as_bad_gateway(payload):
    client = object.__new__(EmbedderClient)
    client.retry_max_attempts = 1
    client.retry_base_delay = 0
    async with httpx.AsyncClient(
        base_url="http://embedder",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)),
    ) as transport:
        client._client = transport
        with pytest.raises(HTTPException) as error:
            await client._embed_batch(["text"])
    assert error.value.status_code == 502
    assert error.value.detail == "Smart Embedder returned invalid embeddings"


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

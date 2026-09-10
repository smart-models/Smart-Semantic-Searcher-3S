"""Rerank capacity errors must reach the caller intact and unretried."""

import httpx
import pytest
from fastapi import HTTPException

from qdrant_middleware.clients.embedder import EmbedderClient
from qdrant_middleware.core.config import Settings


async def _client(handler):
    """Build an EmbedderClient whose transport is a MockTransport.

    EmbedderClient constructs its own httpx.AsyncClient from Settings, so the
    established pattern (tests/test_embedding_contract.py:280) is to close
    that client and swap in one backed by MockTransport.

    RETRY_BASE_DELAY=0 keeps the retry tests from sleeping through real
    exponential backoff.
    """
    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        SMART_EMBEDDER_URL="http://embedder:8010",
        SMART_EMBEDDER_CLIENT_TOKEN="t",
        SMART_EMBEDDER_RERANK_BATCH_SIZE=32,
        RETRY_MAX_ATTEMPTS=3,
        RETRY_BASE_DELAY=0.0,
    )
    client = EmbedderClient(settings)
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url="http://embedder:8010", transport=httpx.MockTransport(handler)
    )
    return client


CAPACITY_BODY = {
    "detail": {
        "error": "rerank_capacity_exceeded",
        "message": "A single query/passage pair exceeds the available GPU memory.",
        "passages": 200,
        "batch_items": 1,
        "batch_tokens": 8100,
        "suggested_max_passages": 1,
    }
}


@pytest.mark.asyncio
async def test_capacity_error_is_not_retried_and_keeps_its_status():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(413, json=CAPACITY_BODY)

    embedder = await _client(handler)
    with pytest.raises(HTTPException) as exc:
        await embedder.rerank("q", ["p1", "p2"])

    assert exc.value.status_code == 413
    assert len(calls) == 1, "413 must not be retried"
    assert exc.value.detail["error"] == "rerank_capacity_exceeded"
    await embedder.close()


@pytest.mark.asyncio
async def test_transient_pressure_is_still_retried():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, json={"detail": {"error": "rerank_vram_pressure"}})

    embedder = await _client(handler)
    with pytest.raises(HTTPException):
        await embedder.rerank("q", ["p1"])

    assert len(calls) == 3, "503 stays retryable"
    await embedder.close()


@pytest.mark.asyncio
async def test_unstructured_upstream_error_still_becomes_502():
    def handler(request):
        return httpx.Response(500, text="boom")

    embedder = await _client(handler)
    with pytest.raises(HTTPException) as exc:
        await embedder.rerank("q", ["p1"])

    assert exc.value.status_code == 502
    await embedder.close()

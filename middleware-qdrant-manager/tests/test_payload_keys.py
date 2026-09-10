"""Tests for the payload-keys sampling repository helper and endpoint."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from qdrant_client.http.exceptions import UnexpectedResponse

from qdrant_middleware.api.dependencies import get_qdrant_client
from qdrant_middleware.core.config import Settings
from qdrant_middleware.main import create_app
from qdrant_middleware.repositories import qdrant_collections


def _test_settings():
    return Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        quantization_enabled=False,
        sparse_idf=False,
    )


def _point(point_id, payload):
    return SimpleNamespace(id=point_id, payload=payload)


class PayloadKeysClient:
    """Fake Qdrant client: existing collection plus a fixed scroll page."""

    def __init__(self, points=None):
        self.points = points if points is not None else []
        self.scroll_kwargs = None

    async def get_collection(self, collection_name):
        return SimpleNamespace(payload_schema={})

    async def scroll(self, **kwargs):
        self.scroll_kwargs = kwargs
        return self.points[: kwargs["limit"]], None


class MissingCollectionClient(PayloadKeysClient):
    """PayloadKeysClient reporting the collection as missing."""

    async def get_collection(self, collection_name):
        raise UnexpectedResponse(
            status_code=404,
            reason_phrase="Not Found",
            content=b"not found",
            headers={},
        )


@pytest.mark.asyncio
async def test_repository_samples_sorted_union_of_payload_keys():
    client = PayloadKeysClient(
        [
            _point(1, {"text": "a", "source_id": "s", "token_count": 3}),
            _point(2, {"text": "b", "Header 1": "H", "custom": True}),
            _point(3, None),
        ]
    )

    keys, sampled = await qdrant_collections.sample_payload_keys(
        client, "demo", 100, settings=_test_settings()
    )

    assert keys == ["custom", "Header 1", "source_id", "text", "token_count"]
    assert sampled == 3
    assert client.scroll_kwargs["with_payload"] is True
    assert client.scroll_kwargs["with_vectors"] is False
    assert client.scroll_kwargs["limit"] == 100


@pytest.mark.asyncio
async def test_repository_returns_empty_for_collection_without_points():
    keys, sampled = await qdrant_collections.sample_payload_keys(
        PayloadKeysClient([]), "demo", 50, settings=_test_settings()
    )

    assert keys == []
    assert sampled == 0


@pytest.mark.asyncio
async def test_repository_raises_not_found_for_missing_collection():
    with pytest.raises(qdrant_collections.PayloadIndexCollectionNotFound):
        await qdrant_collections.sample_payload_keys(
            MissingCollectionClient(), "missing", 10, settings=_test_settings()
        )


def _app_with_client(client):
    app = create_app(_test_settings())
    app.dependency_overrides[get_qdrant_client] = lambda: client
    return app


def test_payload_keys_endpoint_returns_sampled_keys():
    client = PayloadKeysClient(
        [_point(1, {"text": "a", "type": "table", "token_count": 3})]
    )
    response = TestClient(_app_with_client(client)).get(
        "/collections/demo/payload-keys"
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "collection_name": "demo",
        "keys": ["text", "token_count", "type"],
        "count": 3,
        "sampled_points": 1,
    }


def test_payload_keys_endpoint_honours_sample_size():
    client = PayloadKeysClient([_point(i, {f"k{i}": i}) for i in range(5)])
    response = TestClient(_app_with_client(client)).get(
        "/collections/demo/payload-keys",
        params={"sample_size": 2},
    )

    assert response.status_code == 200
    assert response.json()["sampled_points"] == 2
    assert response.json()["keys"] == ["k0", "k1"]


def test_payload_keys_endpoint_returns_404_for_missing_collection():
    response = TestClient(_app_with_client(MissingCollectionClient())).get(
        "/collections/missing/payload-keys"
    )

    assert response.status_code == 404


def test_payload_keys_endpoint_rejects_out_of_range_sample_size():
    response = TestClient(_app_with_client(PayloadKeysClient())).get(
        "/collections/demo/payload-keys",
        params={"sample_size": 0},
    )

    assert response.status_code == 400

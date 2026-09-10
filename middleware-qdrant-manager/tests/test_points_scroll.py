"""Tests for the POST /points/scroll endpoint."""

from types import SimpleNamespace

from fastapi.testclient import TestClient
from qdrant_client import models

from qdrant_middleware.api.dependencies import get_qdrant_client
from qdrant_middleware.core.config import Settings
from qdrant_middleware.main import create_app


class FakeQdrant:
    """Fake Qdrant client serving a fixed page of points to the scroll route."""

    def __init__(self):
        self.scroll_kwargs = None
        self.count_kwargs = None

    async def scroll(self, **kwargs):
        self.scroll_kwargs = kwargs
        return [SimpleNamespace(id=7, payload={"type": "table"})], 9

    async def count(self, **kwargs):
        self.count_kwargs = kwargs
        return SimpleNamespace(count=51)


def test_scroll_points_uses_payload_filter_without_vectors():
    client = FakeQdrant()
    app = create_app(
        Settings(
            QDRANT_COLLECTION="docs",
            MIDDLEWARE_API_TOKENS="",
            MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        )
    )
    app.dependency_overrides[get_qdrant_client] = lambda: client

    response = TestClient(app).post(
        "/points/scroll",
        json={"filters": {"type": "table"}, "limit": 25},
    )

    assert response.status_code == 200
    assert response.json() == {
        "collection_name": "docs",
        "returned": 1,
        "total": 51,
        "next_page_offset": 9,
        "results": [{"id": "7", "payload": {"type": "table"}, "vectors": None}],
    }
    assert client.scroll_kwargs["collection_name"] == "docs"
    assert client.scroll_kwargs["limit"] == 25
    assert client.scroll_kwargs["with_vectors"] is False
    assert isinstance(client.scroll_kwargs["scroll_filter"], models.Filter)
    assert client.count_kwargs["exact"] is True


def test_scroll_points_quotes_payload_fields_containing_spaces():
    client = FakeQdrant()
    app = create_app(
        Settings(
            QDRANT_COLLECTION="docs",
            MIDDLEWARE_API_TOKENS="",
            MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        )
    )
    app.dependency_overrides[get_qdrant_client] = lambda: client

    response = TestClient(app).post(
        "/points/scroll",
        json={"with_payload": ["text", "Header 1", "Header 2"]},
    )

    assert response.status_code == 200
    assert client.scroll_kwargs["with_payload"] == ["text", '"Header 1"', '"Header 2"']


def test_scroll_points_without_filters_returns_unfiltered_page():
    client = FakeQdrant()
    app = create_app(
        Settings(
            QDRANT_COLLECTION="docs",
            MIDDLEWARE_API_TOKENS="",
            MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        )
    )
    app.dependency_overrides[get_qdrant_client] = lambda: client

    response = TestClient(app).post("/points/scroll", json={"limit": 10})

    assert response.status_code == 200
    assert client.scroll_kwargs["scroll_filter"] is None


def test_scroll_points_can_skip_total_count_on_later_pages():
    client = FakeQdrant()
    app = create_app(
        Settings(
            QDRANT_COLLECTION="docs",
            MIDDLEWARE_API_TOKENS="",
            MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        )
    )
    app.dependency_overrides[get_qdrant_client] = lambda: client

    response = TestClient(app).post(
        "/points/scroll", json={"offset": 9, "include_total": False}
    )

    assert response.status_code == 200
    assert response.json()["total"] is None
    assert client.count_kwargs is None


def test_scroll_points_returns_422_for_malformed_geo_filter():
    client = FakeQdrant()
    app = create_app(
        Settings(
            QDRANT_COLLECTION="docs",
            MIDDLEWARE_API_TOKENS="",
            MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        )
    )
    app.dependency_overrides[get_qdrant_client] = lambda: client

    response = TestClient(app).post(
        "/points/scroll",
        json={"filters": {"must": [{"key": "location", "geo_radius": "invalid"}]}},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "geo_radius must be an object"
    assert client.scroll_kwargs is None

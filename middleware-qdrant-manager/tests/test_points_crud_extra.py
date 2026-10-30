"""Tests for the added point CRUD endpoints.

Covers batch retrieve, delete-by-filter, payload key delete, and payload clear.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from qdrant_client import models

from qdrant_middleware.api.dependencies import get_qdrant_client
from qdrant_middleware.api.routers import points
from qdrant_middleware.core.config import Settings
from qdrant_middleware.main import create_app


def _app(client):
    app = create_app(
        Settings(
            QDRANT_COLLECTION="docs",
            MIDDLEWARE_API_TOKENS="",
            MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        )
    )
    app.dependency_overrides[get_qdrant_client] = lambda: client
    return app


class FakeQdrant:
    """Mutable collection with no active reindex job (mutations allowed)."""

    def __init__(self, *, points=None, count=0):
        self._points = points or []
        self._count = count
        self.delete_kwargs = None
        self.delete_payload_kwargs = None
        self.overwrite_kwargs = None
        self.retrieve_ids = None

    async def get_collection(self, collection_name):
        # No middleware metadata → no stored reindex job → mutation allowed.
        return SimpleNamespace(config=SimpleNamespace(metadata=None))

    async def count(self, **kwargs):
        self.count_kwargs = kwargs
        return SimpleNamespace(count=self._count)

    async def retrieve(self, **kwargs):
        self.retrieve_kwargs = kwargs
        self.retrieve_ids = kwargs["ids"]
        wanted = set(kwargs["ids"])
        return [p for p in self._points if p.id in wanted]

    async def delete(self, **kwargs):
        self.delete_kwargs = kwargs

    async def delete_payload(self, **kwargs):
        self.delete_payload_kwargs = kwargs

    async def overwrite_payload(self, **kwargs):
        self.overwrite_kwargs = kwargs


# ── text ownership ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", ["changed", None, "", {"nested": "value"}])
def test_payload_only_update_rejects_text(value):
    """Payload writes cannot replace text without recomputing its vector."""
    with pytest.raises(ValidationError, match="use point upsert"):
        points.PayloadUpdateRequest(payload={"text": value})


def test_filtered_payload_update_rejects_text():
    """Filtered payload writes cannot desynchronize text from vectors."""
    with pytest.raises(ValidationError, match="use point upsert"):
        points.PayloadFilterUpdateRequest(
            filters={"lang": "it"}, payload={"text": "changed"}
        )


def test_upsert_payload_keeps_explicit_text_over_caller_metadata():
    """An upsert stores the text it embedded even with conflicting metadata."""
    assert points._upsert_payload({"text": None, "lang": "it"}, "embedded") == {
        "text": "embedded",
        "lang": "it",
    }


@pytest.mark.parametrize(
    "ids",
    [
        ["1", "01"],
        ["1", "0001"],
        [
            "D9428888-122B-11E1-B85C-61CD3CBB3210",
            "d9428888-122b-11e1-b85c-61cd3cbb3210",
        ],
    ],
)
def test_batch_upsert_rejects_duplicate_normalized_ids(ids):
    """Storage aliases cannot overwrite the same point in one batch."""
    with pytest.raises(ValidationError, match="duplicate"):
        points.BatchUpsertRequest(
            points=[{"point_id": point_id, "text": "body"} for point_id in ids]
        )


# ── batch retrieve ──────────────────────────────────────────────────────────


def test_retrieve_points_returns_found_and_omits_missing():
    client = FakeQdrant(points=[SimpleNamespace(id=5, payload={"text": "a"})])

    response = TestClient(_app(client)).post(
        "/points/retrieve", json={"ids": ["5", "6"]}
    )

    assert response.status_code == 200
    assert response.json() == {
        "collection_name": "docs",
        "returned": 1,
        "results": [{"id": "5", "payload": {"text": "a"}, "vectors": None}],
    }
    assert client.retrieve_ids == [5, 6]


@pytest.mark.parametrize(
    ("selection", "forwarded"),
    [
        ({}, True),
        ({"with_payload": False}, False),
        ({"with_payload": ["text"]}, ["text"]),
    ],
)
def test_retrieve_points_forwards_the_payload_selection(selection, forwarded):
    """Like scroll, retrieve returns every payload field unless told otherwise."""
    client = FakeQdrant(points=[SimpleNamespace(id=5, payload={"text": "a"})])

    response = TestClient(_app(client)).post(
        "/points/retrieve", json={"ids": ["5"], **selection}
    )

    assert response.status_code == 200
    assert client.retrieve_kwargs["with_payload"] == forwarded


def test_retrieve_points_rejects_invalid_id():
    response = TestClient(_app(FakeQdrant())).post(
        "/points/retrieve", json={"ids": ["not-an-id"]}
    )

    # The app maps request-validation errors to 400.
    assert response.status_code == 400


# ── single-point delete by id ───────────────────────────────────────────────


def test_delete_point_is_a_no_op_when_the_id_never_existed():
    """Deleting an absent id is a no-op, not a 404.

    Regression test: this endpoint used to 404 on a miss while the bulk
    endpoint serving the same selector='ids' delete returned 200. Since the
    MCP tool routes by list length alone, one id and two ids took opposite
    outcomes for identical intent. DELETE is idempotent — 'this id must not
    be there' is already satisfied when it isn't.
    """
    client = FakeQdrant(points=[])

    response = TestClient(_app(client)).delete("/points/999")

    assert response.status_code == 200
    assert response.json()["deleted"] == 0


def test_delete_point_reports_one_when_the_id_existed():
    client = FakeQdrant(points=[SimpleNamespace(id=5)])

    response = TestClient(_app(client)).delete("/points/5")

    assert response.status_code == 200
    assert response.json()["deleted"] == 1
    assert client.delete_kwargs is not None


def test_delete_point_still_rejects_a_malformed_id():
    """No-op applies to absent ids, not to ids Qdrant could never hold."""
    response = TestClient(_app(FakeQdrant())).delete("/points/not-an-id")

    assert response.status_code == 400


# ── bulk delete by ids ──────────────────────────────────────────────────────


def test_bulk_delete_by_ids_reports_only_points_that_actually_existed():
    """Regression test.

    The bulk endpoint used to echo back len(ids) as 'deleted' regardless of
    whether those ids were ever real.
    """
    client = FakeQdrant(points=[SimpleNamespace(id=1), SimpleNamespace(id=2)])

    response = TestClient(_app(client)).request(
        "DELETE", "/points", json={"ids": ["1", "2", "3"]}
    )

    assert response.status_code == 200
    # id 3 was never there — only 2 points actually existed to delete.
    assert response.json()["deleted"] == 2


def test_bulk_delete_by_ids_dedupes_the_reported_count():
    client = FakeQdrant(points=[SimpleNamespace(id=1)])

    response = TestClient(_app(client)).request(
        "DELETE", "/points", json={"ids": ["1", "1"]}
    )

    assert response.status_code == 200
    assert response.json()["deleted"] == 1


def test_bulk_delete_by_ids_all_nonexistent_reports_zero():
    client = FakeQdrant(points=[])

    response = TestClient(_app(client)).request(
        "DELETE", "/points", json={"ids": ["1", "2"]}
    )

    assert response.status_code == 200
    assert response.json()["deleted"] == 0
    # the underlying Qdrant delete still ran (idempotent no-op on its side)
    assert client.delete_kwargs is not None


# ── delete by filter ────────────────────────────────────────────────────────


def test_delete_by_filter_deletes_matches_and_reports_count():
    client = FakeQdrant(count=4)

    response = TestClient(_app(client)).post(
        "/points/delete-by-filter", json={"filters": {"type": "table"}}
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True, "deleted": 4, "collection_name": "docs"}
    assert isinstance(client.delete_kwargs["points_selector"], models.FilterSelector)
    assert client.count_kwargs["exact"] is True


def test_delete_by_filter_rejects_empty_filter():
    client = FakeQdrant(count=99)

    response = TestClient(_app(client)).post(
        "/points/delete-by-filter", json={"filters": {}}
    )

    # An empty filter is rejected before any delete runs (never wipes all).
    assert response.status_code == 400
    assert client.delete_kwargs is None


def test_delete_by_filter_dry_run_passes_through_qdrants_own_400():
    """Filtering on an unindexed field is Qdrant's own 400, not a 502.

    Regression test: `_qdrant()` used to map every failure (including a
    genuine client-request error Qdrant itself rejected) to a blanket 502,
    which misreports "missing index" as an upstream/infra failure.
    """
    from qdrant_client.http.exceptions import UnexpectedResponse

    class UnindexedFilterQdrant(FakeQdrant):
        async def count(self, **kwargs):
            raise UnexpectedResponse(
                status_code=400,
                reason_phrase="Bad Request",
                content=(
                    b'{"status":{"error":'
                    b'"Index required but not found for \\"checked\\""}}'
                ),
                headers={},
            )

    client = UnindexedFilterQdrant()

    response = TestClient(_app(client)).post(
        "/points/delete-by-filter", json={"filters": {"checked": True}}
    )

    assert response.status_code == 400
    assert "Index required" in response.json()["detail"]
    assert client.delete_kwargs is None


# ── delete payload keys ─────────────────────────────────────────────────────


def test_delete_payload_keys_removes_named_keys():
    client = FakeQdrant(points=[SimpleNamespace(id=5, payload={"text": "a"})])

    response = TestClient(_app(client)).post(
        "/points/5/payload/delete-keys", json={"keys": ["junk", "tmp"]}
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert client.delete_payload_kwargs["keys"] == ["junk", "tmp"]
    assert client.delete_payload_kwargs["points"] == [5]


def test_delete_payload_keys_rejects_protected_fields():
    client = FakeQdrant(points=[SimpleNamespace(id=5, payload={"text": "a"})])

    response = TestClient(_app(client)).post(
        "/points/5/payload/delete-keys", json={"keys": ["source_id", "junk"]}
    )

    assert response.status_code == 400
    assert "source_id" in response.json()["detail"]
    assert client.delete_payload_kwargs is None


# ── clear payload ───────────────────────────────────────────────────────────


def test_clear_payload_keeps_protected_fields():
    client = FakeQdrant(
        points=[
            SimpleNamespace(id=5, payload={"text": "a", "source_id": "s", "junk": 1})
        ]
    )

    response = TestClient(_app(client)).post("/points/5/payload/clear", json={})

    assert response.status_code == 200
    assert client.overwrite_kwargs["payload"] == {"text": "a", "source_id": "s"}


def test_clear_payload_404_for_missing_point():
    response = TestClient(_app(FakeQdrant(points=[]))).post(
        "/points/5/payload/clear", json={}
    )

    assert response.status_code == 404

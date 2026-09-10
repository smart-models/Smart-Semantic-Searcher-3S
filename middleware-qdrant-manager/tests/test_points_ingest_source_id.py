"""Tests for configurable source_id payload key/type on ingest."""

from types import SimpleNamespace

from fastapi.testclient import TestClient
from qdrant_client import models

from qdrant_middleware.api.dependencies import get_embedder_client, get_qdrant_client
from qdrant_middleware.api.routers import points
from qdrant_middleware.clients.embedder import EmbeddingResult
from qdrant_middleware.core.config import Settings
from qdrant_middleware.main import create_app
from qdrant_middleware.services.collection_schema import make_point_id


class FakeEmbedder:
    """Embedder stand-in returning fixed vectors without a model."""

    max_batch_size = 64

    async def embed(self, texts):
        sparse = models.SparseVector(indices=[0], values=[1.0])
        return [EmbeddingResult(dense=[0.1] * 4, sparse=sparse) for _ in texts]


class FakeQdrant:
    """Fake Qdrant client recording the ingest writes and deletes."""

    def __init__(self):
        self.payload_schema = {}
        self.created_indexes = []
        self.upserted = []
        self.deleted = []
        self.scroll_kwargs = None

    async def get_collection(self, collection_name):
        return SimpleNamespace(
            payload_schema=dict(self.payload_schema), config=SimpleNamespace()
        )

    async def create_payload_index(self, **kwargs):
        self.created_indexes.append(kwargs)
        self.payload_schema[kwargs["field_name"]] = kwargs["field_schema"]

    async def scroll(self, **kwargs):
        self.scroll_kwargs = kwargs
        return [], None

    async def upsert(self, **kwargs):
        self.upserted.extend(kwargs["points"])

    async def delete(self, **kwargs):
        self.deleted.append(kwargs)


def _client(fake):
    app = create_app(
        Settings(
            QDRANT_COLLECTION="docs",
            MIDDLEWARE_API_TOKENS="",
            MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        )
    )
    app.dependency_overrides[get_qdrant_client] = lambda: fake
    app.dependency_overrides[get_embedder_client] = lambda: FakeEmbedder()
    return TestClient(app)


def _body(**extra):
    body = {
        "chunks": [{"id": 1, "text": "alpha"}],
        "source_id": "doc_a",
        "collection_name": "docs",
    }
    body.update(extra)
    return body


def test_default_stores_value_under_source_id_key():
    fake = FakeQdrant()
    resp = _client(fake).post("/points/ingest-chunks", json=_body())
    assert resp.status_code == 200 and resp.json()["ok"] is True
    assert fake.upserted[0].payload["source_id"] == "doc_a"


def test_custom_key_stores_value_under_chosen_key():
    fake = FakeQdrant()
    resp = _client(fake).post(
        "/points/ingest-chunks", json=_body(source_id_key="doc_ref")
    )
    assert resp.status_code == 200 and resp.json()["ok"] is True
    payload = fake.upserted[0].payload
    assert payload["doc_ref"] == "doc_a"
    assert "source_id" not in payload


def test_int_type_coerces_numeric_source_id():
    fake = FakeQdrant()
    resp = _client(fake).post(
        "/points/ingest-chunks",
        json=_body(source_id="12345", source_id_type="integer"),
    )
    assert resp.status_code == 200 and resp.json()["ok"] is True
    assert fake.upserted[0].payload["source_id"] == 12345


def test_replace_source_matches_and_indexes_the_coerced_source_id():
    """Replacement pruning must use the payload type written by ingest."""
    fake = FakeQdrant()
    resp = _client(fake).post(
        "/points/ingest-chunks",
        json=_body(source_id="12345", source_id_type="integer", replace_source=True),
    )

    assert resp.status_code == 200 and resp.json()["ok"] is True
    filter_condition = fake.scroll_kwargs["scroll_filter"].must[0]
    assert filter_condition.match.value == 12345
    assert fake.created_indexes[0]["field_schema"] == models.PayloadSchemaType.INTEGER


def test_replace_source_uses_an_exact_range_for_float_source_ids():
    """A float source id needs a range predicate and a float index."""
    fake = FakeQdrant()
    resp = _client(fake).post(
        "/points/ingest-chunks",
        json=_body(source_id="12.5", source_id_type="float", replace_source=True),
    )

    assert resp.status_code == 200 and resp.json()["ok"] is True
    filter_condition = fake.scroll_kwargs["scroll_filter"].must[0]
    assert filter_condition.range.gte == filter_condition.range.lte == 12.5
    assert fake.created_indexes[0]["field_schema"] == models.PayloadSchemaType.FLOAT


def test_source_predicate_quotes_literal_custom_field_names():
    """A dot in a custom key is a literal name, not a nested path."""
    condition = points._source_id_condition("document.id", 42, "integer")

    assert condition.key == '"document.id"'
    assert condition.match.value == 42


def test_datetime_source_predicate_normalizes_an_naive_timestamp_to_utc():
    """Datetime replacement compares an explicit instant at both range bounds."""
    condition = points._source_id_condition(
        "source_id", "2026-09-05T10:30:00", "datetime"
    )

    assert condition.range.gte.isoformat() == "2026-09-05T10:30:00+00:00"
    assert condition.range.lte.isoformat() == "2026-09-05T10:30:00+00:00"


def test_non_convertible_source_id_type_fails():
    fake = FakeQdrant()
    resp = _client(fake).post(
        "/points/ingest-chunks",
        json=_body(source_id="doc_a", source_id_type="integer"),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is False
    assert "convertible" in (data["error"] or "")


def test_point_id_is_independent_of_key_and_type():
    fake_default = FakeQdrant()
    _client(fake_default).post("/points/ingest-chunks", json=_body())
    fake_custom = FakeQdrant()
    _client(fake_custom).post(
        "/points/ingest-chunks",
        json=_body(source_id_key="doc_ref", source_id_type="untyped"),
    )
    expected = make_point_id("doc_a", 1)
    assert str(fake_default.upserted[0].id) == expected
    assert str(fake_custom.upserted[0].id) == expected


def test_unknown_source_id_type_is_rejected():
    fake = FakeQdrant()
    resp = _client(fake).post(
        "/points/ingest-chunks",
        json=_body(source_id_type="bogus"),
    )
    # This app remaps all pydantic RequestValidationError failures to HTTP 400
    # via a global exception handler in main.py (not FastAPI's default 422).
    assert resp.status_code == 400


def test_non_finite_float_source_id_is_rejected():
    fake = FakeQdrant()
    resp = _client(fake).post(
        "/points/ingest-chunks",
        json=_body(source_id="inf", source_id_type="float"),
    )
    # Coercion happens inside _run_ingest (not pydantic validation), so a
    # non-finite float is reported as an ok=False ingest failure, not a 400.
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is False
    assert "not a float" in (data["error"] or "")
    assert fake.upserted == []


def test_nan_float_source_id_is_rejected():
    fake = FakeQdrant()
    resp = _client(fake).post(
        "/points/ingest-chunks",
        json=_body(source_id="nan", source_id_type="float"),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is False
    assert "not a float" in (data["error"] or "")
    assert fake.upserted == []


def test_clobbering_source_id_key_is_rejected():
    fake = FakeQdrant()
    resp = _client(fake).post(
        "/points/ingest-chunks",
        json=_body(source_id_key="text"),
    )
    # Reserved-key validator raises a pydantic ValueError, which the app's
    # global RequestValidationError handler remaps to HTTP 400.
    assert resp.status_code == 400
    assert fake.upserted == []


def test_default_source_id_key_is_not_rejected():
    fake = FakeQdrant()
    resp = _client(fake).post(
        "/points/ingest-chunks",
        json=_body(source_id_key="source_id"),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert fake.upserted[0].payload["source_id"] == "doc_a"


def test_datetime_type_accepts_iso_and_rejects_non_date():
    ok = _client(FakeQdrant()).post(
        "/points/ingest-chunks",
        json=_body(source_id="2020-01-02", source_id_type="datetime"),
    )
    assert ok.status_code == 200 and ok.json()["ok"] is True

    bad = _client(FakeQdrant()).post(
        "/points/ingest-chunks",
        json=_body(source_id="not-a-date", source_id_type="datetime"),
    )
    assert bad.status_code == 200 and bad.json()["ok"] is False


def test_uuid_type_accepts_uuid_and_rejects_plain_string():
    good = _client(FakeQdrant()).post(
        "/points/ingest-chunks",
        json=_body(
            source_id="12345678-1234-1234-1234-123456789abc", source_id_type="uuid"
        ),
    )
    assert good.status_code == 200 and good.json()["ok"] is True

    bad = _client(FakeQdrant()).post(
        "/points/ingest-chunks",
        json=_body(source_id="doc_a", source_id_type="uuid"),
    )
    assert bad.status_code == 200 and bad.json()["ok"] is False


def test_geo_type_rejects_scalar_source_id():
    resp = _client(FakeQdrant()).post(
        "/points/ingest-chunks",
        json=_body(source_id="doc_a", source_id_type="geo"),
    )
    assert resp.status_code == 200 and resp.json()["ok"] is False


def test_keyword_type_stores_value_as_string():
    fake = FakeQdrant()
    resp = _client(fake).post(
        "/points/ingest-chunks",
        json=_body(source_id="doc_a", source_id_type="keyword"),
    )
    assert resp.status_code == 200 and resp.json()["ok"] is True
    assert fake.upserted[0].payload["source_id"] == "doc_a"

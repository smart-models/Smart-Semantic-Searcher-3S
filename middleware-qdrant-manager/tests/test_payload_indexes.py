"""Tests for payload index management: repository logic and API endpoints."""

from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import PayloadSchemaType, TextIndexParams

from qdrant_middleware.api.dependencies import get_qdrant_client
from qdrant_middleware.core.config import Settings
from qdrant_middleware.main import create_app
from qdrant_middleware.repositories import qdrant_collections
from qdrant_middleware.schemas.collections import PayloadIndexCreateRequest


def test_payload_index_request_accepts_supported_schema():
    request = PayloadIndexCreateRequest.model_validate(
        {"indexes": [{"field_name": " timestamp ", "field_schema": "integer"}]}
    )

    assert request.indexes[0].field_name == "timestamp"
    assert request.indexes[0].field_schema == "integer"


def test_payload_index_request_rejects_empty_indexes():
    with pytest.raises(ValidationError, match="indexes cannot be empty"):
        PayloadIndexCreateRequest.model_validate({"indexes": []})


def test_payload_index_request_rejects_duplicate_fields_after_trim():
    with pytest.raises(ValidationError, match="duplicate payload index field"):
        PayloadIndexCreateRequest.model_validate(
            {
                "indexes": [
                    {"field_name": "timestamp", "field_schema": "integer"},
                    {"field_name": " timestamp ", "field_schema": "integer"},
                ]
            }
        )


def test_payload_index_request_rejects_invalid_schema_type():
    with pytest.raises(ValidationError):
        PayloadIndexCreateRequest.model_validate(
            {"indexes": [{"field_name": "timestamp", "field_schema": "unknown"}]}
        )


@pytest.mark.parametrize("schema_name", ["datetime", "uuid"])
def test_payload_index_request_accepts_datetime_and_uuid(schema_name):
    request = PayloadIndexCreateRequest.model_validate(
        {"indexes": [{"field_name": "created_at", "field_schema": schema_name}]}
    )

    assert request.indexes[0].field_schema == schema_name


def test_payload_index_request_accepts_phrase_matching_on_text():
    request = PayloadIndexCreateRequest.model_validate(
        {
            "indexes": [
                {"field_name": "body", "field_schema": "text", "phrase_matching": True}
            ]
        }
    )

    assert request.indexes[0].phrase_matching is True


def test_payload_index_request_rejects_phrase_matching_on_non_text():
    with pytest.raises(ValidationError, match="phrase_matching"):
        PayloadIndexCreateRequest.model_validate(
            {
                "indexes": [
                    {
                        "field_name": "type",
                        "field_schema": "keyword",
                        "phrase_matching": True,
                    }
                ]
            }
        )


def test_same_payload_schema_compares_phrase_matching_for_text():
    plain_text = SimpleNamespace(data_type=PayloadSchemaType.TEXT, params=None)
    phrase_text = SimpleNamespace(
        data_type=PayloadSchemaType.TEXT,
        params=SimpleNamespace(phrase_matching=True),
    )
    expected_phrase = TextIndexParams(type="text", phrase_matching=True)

    assert not qdrant_collections._same_payload_schema(plain_text, expected_phrase)
    assert qdrant_collections._same_payload_schema(phrase_text, expected_phrase)
    assert qdrant_collections._same_payload_schema(plain_text, PayloadSchemaType.TEXT)


def test_list_payload_indexes_exposes_phrase_matching_for_text():
    entries = {
        "body": SimpleNamespace(
            data_type=PayloadSchemaType.TEXT,
            params=SimpleNamespace(phrase_matching=True),
        ),
        "type": SimpleNamespace(data_type=PayloadSchemaType.KEYWORD, params=None),
    }
    info = SimpleNamespace(payload_schema=entries)

    result = qdrant_collections._payload_index_items(info)

    by_name = {item["field_name"]: item for item in result}
    assert by_name["body"]["phrase_matching"] is True
    assert "phrase_matching" not in by_name["type"]


def _entry(data_type):
    return SimpleNamespace(data_type=data_type)


def _test_settings():
    return Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        quantization_enabled=False,
        sparse_idf=False,
    )


def _compatible_collection(payload_schema=None, settings=None):
    settings = settings or _test_settings()
    dense = SimpleNamespace(
        size=settings.dense_dim,
        distance="Cosine",
        on_disk=settings.dense_on_disk,
        datatype="float16" if settings.dense_float16 else None,
        hnsw_config=SimpleNamespace(
            m=settings.hnsw_m,
            ef_construct=settings.hnsw_ef_construct,
            full_scan_threshold=settings.hnsw_full_scan_threshold,
            inline_storage=None,
        ),
    )
    sparse = SimpleNamespace(
        index=SimpleNamespace(
            on_disk=settings.sparse_on_disk,
            full_scan_threshold=settings.sparse_full_scan_threshold,
            datatype=None,
        ),
        modifier=None,
    )
    config = SimpleNamespace(
        params=SimpleNamespace(
            vectors={settings.dense_vector_name: dense},
            sparse_vectors={settings.sparse_vector_name: sparse},
            shard_number=settings.shard_number,
            replication_factor=settings.replication_factor,
            on_disk_payload=settings.on_disk_payload,
        ),
        quantization_config=None,
        optimizer_config=SimpleNamespace(
            indexing_threshold=settings.indexing_threshold,
            memmap_threshold=None,
            prevent_unoptimized=None,
        ),
        strict_mode_config=SimpleNamespace(enabled=True),
    )
    return SimpleNamespace(
        config=config,
        status="green",
        points_count=0,
        vectors_count=0,
        payload_schema=payload_schema or {},
    )


class SourceIdIndexClient:
    """Minimal client for source-id index creation and schema checks."""

    def __init__(self, payload_schema=None):
        self.info = _compatible_collection(payload_schema)
        self.created = []

    async def get_collection(self, collection_name):
        """Return the configured collection report."""
        return self.info

    async def create_payload_index(self, **kwargs):
        """Record the schema Qdrant would receive."""
        self.created.append(kwargs)


async def test_source_id_index_uses_requested_exact_match_schema():
    """Replacement source filters must index the same type they compare."""
    client = SourceIdIndexClient()

    await qdrant_collections.ensure_source_id_index(
        client,
        "docs",
        _test_settings(),
        field_schema=PayloadSchemaType.INTEGER,
    )

    assert client.created[0]["field_schema"] == PayloadSchemaType.INTEGER


@pytest.mark.parametrize("field", ["source.id", "source[].id", "source name"])
async def test_source_id_index_is_literal_even_when_nested_index_exists(field):
    client = SourceIdIndexClient({field: _entry(PayloadSchemaType.KEYWORD)})
    await qdrant_collections.ensure_source_id_index(
        client, "docs", _test_settings(), field_name=field
    )
    if field == "source name":
        assert client.created == []
    else:
        assert client.created[0]["field_name"] == f'"{field}"'


async def test_source_id_index_rejects_an_existing_mismatched_schema():
    """An old keyword index cannot safely serve an integer source-id filter."""
    client = SourceIdIndexClient({"source_id": _entry(PayloadSchemaType.KEYWORD)})

    with pytest.raises(qdrant_collections.PayloadIndexTypeMismatch):
        await qdrant_collections.ensure_source_id_index(
            client,
            "docs",
            _test_settings(),
            field_schema=PayloadSchemaType.INTEGER,
        )

    assert client.created == []


class PayloadIndexClient:
    """Fake Qdrant client backing payload-index create, read and delete."""

    def __init__(self, payload_schema=None, settings=None):
        self.info = _compatible_collection(payload_schema, settings)
        self.created = []
        self.deleted = []

    async def get_collection(self, collection_name):
        return self.info

    async def create_payload_index(self, **kwargs):
        self.created.append(kwargs)
        field = kwargs["field_name"].strip('"')
        self.info.payload_schema[field] = _entry(kwargs["field_schema"])

    async def delete_payload_index(self, collection_name, field_name):
        self.deleted.append(field_name)
        self.info.payload_schema.pop(field_name.strip('"'), None)


@pytest.mark.asyncio
async def test_repository_creates_payload_index_and_invalidates_cache():
    settings = _test_settings()
    client = PayloadIndexClient()
    key = (id(client), "demo")
    qdrant_collections.repository.mark_indexes_ensured(key)

    result = await qdrant_collections.create_payload_indexes(
        client,
        "demo",
        [("timestamp", PayloadSchemaType.INTEGER)],
        settings=settings,
    )

    assert result == {
        "created": ["timestamp"],
        "updated": [],
        "skipped": [],
        "failed": [],
    }
    assert client.created[0]["field_name"] == "timestamp"
    assert client.created[0]["field_schema"] == PayloadSchemaType.INTEGER
    assert not qdrant_collections.repository.indexes_ensured(key)


@pytest.mark.asyncio
async def test_repository_skips_existing_same_type_payload_index():
    settings = _test_settings()
    client = PayloadIndexClient({"timestamp": _entry(PayloadSchemaType.INTEGER)})

    result = await qdrant_collections.create_payload_indexes(
        client,
        "demo",
        [("timestamp", PayloadSchemaType.INTEGER)],
        settings=settings,
    )

    assert result == {
        "created": [],
        "updated": [],
        "skipped": ["timestamp"],
        "failed": [],
    }
    assert client.created == []


@pytest.mark.asyncio
async def test_repository_replaces_existing_different_type_payload_index():
    settings = _test_settings()
    client = PayloadIndexClient({"timestamp": _entry(PayloadSchemaType.KEYWORD)})

    result = await qdrant_collections.create_payload_indexes(
        client,
        "demo",
        [("timestamp", PayloadSchemaType.INTEGER)],
        settings=settings,
    )

    assert result == {
        "created": [],
        "updated": ["timestamp"],
        "skipped": [],
        "failed": [],
    }
    assert client.deleted == ["timestamp"]
    assert client.created[0]["field_schema"] == PayloadSchemaType.INTEGER


@pytest.mark.asyncio
async def test_repository_quotes_field_names_with_spaces_for_qdrant():
    settings = _test_settings()
    client = PayloadIndexClient()

    await qdrant_collections.create_payload_indexes(
        client,
        "demo",
        [("Header 1", PayloadSchemaType.KEYWORD)],
        settings=settings,
    )

    assert client.created[0]["field_name"] == '"Header 1"'


class MissingCollectionClient(PayloadIndexClient):
    """PayloadIndexClient reporting the collection as missing."""

    async def get_collection(self, collection_name):
        raise UnexpectedResponse(
            status_code=404,
            reason_phrase="Not Found",
            content=b"not found",
            headers={},
        )


@pytest.mark.parametrize("path", ["a.b", "items[].label"])
async def test_literal_and_traversal_indexes_have_independent_lifecycles(path):
    """An old quoted index cannot satisfy or be deleted as a traversal index."""

    class ExactPathClient(PayloadIndexClient):
        """Retain Qdrant path syntax rather than stripping meaningful quotes."""

        async def create_payload_index(self, **kwargs):
            self.created.append(kwargs)
            self.info.payload_schema[kwargs["field_name"]] = _entry(
                kwargs["field_schema"]
            )

        async def delete_payload_index(self, collection_name, field_name):
            self.deleted.append(field_name)
            self.info.payload_schema.pop(field_name, None)

    literal = f'"{path}"'
    client = ExactPathClient({literal: _entry(PayloadSchemaType.KEYWORD)})
    result = await qdrant_collections.create_payload_indexes(
        client, "demo", [(path, PayloadSchemaType.KEYWORD)], _test_settings()
    )
    assert result["created"] == [path]
    assert client.created[0]["field_name"] == path
    listed = await qdrant_collections.list_payload_indexes(
        client, "demo", _test_settings()
    )
    assert {item["field_name"] for item in listed} == {path, literal}
    await qdrant_collections.delete_payload_index(
        client, "demo", path, _test_settings()
    )
    assert client.deleted == [path]
    assert set(client.info.payload_schema) == {literal}
    request = PayloadIndexCreateRequest(
        indexes=[
            {"field_name": path, "field_schema": "keyword"},
            {"field_name": literal, "field_schema": "keyword"},
        ]
    )
    assert len(request.indexes) == 2


@pytest.mark.asyncio
async def test_repository_raises_not_found_for_missing_collection():
    settings = _test_settings()

    with pytest.raises(qdrant_collections.PayloadIndexCollectionNotFound):
        await qdrant_collections.create_payload_indexes(
            MissingCollectionClient(),
            "missing",
            [("timestamp", PayloadSchemaType.INTEGER)],
            settings=settings,
        )


class FlakyPayloadIndexClient(PayloadIndexClient):
    """PayloadIndexClient whose index creation fails once, then succeeds."""

    async def create_payload_index(self, **kwargs):
        self.created.append(kwargs)
        if len(self.created) == 1:
            raise httpx.ConnectError("temporary")
        field = kwargs["field_name"].strip('"')
        self.info.payload_schema[field] = _entry(kwargs["field_schema"])


@pytest.mark.asyncio
async def test_repository_retries_transient_payload_index_creation():
    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        quantization_enabled=False,
        sparse_idf=False,
        RETRY_BASE_DELAY=0,
    )
    client = FlakyPayloadIndexClient()

    result = await qdrant_collections.create_payload_indexes(
        client,
        "demo",
        [("timestamp", PayloadSchemaType.INTEGER)],
        settings=settings,
    )

    assert result["created"] == ["timestamp"]
    assert len(client.created) == 2


def _app_with_client(client):
    app = create_app(_test_settings())
    app.dependency_overrides[get_qdrant_client] = lambda: client
    return app


def test_create_payload_indexes_endpoint_creates_integer_index():
    client = PayloadIndexClient()
    response = TestClient(_app_with_client(client)).post(
        "/collections/demo/payload-indexes",
        json={"indexes": [{"field_name": "timestamp", "field_schema": "integer"}]},
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "collection_name": "demo",
        "created": ["timestamp"],
        "updated": [],
        "skipped": [],
        "failed": [],
    }


def test_create_endpoint_maps_text_phrase_to_text_index_params():
    client = PayloadIndexClient()
    response = TestClient(_app_with_client(client)).post(
        "/collections/demo/payload-indexes",
        json={
            "indexes": [
                {"field_name": "body", "field_schema": "text", "phrase_matching": True}
            ]
        },
    )

    assert response.status_code == 200
    sent = client.created[0]["field_schema"]
    assert isinstance(sent, TextIndexParams)
    assert sent.phrase_matching is True


def test_create_payload_indexes_endpoint_creates_datetime_index():
    client = PayloadIndexClient()
    response = TestClient(_app_with_client(client)).post(
        "/collections/demo/payload-indexes",
        json={"indexes": [{"field_name": "created_at", "field_schema": "datetime"}]},
    )

    assert response.status_code == 200
    assert response.json()["created"] == ["created_at"]
    assert client.created[0]["field_schema"] == PayloadSchemaType.DATETIME


def test_create_payload_indexes_endpoint_creates_uuid_index():
    client = PayloadIndexClient()
    response = TestClient(_app_with_client(client)).post(
        "/collections/demo/payload-indexes",
        json={"indexes": [{"field_name": "doc_id", "field_schema": "uuid"}]},
    )

    assert response.status_code == 200
    assert response.json()["created"] == ["doc_id"]
    assert client.created[0]["field_schema"] == PayloadSchemaType.UUID


def test_create_payload_indexes_endpoint_is_idempotent():
    client = PayloadIndexClient({"timestamp": _entry(PayloadSchemaType.INTEGER)})
    response = TestClient(_app_with_client(client)).post(
        "/collections/demo/payload-indexes",
        json={"indexes": [{"field_name": "timestamp", "field_schema": "integer"}]},
    )

    assert response.status_code == 200
    assert response.json()["created"] == []
    assert response.json()["skipped"] == ["timestamp"]


def test_create_payload_indexes_endpoint_updates_type_of_existing_index():
    client = PayloadIndexClient({"timestamp": _entry(PayloadSchemaType.KEYWORD)})
    response = TestClient(_app_with_client(client)).post(
        "/collections/demo/payload-indexes",
        json={"indexes": [{"field_name": "timestamp", "field_schema": "integer"}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["created"] == []
    assert body["updated"] == ["timestamp"]
    assert body["skipped"] == []
    assert client.deleted == ["timestamp"]


def test_create_payload_indexes_endpoint_returns_400_for_invalid_type():
    response = TestClient(_app_with_client(PayloadIndexClient())).post(
        "/collections/demo/payload-indexes",
        json={"indexes": [{"field_name": "timestamp", "field_schema": "unknown"}]},
    )

    assert response.status_code == 400


def test_create_payload_indexes_endpoint_returns_400_for_empty_indexes():
    response = TestClient(_app_with_client(PayloadIndexClient())).post(
        "/collections/demo/payload-indexes",
        json={"indexes": []},
    )

    assert response.status_code == 400


def test_create_payload_indexes_endpoint_returns_404_for_missing_collection():
    response = TestClient(_app_with_client(MissingCollectionClient())).post(
        "/collections/missing/payload-indexes",
        json={"indexes": [{"field_name": "timestamp", "field_schema": "integer"}]},
    )

    assert response.status_code == 404


class AlreadyExistsRaceClient(PayloadIndexClient):
    """PayloadIndexClient where a concurrent writer created the index."""

    async def create_payload_index(self, **kwargs):
        self.created.append(kwargs)
        self.info.payload_schema[kwargs["field_name"].strip('"')] = _entry(
            kwargs["field_schema"]
        )
        raise UnexpectedResponse(
            status_code=409,
            reason_phrase="Conflict",
            content=b"already exists",
            headers={},
        )


@pytest.mark.asyncio
async def test_repository_treats_create_race_same_type_as_skipped():
    settings = _test_settings()
    client = AlreadyExistsRaceClient()

    result = await qdrant_collections.create_payload_indexes(
        client,
        "demo",
        [("timestamp", PayloadSchemaType.INTEGER)],
        settings=settings,
    )

    assert result == {
        "created": [],
        "updated": [],
        "skipped": ["timestamp"],
        "failed": [],
    }


def test_create_payload_indexes_endpoint_rejects_blank_collection_name():
    response = TestClient(_app_with_client(PayloadIndexClient())).post(
        "/collections/%20%20/payload-indexes",
        json={"indexes": [{"field_name": "timestamp", "field_schema": "integer"}]},
    )

    assert response.status_code == 400


def test_create_payload_indexes_endpoint_rejects_duplicate_normalized_field():
    response = TestClient(_app_with_client(PayloadIndexClient())).post(
        "/collections/demo/payload-indexes",
        json={
            "indexes": [
                {"field_name": "Header 1", "field_schema": "keyword"},
                {"field_name": '"Header 1"', "field_schema": "keyword"},
            ]
        },
    )

    assert response.status_code == 400


def test_create_payload_indexes_endpoint_returns_409_for_non_payload_schema_drift():
    client = PayloadIndexClient(settings=_test_settings())
    client.info.config.params.vectors = {}

    response = TestClient(_app_with_client(client)).post(
        "/collections/demo/payload-indexes",
        json={"indexes": [{"field_name": "timestamp", "field_schema": "integer"}]},
    )

    assert response.status_code == 409
    assert "errors" in response.json()["detail"]


@pytest.mark.asyncio
async def test_repository_lists_payload_indexes_with_normalized_names():
    settings = _test_settings()
    client = PayloadIndexClient(
        {
            "timestamp": _entry(PayloadSchemaType.INTEGER),
            '"Header 1"': _entry(PayloadSchemaType.KEYWORD),
        }
    )

    result = await qdrant_collections.list_payload_indexes(
        client,
        "demo",
        settings=settings,
    )

    assert result == [
        {"field_name": "Header 1", "field_schema": "keyword"},
        {"field_name": "timestamp", "field_schema": "integer"},
    ]


@pytest.mark.asyncio
async def test_repository_lists_payload_indexes_even_with_schema_drift():
    settings = _test_settings()
    client = PayloadIndexClient({"timestamp": _entry(PayloadSchemaType.INTEGER)})
    client.info.config.params.vectors = {}

    result = await qdrant_collections.list_payload_indexes(
        client,
        "demo",
        settings=settings,
    )

    assert result == [{"field_name": "timestamp", "field_schema": "integer"}]


def test_list_payload_indexes_endpoint_returns_indexes():
    client = PayloadIndexClient(
        {
            "timestamp": _entry(PayloadSchemaType.INTEGER),
            '"Header 1"': _entry(PayloadSchemaType.KEYWORD),
        }
    )
    response = TestClient(_app_with_client(client)).get(
        "/collections/demo/payload-indexes",
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "collection_name": "demo",
        "indexes": [
            {"field_name": "Header 1", "field_schema": "keyword"},
            {"field_name": "timestamp", "field_schema": "integer"},
        ],
        "count": 2,
    }


def test_list_payload_indexes_endpoint_returns_404_for_missing_collection():
    response = TestClient(_app_with_client(MissingCollectionClient())).get(
        "/collections/missing/payload-indexes",
    )

    assert response.status_code == 404


class DeletablePayloadIndexClient(PayloadIndexClient):
    """PayloadIndexClient recording the indexes it is asked to delete."""

    def __init__(self, payload_schema=None, settings=None):
        super().__init__(payload_schema, settings)
        self.deleted = []

    async def delete_payload_index(self, **kwargs):
        self.deleted.append(kwargs)
        field = kwargs["field_name"].strip('"')
        self.info.payload_schema.pop(field, None)


@pytest.mark.asyncio
async def test_repository_deletes_payload_index_and_invalidates_cache():
    settings = _test_settings()
    client = DeletablePayloadIndexClient(
        {"timestamp": _entry(PayloadSchemaType.INTEGER)}
    )
    key = (id(client), "demo")
    qdrant_collections.repository.mark_indexes_ensured(key)

    result = await qdrant_collections.delete_payload_index(
        client,
        "demo",
        "timestamp",
        settings=settings,
    )

    assert result == "timestamp"
    assert client.deleted[0]["field_name"] == "timestamp"
    assert "timestamp" not in client.info.payload_schema
    assert not qdrant_collections.repository.indexes_ensured(key)


@pytest.mark.asyncio
async def test_repository_quotes_field_names_with_spaces_on_delete():
    settings = _test_settings()
    client = DeletablePayloadIndexClient(
        {'"Header 1"': _entry(PayloadSchemaType.KEYWORD)}
    )

    await qdrant_collections.delete_payload_index(
        client,
        "demo",
        "Header 1",
        settings=settings,
    )

    assert client.deleted[0]["field_name"] == '"Header 1"'


@pytest.mark.asyncio
async def test_repository_raises_not_found_for_missing_field_on_delete():
    settings = _test_settings()
    client = DeletablePayloadIndexClient(
        {"timestamp": _entry(PayloadSchemaType.INTEGER)}
    )

    with pytest.raises(qdrant_collections.PayloadIndexFieldNotFound) as exc:
        await qdrant_collections.delete_payload_index(
            client,
            "demo",
            "nonexistent",
            settings=settings,
        )

    assert exc.value.field_name == "nonexistent"
    assert client.deleted == []


@pytest.mark.asyncio
async def test_repository_raises_not_found_for_missing_collection_on_delete():
    settings = _test_settings()

    with pytest.raises(qdrant_collections.PayloadIndexCollectionNotFound):
        await qdrant_collections.delete_payload_index(
            MissingCollectionClient(),
            "missing",
            "timestamp",
            settings=settings,
        )


def test_delete_payload_index_endpoint_deletes_index():
    client = DeletablePayloadIndexClient(
        {"timestamp": _entry(PayloadSchemaType.INTEGER)}
    )
    response = TestClient(_app_with_client(client)).delete(
        "/collections/demo/payload-indexes/timestamp",
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "collection_name": "demo",
        "field_name": "timestamp",
        "deleted": True,
    }
    assert "timestamp" not in client.info.payload_schema


def test_delete_payload_index_endpoint_returns_404_for_missing_collection():
    response = TestClient(_app_with_client(MissingCollectionClient())).delete(
        "/collections/missing/payload-indexes/timestamp",
    )

    assert response.status_code == 404


def test_delete_payload_index_endpoint_returns_404_for_missing_field():
    client = DeletablePayloadIndexClient(
        {"timestamp": _entry(PayloadSchemaType.INTEGER)}
    )
    response = TestClient(_app_with_client(client)).delete(
        "/collections/demo/payload-indexes/nonexistent",
    )

    assert response.status_code == 404
    assert "nonexistent" in response.json()["detail"]

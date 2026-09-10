"""Tests for strict-mode alignment and the collection repository caches."""

from types import SimpleNamespace

import httpx
import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from qdrant_middleware.core.config import Settings
from qdrant_middleware.core.lifespan import _align_existing_strict_mode
from qdrant_middleware.repositories import qdrant_collections
from qdrant_middleware.schemas.collections import CollectionSchema


class _Collections:
    """Minimal stand-in for Qdrant's collection-listing response."""

    def __init__(self, names):
        self.collections = [type("Collection", (), {"name": name})() for name in names]


class RaceCreateClient:
    """Fake client where the collection appears between check and create."""

    async def get_collections(self):
        return _Collections([])

    async def create_collection(self, **kwargs):
        raise UnexpectedResponse(
            status_code=409,
            reason_phrase="Conflict",
            content=b"Already exists",
            headers={},
        )

    async def create_payload_index(self, **kwargs):
        return None


class FlakyListClient:
    """Fake client whose first listing fails and whose later ones succeed."""

    def __init__(self):
        self.calls = 0

    async def get_collections(self):
        self.calls += 1
        if self.calls == 1:
            raise httpx.ConnectError("temporary")
        return _Collections(["demo"])


@pytest.mark.asyncio
async def test_create_collection_treats_race_409_as_existing():
    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )

    created = await qdrant_collections.create_collection(
        RaceCreateClient(),
        "demo",
        recreate=False,
        schema=CollectionSchema.from_settings(settings),
        settings=settings,
    )

    assert created is False


class RecordingCreateClient:
    """Fake client recording the create and index calls it receives."""

    def __init__(self):
        self.create_kwargs = None

    async def get_collections(self):
        return _Collections([])

    async def create_collection(self, **kwargs):
        self.create_kwargs = kwargs
        return True

    async def create_payload_index(self, **kwargs):
        return None


@pytest.mark.asyncio
async def test_create_collection_uses_schema_on_disk_payload_override():
    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )
    assert settings.on_disk_payload is True
    client = RecordingCreateClient()

    schema = CollectionSchema.from_settings(settings)
    schema = CollectionSchema(
        dense_on_disk=schema.dense_on_disk,
        quantization=schema.quantization,
        sparse_idf=schema.sparse_idf,
        sparse_on_disk=schema.sparse_on_disk,
        on_disk_payload=False,
    )
    await qdrant_collections.create_collection(
        client, "demo", recreate=False, schema=schema, settings=settings
    )

    assert client.create_kwargs["on_disk_payload"] is False


@pytest.mark.asyncio
async def test_create_collection_defaults_on_disk_payload_from_settings():
    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )
    client = RecordingCreateClient()

    await qdrant_collections.create_collection(
        client, "demo", recreate=False, schema=CollectionSchema(), settings=settings
    )

    assert client.create_kwargs["on_disk_payload"] is settings.on_disk_payload


@pytest.mark.asyncio
async def test_create_collection_enforces_unindexed_filtering_strict_mode():
    # enabled=True alone is a no-op: Qdrant applies only restrictions that
    # are explicitly set, so unindexed filtering must be rejected explicitly.
    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )
    client = RecordingCreateClient()

    await qdrant_collections.create_collection(
        client, "demo", recreate=False, schema=CollectionSchema(), settings=settings
    )

    strict = client.create_kwargs["strict_mode_config"]
    assert strict.enabled is True
    assert strict.unindexed_filtering_retrieve is False
    assert strict.unindexed_filtering_update is False


class StrictModeAlignClient:
    """Fake client exposing per-collection strict mode configs."""

    def __init__(self, strict_configs):
        self._strict = strict_configs
        self.updated = {}

    async def get_collections(self):
        return _Collections(list(self._strict))

    async def get_collection(self, collection_name):
        return SimpleNamespace(
            config=SimpleNamespace(strict_mode_config=self._strict[collection_name])
        )

    async def update_collection(self, collection_name, **kwargs):
        self.updated[collection_name] = kwargs
        return True


def _align_settings():
    return Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        RETRY_BASE_DELAY=0,
    )


@pytest.mark.asyncio
async def test_align_strict_mode_updates_legacy_enabled_only_config():
    client = StrictModeAlignClient(
        {
            "legacy": SimpleNamespace(
                enabled=True,
                unindexed_filtering_retrieve=None,
                unindexed_filtering_update=None,
            )
        }
    )

    updated = await qdrant_collections.align_strict_mode_configs(
        client, settings=_align_settings()
    )

    assert updated == ["legacy"]
    strict = client.updated["legacy"]["strict_mode_config"]
    assert strict.enabled is True
    assert strict.unindexed_filtering_retrieve is False
    assert strict.unindexed_filtering_update is False


@pytest.mark.asyncio
async def test_align_strict_mode_skips_already_aligned_collection():
    client = StrictModeAlignClient(
        {
            "aligned": SimpleNamespace(
                enabled=True,
                unindexed_filtering_retrieve=False,
                unindexed_filtering_update=False,
            )
        }
    )

    updated = await qdrant_collections.align_strict_mode_configs(
        client, settings=_align_settings()
    )

    assert updated == []
    assert client.updated == {}


@pytest.mark.asyncio
async def test_align_strict_mode_updates_collection_without_strict_config():
    client = StrictModeAlignClient({"bare": None})

    updated = await qdrant_collections.align_strict_mode_configs(
        client, settings=_align_settings()
    )

    assert updated == ["bare"]
    assert "bare" in client.updated


@pytest.mark.asyncio
async def test_startup_alignment_tolerates_unreachable_qdrant():
    class DownClient:
        """Fake client whose listing always fails, as if Qdrant were down."""

        async def get_collections(self):
            raise httpx.ConnectError("qdrant down")

    # Must swallow the failure: alignment is best-effort and cannot block startup.
    await _align_existing_strict_mode(DownClient(), _align_settings())


@pytest.mark.asyncio
async def test_list_collections_retries_transient_failures():
    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        RETRY_BASE_DELAY=0,
    )
    client = FlakyListClient()

    result = await qdrant_collections.list_collections(client, settings)

    assert [collection.name for collection in result.collections] == ["demo"]
    assert client.calls == 2


class QuantizationClient:
    """Fake client returning a config that carries quantization."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.calls = 0

    async def get_collection(self, collection_name):
        self.calls += 1
        config = type(
            "Config",
            (),
            {"quantization_config": object() if self.enabled else None},
        )()
        return type("CollectionInfo", (), {"config": config})()


class PayloadIndexClient:
    """Fake client recording the payload indexes it is asked to create."""

    def __init__(self):
        self.calls = 0

    async def create_payload_index(self, **kwargs):
        self.calls += 1
        return None


@pytest.mark.asyncio
async def test_quantization_cache_is_isolated_per_client():
    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )
    qdrant_collections.repository._collection_quantization = {}
    first = QuantizationClient(enabled=False)
    second = QuantizationClient(enabled=True)

    assert (
        await qdrant_collections.collection_has_quantization(first, "demo", settings)
    ) is False
    assert (
        await qdrant_collections.collection_has_quantization(second, "demo", settings)
    ) is True
    assert first.calls == 1
    assert second.calls == 1


@pytest.mark.asyncio
async def test_ensure_payload_indexes_skips_creation_when_default_indexes_empty():
    qdrant_collections.repository._ensured_indexes = set()
    client = PayloadIndexClient()

    await qdrant_collections.ensure_payload_indexes(client, "demo")

    assert client.calls == 0

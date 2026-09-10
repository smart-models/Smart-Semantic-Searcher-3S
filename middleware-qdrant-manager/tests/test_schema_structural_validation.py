"""RED tests: schema validation respects None toggles (structural-only mode)."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from qdrant_client.models import PayloadSchemaType

from qdrant_middleware.api.dependencies import get_qdrant_client
from qdrant_middleware.core.config import Settings
from qdrant_middleware.main import create_app
from qdrant_middleware.repositories import qdrant_collections
from qdrant_middleware.schemas.collections import CollectionSchema
from qdrant_middleware.services.collection_schema import (
    _finding,
    _suggested_fixes_from_findings,
    _validate_schema,
)


def _base_settings():
    return Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )


def _collection_info(
    *,
    quantization_config=None,
    dense_on_disk=True,
    sparse_on_disk=None,
    sparse_modifier=None,
    on_disk_payload=None,
    settings=None,
):
    s = settings or _base_settings()
    dense = SimpleNamespace(
        size=s.dense_dim,
        distance="Cosine",
        on_disk=dense_on_disk,
        datatype="float16",
        hnsw_config=SimpleNamespace(
            m=s.hnsw_m,
            ef_construct=s.hnsw_ef_construct,
            full_scan_threshold=s.hnsw_full_scan_threshold,
            inline_storage=None,
        ),
    )
    sparse = SimpleNamespace(
        index=SimpleNamespace(
            on_disk=sparse_on_disk if sparse_on_disk is not None else s.sparse_on_disk,
            full_scan_threshold=s.sparse_full_scan_threshold,
            datatype=None,
        ),
        modifier=sparse_modifier,
    )
    config = SimpleNamespace(
        params=SimpleNamespace(
            vectors={s.dense_vector_name: dense},
            sparse_vectors={s.sparse_vector_name: sparse},
            shard_number=s.shard_number,
            replication_factor=s.replication_factor,
            on_disk_payload=on_disk_payload
            if on_disk_payload is not None
            else s.on_disk_payload,
        ),
        quantization_config=quantization_config,
        optimizer_config=SimpleNamespace(
            indexing_threshold=s.indexing_threshold,
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
        payload_schema={},
    )


# ---------------------------------------------------------------------------
# _validate_schema unit tests
# ---------------------------------------------------------------------------


def test_structural_schema_skips_dense_on_disk_check():
    settings = _base_settings()
    info = _collection_info(dense_on_disk=False, settings=settings)
    findings = _validate_schema(info, CollectionSchema(), settings)
    assert not any(f.category == "dense" and f.field == "on_disk" for f in findings)


def test_structural_schema_skips_quantization_check():
    settings = _base_settings()
    info = _collection_info(quantization_config=None, settings=settings)
    findings = _validate_schema(info, CollectionSchema(), settings)
    assert not any(f.category == "quantization" for f in findings)


def test_structural_schema_skips_sparse_modifier_check():
    settings = _base_settings()
    info = _collection_info(sparse_modifier=None, settings=settings)
    findings = _validate_schema(info, CollectionSchema(), settings)
    assert not any(f.field == "modifier" for f in findings)


def test_structural_schema_skips_sparse_on_disk_check():
    settings = _base_settings()
    info = _collection_info(
        sparse_on_disk=not settings.sparse_on_disk, settings=settings
    )
    findings = _validate_schema(info, CollectionSchema(), settings)
    assert not any(f.category == "sparse" and f.field == "on_disk" for f in findings)


def test_explicit_quantization_true_flags_missing_quantization():
    settings = _base_settings()
    info = _collection_info(quantization_config=None, settings=settings)
    findings = _validate_schema(info, CollectionSchema(quantization=True), settings)
    assert any(f.category == "quantization" for f in findings)


def test_explicit_quantization_false_flags_unexpected_quantization():
    settings = _base_settings()
    quant = SimpleNamespace(turbo=SimpleNamespace(bits="B4", always_ram=True))
    info = _collection_info(quantization_config=quant, settings=settings)
    findings = _validate_schema(info, CollectionSchema(quantization=False), settings)
    assert any(f.category == "quantization" for f in findings)


def test_explicit_dense_on_disk_true_flags_mismatch():
    settings = _base_settings()
    info = _collection_info(dense_on_disk=False, settings=settings)
    findings = _validate_schema(info, CollectionSchema(dense_on_disk=True), settings)
    assert any(f.category == "dense" and f.field == "on_disk" for f in findings)


def test_explicit_sparse_idf_true_flags_missing_modifier():
    settings = _base_settings()
    info = _collection_info(sparse_modifier=None, settings=settings)
    findings = _validate_schema(info, CollectionSchema(sparse_idf=True), settings)
    assert any(f.field == "modifier" for f in findings)


def test_explicit_sparse_on_disk_true_flags_mismatch():
    settings = _base_settings()
    info = _collection_info(sparse_on_disk=False, settings=settings)
    findings = _validate_schema(info, CollectionSchema(sparse_on_disk=True), settings)
    assert any(f.category == "sparse" and f.field == "on_disk" for f in findings)


def test_structural_schema_skips_on_disk_payload_check():
    settings = _base_settings()
    info = _collection_info(
        on_disk_payload=not settings.on_disk_payload, settings=settings
    )
    findings = _validate_schema(info, CollectionSchema(), settings)
    assert not any(f.field == "on_disk_payload" for f in findings)


def test_explicit_on_disk_payload_true_flags_mismatch():
    settings = _base_settings()
    info = _collection_info(on_disk_payload=False, settings=settings)
    findings = _validate_schema(info, CollectionSchema(on_disk_payload=True), settings)
    assert any(f.field == "on_disk_payload" for f in findings)


def test_explicit_on_disk_payload_matching_passes():
    settings = _base_settings()
    info = _collection_info(on_disk_payload=False, settings=settings)
    findings = _validate_schema(info, CollectionSchema(on_disk_payload=False), settings)
    assert not any(f.field == "on_disk_payload" for f in findings)


# ---------------------------------------------------------------------------
# _suggested_fixes_from_findings: non-auto-fixable when toggles unknown
# ---------------------------------------------------------------------------


def test_suggested_fix_not_auto_fixable_when_schema_toggles_unknown():
    schema = CollectionSchema()  # all None
    findings = [
        _finding(
            "collection",
            "exists",
            True,
            False,
            "critical",
            "collection 'demo' not found",
        )
    ]
    fixes = _suggested_fixes_from_findings("demo", schema, findings)
    assert fixes
    assert not any(f.auto_fixable for f in fixes)


# ---------------------------------------------------------------------------
# _suggested_fixes_from_findings: optimizer drift is fixable online
# ---------------------------------------------------------------------------


def _optimizer_finding():
    return _finding(
        "optimizer",
        "memmap_threshold",
        20_000,
        50_000,
        "medium",
        "memmap_threshold mismatch: expected 20000, got 50000",
    )


def test_optimizer_only_finding_suggests_online_update_not_recreate():
    fixes = _suggested_fixes_from_findings(
        "demo", CollectionSchema(), [_optimizer_finding()]
    )
    assert not any(f.type == "recreate_collection" for f in fixes)
    updates = [f for f in fixes if f.type == "update_optimizers"]
    assert len(updates) == 1
    assert updates[0].destructive is False
    assert "memmap_threshold" in updates[0].description


def test_optimizer_finding_with_structural_drift_still_suggests_recreate():
    findings = [
        _optimizer_finding(),
        _finding(
            "dense",
            "size",
            1024,
            768,
            "critical",
            "dense vector size mismatch: expected 1024, got 768",
        ),
    ]
    fixes = _suggested_fixes_from_findings("demo", CollectionSchema(), findings)
    assert any(f.type == "recreate_collection" for f in fixes)
    assert any(f.type == "update_optimizers" for f in fixes)


# ---------------------------------------------------------------------------
# Endpoint tests via TestClient
# ---------------------------------------------------------------------------


class StructuralClient:
    """Fake Qdrant client returning a collection with all toggles off."""

    def __init__(self):
        self.info = _collection_info(
            quantization_config=None,
            dense_on_disk=False,
            sparse_modifier=None,
        )
        self.created = []

    async def get_collection(self, collection_name):
        return self.info

    async def create_payload_index(self, **kwargs):
        self.created.append(kwargs)
        field = kwargs["field_name"].strip('"')
        self.info.payload_schema[field] = SimpleNamespace(
            data_type=kwargs["field_schema"]
        )


def _app(client):
    app = create_app(_base_settings())
    app.dependency_overrides[get_qdrant_client] = lambda: client
    return app


def test_verify_without_params_does_not_flag_toggle_mismatches():
    response = TestClient(_app(StructuralClient())).post(
        "/collections/verify",
        json={"collection_name": "demo"},
    )
    assert response.status_code == 200
    data = response.json()
    toggle_findings = [
        f
        for f in data.get("findings", [])
        if (
            f["category"] in {"dense", "quantization"}
            and f["field"] in {"on_disk", "turbo", "enabled"}
        )
        or f["field"] == "modifier"
    ]
    assert toggle_findings == []


def test_verify_with_quantization_true_flags_missing_quantization():
    response = TestClient(_app(StructuralClient())).post(
        "/collections/verify",
        json={"collection_name": "demo", "quantization": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert any(f["category"] == "quantization" for f in data.get("findings", []))


def test_verify_with_quantization_false_passes_when_quant_absent():
    response = TestClient(_app(StructuralClient())).post(
        "/collections/verify",
        json={"collection_name": "demo", "quantization": False},
    )
    assert response.status_code == 200
    data = response.json()
    assert not any(f["category"] == "quantization" for f in data.get("findings", []))


# ---------------------------------------------------------------------------
# create_payload_indexes: structural guard ignores toggle differences
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_payload_indexes_not_blocked_by_toggle_differences():
    settings = _base_settings()
    client = StructuralClient()

    result = await qdrant_collections.create_payload_indexes(
        client,
        "demo",
        [("tag", PayloadSchemaType.KEYWORD)],
        settings=settings,
    )

    assert result["created"] == ["tag"]


# ---------------------------------------------------------------------------
# PATCH schema toggles: online collection mutation
# ---------------------------------------------------------------------------


class ToggleUpdateClient(StructuralClient):
    """StructuralClient recording the update payload sent for schema toggles."""

    def __init__(self):
        super().__init__()
        self.update_calls = []

    async def update_collection(self, **kwargs):
        self.update_calls.append(kwargs)
        vectors_config = kwargs.get("vectors_config") or {}
        dense_diff = vectors_config.get(_base_settings().dense_vector_name)
        if dense_diff is not None and dense_diff.on_disk is not None:
            dense = self.info.config.params.vectors[_base_settings().dense_vector_name]
            dense.on_disk = dense_diff.on_disk

        sparse_config = kwargs.get("sparse_vectors_config") or {}
        sparse_params = sparse_config.get(_base_settings().sparse_vector_name)
        if sparse_params is not None:
            sparse = self.info.config.params.sparse_vectors[
                _base_settings().sparse_vector_name
            ]
            # Mirror Qdrant PATCH semantics: null fields mean "no change",
            # so clearing the modifier requires an explicit Modifier.NONE.
            if sparse_params.modifier is not None:
                sparse.modifier = sparse_params.modifier
            if (
                sparse_params.index is not None
                and sparse_params.index.on_disk is not None
            ):
                sparse.index.on_disk = sparse_params.index.on_disk

        if "quantization_config" in kwargs:
            quantization_config = kwargs["quantization_config"]
            self.info.config.quantization_config = (
                None
                if str(quantization_config) == "Disabled.DISABLED"
                else quantization_config
            )

        collection_params = kwargs.get("collection_params")
        if (
            collection_params is not None
            and collection_params.on_disk_payload is not None
        ):
            self.info.config.params.on_disk_payload = collection_params.on_disk_payload
        return True


@pytest.mark.asyncio
async def test_repository_updates_only_requested_collection_toggles():
    settings = _base_settings()
    client = ToggleUpdateClient()
    key = (id(client), "demo")
    qdrant_collections.repository.remember_quantization(key, False)

    await qdrant_collections.update_collection_schema_toggles(
        client,
        "demo",
        CollectionSchema(
            dense_on_disk=True,
            quantization=True,
            sparse_idf=True,
            sparse_on_disk=True,
        ),
        settings=settings,
    )

    assert len(client.update_calls) == 1
    call = client.update_calls[0]
    assert set(call) == {
        "collection_name",
        "vectors_config",
        "quantization_config",
        "sparse_vectors_config",
    }
    assert call["collection_name"] == "demo"
    dense_diff = call["vectors_config"][settings.dense_vector_name]
    assert dense_diff.on_disk is True
    assert call["quantization_config"] is not None
    sparse_params = call["sparse_vectors_config"][settings.sparse_vector_name]
    assert str(sparse_params.modifier).lower().endswith("idf")
    assert qdrant_collections.repository.cached_quantization(key) is None


@pytest.mark.asyncio
async def test_repository_updates_on_disk_payload_toggle_only():
    settings = _base_settings()
    client = ToggleUpdateClient()
    client.info.config.params.on_disk_payload = True

    await qdrant_collections.update_collection_schema_toggles(
        client,
        "demo",
        CollectionSchema(on_disk_payload=False),
        settings=settings,
    )

    assert len(client.update_calls) == 1
    call = client.update_calls[0]
    assert set(call) == {"collection_name", "collection_params"}
    assert call["collection_params"].on_disk_payload is False
    assert client.info.config.params.on_disk_payload is False


@pytest.mark.asyncio
async def test_repository_uses_disabled_marker_when_quantization_false():
    settings = _base_settings()
    client = ToggleUpdateClient()

    await qdrant_collections.update_collection_schema_toggles(
        client,
        "demo",
        CollectionSchema(quantization=False),
        settings=settings,
    )

    call = client.update_calls[0]
    assert set(call) == {"collection_name", "quantization_config"}
    assert str(call["quantization_config"]) == "Disabled.DISABLED"


@pytest.mark.asyncio
async def test_repository_sparse_on_disk_only_preserves_live_sparse_modifier():
    settings = _base_settings()
    client = ToggleUpdateClient()
    sparse = client.info.config.params.sparse_vectors[settings.sparse_vector_name]
    sparse.modifier = None
    sparse.index.on_disk = False

    await qdrant_collections.update_collection_schema_toggles(
        client,
        "demo",
        CollectionSchema(sparse_on_disk=True),
        settings=settings,
    )

    assert sparse.index.on_disk is True
    # The live "no IDF" state must survive; an explicit Modifier.NONE is the
    # PATCH-safe way to express it, so both spellings are acceptable.
    assert not qdrant_collections._sparse_modifier_enabled(sparse.modifier)


@pytest.mark.asyncio
async def test_repository_sparse_idf_false_sends_explicit_none_modifier():
    """Disabling IDF sends Modifier.NONE rather than a null modifier.

    Qdrant PATCH treats a null modifier as "no change", so omitting the
    explicit Modifier.NONE would leave the live idf modifier in place.
    """
    settings = _base_settings()
    client = ToggleUpdateClient()
    sparse = client.info.config.params.sparse_vectors[settings.sparse_vector_name]
    sparse.modifier = "idf"

    await qdrant_collections.update_collection_schema_toggles(
        client,
        "demo",
        CollectionSchema(sparse_idf=False),
        settings=settings,
    )

    sent = client.update_calls[0]["sparse_vectors_config"][settings.sparse_vector_name]
    assert sent.modifier is not None
    assert str(getattr(sent.modifier, "value", sent.modifier)).lower() == "none"


def test_patch_schema_toggles_disables_idf_on_collection_with_idf():
    client = ToggleUpdateClient()
    sparse = client.info.config.params.sparse_vectors[
        _base_settings().sparse_vector_name
    ]
    sparse.modifier = "idf"

    response = TestClient(_app(client)).patch(
        "/collections/demo/schema-toggles",
        json={"sparse_idf": False},
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True


@pytest.mark.asyncio
async def test_repository_sparse_idf_only_preserves_live_sparse_on_disk():
    settings = _base_settings()
    client = ToggleUpdateClient()
    sparse = client.info.config.params.sparse_vectors[settings.sparse_vector_name]
    sparse.modifier = None
    sparse.index.on_disk = False

    await qdrant_collections.update_collection_schema_toggles(
        client,
        "demo",
        CollectionSchema(sparse_idf=True),
        settings=settings,
    )

    assert str(sparse.modifier).lower().endswith("idf")
    assert sparse.index.on_disk is False


def test_patch_schema_toggles_endpoint_updates_and_reports_requested_schema():
    client = ToggleUpdateClient()
    response = TestClient(_app(client)).patch(
        "/collections/demo/schema-toggles",
        json={
            "dense_on_disk": True,
            "quantization": True,
            "sparse_idf": True,
            "sparse_on_disk": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["collection_name"] == "demo"
    assert data["report"]["schema_ok"] is True
    assert len(client.update_calls) == 1


def test_put_schema_toggles_endpoint_updates_and_reports_requested_schema():
    client = ToggleUpdateClient()
    response = TestClient(_app(client)).put(
        "/collections/demo/schema-toggles",
        json={"sparse_on_disk": True},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["collection_name"] == "demo"
    assert data["report"]["schema_ok"] is True
    assert len(client.update_calls) == 1


def test_patch_schema_toggles_endpoint_accepts_on_disk_payload_only():
    client = ToggleUpdateClient()
    client.info.config.params.on_disk_payload = True
    response = TestClient(_app(client)).patch(
        "/collections/demo/schema-toggles",
        json={"on_disk_payload": False},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["report"]["schema_ok"] is True
    assert data["report"]["on_disk_payload"] is False
    assert len(client.update_calls) == 1


def test_patch_schema_toggles_succeeds_despite_unrelated_schema_drift():
    """Drift on non-toggle settings (optimizer, hnsw) must not block toggle updates."""
    client = ToggleUpdateClient()
    client.info.config.optimizer_config.indexing_threshold = 999_999
    client.info.config.params.vectors[
        _base_settings().dense_vector_name
    ].hnsw_config.m = 4

    response = TestClient(_app(client)).patch(
        "/collections/demo/schema-toggles",
        json={"dense_on_disk": True},
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert len(client.update_calls) == 1


class IgnoringUpdateClient(StructuralClient):
    """Fake that accepts the update call but never applies it."""

    def __init__(self):
        super().__init__()
        self.update_calls = []

    async def update_collection(self, **kwargs):
        self.update_calls.append(kwargs)
        return True


def test_patch_schema_toggles_409_when_requested_toggle_not_applied():
    client = IgnoringUpdateClient()
    client.info.config.params.vectors[
        _base_settings().dense_vector_name
    ].on_disk = False

    response = TestClient(_app(client)).patch(
        "/collections/demo/schema-toggles",
        json={"dense_on_disk": True},
    )

    assert response.status_code == 409


def test_patch_schema_toggles_endpoint_rejects_empty_body():
    response = TestClient(_app(ToggleUpdateClient())).patch(
        "/collections/demo/schema-toggles",
        json={},
    )

    assert response.status_code == 400
    assert "at least one schema toggle" in response.text

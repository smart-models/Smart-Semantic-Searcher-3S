"""Pure collection schema helpers (no Qdrant I/O)."""

import logging
import uuid
from typing import Any, Literal

from qdrant_client.models import (
    Datatype,
    Distance,
    Modifier,
    PayloadSchemaType,
    SparseIndexParams,
    SparseVectorParams,
    TurboQuantBitSize,
    TurboQuantization,
    TurboQuantQuantizationConfig,
)

from qdrant_middleware.core.config import Settings, get_settings
from qdrant_middleware.schemas.collections import (
    CollectionReport,
    CollectionSchema,
    SchemaFinding,
    SuggestedFix,
)

log = logging.getLogger(__name__)


def _runtime_settings(settings: Settings | None) -> Settings:
    """Return ``settings`` or fall back to the cached process defaults."""
    return settings or get_settings()


TURBO_BITS = TurboQuantBitSize.BITS4

_POINT_NS = uuid.NAMESPACE_URL

PAYLOAD_INDEXES: list[tuple[str, PayloadSchemaType]] = []


def make_point_id(source_id: str, chunk_id: int) -> str:
    """Build a deterministic UUID5 point id from ``source_id`` and ``chunk_id``."""
    return str(uuid.uuid5(_POINT_NS, f"{source_id}:{chunk_id}"))


def _strip_payload_quotes(field_name: str) -> str:
    """Remove the surrounding double quotes from a payload field name, if any."""
    return field_name.strip('"')


def _enum_value(value: Any) -> Any:
    """Return the ``.value`` of an enum, or the value itself if not an enum."""
    return getattr(value, "value", value)


def _quantization_label(quant: Any) -> str:
    """Human-readable quantization summary for reports and the dashboard."""
    if quant is None:
        return "disabled"
    turbo = getattr(quant, "turbo", None)
    if turbo is not None:
        bits = _enum_value(getattr(turbo, "bits", None))
        return f"turbo ({bits})" if bits else "turbo"
    return "enabled"


def _same_enum(actual: Any, expected: Any) -> bool:
    """Compare two enum-or-scalar values case-insensitively by string value."""
    return str(_enum_value(actual)).lower() == str(_enum_value(expected)).lower()


def _same_optional_bool(actual: Any, expected: bool) -> bool:
    """Return True if ``actual`` (truthiness) matches the expected boolean."""
    return bool(actual) is bool(expected)


def _same_optional_int(actual: Any, expected: int) -> bool:
    """Return True if ``actual`` is reported and equals the expected integer."""
    return actual is not None and int(actual) == expected


def _same_optional_float(actual: Any, expected: float) -> bool:
    """Return True if ``actual`` is reported and matches the expected float."""
    return actual is not None and abs(float(actual) - expected) < 1e-9


def _quantization_config(
    enabled: bool, settings: Settings | None = None
) -> TurboQuantization | None:
    """Build the TurboQuant quantization config, or None when disabled."""
    if not enabled:
        return None
    resolved_settings = _runtime_settings(settings)
    return TurboQuantization(
        turbo=TurboQuantQuantizationConfig(
            bits=TURBO_BITS,
            always_ram=resolved_settings.quantization_always_ram,
        ),
    )


def _sparse_vector_params(
    sparse_idf: bool,
    settings: Settings | None = None,
    on_disk: bool | None = None,
) -> SparseVectorParams:
    """Build sparse vector params, enabling the IDF modifier when requested.

    The modifier is always explicit (IDF or NONE): Qdrant's collection PATCH
    treats a null modifier as "leave unchanged", so a null could never turn
    IDF off on an existing collection.
    """
    resolved_settings = _runtime_settings(settings)
    return SparseVectorParams(
        index=SparseIndexParams(
            on_disk=on_disk
            if on_disk is not None
            else resolved_settings.sparse_on_disk,
            full_scan_threshold=resolved_settings.sparse_full_scan_threshold,
            datatype=Datatype.FLOAT16 if resolved_settings.sparse_float16 else None,
        ),
        modifier=Modifier.IDF if sparse_idf else Modifier.NONE,
    )


def _get_named_config(configs: Any, name: str) -> Any:
    """Return the named-vector config for ``name`` from a mapping, else None."""
    if isinstance(configs, dict):
        return configs.get(name)
    return None


def _finding(
    category: str,
    field: str,
    expected: Any,
    actual: Any,
    severity: Literal["critical", "high", "medium", "low"],
    message: str,
) -> SchemaFinding:
    """Shorthand constructor for a :class:`SchemaFinding`."""
    return SchemaFinding(
        category=category,
        field=field,
        expected=expected,
        actual=actual,
        severity=severity,
        message=message,
    )


def _collection_body(
    collection_name: str, schema: CollectionSchema, recreate: bool
) -> dict[str, Any]:
    """Build the ``POST /collections`` body suggested by an auto-fix."""
    body: dict[str, Any] = {
        "collection_name": collection_name,
        "recreate": recreate,
        "dense_on_disk": schema.dense_on_disk,
        "quantization": schema.quantization,
        "sparse_idf": schema.sparse_idf,
        "sparse_on_disk": schema.sparse_on_disk,
        "on_disk_payload": schema.on_disk_payload,
    }
    if recreate:
        body["confirm_recreate"] = collection_name
    return body


def _toggles_known(schema: CollectionSchema) -> bool:
    """Return True only if all four schema toggles carry explicit values."""
    return (
        schema.dense_on_disk is not None
        and schema.quantization is not None
        and schema.sparse_idf is not None
        and schema.sparse_on_disk is not None
    )


def _suggested_fixes_from_findings(
    collection_name: str,
    schema: CollectionSchema,
    findings: list[SchemaFinding],
) -> list[SuggestedFix]:
    """Derive remediation options (create, index repair, recreate) from findings."""
    fixes: list[SuggestedFix] = []
    known = _toggles_known(schema)

    if any(f.category == "collection" and f.field == "exists" for f in findings):
        fixes.append(
            SuggestedFix(
                type="create_collection",
                destructive=False,
                auto_fixable=known,
                impact="availability",
                description=(
                    f"Collection '{collection_name}' does not exist. Create it."
                ),
                endpoint="/collections",
                method="POST",
                body=_collection_body(collection_name, schema, recreate=False),
            )
        )
        return fixes

    missing_indexes = [
        f.field for f in findings if f.category == "payload_index" and f.actual is None
    ]
    if missing_indexes:
        count = len(missing_indexes)
        fixes.append(
            SuggestedFix(
                type="create_payload_indexes",
                destructive=False,
                auto_fixable=known,
                impact="performance",
                description=(
                    f"Create {count} missing payload indexes: "
                    f"{', '.join(missing_indexes)}"
                ),
                missing_indexes=missing_indexes,
                endpoint="/collections",
                method="POST",
                body=_collection_body(collection_name, schema, recreate=False),
            )
        )

    optimizer_findings = [f for f in findings if f.category == "optimizer"]
    if optimizer_findings:
        drifted = ", ".join(
            f"{f.field} (expected {f.expected}, got {f.actual})"
            for f in optimizer_findings
        )
        fixes.append(
            SuggestedFix(
                type="update_optimizers",
                destructive=False,
                auto_fixable=False,
                impact="performance",
                description=(
                    f"Update optimizer settings online, no data loss: {drifted}"
                ),
            )
        )

    # Optimizer settings are mutable via update_collection; only structural
    # drift (vectors, quantization, shard layout, ...) forces a recreate.
    needs_recreate = any(
        not (
            (f.category == "payload_index" and f.actual is None)
            or f.category == "optimizer"
        )
        for f in findings
    )
    if needs_recreate:
        fixes.append(
            SuggestedFix(
                type="recreate_collection",
                destructive=True,
                auto_fixable=False,
                impact="dataloss",
                description="Recreate collection with correct Smart Embedder schema",
                warning="Deletes all existing points",
                endpoint="/collections",
                method="POST",
                body=_collection_body(collection_name, schema, recreate=True),
            )
        )

    return fixes


def _validate_dense_hnsw(hnsw: Any, settings: Settings) -> list[SchemaFinding]:
    """Validate dense-vector HNSW settings reported by Qdrant."""
    errors: list[SchemaFinding] = []
    if hnsw is None:
        return [
            _finding(
                "hnsw",
                "config",
                "present",
                None,
                "low",
                "missing dense HNSW config",
            )
        ]
    if not _same_optional_int(hnsw.m, settings.hnsw_m):
        errors.append(
            _finding(
                "hnsw",
                "m",
                settings.hnsw_m,
                hnsw.m,
                "low",
                f"HNSW m mismatch: expected {settings.hnsw_m}, got {hnsw.m}",
            )
        )
    if not _same_optional_int(hnsw.ef_construct, settings.hnsw_ef_construct):
        errors.append(
            _finding(
                "hnsw",
                "ef_construct",
                settings.hnsw_ef_construct,
                hnsw.ef_construct,
                "low",
                (
                    f"HNSW ef_construct mismatch: expected "
                    f"{settings.hnsw_ef_construct}, got {hnsw.ef_construct}"
                ),
            )
        )
    if not _same_optional_int(
        hnsw.full_scan_threshold, settings.hnsw_full_scan_threshold
    ):
        errors.append(
            _finding(
                "hnsw",
                "full_scan_threshold",
                settings.hnsw_full_scan_threshold,
                hnsw.full_scan_threshold,
                "low",
                (
                    f"HNSW full_scan_threshold mismatch: "
                    f"expected {settings.hnsw_full_scan_threshold}, "
                    f"got {hnsw.full_scan_threshold}"
                ),
            )
        )
    inline_storage = getattr(hnsw, "inline_storage", None)
    if inline_storage is not None and not _same_optional_bool(
        inline_storage, settings.hnsw_inline_storage
    ):
        errors.append(
            _finding(
                "hnsw",
                "inline_storage",
                settings.hnsw_inline_storage,
                inline_storage,
                "low",
                (
                    f"HNSW inline_storage mismatch: expected "
                    f"{settings.hnsw_inline_storage}, got {inline_storage}"
                ),
            )
        )
    return errors


def _validate_dense_schema(
    cfg: Any,
    schema: CollectionSchema,
    settings: Settings,
    vector_name: str | None = None,
    dimension: int | None = None,
) -> list[SchemaFinding]:
    """Validate dense vector config and its HNSW settings."""
    expected_name = vector_name or settings.dense_vector_name
    expected_dimension = dimension or settings.dense_dim
    dense_cfg = _get_named_config(cfg.params.vectors, expected_name)
    if dense_cfg is None:
        return [
            _finding(
                "dense",
                "vector",
                expected_name,
                None,
                "critical",
                f"missing dense vector '{expected_name}'",
            )
        ]

    errors: list[SchemaFinding] = []
    if int(dense_cfg.size) != expected_dimension:
        errors.append(
            _finding(
                "dense",
                "size",
                expected_dimension,
                dense_cfg.size,
                "critical",
                (
                    f"dense size mismatch: expected {expected_dimension}, "
                    f"got {dense_cfg.size}"
                ),
            )
        )
    if not _same_enum(dense_cfg.distance, Distance.COSINE):
        errors.append(
            _finding(
                "dense",
                "distance",
                Distance.COSINE,
                dense_cfg.distance,
                "critical",
                (
                    f"dense distance mismatch: expected "
                    f"{Distance.COSINE}, got {dense_cfg.distance}"
                ),
            )
        )
    if schema.dense_on_disk is not None and not _same_optional_bool(
        dense_cfg.on_disk, schema.dense_on_disk
    ):
        errors.append(
            _finding(
                "dense",
                "on_disk",
                schema.dense_on_disk,
                dense_cfg.on_disk,
                "medium",
                (
                    f"dense on_disk mismatch: expected "
                    f"{schema.dense_on_disk}, got {dense_cfg.on_disk}"
                ),
            )
        )
    if settings.dense_float16:
        datatype = getattr(dense_cfg, "datatype", None)
        if datatype is None or not _same_enum(datatype, Datatype.FLOAT16):
            errors.append(
                _finding(
                    "dense",
                    "datatype",
                    Datatype.FLOAT16,
                    datatype,
                    "high",
                    (
                        f"dense datatype mismatch: expected "
                        f"{Datatype.FLOAT16}, got {datatype}"
                    ),
                )
            )
    errors.extend(_validate_dense_hnsw(dense_cfg.hnsw_config, settings))
    return errors


def _validate_sparse_modifier(
    sparse_cfg: Any, schema: CollectionSchema
) -> list[SchemaFinding]:
    """Validate sparse IDF modifier when the request supplied an expectation."""
    if schema.sparse_idf is None:
        return []
    expected_modifier = Modifier.IDF if schema.sparse_idf else None
    actual_modifier = sparse_cfg.modifier
    if expected_modifier is None:
        if actual_modifier is None or _same_enum(actual_modifier, "none"):
            return []
        return [
            _finding(
                "sparse",
                "modifier",
                "none",
                actual_modifier,
                "high",
                f"sparse modifier mismatch: expected none, got {actual_modifier}",
            )
        ]
    if _same_enum(actual_modifier, expected_modifier):
        return []
    return [
        _finding(
            "sparse",
            "modifier",
            str(expected_modifier),
            actual_modifier,
            "high",
            (
                f"sparse modifier mismatch: expected "
                f"{expected_modifier}, got {actual_modifier}"
            ),
        )
    ]


def _validate_sparse_index(
    sparse_cfg: Any, schema: CollectionSchema, settings: Settings
) -> list[SchemaFinding]:
    """Validate sparse index settings reported by Qdrant."""
    if sparse_cfg.index is None:
        return [
            _finding(
                "sparse",
                "index",
                "present",
                None,
                "high",
                "missing sparse index config",
            )
        ]

    errors: list[SchemaFinding] = []
    if schema.sparse_on_disk is not None and not _same_optional_bool(
        sparse_cfg.index.on_disk, schema.sparse_on_disk
    ):
        errors.append(
            _finding(
                "sparse",
                "on_disk",
                schema.sparse_on_disk,
                sparse_cfg.index.on_disk,
                "medium",
                (
                    f"sparse on_disk mismatch: expected "
                    f"{schema.sparse_on_disk}, got {sparse_cfg.index.on_disk}"
                ),
            )
        )
    if settings.sparse_float16:
        sparse_datatype = getattr(sparse_cfg.index, "datatype", None)
        if sparse_datatype is not None and not _same_enum(
            sparse_datatype, Datatype.FLOAT16
        ):
            errors.append(
                _finding(
                    "sparse",
                    "datatype",
                    Datatype.FLOAT16,
                    sparse_datatype,
                    "medium",
                    (
                        f"sparse datatype mismatch: expected "
                        f"{Datatype.FLOAT16}, got {sparse_datatype}"
                    ),
                )
            )
    if not _same_optional_int(
        sparse_cfg.index.full_scan_threshold, settings.sparse_full_scan_threshold
    ):
        errors.append(
            _finding(
                "sparse",
                "full_scan_threshold",
                settings.sparse_full_scan_threshold,
                sparse_cfg.index.full_scan_threshold,
                "low",
                (
                    f"sparse full_scan_threshold mismatch: "
                    f"expected {settings.sparse_full_scan_threshold}, "
                    f"got {sparse_cfg.index.full_scan_threshold}"
                ),
            )
        )
    return errors


def _validate_sparse_schema(
    cfg: Any,
    schema: CollectionSchema,
    settings: Settings,
    vector_name: str | None = None,
) -> list[SchemaFinding]:
    """Validate sparse vector config, index settings, and modifier."""
    expected_name = vector_name or settings.sparse_vector_name
    sparse_cfg = _get_named_config(cfg.params.sparse_vectors, expected_name)
    if sparse_cfg is None:
        return [
            _finding(
                "sparse",
                "vector",
                expected_name,
                None,
                "critical",
                f"missing sparse vector '{expected_name}'",
            )
        ]
    return _validate_sparse_index(
        sparse_cfg, schema, settings
    ) + _validate_sparse_modifier(sparse_cfg, schema)


def _validate_quantization_schema(
    cfg: Any, schema: CollectionSchema, settings: Settings
) -> list[SchemaFinding]:
    """Validate quantization config when the request supplied an expectation."""
    if schema.quantization is None:
        return []
    quant = cfg.quantization_config
    if not schema.quantization:
        if quant is None:
            return []
        return [
            _finding(
                "quantization",
                "enabled",
                "disabled",
                quant,
                "high",
                f"quantization mismatch: expected disabled, got {quant}",
            )
        ]

    turbo = getattr(quant, "turbo", None) if quant is not None else None
    if turbo is None:
        return [
            _finding(
                "quantization",
                "turbo",
                "present",
                None,
                "high",
                "missing TurboQuant quantization config",
            )
        ]

    errors: list[SchemaFinding] = []
    if not _same_enum(turbo.bits, TURBO_BITS):
        errors.append(
            _finding(
                "quantization",
                "bits",
                TURBO_BITS,
                turbo.bits,
                "high",
                (
                    f"quantization bits mismatch: expected "
                    f"{TURBO_BITS}, got {turbo.bits}"
                ),
            )
        )
    if not _same_optional_bool(turbo.always_ram, settings.quantization_always_ram):
        errors.append(
            _finding(
                "quantization",
                "always_ram",
                settings.quantization_always_ram,
                turbo.always_ram,
                "high",
                (
                    f"quantization always_ram mismatch: expected "
                    f"{settings.quantization_always_ram}, got {turbo.always_ram}"
                ),
            )
        )
    return errors


def _validate_collection_params(
    cfg: Any, schema: CollectionSchema, settings: Settings
) -> list[SchemaFinding]:
    """Validate shard, replica, and payload storage collection params."""
    errors: list[SchemaFinding] = []
    if cfg.params.shard_number is None:
        if settings.shard_number > 1:
            errors.append(
                _finding(
                    "shard",
                    "shard_number",
                    settings.shard_number,
                    "unknown",
                    "low",
                    (
                        f"shard_number mismatch: expected {settings.shard_number}, "
                        "got unknown"
                    ),
                )
            )
    elif int(cfg.params.shard_number) != settings.shard_number:
        errors.append(
            _finding(
                "shard",
                "shard_number",
                settings.shard_number,
                cfg.params.shard_number,
                "low",
                (
                    f"shard_number mismatch: expected {settings.shard_number}, "
                    f"got {cfg.params.shard_number}"
                ),
            )
        )

    if cfg.params.replication_factor is None:
        if settings.replication_factor > 1:
            errors.append(
                _finding(
                    "replica",
                    "replication_factor",
                    settings.replication_factor,
                    "unknown",
                    "low",
                    (
                        f"replication_factor mismatch: expected "
                        f"{settings.replication_factor}, got unknown"
                    ),
                )
            )
    elif int(cfg.params.replication_factor) != settings.replication_factor:
        errors.append(
            _finding(
                "replica",
                "replication_factor",
                settings.replication_factor,
                cfg.params.replication_factor,
                "low",
                (
                    f"replication_factor mismatch: expected "
                    f"{settings.replication_factor}, got "
                    f"{cfg.params.replication_factor}"
                ),
            )
        )

    on_disk_payload = getattr(cfg.params, "on_disk_payload", None)
    if (
        schema.on_disk_payload is not None
        and on_disk_payload is not None
        and not _same_optional_bool(on_disk_payload, schema.on_disk_payload)
    ):
        errors.append(
            _finding(
                "payload",
                "on_disk_payload",
                schema.on_disk_payload,
                on_disk_payload,
                "low",
                (
                    f"on_disk_payload mismatch: expected {schema.on_disk_payload}, "
                    f"got {on_disk_payload}"
                ),
            )
        )
    return errors


def _validate_optimizer_schema(cfg: Any, settings: Settings) -> list[SchemaFinding]:
    """Validate collection optimizer settings reported by Qdrant."""
    optimizer = getattr(cfg, "optimizer_config", None)
    if optimizer is None:
        return [
            _finding(
                "optimizer",
                "config",
                "present",
                None,
                "medium",
                "missing optimizer config",
            )
        ]

    errors: list[SchemaFinding] = []
    indexing_threshold = getattr(optimizer, "indexing_threshold", None)
    if not _same_optional_int(indexing_threshold, settings.indexing_threshold):
        errors.append(
            _finding(
                "optimizer",
                "indexing_threshold",
                settings.indexing_threshold,
                indexing_threshold,
                "medium",
                (
                    f"indexing_threshold mismatch: expected "
                    f"{settings.indexing_threshold}, got {indexing_threshold}"
                ),
            )
        )
    memmap = getattr(optimizer, "memmap_threshold", None)
    if memmap is not None and int(memmap) != settings.memmap_threshold:
        errors.append(
            _finding(
                "optimizer",
                "memmap_threshold",
                settings.memmap_threshold,
                memmap,
                "medium",
                (
                    f"memmap_threshold mismatch: expected "
                    f"{settings.memmap_threshold}, got {memmap}"
                ),
            )
        )
    prevent = getattr(optimizer, "prevent_unoptimized", None)
    if prevent is not None and not _same_optional_bool(
        prevent, settings.prevent_unoptimized
    ):
        errors.append(
            _finding(
                "optimizer",
                "prevent_unoptimized",
                settings.prevent_unoptimized,
                prevent,
                "medium",
                (
                    f"prevent_unoptimized mismatch: expected "
                    f"{settings.prevent_unoptimized}, got {prevent}"
                ),
            )
        )
    return errors


def _validate_strict_mode_schema(cfg: Any) -> list[SchemaFinding]:
    """Validate strict mode when Qdrant reports a strict-mode config."""
    strict = getattr(cfg, "strict_mode_config", None)
    if strict is None:
        return []
    strict_enabled = getattr(strict, "enabled", None)
    if strict_enabled is None or bool(strict_enabled):
        return []
    return [
        _finding(
            "strict_mode",
            "enabled",
            True,
            strict_enabled,
            "low",
            "strict_mode mismatch: expected enabled, got disabled",
        )
    ]


def _validate_payload_indexes(info: Any) -> list[SchemaFinding]:
    """Validate configured payload indexes against the reported payload schema."""
    payload_schema = getattr(info, "payload_schema", None) or {}
    indexed = {
        _strip_payload_quotes(name): entry for name, entry in payload_schema.items()
    }
    errors: list[SchemaFinding] = []
    for field_name, expected_type in PAYLOAD_INDEXES:
        expected = _strip_payload_quotes(field_name)
        if expected not in indexed:
            errors.append(
                _finding(
                    "payload_index",
                    expected,
                    expected_type,
                    None,
                    "medium",
                    f"missing payload index '{expected}'",
                )
            )
            continue
        actual_type = getattr(indexed[expected], "data_type", None)
        if actual_type is not None and not _same_enum(actual_type, expected_type):
            errors.append(
                _finding(
                    "payload_index",
                    expected,
                    expected_type,
                    actual_type,
                    "medium",
                    (
                        f"payload index '{expected}' type mismatch: "
                        f"expected {expected_type}, got {actual_type}"
                    ),
                )
            )
    return errors


def _validate_schema(
    info: Any,
    schema: CollectionSchema,
    settings: Settings | None = None,
    *,
    dense_vector_name: str | None = None,
    sparse_vector_name: str | None = None,
    dense_dimension: int | None = None,
) -> list[SchemaFinding]:
    """Compare a live collection against the expected schema."""
    resolved_settings = _runtime_settings(settings)
    cfg = info.config
    return (
        _validate_dense_schema(
            cfg,
            schema,
            resolved_settings,
            dense_vector_name,
            dense_dimension,
        )
        + _validate_sparse_schema(
            cfg,
            schema,
            resolved_settings,
            sparse_vector_name,
        )
        + _validate_quantization_schema(cfg, schema, resolved_settings)
        + _validate_collection_params(cfg, schema, resolved_settings)
        + _validate_optimizer_schema(cfg, resolved_settings)
        + _validate_strict_mode_schema(cfg)
        + _validate_payload_indexes(info)
    )


def _missing_collection_report(
    collection_name: str, schema: CollectionSchema
) -> CollectionReport:
    """Build the report returned when the collection does not exist."""
    message = f"collection '{collection_name}' not found"
    findings = [
        _finding(
            "collection",
            "exists",
            True,
            False,
            "critical",
            message,
        )
    ]
    return CollectionReport(
        collection_name=collection_name,
        exists=False,
        schema_ok=False,
        errors=[message],
        findings=findings,
        suggested_fixes=_suggested_fixes_from_findings(
            collection_name, schema, findings
        ),
        quantization="unknown",
    )

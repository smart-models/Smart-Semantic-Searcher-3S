"""Qdrant collection I/O repository."""

import asyncio
import logging
from datetime import datetime, timezone

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import (
    CollectionParamsDiff,
    Datatype,
    Disabled,
    Distance,
    HnswConfigDiff,
    Modifier,
    OptimizersConfigDiff,
    PayloadSchemaType,
    StrictModeConfig,
    TextIndexParams,
    VectorParams,
    VectorParamsDiff,
)

from qdrant_middleware.clients.retry import RETRYABLE_HTTP_STATUS, retry_async
from qdrant_middleware.core.config import Settings, get_settings
from qdrant_middleware.schemas.collections import CollectionReport, CollectionSchema
from qdrant_middleware.schemas.embedding_contract import (
    ActiveVectorNames,
    EmbedderContractInfo,
    EmbeddingCompatibility,
    EmbeddingContract,
    ReindexJobState,
)
from qdrant_middleware.services.collection_schema import (
    PAYLOAD_INDEXES,
    _get_named_config,
    _missing_collection_report,
    _quantization_config,
    _quantization_label,
    _sparse_vector_params,
    _suggested_fixes_from_findings,
    _validate_schema,
)
from qdrant_middleware.services.payload_paths import (
    canonical_payload_index_field,
    qdrant_filter_index_path,
    qdrant_payload_path,
)

log = logging.getLogger(__name__)


def _runtime_settings(settings: Settings | None) -> Settings:
    """Return ``settings`` or fall back to the cached process defaults."""
    return settings or get_settings()


CacheKey = tuple[int, str]
EMBEDDING_METADATA_NAMESPACE = "hybrid_manager"
ACTIVE_REINDEX_STATUSES = {
    "running",
    "paused",
    "interrupted",
    "finalizing",
    "failed",
}


def _cache_key(client: AsyncQdrantClient, collection_name: str) -> CacheKey:
    """Build the per-client/per-collection cache key used by the repository."""
    return (id(client), collection_name)


_payload_index_locks: dict[CacheKey, asyncio.Lock] = {}


def _payload_index_lock(
    client: AsyncQdrantClient, collection_name: str
) -> asyncio.Lock:
    """Serialise payload index changes for one collection within this process.

    Qdrant accepts a duplicate ``create_payload_index`` without complaining, so
    two concurrent requests would both report the field as newly created. The
    lock makes the second one observe the first one's result and report it as
    skipped instead.
    """
    key = _cache_key(client, collection_name)
    lock = _payload_index_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _payload_index_locks[key] = lock
    return lock


def _default_active_vectors(settings: Settings) -> ActiveVectorNames:
    """Return the configured vector names used by metadata schema v1."""
    return ActiveVectorNames(
        dense=settings.dense_vector_name,
        sparse=settings.sparse_vector_name,
    )


def _collection_metadata(
    contract: EmbedderContractInfo,
    active_vectors: ActiveVectorNames,
    reindex_job: ReindexJobState | None = None,
) -> dict:
    """Build the middleware-owned metadata written with a new collection."""
    owned = {
        "metadata_schema": 2,
        "embedding_contract": contract.contract.model_dump(mode="json"),
        "active_vectors": active_vectors.model_dump(mode="json"),
        "smart_embedder_version": contract.server_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if reindex_job is not None:
        owned["reindex_job"] = reindex_job.model_dump(mode="json")
    return {EMBEDDING_METADATA_NAMESPACE: owned}


def _owned_embedding_metadata(config: object) -> tuple[dict | None, bool]:
    """Return middleware-owned metadata and whether its container is malformed."""
    metadata = getattr(config, "metadata", None) or {}
    if not isinstance(metadata, dict):
        return None, True
    owned = metadata.get(EMBEDDING_METADATA_NAMESPACE)
    if owned is None:
        return None, False
    if not isinstance(owned, dict):
        return None, True
    return owned, False


def _stored_embedding_contract(config: object) -> tuple[EmbeddingContract | None, bool]:
    """Return the stored contract and whether malformed metadata was present."""
    owned, malformed = _owned_embedding_metadata(config)
    if malformed:
        return None, True
    if owned is None:
        return None, False
    if "embedding_contract" not in owned:
        # A schema-v2 reindex job may start from a collection created before
        # embedding metadata was persisted.
        return None, owned.get("metadata_schema") != 2
    try:
        return EmbeddingContract.model_validate(owned["embedding_contract"]), False
    except (TypeError, ValueError):
        return None, True


def stored_active_vectors(
    config: object,
    settings: Settings | None = None,
) -> tuple[ActiveVectorNames, bool]:
    """Read schema-v2 active names, falling back for older metadata."""
    resolved_settings = _runtime_settings(settings)
    fallback = _default_active_vectors(resolved_settings)
    owned, malformed = _owned_embedding_metadata(config)
    if malformed:
        return fallback, True
    if owned is None or "active_vectors" not in owned:
        return fallback, False
    try:
        return ActiveVectorNames.model_validate(owned["active_vectors"]), False
    except (TypeError, ValueError):
        return fallback, True


def stored_reindex_job(config: object) -> tuple[ReindexJobState | None, bool]:
    """Read a persisted reindex job from collection metadata."""
    owned, malformed = _owned_embedding_metadata(config)
    if malformed:
        return None, True
    if owned is None or "reindex_job" not in owned:
        return None, False
    try:
        return ReindexJobState.model_validate(owned["reindex_job"]), False
    except (TypeError, ValueError):
        return None, True


def embedding_compatibility(
    config: object,
    active: EmbeddingContract | None,
    settings: Settings | None = None,
) -> EmbeddingCompatibility:
    """Compare collection metadata with the active Smart Embedder contract."""
    stored, malformed = _stored_embedding_contract(config)
    vectors, malformed_vectors = stored_active_vectors(config, settings)
    job, malformed_job = stored_reindex_job(config)
    if malformed or malformed_vectors or malformed_job:
        return EmbeddingCompatibility(
            status="incompatible",
            active=active,
            active_vectors=vectors,
            message="Collection embedding metadata is invalid; re-embed required.",
        )
    if stored is None:
        return EmbeddingCompatibility(
            status="incompatible",
            active=active,
            active_vectors=vectors,
            reindex_job=job,
            message="Collection has no verified embedding contract; re-embed required.",
        )
    if active is None:
        return EmbeddingCompatibility(
            status="incompatible",
            stored=stored,
            active_vectors=vectors,
            reindex_job=job,
            message=(
                "Active Smart Embedder contract is unavailable; compatibility "
                "cannot be verified."
            ),
        )
    if stored.fingerprint != active.fingerprint:
        return EmbeddingCompatibility(
            status="incompatible",
            stored=stored,
            active=active,
            active_vectors=vectors,
            reindex_job=job,
            message="Collection and active Smart Embedder use different vector spaces.",
        )
    return EmbeddingCompatibility(
        status="compatible",
        stored=stored,
        active=active,
        active_vectors=vectors,
        reindex_job=job,
        message="Collection matches the active Smart Embedder contract.",
    )


def qdrant_error_status(exc: Exception) -> int | None:
    """Unwrap a Qdrant client exception to the HTTP status Qdrant itself sent.

    ``ResponseHandlingException`` wraps a transport-layer failure and carries
    no status of its own, so it recurses into ``exc.source``.
    ``UnexpectedResponse`` (and similar) carry ``status_code`` directly —
    Qdrant's own verdict on the request, e.g. 400 for a filter referencing an
    unindexed field. Returns ``None`` when the exception isn't a Qdrant HTTP
    response at all (connection/timeout failures have no status to report).
    """
    if isinstance(exc, ResponseHandlingException):
        return qdrant_error_status(exc.source)
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


def is_transient_qdrant(exc: Exception) -> bool:
    """Return whether a Qdrant error is transient and worth retrying."""
    status = qdrant_error_status(exc)
    if status is not None:
        return status in RETRYABLE_HTTP_STATUS
    return isinstance(
        exc, (httpx.TransportError, httpx.TimeoutException, asyncio.TimeoutError)
    )


def _is_already_exists(exc: Exception) -> bool:
    """Return whether a Qdrant error reports an already-existing resource."""
    status = getattr(exc, "status_code", None)
    return status == 409 or "already exists" in str(exc).lower()


class PayloadIndexCollectionNotFound(Exception):
    """Raised when payload indexes are requested for a missing collection."""

    def __init__(self, message: str = "collection not found") -> None:
        """Initialize the error with a non-empty client-safe message."""
        # A default message keeps any generic ``except`` from producing an
        # empty detail string.
        super().__init__(message)


class PayloadIndexSchemaMismatch(Exception):
    """Raised when a collection has non-payload-index schema drift."""

    def __init__(self, errors: list[str]) -> None:
        """Store the schema errors that prevented index repair."""
        self.errors = errors
        super().__init__("Collection schema does not match expected config.")


class PayloadIndexTypeMismatch(Exception):
    """Raised when an existing payload index has a different schema type."""

    def __init__(
        self, field_name: str, expected: PayloadSchemaType, actual: object
    ) -> None:
        """Store normalized expected and actual payload index types."""
        self.field_name = field_name
        self.expected = str(getattr(expected, "value", expected))
        self.actual = str(getattr(actual, "value", actual))
        super().__init__(
            f"payload index '{field_name}' type mismatch: "
            f"expected {self.expected}, got {self.actual}"
        )


class PayloadIndexFieldNotFound(Exception):
    """Raised when a payload index to delete does not exist."""

    def __init__(self, field_name: str) -> None:
        """Store the missing payload field name."""
        self.field_name = field_name
        super().__init__(f"payload index '{field_name}' not found")


class EmbeddingContractViolation(Exception):
    """Collection metadata is unsafe for the embedding fingerprint in use."""

    def __init__(self, status: str, message: str) -> None:
        """Store the compatibility status and client-safe message."""
        self.status = status
        self.message = message
        super().__init__(message)


def _require_index_mutable(info: object) -> None:
    """Reject payload index changes while a regeneration generation is pending.

    Building or dropping an index over segments that a reindex job is bulk
    rewriting is pure extra load, and leaving these two routes unguarded made
    them the only mutations that ignored the lockdown every other one honours.
    """
    job, malformed_job = stored_reindex_job(getattr(info, "config", None))
    if malformed_job:
        raise EmbeddingContractViolation(
            "invalid_metadata",
            "Collection embedding metadata is invalid; payload index changes blocked.",
        )
    if job is not None and job.status in ACTIVE_REINDEX_STATUSES:
        raise EmbeddingContractViolation(
            "reindex_in_progress",
            "Collection vectors are being regenerated; "
            "payload index changes are temporarily blocked.",
        )


def _payload_schema_map(info: object) -> dict[str, object]:
    """Map a collection's reported payload schema by canonical field name."""
    payload_schema = getattr(info, "payload_schema", None) or {}
    return {
        canonical_payload_index_field(name): entry
        for name, entry in payload_schema.items()
    }


def _schema_value(value: object) -> str:
    """Return a lowercase string form of an enum-or-scalar schema value."""
    return str(getattr(value, "value", value)).lower()


def _same_payload_schema(actual: object, expected: object) -> bool:
    actual_type = getattr(actual, "data_type", actual)
    if isinstance(expected, TextIndexParams):
        # Params-aware comparison: a text index only matches a TextIndexParams
        # request when the phrase_matching flag agrees (absent counts as off).
        if _schema_value(actual_type) != "text":
            return False
        actual_params = getattr(actual, "params", None)
        actual_phrase = bool(getattr(actual_params, "phrase_matching", False))
        return actual_phrase == bool(expected.phrase_matching)
    return _schema_value(actual_type) == _schema_value(expected)


def _non_payload_schema_errors(
    info: object,
    schema: CollectionSchema,
    settings: Settings,
) -> list[str]:
    """Return schema-drift messages excluding payload-index findings."""
    active_vectors, _ = stored_active_vectors(info.config, settings)
    contract, _ = _stored_embedding_contract(info.config)
    return [
        finding.message
        for finding in _validate_schema(
            info,
            schema,
            settings,
            dense_vector_name=active_vectors.dense,
            sparse_vector_name=active_vectors.sparse,
            dense_dimension=contract.dense.dimension if contract else None,
        )
        if finding.category != "payload_index"
    ]


def _sparse_modifier_enabled(modifier: object) -> bool:
    """Return whether a reported sparse modifier value means IDF is on."""
    return str(getattr(modifier, "value", modifier)).lower() == str(Modifier.IDF.value)


async def _get_collection_info(
    client: AsyncQdrantClient,
    collection_name: str,
    settings: Settings,
) -> object:
    """Fetch collection info, mapping Qdrant not-found errors to domain errors."""
    try:
        return await retry_async(
            lambda: client.get_collection(collection_name),
            attempts=settings.retry_max_attempts,
            base_delay=settings.retry_base_delay,
            should_retry=is_transient_qdrant,
            description="qdrant.get_collection",
        )
    except UnexpectedResponse as exc:
        if exc.status_code == 404:
            raise PayloadIndexCollectionNotFound() from exc
        raise
    except ValueError as exc:
        if "not found" in str(exc).lower():
            raise PayloadIndexCollectionNotFound() from exc
        raise


async def require_embedding_fingerprint(
    client: AsyncQdrantClient,
    collection_name: str,
    fingerprint: str,
    settings: Settings | None = None,
    allow_during_reindex: bool = False,
) -> ActiveVectorNames:
    """Validate the fingerprint and return the active named-vector generation."""
    resolved_settings = _runtime_settings(settings)
    info = await _get_collection_info(client, collection_name, resolved_settings)
    stored, malformed = _stored_embedding_contract(info.config)
    vectors, malformed_vectors = stored_active_vectors(info.config, resolved_settings)
    job, malformed_job = stored_reindex_job(info.config)
    if malformed or malformed_vectors or malformed_job:
        raise EmbeddingContractViolation(
            "invalid_metadata",
            "Collection embedding metadata is invalid; vector operation blocked.",
        )
    if (
        not allow_during_reindex
        and job is not None
        and job.status in ACTIVE_REINDEX_STATUSES
    ):
        raise EmbeddingContractViolation(
            "reindex_in_progress",
            "Collection vectors are being regenerated; mutations are "
            "temporarily blocked",
        )
    if stored is None:
        raise EmbeddingContractViolation(
            "incompatible",
            "Collection has no verified embedding contract",
        )
    if stored.fingerprint != fingerprint:
        raise EmbeddingContractViolation(
            "incompatible",
            "Active embedder and collection use different vector spaces.",
        )
    return vectors


async def require_collection_mutable(
    client: AsyncQdrantClient,
    collection_name: str,
    settings: Settings | None = None,
    allow_invalid_metadata: bool = False,
) -> None:
    """Reject mutations while a persisted reindex generation is active.

    ``allow_invalid_metadata`` lets confirmed destructive operations (delete,
    recreate) through when the metadata namespace cannot be parsed: unreadable
    metadata makes a reset the remedy, not the risk, and blocking it leaves the
    collection unmanageable through this API.
    """
    resolved_settings = _runtime_settings(settings)
    info = await _get_collection_info(client, collection_name, resolved_settings)
    job, malformed_job = stored_reindex_job(info.config)
    if malformed_job and not allow_invalid_metadata:
        raise EmbeddingContractViolation(
            "invalid_metadata",
            "Collection embedding metadata is invalid; mutation blocked.",
        )
    if job is not None and job.status in ACTIVE_REINDEX_STATUSES:
        raise EmbeddingContractViolation(
            "reindex_in_progress",
            "Collection vectors are being regenerated; mutations are "
            "temporarily blocked.",
        )


async def get_embedding_metadata_state(
    client: AsyncQdrantClient,
    collection_name: str,
    settings: Settings | None = None,
) -> tuple[object, EmbeddingContract | None, ActiveVectorNames, ReindexJobState | None]:
    """Return collection info plus validated middleware embedding metadata."""
    resolved_settings = _runtime_settings(settings)
    info = await _get_collection_info(client, collection_name, resolved_settings)
    contract, malformed_contract = _stored_embedding_contract(info.config)
    vectors, malformed_vectors = stored_active_vectors(info.config, resolved_settings)
    job, malformed_job = stored_reindex_job(info.config)
    if malformed_contract or malformed_vectors or malformed_job:
        raise EmbeddingContractViolation(
            "invalid_metadata",
            "Collection embedding metadata is invalid.",
        )
    return info, contract, vectors, job


async def patch_embedding_metadata(
    client: AsyncQdrantClient,
    collection_name: str,
    *,
    settings: Settings | None = None,
    contract: EmbeddingContract | None = None,
    active_vectors: ActiveVectorNames | None = None,
    reindex_job: ReindexJobState | None = None,
    clear_reindex_job: bool = False,
    smart_embedder_version: str | None = None,
) -> None:
    """Merge validated embedding state into the middleware metadata namespace."""
    resolved_settings = _runtime_settings(settings)
    info = await _get_collection_info(client, collection_name, resolved_settings)
    owned, malformed = _owned_embedding_metadata(info.config)
    if malformed:
        raise EmbeddingContractViolation(
            "invalid_metadata",
            "Collection embedding metadata is invalid.",
        )
    merged = dict(owned or {})
    merged["metadata_schema"] = 2
    merged.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    if contract is not None:
        merged["embedding_contract"] = contract.model_dump(mode="json")
    if active_vectors is not None:
        merged["active_vectors"] = active_vectors.model_dump(mode="json")
    if smart_embedder_version is not None:
        merged["smart_embedder_version"] = smart_embedder_version
    if clear_reindex_job:
        merged.pop("reindex_job", None)
    elif reindex_job is not None:
        merged["reindex_job"] = reindex_job.model_dump(mode="json")
    merged["updated_at"] = datetime.now(timezone.utc).isoformat()

    updated = await retry_async(
        lambda: client.update_collection(
            collection_name=collection_name,
            metadata={EMBEDDING_METADATA_NAMESPACE: merged},
        ),
        attempts=resolved_settings.retry_max_attempts,
        base_delay=resolved_settings.retry_base_delay,
        should_retry=is_transient_qdrant,
        description="qdrant.patch_embedding_metadata",
    )
    if updated is False:
        raise RuntimeError("Qdrant collection metadata update returned false")


async def reset_embedding_metadata(
    client: AsyncQdrantClient,
    collection_name: str,
    settings: Settings | None = None,
) -> None:
    """Overwrite an unreadable metadata namespace with an empty schema-v2 one.

    Unlike :func:`patch_embedding_metadata` this never reads the existing
    namespace, so it is the one operation that still works once the metadata is
    corrupt. The contract and any job state are dropped: the collection is left
    in the same shape as one created before embedding metadata existed, which
    the compatibility report already knows how to describe.
    """
    resolved_settings = _runtime_settings(settings)
    now = datetime.now(timezone.utc).isoformat()
    updated = await retry_async(
        lambda: client.update_collection(
            collection_name=collection_name,
            metadata={
                EMBEDDING_METADATA_NAMESPACE: {
                    "metadata_schema": 2,
                    "created_at": now,
                    "updated_at": now,
                }
            },
        ),
        attempts=resolved_settings.retry_max_attempts,
        base_delay=resolved_settings.retry_base_delay,
        should_retry=is_transient_qdrant,
        description="qdrant.reset_embedding_metadata",
    )
    if updated is False:
        raise RuntimeError("Qdrant collection metadata reset returned false")


async def _delete_payload_index_call(
    client: AsyncQdrantClient,
    collection_name: str,
    field_name: str,
    settings: Settings,
) -> None:
    """Delete one payload index with the shared transient retry policy."""
    await retry_async(
        lambda: client.delete_payload_index(
            collection_name=collection_name,
            field_name=qdrant_filter_index_path(field_name),
        ),
        attempts=settings.retry_max_attempts,
        base_delay=settings.retry_base_delay,
        should_retry=is_transient_qdrant,
        description="qdrant.delete_payload_index",
    )


async def _create_missing_payload_index(
    client: AsyncQdrantClient,
    collection_name: str,
    field_name: str,
    field_schema: PayloadSchemaType,
    settings: Settings,
) -> tuple[bool, dict[str, object] | None]:
    """Create one missing payload index, handling create races idempotently."""
    canonical = canonical_payload_index_field(field_name)
    try:
        await retry_async(
            lambda field_name=field_name, field_schema=field_schema: (
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=qdrant_filter_index_path(field_name),
                    field_schema=field_schema,
                )
            ),
            attempts=settings.retry_max_attempts,
            base_delay=settings.retry_base_delay,
            should_retry=is_transient_qdrant,
            description="qdrant.create_payload_index",
        )
        return True, None
    except (UnexpectedResponse, ValueError) as exc:
        if not _is_already_exists(exc):
            raise
        refreshed = await _get_collection_info(client, collection_name, settings)
        refreshed_existing = _payload_schema_map(refreshed)
        refreshed_entry = refreshed_existing.get(canonical)
        if refreshed_entry is not None and _same_payload_schema(
            refreshed_entry, field_schema
        ):
            return False, refreshed_existing
        actual = (
            getattr(refreshed_entry, "data_type", refreshed_entry)
            if refreshed_entry is not None
            else "unknown"
        )
        raise PayloadIndexTypeMismatch(canonical, field_schema, actual) from exc


async def list_collections(client: AsyncQdrantClient, settings: Settings | None = None):
    """Return Qdrant collections using the shared transient retry policy."""
    resolved_settings = _runtime_settings(settings)
    return await retry_async(
        lambda: client.get_collections(),
        attempts=resolved_settings.retry_max_attempts,
        base_delay=resolved_settings.retry_base_delay,
        should_retry=is_transient_qdrant,
        description="qdrant.get_collections",
    )


async def ensure_payload_indexes(
    client: AsyncQdrantClient,
    collection_name: str,
    settings: Settings | None = None,
) -> None:
    """Create the expected payload indexes, idempotently.

    Skips the 5 Qdrant round-trips when indexes for ``collection_name`` were
    already ensured in this process and the cache has not been invalidated.
    """
    key = _cache_key(client, collection_name)
    if repository.indexes_ensured(key):
        return
    resolved_settings = _runtime_settings(settings)
    for field_name, field_schema in PAYLOAD_INDEXES:
        try:
            await retry_async(
                lambda field_name=field_name, field_schema=field_schema: (
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_schema,
                    )
                ),
                attempts=resolved_settings.retry_max_attempts,
                base_delay=resolved_settings.retry_base_delay,
                should_retry=is_transient_qdrant,
                description="qdrant.create_payload_index",
            )
        except (UnexpectedResponse, ValueError) as e:  # noqa: PERF203
            if _is_already_exists(e):
                log.debug("Payload index '%s' already exists, skipping.", field_name)
                continue
            raise
    repository.mark_indexes_ensured(key)


async def ensure_source_id_index(
    client: AsyncQdrantClient,
    collection_name: str,
    settings: Settings | None = None,
    field_name: str = "source_id",
    field_schema: PayloadSchemaType | None = None,
) -> None:
    """Ensure the source-id index supports the replacement filter.

    ``replace_source`` pruning scrolls with a ``source_id`` filter, which
    strict-mode collections reject unless the field is indexed. The field is
    middleware-owned (always written on ingest), so creating its index here is
    safe even though other payload indexes are user-managed. Callers may use
    any exact-match payload type supported by Qdrant; its schema must agree
    with the values they write, rather than silently falling back to keyword.
    """
    resolved_settings = _runtime_settings(settings)
    expected_schema = field_schema or PayloadSchemaType.KEYWORD
    index_path = qdrant_payload_path(field_name)
    info = await _get_collection_info(client, collection_name, resolved_settings)
    existing = _payload_schema_map(info).get(canonical_payload_index_field(index_path))
    if existing is not None:
        if not _same_payload_schema(existing, expected_schema):
            actual = getattr(existing, "data_type", existing)
            raise PayloadIndexTypeMismatch(field_name, expected_schema, actual)
        return
    try:
        await retry_async(
            lambda: client.create_payload_index(
                collection_name=collection_name,
                field_name=index_path,
                field_schema=expected_schema,
            ),
            attempts=resolved_settings.retry_max_attempts,
            base_delay=resolved_settings.retry_base_delay,
            should_retry=is_transient_qdrant,
            description="qdrant.create_payload_index",
        )
    except (UnexpectedResponse, ValueError) as e:
        if not _is_already_exists(e):
            raise
    invalidate_payload_indexes(collection_name)


class QdrantCollectionRepository:
    """In-memory cache of per-collection quantization and payload-index state."""

    def __init__(self) -> None:
        """Initialize the repository with empty caches."""
        self._collection_quantization: dict[CacheKey, bool] = {}
        self._ensured_indexes: set[CacheKey] = set()

    def cached_quantization(self, key: CacheKey) -> bool | None:
        """Return the cached quantization flag, or ``None`` if unknown."""
        return self._collection_quantization.get(key)

    def remember_quantization(self, key: CacheKey, enabled: bool) -> None:
        """Cache whether a client/collection pair has quantization enabled."""
        self._collection_quantization[key] = enabled

    def invalidate_collection_quantization(self, collection_name: str) -> None:
        """Drop cached quantization flags for ``collection_name`` across clients."""
        self._collection_quantization = {
            key: value
            for key, value in self._collection_quantization.items()
            if key[1] != collection_name
        }

    def indexes_ensured(self, key: CacheKey) -> bool:
        """Return whether indexes were ensured for a client and collection."""
        return key in self._ensured_indexes

    def mark_indexes_ensured(self, key: CacheKey) -> None:
        """Mark payload indexes as ensured for a client/collection."""
        self._ensured_indexes.add(key)

    def invalidate_payload_indexes(self, collection_name: str) -> None:
        """Drop payload-index cache entries for ``collection_name`` across clients."""
        self._ensured_indexes = {
            key for key in self._ensured_indexes if key[1] != collection_name
        }


repository = QdrantCollectionRepository()


def invalidate_collection_quantization(collection_name: str) -> None:
    """Invalidate the shared repository's cached quantization flag."""
    repository.invalidate_collection_quantization(collection_name)


def invalidate_payload_indexes(collection_name: str) -> None:
    """Invalidate the shared repository's payload-index cache."""
    repository.invalidate_payload_indexes(collection_name)


async def collection_has_quantization(
    client: AsyncQdrantClient,
    collection_name: str,
    settings: Settings | None = None,
) -> bool:
    """Return whether ``collection_name`` has quantization configured (cached)."""
    key = _cache_key(client, collection_name)
    cached = repository.cached_quantization(key)
    if cached is not None:
        return cached
    try:
        info = await client.get_collection(collection_name)
        has_quant = getattr(info.config, "quantization_config", None) is not None
    except Exception:
        return _runtime_settings(settings).quantization_enabled
    repository.remember_quantization(key, has_quant)
    return has_quant


def _payload_index_items(info: object) -> list[dict]:
    """Project reported payload indexes; text entries carry phrase_matching."""
    items: list[dict] = []
    for field_name, entry in _payload_schema_map(info).items():
        item: dict = {
            "field_name": field_name,
            "field_schema": _schema_value(getattr(entry, "data_type", entry)),
        }
        if item["field_schema"] == "text":
            params = getattr(entry, "params", None)
            item["phrase_matching"] = bool(getattr(params, "phrase_matching", False))
        items.append(item)
    return sorted(items, key=lambda item: item["field_name"].casefold())


async def list_payload_indexes(
    client: AsyncQdrantClient,
    collection_name: str,
    settings: Settings | None = None,
) -> list[dict]:
    """Return payload indexes for a collection without validating middleware schema."""
    resolved_settings = _runtime_settings(settings)
    info = await _get_collection_info(client, collection_name, resolved_settings)
    return _payload_index_items(info)


async def sample_payload_keys(
    client: AsyncQdrantClient,
    collection_name: str,
    sample_size: int,
    settings: Settings | None = None,
) -> tuple[list[str], int]:
    """Return the payload keys observed on up to ``sample_size`` points.

    Existence is checked first so a missing collection surfaces as
    :class:`PayloadIndexCollectionNotFound` instead of a scroll error.
    """
    resolved_settings = _runtime_settings(settings)
    await _get_collection_info(client, collection_name, resolved_settings)
    points, _offset = await retry_async(
        lambda: client.scroll(
            collection_name=collection_name,
            with_payload=True,
            with_vectors=False,
            limit=sample_size,
        ),
        attempts=resolved_settings.retry_max_attempts,
        base_delay=resolved_settings.retry_base_delay,
        should_retry=is_transient_qdrant,
        description="qdrant.scroll_payload_keys",
    )
    keys = {key for point in points for key in (point.payload or {})}
    return sorted(keys, key=str.casefold), len(points)


async def create_payload_indexes(
    client: AsyncQdrantClient,
    collection_name: str,
    indexes: list[tuple[str, PayloadSchemaType]],
    settings: Settings | None = None,
) -> dict[str, list[str]]:
    """Create user-requested payload indexes idempotently.

    An existing index with a different schema type is replaced (Qdrant has no
    in-place index type change, so it is dropped and recreated) and reported
    under ``updated``.
    """
    resolved_settings = _runtime_settings(settings)
    # Read the live schema inside the lock: a concurrent request for the same
    # field must see what the first one committed, not a stale snapshot.
    async with _payload_index_lock(client, collection_name):
        info = await _get_collection_info(client, collection_name, resolved_settings)
        _require_index_mutable(info)
        non_payload_errors = _non_payload_schema_errors(
            info, CollectionSchema(), resolved_settings
        )
        if non_payload_errors:
            raise PayloadIndexSchemaMismatch(non_payload_errors)

        existing = _payload_schema_map(info)
        created: list[str] = []
        updated: list[str] = []
        skipped: list[str] = []
        for field_name, field_schema in indexes:
            canonical = canonical_payload_index_field(field_name)
            existing_entry = existing.get(canonical)
            if existing_entry is not None:
                if _same_payload_schema(existing_entry, field_schema):
                    skipped.append(canonical)
                    continue
                await _delete_payload_index_call(
                    client, collection_name, field_name, resolved_settings
                )
                await _create_missing_payload_index(
                    client, collection_name, field_name, field_schema, resolved_settings
                )
                updated.append(canonical)
                existing[canonical] = field_schema
                continue

            was_created, refreshed_existing = await _create_missing_payload_index(
                client,
                collection_name,
                field_name,
                field_schema,
                resolved_settings,
            )
            if was_created:
                created.append(canonical)
                existing[canonical] = field_schema
            else:
                skipped.append(canonical)
                existing = refreshed_existing or existing

        if created or updated:
            invalidate_payload_indexes(collection_name)
        return {
            "created": created,
            "updated": updated,
            "skipped": skipped,
            "failed": [],
        }


async def delete_payload_index(
    client: AsyncQdrantClient,
    collection_name: str,
    field_name: str,
    settings: Settings | None = None,
) -> str:
    """Delete a user-managed payload index from a collection."""
    resolved_settings = _runtime_settings(settings)
    async with _payload_index_lock(client, collection_name):
        info = await _get_collection_info(client, collection_name, resolved_settings)
        _require_index_mutable(info)

        canonical = canonical_payload_index_field(field_name)
        existing = _payload_schema_map(info)
        if canonical not in existing:
            raise PayloadIndexFieldNotFound(canonical)

        await _delete_payload_index_call(
            client, collection_name, field_name, resolved_settings
        )
        invalidate_payload_indexes(collection_name)
        return canonical


async def update_collection_schema_toggles(
    client: AsyncQdrantClient,
    collection_name: str,
    schema: CollectionSchema,
    settings: Settings | None = None,
) -> None:
    """Update mutable collection schema toggles without recreating the collection."""
    resolved_settings = _runtime_settings(settings)
    info = await _get_collection_info(client, collection_name, resolved_settings)
    job, malformed_job = stored_reindex_job(info.config)
    if malformed_job:
        raise EmbeddingContractViolation(
            "invalid_metadata",
            "Collection embedding metadata is invalid; schema update blocked.",
        )
    if job is not None and job.status in ACTIVE_REINDEX_STATUSES:
        raise EmbeddingContractViolation(
            "reindex_in_progress",
            "Collection vectors are being regenerated; schema updates are "
            "temporarily blocked.",
        )
    active_vectors, malformed_vectors = stored_active_vectors(
        info.config, resolved_settings
    )
    if malformed_vectors:
        raise EmbeddingContractViolation(
            "invalid_metadata",
            "Collection active-vector metadata is invalid; schema update blocked.",
        )
    # Toggling a vector the collection does not have makes Qdrant answer "Not
    # existing vector name error", which used to surface as a 502. Check the
    # live config first and report it as the schema mismatch it is.
    if schema.dense_on_disk is not None and not _get_named_config(
        info.config.params.vectors, active_vectors.dense
    ):
        raise EmbeddingContractViolation(
            "schema_mismatch",
            f"Collection has no dense vector '{active_vectors.dense}'; "
            "schema update blocked.",
        )
    if (
        schema.sparse_idf is not None or schema.sparse_on_disk is not None
    ) and not _get_named_config(
        info.config.params.sparse_vectors, active_vectors.sparse
    ):
        raise EmbeddingContractViolation(
            "schema_mismatch",
            f"Collection has no sparse vector '{active_vectors.sparse}'; "
            "schema update blocked.",
        )
    update_kwargs = {}

    if schema.dense_on_disk is not None:
        update_kwargs["vectors_config"] = {
            active_vectors.dense: VectorParamsDiff(on_disk=schema.dense_on_disk)
        }

    if schema.quantization is not None:
        update_kwargs["quantization_config"] = (
            _quantization_config(True, resolved_settings)
            if schema.quantization
            else Disabled.DISABLED
        )

    if schema.on_disk_payload is not None:
        update_kwargs["collection_params"] = CollectionParamsDiff(
            on_disk_payload=schema.on_disk_payload
        )

    if schema.sparse_idf is not None or schema.sparse_on_disk is not None:
        sparse_cfg = _get_named_config(
            info.config.params.sparse_vectors,
            active_vectors.sparse,
        )
        sparse_index = getattr(sparse_cfg, "index", None) if sparse_cfg else None
        sparse_idf = (
            schema.sparse_idf
            if schema.sparse_idf is not None
            else _sparse_modifier_enabled(getattr(sparse_cfg, "modifier", None))
        )
        sparse_on_disk = (
            schema.sparse_on_disk
            if schema.sparse_on_disk is not None
            else getattr(sparse_index, "on_disk", None)
        )
        update_kwargs["sparse_vectors_config"] = {
            active_vectors.sparse: _sparse_vector_params(
                sparse_idf,
                resolved_settings,
                on_disk=sparse_on_disk,
            )
        }

    if not update_kwargs:
        raise ValueError("at least one schema toggle must be provided")

    updated = await retry_async(
        lambda: client.update_collection(
            collection_name=collection_name,
            **update_kwargs,
        ),
        attempts=resolved_settings.retry_max_attempts,
        base_delay=resolved_settings.retry_base_delay,
        should_retry=is_transient_qdrant,
        description="qdrant.update_collection",
    )
    if updated is False:
        raise RuntimeError("Qdrant collection update returned false")
    if schema.quantization is not None:
        invalidate_collection_quantization(collection_name)


def _required_strict_mode_config() -> StrictModeConfig:
    """Strict mode every collection must run with.

    enabled=True alone enforces nothing: Qdrant applies only the restrictions
    set explicitly. Unindexed filtering is rejected so the console's
    indexed-fields-only dropdown matches server truth.
    """
    return StrictModeConfig(
        enabled=True,
        unindexed_filtering_retrieve=False,
        unindexed_filtering_update=False,
    )


def _strict_mode_aligned(strict: object) -> bool:
    """Return whether a reported strict-mode config matches the required one."""
    return (
        bool(getattr(strict, "enabled", False))
        and getattr(strict, "unindexed_filtering_retrieve", None) is False
        and getattr(strict, "unindexed_filtering_update", None) is False
    )


async def align_strict_mode_configs(
    client: AsyncQdrantClient,
    settings: Settings | None = None,
) -> list[str]:
    """Bring existing collections up to the required strict mode config.

    Collections created before the unindexed-filtering restrictions were
    added carry a bare ``enabled=True`` config, which enforces nothing.
    Returns the names of the collections that were updated.
    """
    resolved_settings = _runtime_settings(settings)
    collections = await list_collections(client, resolved_settings)
    updated: list[str] = []
    for entry in collections.collections:
        name = entry.name
        info = await retry_async(
            lambda name=name: client.get_collection(name),
            attempts=resolved_settings.retry_max_attempts,
            base_delay=resolved_settings.retry_base_delay,
            should_retry=is_transient_qdrant,
            description="qdrant.get_collection",
        )
        strict = getattr(info.config, "strict_mode_config", None)
        if strict is not None and _strict_mode_aligned(strict):
            continue
        await retry_async(
            lambda name=name: client.update_collection(
                collection_name=name,
                strict_mode_config=_required_strict_mode_config(),
            ),
            attempts=resolved_settings.retry_max_attempts,
            base_delay=resolved_settings.retry_base_delay,
            should_retry=is_transient_qdrant,
            description="qdrant.update_collection",
        )
        updated.append(name)
    return updated


async def create_collection(
    client: AsyncQdrantClient,
    collection_name: str,
    recreate: bool = False,
    schema: CollectionSchema | None = None,
    settings: Settings | None = None,
    embedding_contract: EmbedderContractInfo | None = None,
) -> bool:
    """Create the collection, optionally recreating it, and ensure indexes."""
    resolved_settings = _runtime_settings(settings)
    schema = schema or CollectionSchema.from_settings(resolved_settings)
    existing = {
        c.name for c in (await list_collections(client, resolved_settings)).collections
    }
    if collection_name in existing:
        if recreate:
            # Recreate is the documented repair for a collection whose metadata
            # became unreadable, so it must not be blocked by that same state.
            await require_collection_mutable(
                client,
                collection_name,
                settings=resolved_settings,
                allow_invalid_metadata=True,
            )
        if not recreate:
            log.warning("Collection '%s' already exists.", collection_name)
            await ensure_payload_indexes(
                client, collection_name, settings=resolved_settings
            )
            return False
        log.info("Deleting existing collection '%s'...", collection_name)
        await client.delete_collection(collection_name)

    log.info("Creating collection '%s'...", collection_name)
    try:
        await client.create_collection(
            collection_name=collection_name,
            vectors_config={
                resolved_settings.dense_vector_name: VectorParams(
                    size=resolved_settings.dense_dim,
                    distance=Distance.COSINE,
                    on_disk=schema.dense_on_disk,
                    datatype=Datatype.FLOAT16
                    if resolved_settings.dense_float16
                    else None,
                    hnsw_config=HnswConfigDiff(
                        m=resolved_settings.hnsw_m,
                        ef_construct=resolved_settings.hnsw_ef_construct,
                        full_scan_threshold=resolved_settings.hnsw_full_scan_threshold,
                        inline_storage=resolved_settings.hnsw_inline_storage,
                    ),
                ),
            },
            sparse_vectors_config={
                resolved_settings.sparse_vector_name: _sparse_vector_params(
                    schema.sparse_idf,
                    resolved_settings,
                    on_disk=schema.sparse_on_disk,
                ),
            },
            quantization_config=_quantization_config(
                schema.quantization, resolved_settings
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=resolved_settings.indexing_threshold,
                memmap_threshold=resolved_settings.memmap_threshold,
                prevent_unoptimized=resolved_settings.prevent_unoptimized,
            ),
            shard_number=resolved_settings.shard_number,
            replication_factor=resolved_settings.replication_factor,
            on_disk_payload=schema.on_disk_payload
            if schema.on_disk_payload is not None
            else resolved_settings.on_disk_payload,
            strict_mode_config=_required_strict_mode_config(),
            metadata=_collection_metadata(
                embedding_contract,
                _default_active_vectors(resolved_settings),
            )
            if embedding_contract is not None
            else None,
        )
    except (UnexpectedResponse, ValueError) as e:
        if not recreate and _is_already_exists(e):
            log.warning(
                "Collection '%s' already exists after create race.", collection_name
            )
            await ensure_payload_indexes(
                client, collection_name, settings=resolved_settings
            )
            return False
        raise

    invalidate_payload_indexes(collection_name)
    await ensure_payload_indexes(client, collection_name, settings=resolved_settings)
    invalidate_collection_quantization(collection_name)
    log.info("Collection '%s' created.", collection_name)
    return True


async def collection_report(
    client: AsyncQdrantClient,
    collection_name: str,
    schema: CollectionSchema | None = None,
    validate: bool = True,
    settings: Settings | None = None,
    active_embedding_contract: EmbeddingContract | None = None,
) -> CollectionReport:
    """Fetch a collection and return a :class:`CollectionReport` for it."""
    resolved_settings = _runtime_settings(settings)
    schema = schema or CollectionSchema.from_settings(resolved_settings)
    try:
        info = await _get_collection_info(client, collection_name, resolved_settings)
    except PayloadIndexCollectionNotFound:
        log.debug("Collection '%s' not found for report.", collection_name)
        return _missing_collection_report(collection_name, schema)

    cfg = info.config
    active_vectors, _ = stored_active_vectors(cfg, resolved_settings)
    stored_contract, _ = _stored_embedding_contract(cfg)
    dense_cfg = _get_named_config(cfg.params.vectors, active_vectors.dense)
    sparse_cfg = _get_named_config(cfg.params.sparse_vectors, active_vectors.sparse)
    sparse_index = sparse_cfg.index if sparse_cfg else None
    findings = (
        _validate_schema(
            info,
            schema,
            resolved_settings,
            dense_vector_name=active_vectors.dense,
            sparse_vector_name=active_vectors.sparse,
            dense_dimension=(
                stored_contract.dense.dimension if stored_contract else None
            ),
        )
        if validate
        else []
    )
    errors = [finding.message for finding in findings]

    return CollectionReport(
        collection_name=collection_name,
        exists=True,
        # None, not True: with validate=false nothing was checked, and a bare
        # "true" reads as a clearance that was never given.
        schema_ok=(not errors) if validate else None,
        errors=errors,
        findings=findings,
        suggested_fixes=_suggested_fixes_from_findings(
            collection_name, schema, findings
        ),
        status=str(info.status),
        points_count=info.points_count,
        vectors_count=getattr(info, "vectors_count", None),
        dense={
            "name": active_vectors.dense,
            "size": dense_cfg.size,
            "distance": str(dense_cfg.distance),
            "on_disk": dense_cfg.on_disk,
        }
        if dense_cfg
        else None,
        sparse={
            "name": active_vectors.sparse,
            "on_disk": getattr(sparse_index, "on_disk", None),
            "full_scan_threshold": getattr(sparse_index, "full_scan_threshold", None),
            "modifier": str(getattr(sparse_cfg, "modifier", None) or "none"),
        }
        if sparse_cfg
        else None,
        quantization=_quantization_label(cfg.quantization_config),
        shards=cfg.params.shard_number,
        replicas=cfg.params.replication_factor,
        on_disk_payload=getattr(cfg.params, "on_disk_payload", None),
        embedding_compatibility=embedding_compatibility(
            cfg,
            active_embedding_contract,
            resolved_settings,
        ),
    )


async def verify_collection(
    client: AsyncQdrantClient,
    collection_name: str,
    schema: CollectionSchema | None = None,
    settings: Settings | None = None,
) -> bool:
    """Log and return whether the collection exists and matches the schema."""
    report = await collection_report(client, collection_name, schema, settings=settings)
    if not report.exists:
        log.error(report.errors[0])
        return False
    if not report.schema_ok:
        log.error("--- Schema mismatch ---")
        for error in report.errors:
            log.error("  - %s", error)
        return False
    log.info("Collection '%s' schema OK.", collection_name)
    return True

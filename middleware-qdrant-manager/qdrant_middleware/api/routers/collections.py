"""Collection management endpoints (list, get, verify, create, delete).

Create and recreate are gated behind confirmation tokens; ``POST /collections``
auto-repairs missing payload indexes on the "already exists" path, while
non-repairable schema drift yields a 409. ``/collections/verify`` reports drift
but never repairs.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Annotated, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from prometheus_client import Counter
from pydantic import BaseModel, field_validator, model_validator
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PayloadSchemaType, TextIndexParams

from qdrant_middleware.api.dependencies import (
    get_embedder_client,
    get_mutation_coordinator,
    get_vector_reindex_manager,
    get_qdrant_client,
    get_request_settings,
)
from qdrant_middleware.clients.embedder import EmbedderClient
from qdrant_middleware.api.utils import resolve_collection_name
from qdrant_middleware.core.config import Settings
from qdrant_middleware.core.mutation_coordinator import MutationCoordinator
from qdrant_middleware.repositories.qdrant_collections import (
    EmbeddingContractViolation,
    PayloadIndexCollectionNotFound,
    PayloadIndexFieldNotFound,
    PayloadIndexSchemaMismatch,
    PayloadIndexTypeMismatch,
    collection_report,
    create_collection,
    create_payload_indexes,
    delete_payload_index,
    invalidate_collection_quantization,
    invalidate_payload_indexes,
    get_embedding_metadata_state,
    list_payload_indexes,
    sample_payload_keys,
    patch_embedding_metadata,
    reset_embedding_metadata,
    update_collection_schema_toggles,
)
from qdrant_middleware.repositories.qdrant_collections import (
    list_collections as list_qdrant_collections,
)
from qdrant_middleware.schemas.collections import (
    CollectionReport,
    CollectionSchema,
    PayloadIndexCreateRequest,
    PayloadIndexCreateResponse,
    PayloadIndexRequest,
    PayloadIndexDeleteResponse,
    PayloadIndexListResponse,
    PayloadKeysResponse,
)
from qdrant_middleware.schemas.embedding_contract import EmbeddingContract
from qdrant_middleware.schemas.reindex import (
    ReindexActionResponse,
    ReindexPreflightResponse,
    ReindexStartRequest,
    ReindexStatusResponse,
)
from qdrant_middleware.services.vector_reindex import (
    VectorReindexManager,
    preflight_reindex,
)
from qdrant_middleware.services.mutation_guard import mutation_scope

log = logging.getLogger(__name__)

router = APIRouter(prefix="/collections", tags=["COLLECTIONS"])

collection_operations = Counter(
    "qdrant_collection_operations_total",
    "Qdrant collection operations",
    ["operation", "status"],
)


class _CollectionNameOptions(BaseModel):
    """Base for request bodies carrying an optional ``collection_name``."""

    collection_name: Optional[str] = None

    @field_validator("collection_name")
    @classmethod
    def validate_collection_name(cls, value: Optional[str]) -> Optional[str]:
        """Strip whitespace and reject an empty collection name (None is OK)."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("collection_name cannot be empty")
        return stripped


class CollectionOptions(_CollectionNameOptions):
    """Request body for creating (or recreating) a collection."""

    recreate: bool = False
    confirm_recreate: Optional[str] = None
    dense_on_disk: bool
    quantization: bool
    sparse_idf: bool
    sparse_on_disk: bool
    # Optional so older clients keep working; None falls back to settings.
    on_disk_payload: Optional[bool] = None


class VerifyOptions(_CollectionNameOptions):
    """Request body for verifying a collection against the expected schema."""

    dense_on_disk: Optional[bool] = None
    quantization: Optional[bool] = None
    sparse_idf: Optional[bool] = None
    sparse_on_disk: Optional[bool] = None
    on_disk_payload: Optional[bool] = None


class SchemaToggleUpdateOptions(BaseModel):
    """Request body for online updates to mutable collection schema toggles."""

    dense_on_disk: Optional[bool] = None
    quantization: Optional[bool] = None
    sparse_idf: Optional[bool] = None
    sparse_on_disk: Optional[bool] = None
    on_disk_payload: Optional[bool] = None

    @model_validator(mode="after")
    def validate_at_least_one_toggle(self) -> "SchemaToggleUpdateOptions":
        """Reject updates that do not request any mutable schema change."""
        if (
            self.dense_on_disk is None
            and self.quantization is None
            and self.sparse_idf is None
            and self.sparse_on_disk is None
            and self.on_disk_payload is None
        ):
            raise ValueError("at least one schema toggle must be provided")
        return self


class OperationResponse(BaseModel):
    """Response for a create/verify operation, including the resulting report."""

    ok: bool
    created: bool = False
    collection_name: str
    report: CollectionReport


class DeleteCollectionResponse(BaseModel):
    """Response confirming a collection deletion."""

    ok: bool
    deleted: bool
    collection_name: str


class CollectionListResponse(BaseModel):
    """Response listing the names of all collections."""

    ok: bool = True
    collections: list[str]
    count: int


def _schema_from_options(
    options: CollectionOptions | VerifyOptions | SchemaToggleUpdateOptions,
) -> CollectionSchema:
    """Build a schema from request toggles; ``None`` skips validation."""
    return CollectionSchema(
        dense_on_disk=options.dense_on_disk,
        quantization=options.quantization,
        sparse_idf=options.sparse_idf,
        sparse_on_disk=options.sparse_on_disk,
        on_disk_payload=options.on_disk_payload,
    )


def _cached_active_contract(request: Request) -> EmbeddingContract | None:
    """Read the latest validated contract cached by the health probe."""
    probe = getattr(request.app.state, "health_probe", None)
    raw = getattr(probe, "embedder_contract", None)
    if raw is None:
        return None
    try:
        return EmbeddingContract.model_validate(raw)
    except (TypeError, ValueError):
        return None


def _reindex_manager(request: Request) -> VectorReindexManager:
    """Return the manager paired with the app mutation coordinator."""
    return get_vector_reindex_manager(request)


# Findings that concern the five mutable schema toggles. The toggle-update
# endpoint must only 409 when a *requested* toggle failed to apply; unrelated
# drift (optimizer, HNSW, shards, ...) is reported but does not block.
_TOGGLE_FINDING_FIELDS = {
    ("dense", "on_disk"),
    ("sparse", "on_disk"),
    ("sparse", "modifier"),
    ("payload", "on_disk_payload"),
}


def _toggle_findings(report: CollectionReport) -> list[str]:
    """Return finding messages that concern the mutable schema toggles."""
    return [
        finding.message
        for finding in report.findings
        if finding.category == "quantization"
        or (finding.category, finding.field) in _TOGGLE_FINDING_FIELDS
    ]


_PAYLOAD_SCHEMA_TYPES = {
    "keyword": PayloadSchemaType.KEYWORD,
    "text": PayloadSchemaType.TEXT,
    "integer": PayloadSchemaType.INTEGER,
    "float": PayloadSchemaType.FLOAT,
    "bool": PayloadSchemaType.BOOL,
    "geo": PayloadSchemaType.GEO,
    "datetime": PayloadSchemaType.DATETIME,
    "uuid": PayloadSchemaType.UUID,
}


def _requested_index_schema(item: PayloadIndexRequest):
    """Map one request item to the qdrant-client field schema."""
    if item.field_schema == "text" and item.phrase_matching:
        return TextIndexParams(type="text", phrase_matching=True)
    return _PAYLOAD_SCHEMA_TYPES[item.field_schema]


def _validated_payload_index_request(
    payload_request: Annotated[PayloadIndexCreateRequest, Body(...)],
) -> PayloadIndexCreateRequest:
    """Pass-through dependency that validates the request body model."""
    return payload_request


# Qdrant's own limit; a longer name is rejected by the server as a path
# validation error, which used to reach the client as a 502.
_MAX_COLLECTION_NAME_LENGTH = 255


def _resolve_collection(value: Optional[str], settings: Settings) -> str:
    """Resolve a collection name and reject shapes Qdrant cannot accept.

    Without this, an over-long name or one containing a path separator reaches
    Qdrant and comes back as a gateway error, so a caller mistake is reported as
    a backend failure.
    """
    name = resolve_collection_name(value, settings)
    if len(name) > _MAX_COLLECTION_NAME_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=(
                "collection_name must be at most "
                f"{_MAX_COLLECTION_NAME_LENGTH} characters"
            ),
        )
    if "/" in name or "\\" in name:
        raise HTTPException(
            status_code=400,
            detail="collection_name cannot contain '/' or '\\'",
        )
    if any(char.isspace() or ord(char) < 32 for char in name):
        raise HTTPException(
            status_code=400,
            detail="collection_name cannot contain whitespace or control characters",
        )
    return name


def _qdrant_failure(exc: Exception, message: str) -> HTTPException:
    """Map a Qdrant error to 400 when it rejected the input, else 502.

    Qdrant's raw response body is never forwarded: it carries server internals
    and is unusable by a client.
    """
    status_code = getattr(exc, "status_code", None)
    if status_code not in {400, 422}:
        return HTTPException(status_code=502, detail=message)
    reason = "rejected by Qdrant"
    content = getattr(exc, "content", None)
    if content:
        try:
            parsed = json.loads(content)
            reason = str(parsed["status"]["error"]) or reason
        except (TypeError, ValueError, KeyError):
            pass
    return HTTPException(status_code=400, detail=f"{message}: {reason}")


@router.get("", response_model=CollectionListResponse)
async def list_collections(
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
):
    """Return the sorted names of all collections."""
    try:
        result = await list_qdrant_collections(client, settings)
    except HTTPException:
        raise
    except Exception as e:
        collection_operations.labels(operation="list", status="error").inc()
        raise HTTPException(
            status_code=502, detail="Qdrant collection list failed"
        ) from e
    names = sorted(c.name for c in result.collections)
    collection_operations.labels(operation="list", status="ok").inc()
    return CollectionListResponse(collections=names, count=len(names))


@router.get("/{collection_name}", response_model=CollectionReport)
async def get_collection(
    collection_name: str,
    request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    validate: bool = Query(
        True,
        description=(
            "Set to false to skip schema validation and return an info-only report"
        ),
    ),
):
    """Return the schema report for a single collection (404 if missing)."""
    collection_name = _resolve_collection(collection_name, settings)
    try:
        report = await collection_report(
            client,
            collection_name,
            CollectionSchema(),
            validate=validate,
            settings=settings,
            active_embedding_contract=_cached_active_contract(request),
        )
    except Exception as e:
        collection_operations.labels(operation="get", status="error").inc()
        raise _qdrant_failure(e, "Qdrant collection lookup failed") from e
    if not report.exists:
        collection_operations.labels(operation="get", status="not_found").inc()
        raise HTTPException(
            status_code=404,
            detail=report.errors[0]
            if report.errors
            else f"collection '{collection_name}' not found",
        )
    collection_operations.labels(operation="get", status="ok").inc()
    return report


@router.post("/verify", response_model=CollectionReport)
async def verify_collection_endpoint(
    request: VerifyOptions,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
):
    """Report whether a collection matches the expected schema (never repairs)."""
    collection_name = _resolve_collection(request.collection_name, settings)
    http_request.state.log_collection = collection_name
    schema = _schema_from_options(request)
    try:
        report = await collection_report(
            client,
            collection_name,
            schema,
            settings=settings,
            active_embedding_contract=_cached_active_contract(http_request),
        )
    except Exception as e:
        collection_operations.labels(operation="verify", status="error").inc()
        raise _qdrant_failure(e, "Qdrant verification failed") from e
    collection_operations.labels(
        operation="verify",
        status="ok" if report.schema_ok else "mismatch",
    ).inc()
    return report


@router.post("", response_model=OperationResponse)
async def create_collection_endpoint(
    request: CollectionOptions,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    embedder: Annotated[EmbedderClient, Depends(get_embedder_client)],
    coordinator: Annotated[MutationCoordinator, Depends(get_mutation_coordinator)],
    manager: Annotated[VectorReindexManager, Depends(get_vector_reindex_manager)],
):
    """Create a collection (or repair indexes if it exists), gated by confirmation.

    ``recreate=true`` requires ``confirm_recreate`` to equal the collection
    name. Existing collections with non-repairable schema drift yield a 409.
    """
    collection_name = _resolve_collection(request.collection_name, settings)
    http_request.state.log_collection = collection_name
    if request.recreate and request.confirm_recreate != collection_name:
        raise HTTPException(
            status_code=400,
            detail=(
                "recreate=true is destructive. Set confirm_recreate equal to "
                "collection_name to confirm deletion and recreation."
            ),
        )
    schema = _schema_from_options(request)
    try:
        active_contract = await embedder.get_embedding_contract()
        async with mutation_scope(
            client,
            collection_name,
            settings,
            coordinator,
            manager,
            allow_missing=True,
            allow_invalid_metadata=request.recreate,
        ):
            created = await create_collection(
                client,
                collection_name,
                request.recreate,
                schema,
                settings=settings,
                embedding_contract=active_contract,
            )
            report = await collection_report(
                client,
                collection_name,
                schema,
                settings=settings,
                active_embedding_contract=active_contract.contract,
            )
    except HTTPException:
        raise
    except EmbeddingContractViolation as exc:
        collection_operations.labels(
            operation="create", status="mutation_blocked"
        ).inc()
        raise HTTPException(status_code=409, detail=exc.message) from exc
    except Exception as e:
        collection_operations.labels(operation="create", status="error").inc()
        raise _qdrant_failure(e, "Qdrant collection operation failed") from e

    if not report.schema_ok:
        collection_operations.labels(operation="create", status="mismatch").inc()
        raise HTTPException(
            status_code=409,
            detail={
                "message": (
                    "Collection exists but schema does not "
                    "match expected Smart Embedder config."
                ),
                "report": report.model_dump(),
            },
        )
    compatibility = report.embedding_compatibility
    if compatibility is None or compatibility.status != "compatible":
        collection_operations.labels(
            operation="create", status="contract_mismatch"
        ).inc()
        raise HTTPException(
            status_code=409,
            detail={
                "message": (
                    "Collection does not have a compatible Smart Embedder "
                    "contract. Recreate it or adopt the contract explicitly."
                ),
                "report": report.model_dump(mode="json"),
            },
        )
    collection_operations.labels(
        operation="create", status="created" if created else "exists"
    ).inc()
    return OperationResponse(
        ok=True,
        created=created,
        collection_name=collection_name,
        report=report,
    )


@router.put("/{collection_name}/schema-toggles", response_model=OperationResponse)
@router.patch("/{collection_name}/schema-toggles", response_model=OperationResponse)
async def update_collection_schema_toggles_endpoint(
    collection_name: str,
    request: SchemaToggleUpdateOptions,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    coordinator: Annotated[MutationCoordinator, Depends(get_mutation_coordinator)],
    manager: Annotated[VectorReindexManager, Depends(get_vector_reindex_manager)],
):
    """Update mutable collection schema toggles without recreating the collection."""
    collection_name = _resolve_collection(collection_name, settings)
    schema = _schema_from_options(request)
    try:
        async with mutation_scope(
            client, collection_name, settings, coordinator, manager
        ):
            await update_collection_schema_toggles(
                client,
                collection_name,
                schema,
                settings=settings,
            )
            report = await collection_report(
                client,
                collection_name,
                schema,
                settings=settings,
                active_embedding_contract=_cached_active_contract(http_request),
            )
    except PayloadIndexCollectionNotFound as exc:
        collection_operations.labels(
            operation="schema_toggle", status="not_found"
        ).inc()
        raise HTTPException(
            status_code=404,
            detail=f"collection '{collection_name}' not found",
        ) from exc
    except EmbeddingContractViolation as exc:
        collection_operations.labels(
            operation="schema_toggle", status="mutation_blocked"
        ).inc()
        raise HTTPException(status_code=409, detail=exc.message) from exc
    except HTTPException:
        raise
    except Exception as exc:
        collection_operations.labels(operation="schema_toggle", status="error").inc()
        raise _qdrant_failure(
            exc, "Qdrant collection schema toggle update failed"
        ) from exc

    toggle_errors = _toggle_findings(report)
    if toggle_errors:
        collection_operations.labels(operation="schema_toggle", status="mismatch").inc()
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Collection update did not match requested schema toggles.",
                "errors": toggle_errors,
                "report": report.model_dump(),
            },
        )

    collection_operations.labels(operation="schema_toggle", status="ok").inc()
    return OperationResponse(
        ok=True,
        created=False,
        collection_name=collection_name,
        report=report,
    )


@router.post(
    "/{collection_name}/reindex/preflight",
    response_model=ReindexPreflightResponse,
)
async def reindex_preflight_endpoint(
    collection_name: str,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    embedder: Annotated[EmbedderClient, Depends(get_embedder_client)],
):
    """Validate the complete source collection without changing it."""
    collection_name = _resolve_collection(collection_name, settings)
    try:
        return await preflight_reindex(
            client,
            embedder,
            collection_name,
            settings,
        )
    except PayloadIndexCollectionNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail=f"collection '{collection_name}' not found",
        ) from exc
    except EmbeddingContractViolation as exc:
        raise HTTPException(status_code=409, detail=exc.message) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise _qdrant_failure(exc, "Collection reindex preflight failed") from exc


@router.post(
    "/{collection_name}/reindex",
    response_model=ReindexActionResponse,
)
async def start_reindex_endpoint(
    collection_name: str,
    request: ReindexStartRequest,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    embedder: Annotated[EmbedderClient, Depends(get_embedder_client)],
):
    """Start background regeneration after exact collection-name confirmation."""
    collection_name = _resolve_collection(collection_name, settings)
    if request.confirm_collection != collection_name:
        raise HTTPException(
            status_code=400,
            detail="confirm_collection must exactly match collection_name",
        )
    manager = _reindex_manager(http_request)
    try:
        job = await manager.start(
            client,
            embedder,
            collection_name,
            settings,
        )
    except PayloadIndexCollectionNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail=f"collection '{collection_name}' not found",
        ) from exc
    except (EmbeddingContractViolation, ValueError) as exc:
        detail = (
            exc.message if isinstance(exc, EmbeddingContractViolation) else str(exc)
        )
        raise HTTPException(status_code=409, detail=detail) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise _qdrant_failure(exc, "Collection reindex start failed") from exc
    return ReindexActionResponse(
        collection_name=collection_name,
        job=job,
        task_running=manager.is_running(collection_name),
    )


@router.get(
    "/{collection_name}/reindex",
    response_model=ReindexStatusResponse,
)
async def get_reindex_status_endpoint(
    collection_name: str,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
):
    """Return persisted progress and normalize orphaned running jobs."""
    collection_name = _resolve_collection(collection_name, settings)
    manager = _reindex_manager(http_request)
    try:
        _, _, _, job = await get_embedding_metadata_state(
            client, collection_name, settings
        )
        if (
            job is not None
            and job.status in {"running", "finalizing"}
            and not manager.is_running(collection_name)
        ):
            job = job.model_copy(
                update={
                    "status": "interrupted",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            await patch_embedding_metadata(
                client,
                collection_name,
                settings=settings,
                reindex_job=job,
            )
    except PayloadIndexCollectionNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail=f"collection '{collection_name}' not found",
        ) from exc
    except EmbeddingContractViolation as exc:
        if exc.status != "invalid_metadata":
            raise HTTPException(status_code=409, detail=exc.message) from exc
        # Reading the status must keep working when the metadata is corrupt:
        # answering 409 here left the console unable to say what is wrong, and
        # cancel is what clears it.
        return ReindexStatusResponse(
            collection_name=collection_name,
            job=None,
            task_running=manager.is_running(collection_name),
            metadata_invalid=True,
        )
    except Exception as exc:
        raise _qdrant_failure(exc, "Collection reindex status failed") from exc
    return ReindexStatusResponse(
        collection_name=collection_name,
        job=job,
        task_running=manager.is_running(collection_name),
    )


async def _reindex_control(
    *,
    collection_name: str,
    action: str,
    http_request: Request,
    client: AsyncQdrantClient,
    settings: Settings,
    embedder: EmbedderClient,
) -> ReindexActionResponse:
    """Apply pause/cancel/resume with consistent HTTP error mapping."""
    manager = _reindex_manager(http_request)
    try:
        if action == "resume":
            job = await manager.resume(client, embedder, collection_name, settings)
        else:
            job = await manager.set_control_status(
                client, collection_name, settings, action
            )
    except PayloadIndexCollectionNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail=f"collection '{collection_name}' not found",
        ) from exc
    except (EmbeddingContractViolation, ValueError) as exc:
        if (
            action == "cancelled"
            and isinstance(exc, EmbeddingContractViolation)
            and exc.status == "invalid_metadata"
        ):
            # Cancel is the escape hatch for a corrupt namespace: reset it so
            # preflight, recreate and the compatibility report work again.
            await reset_embedding_metadata(client, collection_name, settings=settings)
            invalidate_collection_quantization(collection_name)
            invalidate_payload_indexes(collection_name)
            log.warning(
                "Reset unreadable embedding metadata for collection '%s' on cancel.",
                collection_name,
            )
            return ReindexActionResponse(
                collection_name=collection_name,
                job=None,
                task_running=manager.is_running(collection_name),
                metadata_invalid=True,
            )
        detail = (
            exc.message if isinstance(exc, EmbeddingContractViolation) else str(exc)
        )
        raise HTTPException(status_code=409, detail=detail) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise _qdrant_failure(exc, f"Collection reindex {action} failed") from exc
    return ReindexActionResponse(
        collection_name=collection_name,
        job=job,
        task_running=manager.is_running(collection_name),
    )


@router.post(
    "/{collection_name}/reindex/pause",
    response_model=ReindexActionResponse,
)
async def pause_reindex_endpoint(
    collection_name: str,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    embedder: Annotated[EmbedderClient, Depends(get_embedder_client)],
):
    """Pause a running job at the next durable batch boundary."""
    collection_name = _resolve_collection(collection_name, settings)
    return await _reindex_control(
        collection_name=collection_name,
        action="paused",
        http_request=http_request,
        client=client,
        settings=settings,
        embedder=embedder,
    )


@router.post(
    "/{collection_name}/reindex/resume",
    response_model=ReindexActionResponse,
)
async def resume_reindex_endpoint(
    collection_name: str,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    embedder: Annotated[EmbedderClient, Depends(get_embedder_client)],
):
    """Resume from the last offset committed after ``update_vectors``."""
    collection_name = _resolve_collection(collection_name, settings)
    return await _reindex_control(
        collection_name=collection_name,
        action="resume",
        http_request=http_request,
        client=client,
        settings=settings,
        embedder=embedder,
    )


@router.post(
    "/{collection_name}/reindex/cancel",
    response_model=ReindexActionResponse,
)
async def cancel_reindex_endpoint(
    collection_name: str,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    embedder: Annotated[EmbedderClient, Depends(get_embedder_client)],
):
    """Cancel after the current batch while leaving the source generation active."""
    collection_name = _resolve_collection(collection_name, settings)
    return await _reindex_control(
        collection_name=collection_name,
        action="cancelled",
        http_request=http_request,
        client=client,
        settings=settings,
        embedder=embedder,
    )


@router.get(
    "/{collection_name}/payload-indexes",
    response_model=PayloadIndexListResponse,
    # phrase_matching exists only on text entries; hide the null on the rest.
    response_model_exclude_none=True,
)
async def list_payload_indexes_endpoint(
    collection_name: str,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
):
    """List payload indexes for an existing collection."""
    collection_name = _resolve_collection(collection_name, settings)
    try:
        indexes = await list_payload_indexes(
            client,
            collection_name,
            settings=settings,
        )
    except PayloadIndexCollectionNotFound as exc:
        collection_operations.labels(
            operation="payload_index_list", status="not_found"
        ).inc()
        raise HTTPException(
            status_code=404,
            detail=f"collection '{collection_name}' not found",
        ) from exc
    except Exception as exc:
        collection_operations.labels(
            operation="payload_index_list", status="error"
        ).inc()
        raise _qdrant_failure(exc, "Qdrant payload index lookup failed") from exc

    collection_operations.labels(operation="payload_index_list", status="ok").inc()
    return PayloadIndexListResponse(
        collection_name=collection_name,
        indexes=indexes,
        count=len(indexes),
    )


@router.get(
    "/{collection_name}/payload-keys",
    response_model=PayloadKeysResponse,
)
async def list_payload_keys_endpoint(
    collection_name: str,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    sample_size: int = Query(
        100,
        ge=1,
        le=1000,
        description="Number of points sampled to collect payload keys",
    ),
):
    """Return the payload keys observed on a sample of the collection's points."""
    collection_name = _resolve_collection(collection_name, settings)
    try:
        keys, sampled = await sample_payload_keys(
            client,
            collection_name,
            sample_size,
            settings=settings,
        )
    except PayloadIndexCollectionNotFound as exc:
        collection_operations.labels(operation="payload_keys", status="not_found").inc()
        raise HTTPException(
            status_code=404,
            detail=f"collection '{collection_name}' not found",
        ) from exc
    except Exception as exc:
        collection_operations.labels(operation="payload_keys", status="error").inc()
        raise _qdrant_failure(exc, "Qdrant payload key sampling failed") from exc

    collection_operations.labels(operation="payload_keys", status="ok").inc()
    return PayloadKeysResponse(
        collection_name=collection_name,
        keys=keys,
        count=len(keys),
        sampled_points=sampled,
    )


@router.post(
    "/{collection_name}/payload-indexes",
    response_model=PayloadIndexCreateResponse,
    responses={400: {"description": "Invalid request body"}},
)
async def create_payload_indexes_endpoint(
    collection_name: str,
    payload_request: Annotated[
        PayloadIndexCreateRequest, Depends(_validated_payload_index_request)
    ],
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    coordinator: Annotated[MutationCoordinator, Depends(get_mutation_coordinator)],
    manager: Annotated[VectorReindexManager, Depends(get_vector_reindex_manager)],
):
    """Create user-managed payload indexes for an existing collection."""
    collection_name = _resolve_collection(collection_name, settings)
    indexes = [
        (item.field_name, _requested_index_schema(item))
        for item in payload_request.indexes
    ]
    try:
        async with mutation_scope(
            client, collection_name, settings, coordinator, manager
        ):
            result = await create_payload_indexes(
                client,
                collection_name,
                indexes,
                settings=settings,
            )
    except PayloadIndexCollectionNotFound as exc:
        collection_operations.labels(
            operation="payload_index", status="not_found"
        ).inc()
        raise HTTPException(
            status_code=404,
            detail=f"collection '{collection_name}' not found",
        ) from exc
    except PayloadIndexSchemaMismatch as exc:
        collection_operations.labels(operation="payload_index", status="mismatch").inc()
        raise HTTPException(
            status_code=409,
            detail={
                "message": (
                    "Collection exists but schema does not match expected config."
                ),
                "errors": exc.errors,
            },
        ) from exc
    except EmbeddingContractViolation as exc:
        collection_operations.labels(
            operation="payload_index", status="mutation_blocked"
        ).inc()
        raise HTTPException(status_code=409, detail=exc.message) from exc
    except PayloadIndexTypeMismatch as exc:
        collection_operations.labels(
            operation="payload_index", status="type_mismatch"
        ).inc()
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Payload index type mismatch.",
                "collection_name": collection_name,
                "field_name": exc.field_name,
                "expected": exc.expected,
                "actual": exc.actual,
            },
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        collection_operations.labels(operation="payload_index", status="error").inc()
        raise _qdrant_failure(exc, "Qdrant payload index operation failed") from exc

    collection_operations.labels(operation="payload_index", status="ok").inc()
    return PayloadIndexCreateResponse(
        collection_name=collection_name,
        created=result["created"],
        updated=result["updated"],
        skipped=result["skipped"],
        failed=result["failed"],
    )


@router.delete(
    "/{collection_name}/payload-indexes/{field_name}",
    response_model=PayloadIndexDeleteResponse,
)
async def delete_payload_index_endpoint(
    collection_name: str,
    field_name: str,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    coordinator: Annotated[MutationCoordinator, Depends(get_mutation_coordinator)],
    manager: Annotated[VectorReindexManager, Depends(get_vector_reindex_manager)],
):
    """Delete a user-managed payload index from an existing collection."""
    collection_name = _resolve_collection(collection_name, settings)
    try:
        async with mutation_scope(
            client, collection_name, settings, coordinator, manager
        ):
            deleted_field = await delete_payload_index(
                client,
                collection_name,
                field_name,
                settings=settings,
            )
    except PayloadIndexCollectionNotFound as exc:
        collection_operations.labels(
            operation="payload_index_delete", status="not_found"
        ).inc()
        raise HTTPException(
            status_code=404,
            detail=f"collection '{collection_name}' not found",
        ) from exc
    except EmbeddingContractViolation as exc:
        collection_operations.labels(
            operation="payload_index_delete", status="mutation_blocked"
        ).inc()
        raise HTTPException(status_code=409, detail=exc.message) from exc
    except PayloadIndexFieldNotFound as exc:
        collection_operations.labels(
            operation="payload_index_delete", status="field_not_found"
        ).inc()
        raise HTTPException(
            status_code=404,
            detail=(
                f"payload index '{exc.field_name}' not found in collection "
                f"'{collection_name}'"
            ),
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        collection_operations.labels(
            operation="payload_index_delete", status="error"
        ).inc()
        raise _qdrant_failure(exc, "Qdrant payload index deletion failed") from exc

    collection_operations.labels(operation="payload_index_delete", status="ok").inc()
    return PayloadIndexDeleteResponse(
        collection_name=collection_name,
        field_name=deleted_field,
    )


@router.delete("/{collection_name}", response_model=DeleteCollectionResponse)
async def delete_collection_endpoint(
    collection_name: str,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    coordinator: Annotated[MutationCoordinator, Depends(get_mutation_coordinator)],
    manager: Annotated[VectorReindexManager, Depends(get_vector_reindex_manager)],
    confirm_delete: Optional[str] = Query(default=None, max_length=255),
):
    """Delete a collection, gated by a ``confirm_delete`` token matching its name."""
    collection_name = _resolve_collection(collection_name, settings)
    if confirm_delete != collection_name:
        raise HTTPException(
            status_code=400,
            detail=(
                "delete is destructive. Set confirm_delete equal to "
                "collection_name to confirm deletion."
            ),
        )

    try:
        async with mutation_scope(
            client,
            collection_name,
            settings,
            coordinator,
            manager,
            allow_invalid_metadata=True,
        ):
            report = await collection_report(
                client, collection_name, CollectionSchema(), settings=settings
            )
            if not report.exists:
                collection_operations.labels(
                    operation="delete", status="not_found"
                ).inc()
                raise HTTPException(
                    status_code=404,
                    detail=report.errors[0]
                    if report.errors
                    else f"collection '{collection_name}' not found",
                )

            await client.delete_collection(collection_name=collection_name)
            invalidate_collection_quantization(collection_name)
            invalidate_payload_indexes(collection_name)
    except HTTPException:
        raise
    except EmbeddingContractViolation as exc:
        collection_operations.labels(
            operation="delete", status="mutation_blocked"
        ).inc()
        raise HTTPException(status_code=409, detail=exc.message) from exc
    except Exception as e:
        collection_operations.labels(operation="delete", status="error").inc()
        raise _qdrant_failure(e, "Qdrant collection deletion failed") from e

    collection_operations.labels(operation="delete", status="deleted").inc()
    log.info("Collection '%s' deleted.", collection_name)
    return DeleteCollectionResponse(
        ok=True, deleted=True, collection_name=collection_name
    )

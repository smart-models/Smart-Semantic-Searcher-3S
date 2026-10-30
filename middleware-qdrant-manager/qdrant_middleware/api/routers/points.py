"""Point ingestion and CRUD endpoints.

Covers bulk file ingestion (``/points/ingest``, reading only from the safe
``data/`` root), single/batch upserts, retrieval, deletion, payload updates,
and whole-collection export (``/points/export``, writing only under the
separate read-write ``exports/`` root, at a filename this service derives).
Point ids are deterministic (UUID5 of ``source_id:chunk_id``), so
re-ingesting a source overwrites rather than duplicates. Ingest and upsert
paths finish embedding before writing, so an embedding failure leaves stored
data untouched.
"""

import asyncio
import json
import logging
import math
import os
import re
import tempfile
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    Range,
    DatetimeRange,
)

from qdrant_middleware.api.dependencies import (
    get_embedder_client,
    get_mutation_coordinator,
    get_qdrant_client,
    get_request_settings,
    get_vector_reindex_manager,
    reserve_export_slot,
)
from qdrant_middleware.api.utils import resolve_collection_name
from qdrant_middleware.clients.embedder import EmbedderClient, EmbeddingResult
from qdrant_middleware.clients.retry import RETRYABLE_HTTP_STATUS, retry_async
from qdrant_middleware.core.config import Settings
from qdrant_middleware.core.mutation_coordinator import MutationCoordinator
from qdrant_middleware.repositories.qdrant_collections import (
    PayloadIndexTypeMismatch,
    ensure_payload_indexes,
    ensure_source_id_index,
    is_transient_qdrant,
    qdrant_error_status,
    sample_payload_keys,
)
from qdrant_middleware.services.collection_export import (
    ExportLimitExceeded,
    build_export_document,
)
from qdrant_middleware.services.export_storage import (
    attachment_disposition,
    export_target,
)
from qdrant_middleware.services.collection_schema import make_point_id
from qdrant_middleware.services.embedding_guard import guard_vector_operation
from qdrant_middleware.services.mutation_guard import mutation_scope
from qdrant_middleware.services.vector_reindex import VectorReindexManager
from qdrant_middleware.schemas.embedding_contract import ActiveVectorNames
from qdrant_middleware.services.filter_builder import build_qdrant_filter
from qdrant_middleware.services.payload_paths import (
    payload_path_root,
    qdrant_payload_path,
    qdrant_payload_selector,
)

_STALE_DELETE_CHUNK = 1000
# Points sampled to learn which payload keys a filtered clear must delete. A key
# absent from the sample survives the clear, so this stays at the ceiling the
# payload-keys endpoint allows.
_CLEAR_PAYLOAD_KEY_SAMPLE = 1000


async def _qdrant(operation, description: str, settings: Settings | None = None):
    """Run an idempotent Qdrant op with transient-failure retries.

    Any failure remaining after the retry budget is exhausted is mapped to an
    HTTP response. When the failure carries a genuine non-retryable 4xx
    status from Qdrant itself (e.g. a filter referencing an unindexed field
    comes back 400 Bad Request), that status is passed through — it's a
    request problem, not something wrong with the middleware or Qdrant.
    Everything else (connection failures, exhausted 5xx/429 retries, unknown
    exceptions) maps to HTTP 502, unchanged from before.
    """
    retry_kwargs = {}
    if settings is not None:
        retry_kwargs = {
            "attempts": settings.retry_max_attempts,
            "base_delay": settings.retry_base_delay,
        }
    try:
        return await retry_async(
            operation,
            should_retry=is_transient_qdrant,
            description=description,
            **retry_kwargs,
        )
    except Exception as e:
        msg = str(e) or repr(e)
        status = qdrant_error_status(e)
        is_client_error = (
            status is not None
            and 400 <= status < 500
            and status not in RETRYABLE_HTTP_STATUS
        )
        if is_client_error:
            raise HTTPException(status, f"Qdrant error: {msg}") from e
        raise HTTPException(502, f"Qdrant error: {msg}") from e


async def _retrieve_point_or_404(
    client: AsyncQdrantClient,
    collection_name: str,
    point_id: str,
    settings: Settings,
    *,
    with_payload: bool,
    with_vectors: bool = False,
) -> tuple[Union[int, str], list]:
    """Retrieve one point by id, raising HTTP 404 when it does not exist.

    ``point_id`` is the raw path/body value (used verbatim in the 404 detail);
    it is normalized to the Qdrant id form (HTTP 400 if invalid) before the
    lookup. Returns the normalized id and the retrieved records.
    """
    pid = _normalize_point_id(point_id)
    results = await _qdrant(
        lambda: client.retrieve(
            collection_name=collection_name,
            ids=[pid],
            with_vectors=with_vectors,
            with_payload=with_payload,
        ),
        "retrieve point",
        settings=settings,
    )
    if not results:
        raise HTTPException(
            404, f"Point {point_id} not found in collection {collection_name}"
        )
    return pid, results


def _summarize_vectors(vectors: object) -> Optional[dict[str, dict[str, Any]]]:
    """Describe a point's stored vectors by name/shape, never the raw numbers.

    A dense entry is a list of floats; a sparse entry is a
    :class:`~qdrant_client.models.SparseVector` with ``indices``/``values``.
    Anything else (unnamed single vector, multivector) is reported by type only.
    """
    if not isinstance(vectors, dict):
        return None
    summary: dict[str, dict[str, Any]] = {}
    for name, value in vectors.items():
        indices = getattr(value, "indices", None)
        if indices is not None:
            summary[name] = {"type": "sparse", "nonzero": len(indices)}
        elif isinstance(value, list):
            summary[name] = {"type": "dense", "dimension": len(value)}
        else:
            summary[name] = {"type": type(value).__name__}
    return summary


async def _existing_point_ids(
    client: AsyncQdrantClient,
    collection_name: str,
    ids: list[Union[int, str]],
    settings: Settings,
) -> set[Union[int, str]]:
    """Return which of ``ids`` already exist, for insert_only/update_only filtering."""
    results = await _qdrant(
        lambda: client.retrieve(
            collection_name=collection_name,
            ids=ids,
            with_payload=False,
            with_vectors=False,
        ),
        "retrieve existing point ids",
        settings=settings,
    )
    return {r.id for r in results}


async def _scroll_source_point_ids(
    client,
    collection_name: str,
    source_id: Any,
    source_id_key: str = "source_id",
    source_id_type: str = "keyword",
) -> set[str]:
    """Return all point ids currently stored for a source_id (ids only)."""
    ids: set[str] = set()
    offset = None
    flt = Filter(must=[_source_id_condition(source_id_key, source_id, source_id_type)])
    while True:
        points, offset = await client.scroll(
            collection_name=collection_name,
            scroll_filter=flt,
            with_payload=False,
            with_vectors=False,
            limit=1000,
            offset=offset,
        )
        ids.update(str(p.id) for p in points)
        if offset is None:
            break
    return ids


log = logging.getLogger(__name__)

router = APIRouter(prefix="/points", tags=["POINTS"])

# Chunk keys the ingest path treats as structural rather than user payload:
# ``id`` is the chunker's per-source ordinal that the point id derives from
# (UUID5(source_id:id)) and the other three are chunker bookkeeping. Kept as a
# named set because ``_validate_source_id_key`` must keep rejecting them
# regardless of whether they are stripped or not.
CHUNK_RESERVED_FIELDS = frozenset(
    {"id", "overflow_details", "overlap_sentences_count", "split_metadata"}
)
# Chunk keys dropped from the payload before writing. ``id`` is not among them:
# it is not stripped but *relocated*, stored under ``chunk_id_key`` (default
# ``"id"``) by ``_build_ingest_batches`` — the ordinal is ordinary payload whose
# name the caller owns, exactly like ``source_id_key``. The three chunker
# bookkeeping keys are not stripped either, on request; put
# ``CHUNK_RESERVED_FIELDS - {"id"}`` here to start dropping them again.
CHUNK_EXCLUDE_FIELDS: frozenset[str] = frozenset()

# The chunk key that is never copied verbatim into the payload: it is written
# back under ``chunk_id_key`` instead, so a renamed ordinal leaves no duplicate.
CHUNK_ORDINAL_FIELD = "id"
# Payload keys the service owns; overwrite_payload (PUT) carries these forward so a
# full replace cannot silently strip text/source_id and break retrieval.
PROTECTED_PAYLOAD_FIELDS = ("text", "source_id")

SourceIdType = Literal[
    "keyword", "text", "integer", "float", "bool", "datetime", "uuid", "geo", "untyped"
]

_SOURCE_ID_INDEX_SCHEMAS = {
    "keyword": PayloadSchemaType.KEYWORD,
    "text": PayloadSchemaType.KEYWORD,
    "untyped": PayloadSchemaType.KEYWORD,
    "integer": PayloadSchemaType.INTEGER,
    "float": PayloadSchemaType.FLOAT,
    "bool": PayloadSchemaType.BOOL,
    "datetime": PayloadSchemaType.DATETIME,
    "uuid": PayloadSchemaType.UUID,
}

_SOURCE_ID_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


def _coerce_source_id_value(sid: str, source_id_type: str) -> Any:
    """Coerce a source ID string to the chosen Qdrant payload type.

    Mirror the frontend ``coercePayloadValue`` rules and raise ``ValueError``
    when the value is not convertible.
    """
    if source_id_type == "untyped":
        return sid
    if source_id_type in ("keyword", "text"):
        return str(sid)
    if source_id_type == "integer":
        s = sid.strip()
        if re.fullmatch(r"[+-]?\d+", s, flags=re.ASCII):
            return int(s)
        raise ValueError(f"'{sid}' is not convertible to integer")
    if source_id_type == "float":
        s = sid.strip()
        if s:
            try:
                value = float(s)
            except ValueError:
                pass
            else:
                if not math.isfinite(value):
                    raise ValueError(f"'{sid}' is not a float")
                return value
        raise ValueError(f"'{sid}' is not convertible to float")
    if source_id_type == "bool":
        v = sid.strip().lower()
        if v == "true":
            return True
        if v == "false":
            return False
        raise ValueError(f"'{sid}' is not convertible to bool")
    if source_id_type == "datetime":
        try:
            datetime.fromisoformat(sid)
        except ValueError as exc:
            raise ValueError(f"'{sid}' is not convertible to datetime") from exc
        return sid
    if source_id_type == "uuid":
        if _SOURCE_ID_UUID_RE.match(sid):
            return sid
        raise ValueError(f"'{sid}' is not convertible to uuid")
    # geo: a scalar source_id string is never a geo object.
    raise ValueError(f"'{sid}' is not convertible to {source_id_type}")


def _source_id_index_schema(source_id_type: str) -> PayloadSchemaType:
    """Return the Qdrant index schema that supports a source predicate."""
    try:
        return _SOURCE_ID_INDEX_SCHEMAS[source_id_type]
    except KeyError as exc:
        raise ValueError(f"unsupported source_id_type: {source_id_type}") from exc


def _source_id_condition(
    field_name: str, value: Any, source_id_type: str
) -> FieldCondition:
    """Build the typed Qdrant predicate used to find a source's old points."""
    key = qdrant_payload_path(field_name)
    if source_id_type == "float":
        return FieldCondition(key=key, range=Range(gte=value, lte=value))
    if source_id_type == "datetime":
        instant = datetime.fromisoformat(str(value))
        if instant.tzinfo is None:
            instant = instant.replace(tzinfo=timezone.utc)
        normalized = instant.astimezone(timezone.utc).isoformat()
        return FieldCondition(
            key=key, range=DatetimeRange(gte=normalized, lte=normalized)
        )
    return FieldCondition(key=key, match=MatchValue(value=value))


class OrderBySpec(BaseModel):
    """Ordered-scroll spec: order points by an indexed payload field.

    Qdrant refuses ``offset`` together with ``order_by``; paging an ordered
    scroll goes through ``start_from`` instead (resume at the first value not
    yet seen), which is why both are exposed here.
    """

    key: str = Field(min_length=1)
    direction: Literal["asc", "desc"] = "asc"
    start_from: Optional[Union[int, float, str]] = None

    @field_validator("key")
    @classmethod
    def strip_key(cls, value: str) -> str:
        """Reject a blank order key (it cannot address a payload field)."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("order_by.key cannot be blank")
        return stripped


class _OptionalCollectionRequest(BaseModel):
    """Base for request bodies with a validated optional collection name."""

    collection_name: Optional[str] = None

    @field_validator("collection_name")
    @classmethod
    def strip_collection_name(cls, value: Optional[str]) -> Optional[str]:
        """Strip whitespace and reject blank collection values (None is OK)."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("collection_name cannot be blank")
        return stripped


class ScrollPointsRequest(_OptionalCollectionRequest):
    """Retrieve points by payload/id filters without vector search."""

    filters: Optional[dict[str, Any]] = None
    limit: int = Field(default=10, ge=1, le=1000)
    offset: Optional[Union[int, str]] = None
    order_by: Optional[OrderBySpec] = None
    with_payload: Union[bool, list[str]] = True
    include_total: bool = True


class CountPointsRequest(_OptionalCollectionRequest):
    """Count the points matching a payload/id filter, without fetching them."""

    filters: Optional[dict[str, Any]] = None
    exact: bool = True


class CountPointsResponse(BaseModel):
    """Response for ``POST /points/count``."""

    ok: bool
    collection_name: str
    count: int
    exact: bool


class FacetPointsRequest(_OptionalCollectionRequest):
    """Request body for ``POST /points/facet``: distinct values of one field."""

    key: str = Field(min_length=1)
    filters: Optional[dict[str, Any]] = None
    limit: int = Field(default=20, ge=1, le=1000)
    exact: bool = True

    @field_validator("key")
    @classmethod
    def strip_key(cls, value: str) -> str:
        """Reject a blank facet key (it cannot address a payload field)."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("key cannot be blank")
        return stripped


class FacetHit(BaseModel):
    """One distinct value and how many points carry it."""

    value: Union[str, int, bool]
    count: int


class FacetPointsResponse(BaseModel):
    """Response for ``POST /points/facet``."""

    ok: bool
    collection_name: str
    key: str
    exact: bool
    hits: list[FacetHit]


class ScrolledPoint(BaseModel):
    """A single point (id + payload) returned by ``POST /points/scroll``."""

    id: str
    payload: dict[str, Any]
    # Populated only by /points/retrieve when with_vectors is requested; scroll
    # never requests vectors, so this stays None there.
    vectors: Optional[dict[str, dict[str, Any]]] = None


class ScrollPointsResponse(BaseModel):
    """Response page for ``POST /points/scroll``."""

    collection_name: str
    returned: int
    total: Optional[int] = None
    next_page_offset: Optional[Union[int, str]] = None
    results: list[ScrolledPoint]


class ExportCollectionRequest(_OptionalCollectionRequest):
    """Request body for ``POST /points/export``: a whole-collection dump.

    There is deliberately no ``filters`` and no ``with_vectors``: the export is
    the entire collection's payload, and scroll never returns vectors.
    """

    destination: Literal["file", "body"] = "file"


class ExportCollectionResponse(BaseModel):
    """Result of an export written to disk (``destination="file"``)."""

    ok: bool
    collection_name: str
    n_chunks: int
    path: Optional[str] = None
    bytes: int


def _coerce_point_id(raw: str) -> Union[int, str]:
    """Coerce a raw point ID into the form Qdrant expects.

    Qdrant point IDs must be an unsigned integer or a UUID. Returns the value as
    an ``int`` or canonical ``str``, raising ``ValueError`` otherwise so pydantic
    surfaces it as a request-validation error — which ``create_app`` normalises
    to a 400, not FastAPI's default 422.
    """
    s = str(raw).strip()
    if not s:
        raise ValueError("point_id cannot be blank")
    if re.fullmatch(r"\d+", s, flags=re.ASCII):
        return int(s)
    try:
        return str(uuid.UUID(s))
    except ValueError as e:
        raise ValueError(f"point_id must be a UUID or unsigned integer: {raw}") from e


def _normalize_point_id(raw: str) -> Union[int, str]:
    """Normalize a point ID supplied as a path parameter.

    Same validation as ``_coerce_point_id`` but maps the failure to an HTTP 400,
    since path params bypass pydantic body validation.
    """
    try:
        return _coerce_point_id(raw)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def _resolve_batch_size(value: Optional[int], settings: Settings) -> int:
    """Resolve a possibly-None batch size from runtime settings."""
    if value is None:
        return settings.ingest_default_batch_size
    if not (1 <= value <= settings.ingest_max_batch_size):
        raise HTTPException(
            status_code=400,
            detail=f"batch_size must be between 1 and {settings.ingest_max_batch_size}",
        )
    return value


def _validate_source_id_key(value: str) -> str:
    """Reject a blank or reserved ``source_id_key``.

    Reserved values would clobber a chunk-critical payload field
    (``text``/``id``/the chunker bookkeeping keys in
    ``CHUNK_RESERVED_FIELDS``).

    ``source_id`` itself is the legitimate default and must stay allowed.
    """
    stripped = value.strip()
    if not stripped:
        raise ValueError("source_id_key cannot be blank")
    reserved = {"text", "id"} | CHUNK_RESERVED_FIELDS
    if stripped in reserved:
        raise ValueError(f"source_id_key cannot be '{stripped}' (reserved field)")
    return stripped


def _validate_chunk_id_key(value: str) -> str:
    """Reject a blank or reserved ``chunk_id_key``.

    The chunk ordinal is ordinary payload whose name the caller picks, so any
    name is allowed except ones that would clobber another chunk-critical
    field. ``id`` itself is the legitimate default and must stay allowed —
    unlike for ``source_id_key``, where it would overwrite the ordinal.
    """
    stripped = value.strip()
    if not stripped:
        raise ValueError("chunk_id_key cannot be blank")
    reserved = {"text"} | (CHUNK_RESERVED_FIELDS - {CHUNK_ORDINAL_FIELD})
    if stripped in reserved:
        raise ValueError(f"chunk_id_key cannot be '{stripped}' (reserved field)")
    return stripped


def _reject_ingest_key_collision(request: Any) -> Any:
    """Reject an ingest request whose two payload keys are the same name.

    Both are written after the caller's ``metadata`` is merged, so a shared
    name would silently drop one of the two values.
    """
    if request.chunk_id_key == request.source_id_key:
        raise ValueError("chunk_id_key and source_id_key cannot be the same field")
    return request


class IngestRequest(BaseModel):
    """Request body for ingesting chunks from a JSON file under ``data/``."""

    file_path: str
    collection_name: Optional[str] = None
    batch_size: Optional[int] = None
    replace_source: bool = False
    source_id: Optional[str] = None
    source_id_key: str = "source_id"
    source_id_type: SourceIdType = "keyword"
    chunk_id_key: str = "id"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_id_key")
    @classmethod
    def validate_source_id_key(cls, value: str) -> str:
        """Reject blank/clobbering source_id_key values."""
        return _validate_source_id_key(value)

    @field_validator("chunk_id_key")
    @classmethod
    def validate_chunk_id_key(cls, value: str) -> str:
        """Reject blank/clobbering chunk_id_key values."""
        return _validate_chunk_id_key(value)

    @model_validator(mode="after")
    def validate_key_collision(self) -> "IngestRequest":
        """Reject a chunk_id_key that would land on the source_id key."""
        return _reject_ingest_key_collision(self)


class IngestChunksRequest(BaseModel):
    """Request body for ingesting chunks supplied inline (no file on disk).

    Mirrors ``IngestRequest`` but carries the ``chunks`` array directly and
    requires ``source_id`` (there is no filename to fall back to).
    """

    chunks: list[dict[str, Any]] = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_id_key: str = "source_id"
    source_id_type: SourceIdType = "keyword"
    chunk_id_key: str = "id"
    collection_name: Optional[str] = None
    batch_size: Optional[int] = None
    replace_source: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_id_key")
    @classmethod
    def validate_source_id_key(cls, value: str) -> str:
        """Reject blank/clobbering source_id_key values."""
        return _validate_source_id_key(value)

    @field_validator("chunk_id_key")
    @classmethod
    def validate_chunk_id_key(cls, value: str) -> str:
        """Reject blank/clobbering chunk_id_key values."""
        return _validate_chunk_id_key(value)

    @model_validator(mode="after")
    def validate_key_collision(self) -> "IngestChunksRequest":
        """Reject a chunk_id_key that would land on the source_id key."""
        return _reject_ingest_key_collision(self)


class IngestResponse(BaseModel):
    """Result of an ingest run, including failure position on partial failure."""

    ok: bool
    ingested: int
    skipped: int = 0
    collection_name: str
    failed_at_chunk: Optional[int] = None
    error: Optional[str] = None


def _require_non_blank_text(value: str) -> str:
    """Reject text that is empty or whitespace-only (embedding it is useless)."""
    if not value.strip():
        raise ValueError("text must be a non-empty string")
    return value


def _reject_payload_text(payload: dict[str, Any]) -> dict[str, Any]:
    """Require text changes to go through an embedding upsert."""
    if "text" in payload:
        raise ValueError("text is service-owned; use point upsert to change it")
    return payload


def _upsert_payload(payload: dict[str, Any], text: str) -> dict[str, Any]:
    """Build a point payload with embedded text taking precedence."""
    return {**payload, "text": text}


def _clean_payload_delete_keys(keys: list[str]) -> list[str]:
    """Normalize and parse payload paths before a delete reaches Qdrant."""
    cleaned = [key.strip() for key in keys]
    if any(not key for key in cleaned):
        raise ValueError("payload keys cannot be blank")
    for key in cleaned:
        payload_path_root(key)
    return cleaned


def _validate_payload_delete_keys(keys: list[str]) -> list[str]:
    """Reject protected Qdrant payload paths without changing their spelling."""
    cleaned = _clean_payload_delete_keys(keys)
    roots = {payload_path_root(key) for key in cleaned}
    protected = sorted(roots.intersection(PROTECTED_PAYLOAD_FIELDS))
    if protected:
        raise ValueError(
            f"Cannot delete protected payload fields: {', '.join(protected)}"
        )
    return cleaned


# Qdrant skips rather than errors on a point an insert_only/update_only write
# does not apply to, and reports no per-point detail — the ids straddling that
# line are found here with a cheap preflight retrieve, before any embedding
# work is spent on a point that will not be written.
UpsertMode = Literal["upsert", "insert_only", "update_only"]


class UpsertPointRequest(BaseModel):
    """Request body for upserting a single point from raw text."""

    collection_name: Optional[str] = None
    point_id: str
    text: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    update_mode: UpsertMode = "upsert"

    @field_validator("point_id")
    @classmethod
    def validate_point_id(cls, value: str) -> str:
        """Validate that ``point_id`` is a UUID or unsigned integer."""
        _coerce_point_id(value)
        return value

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Reject whitespace-only text."""
        return _require_non_blank_text(value)


class UpsertPointResponse(BaseModel):
    """Response confirming a single-point upsert."""

    ok: bool
    point_id: str
    collection_name: str
    # True when update_mode caused Qdrant not to write this point.
    skipped: bool = False


class BatchUpsertItem(BaseModel):
    """One point in a batch upsert request."""

    point_id: str
    text: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("point_id")
    @classmethod
    def validate_point_id(cls, value: str) -> str:
        """Validate that ``point_id`` is a UUID or unsigned integer."""
        _coerce_point_id(value)
        return value

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Reject whitespace-only text."""
        return _require_non_blank_text(value)


class BatchUpsertRequest(BaseModel):
    """Request body for upserting a batch of points."""

    collection_name: Optional[str] = None
    points: list[BatchUpsertItem] = Field(min_length=1, max_length=10000)
    update_mode: UpsertMode = "upsert"

    @field_validator("points")
    @classmethod
    def validate_unique_ids(cls, value: list[BatchUpsertItem]) -> list[BatchUpsertItem]:
        """Reject a batch that contains duplicate point ids."""
        ids = [_coerce_point_id(item.point_id) for item in value]
        if len(set(ids)) != len(ids):
            raise ValueError("points contains duplicate point_id values")
        return value


class BatchUpsertResponse(BaseModel):
    """Response confirming a batch upsert."""

    ok: bool
    upserted: int
    collection_name: str
    # Points update_mode kept out of this write (already existed for
    # insert_only, or did not exist for update_only).
    skipped: int = 0


class BulkDeleteRequest(BaseModel):
    """Request body for deleting many points by id."""

    collection_name: Optional[str] = None
    ids: list[str] = Field(min_length=1, max_length=10_000)

    @field_validator("ids")
    @classmethod
    def validate_ids(cls, value: list[str]) -> list[str]:
        """Validate that every id is a UUID or unsigned integer."""
        for item in value:
            _coerce_point_id(item)
        return value


class PayloadUpdateRequest(BaseModel):
    """Request body for a payload patch (PATCH) or full replace (PUT)."""

    collection_name: Optional[str] = None
    payload: dict[str, Any]

    @field_validator("payload")
    @classmethod
    def validate_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Require payload-only writes to leave embedded text unchanged."""
        return _reject_payload_text(value)


class RetrievePointsRequest(BaseModel):
    """Request body for retrieving many points by id in one call."""

    collection_name: Optional[str] = None
    ids: list[str] = Field(min_length=1, max_length=1000)
    with_vectors: bool = False
    with_payload: Union[bool, list[str]] = True

    @field_validator("ids")
    @classmethod
    def validate_ids(cls, value: list[str]) -> list[str]:
        """Validate that every id is a UUID or unsigned integer."""
        for item in value:
            _coerce_point_id(item)
        return value


class DeleteByFilterRequest(BaseModel):
    """Request body for deleting every point matching a payload/id filter."""

    collection_name: Optional[str] = None
    filters: dict[str, Any] = Field(min_length=1)


class PayloadKeysDeleteRequest(BaseModel):
    """Request body for removing named keys from a point's payload."""

    collection_name: Optional[str] = None
    keys: list[str] = Field(min_length=1)

    @field_validator("keys")
    @classmethod
    def validate_keys(cls, value: list[str]) -> list[str]:
        """Reject blank or malformed payload paths."""
        return _clean_payload_delete_keys(value)


class ClearPayloadRequest(BaseModel):
    """Request body for clearing a point's payload (protected fields kept)."""

    collection_name: Optional[str] = None


class PayloadFilterUpdateRequest(BaseModel):
    """Merge a payload into every point matching a filter.

    The filter must resolve to at least one condition: an empty filter would
    rewrite the whole collection, so it is refused the same way
    ``/points/delete-by-filter`` refuses a match-all delete.
    """

    collection_name: Optional[str] = None
    filters: dict[str, Any] = Field(min_length=1)
    payload: dict[str, Any] = Field(min_length=1)

    @field_validator("payload")
    @classmethod
    def validate_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Require payload-only writes to leave embedded text unchanged."""
        return _reject_payload_text(value)


class PayloadFilterKeysDeleteRequest(BaseModel):
    """Remove named keys from every point matching a filter."""

    collection_name: Optional[str] = None
    filters: dict[str, Any] = Field(min_length=1)
    keys: list[str] = Field(min_length=1)

    @field_validator("keys")
    @classmethod
    def validate_keys(cls, value: list[str]) -> list[str]:
        """Reject blank or malformed payload paths."""
        return _clean_payload_delete_keys(value)


class PayloadFilterClearRequest(BaseModel):
    """Clear the payload of every point matching a filter (protected kept)."""

    collection_name: Optional[str] = None
    filters: dict[str, Any] = Field(min_length=1)


class PayloadFilterUpdateResponse(BaseModel):
    """Response reporting how many points a filtered payload write matched."""

    ok: bool
    matched: int
    collection_name: str


def _validate_ingest_path(file_path: str, settings: Settings) -> Path:
    """Resolve and validate an ingest path, confining it to ``data/``.

    Raises HTTP 400 if the path escapes the data root, does not point at an
    existing file, or exceeds the max file size from settings.
    """
    try:
        root = settings.ingest_data_root.resolve()
        resolved = (settings.ingest_data_root / file_path).resolve()
    except Exception as e:
        raise HTTPException(400, f"Invalid file_path: {e}") from e
    if not resolved.is_relative_to(root):
        raise HTTPException(400, "file_path must be inside the ingest data/ directory")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(400, f"File not found: {file_path}")
    if resolved.stat().st_size > settings.ingest_max_file_bytes:
        raise HTTPException(
            400, f"File too large: max {settings.ingest_max_file_bytes} bytes"
        )
    return resolved


def _validate_chunks(chunks: Any, settings: Settings) -> list[dict[str, Any]]:
    """Validate a ``chunks`` array.

    Each chunk must be an object with an integer ``id`` and non-empty ``text``,
    and ids must be unique. Raises HTTP 400 on any structural problem. Shared by
    file-based and inline ingestion.

    The count is checked first, and against ``INGEST_MAX_CHUNKS`` rather than
    any byte size, because the memory this request will peak at scales with the
    number of points and not with the bytes that described them: every chunk
    becomes a dense and a sparse vector, all of which are resident together
    before the first write. Over the limit is HTTP 413 with nothing embedded --
    the alternative is not a slow request but an OOM-killed container, which
    the caller sees as a dropped connection rather than an error.

    The inline endpoint's Pydantic schema deliberately carries no
    ``max_length`` to match: that would be a second, static source of truth
    that silently vetoes an operator who raised ``INGEST_MAX_CHUNKS``, and it
    would reject no earlier -- the body is already parsed by the time a field
    validator runs.
    """
    if not isinstance(chunks, list):
        raise HTTPException(400, "'chunks' must be an array")
    if len(chunks) > settings.ingest_max_chunks:
        raise HTTPException(
            413,
            f"Too many chunks: {len(chunks)} exceeds INGEST_MAX_CHUNKS "
            f"({settings.ingest_max_chunks}). Split the source into smaller "
            "ingests; point ids are deterministic, so the parts do not "
            "overlap or duplicate.",
        )

    seen_ids: set[int] = set()
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise HTTPException(400, f"chunks[{i}] must be an object")
        chunk_id = chunk.get("id")
        if isinstance(chunk_id, bool) or not isinstance(chunk_id, int):
            raise HTTPException(400, f"chunks[{i}].id must be an integer")
        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(400, f"chunks[{i}].text is empty or missing")
        if chunk_id in seen_ids:
            raise HTTPException(400, f"Duplicate chunk id {chunk_id} at index {i}")
        seen_ids.add(chunk_id)

    return chunks


def _read_json_file(path: Path) -> dict[str, Any]:
    """Read and parse a JSON file, mapping I/O and parse errors to HTTP 400."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid JSON: {e}") from e
    except OSError as e:
        raise HTTPException(400, f"Cannot read file: {e}") from e


def _load_and_validate_chunks(path: Path, settings: Settings) -> list[dict[str, Any]]:
    """Load the JSON file and validate its ``chunks`` array.

    See :func:`_validate_chunks` for the validation contract.
    """
    data = _read_json_file(path)
    if (
        not isinstance(data, dict)
        or "chunks" not in data
        or not isinstance(data["chunks"], list)
    ):
        raise HTTPException(400, "JSON must have a top-level 'chunks' array")

    return _validate_chunks(data["chunks"], settings)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_chunks(
    request: IngestRequest,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    coordinator: Annotated[MutationCoordinator, Depends(get_mutation_coordinator)],
    manager: Annotated[VectorReindexManager, Depends(get_vector_reindex_manager)],
    embedder: EmbedderClient = Depends(get_embedder_client),
):
    """Ingest chunks from a JSON file under ``data/``: embed, upsert, then prune.

    Thin wrapper that resolves+validates the file, then delegates to
    ``_run_ingest``. See that function for the embed/upsert/prune contract.
    """
    collection_name = resolve_collection_name(request.collection_name, settings)
    http_request.state.log_collection = collection_name
    batch_size = _resolve_batch_size(request.batch_size, settings)
    path = _validate_ingest_path(request.file_path, settings)
    chunks = await asyncio.to_thread(_load_and_validate_chunks, path, settings)
    sid = request.source_id or path.stem
    response = await _run_ingest(
        chunks=chunks,
        collection_name=collection_name,
        batch_size=batch_size,
        replace_source=request.replace_source,
        sid=sid,
        source_id_key=request.source_id_key,
        source_id_type=request.source_id_type,
        chunk_id_key=request.chunk_id_key,
        metadata=request.metadata,
        embedder=embedder,
        client=client,
        settings=settings,
        coordinator=coordinator,
        manager=manager,
    )
    http_request.state.log_points = response.ingested
    return response


@router.post("/ingest-chunks", response_model=IngestResponse)
async def ingest_inline_chunks(
    request: IngestChunksRequest,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    coordinator: Annotated[MutationCoordinator, Depends(get_mutation_coordinator)],
    manager: Annotated[VectorReindexManager, Depends(get_vector_reindex_manager)],
    embedder: EmbedderClient = Depends(get_embedder_client),
):
    """Ingest chunks supplied inline in the request body (no file on disk).

    Same embed/upsert/prune contract as ``/ingest``, but the caller sends the
    ``chunks`` array directly. Used by the dashboard, which has no shared
    filesystem with this service. ``source_id`` is required since there is no
    filename to derive it from.
    """
    collection_name = resolve_collection_name(request.collection_name, settings)
    http_request.state.log_collection = collection_name
    batch_size = _resolve_batch_size(request.batch_size, settings)
    chunks = _validate_chunks(request.chunks, settings)
    response = await _run_ingest(
        chunks=chunks,
        collection_name=collection_name,
        batch_size=batch_size,
        replace_source=request.replace_source,
        sid=request.source_id,
        source_id_key=request.source_id_key,
        source_id_type=request.source_id_type,
        chunk_id_key=request.chunk_id_key,
        metadata=request.metadata,
        embedder=embedder,
        client=client,
        settings=settings,
        coordinator=coordinator,
        manager=manager,
    )
    http_request.state.log_points = response.ingested
    return response


@dataclass(frozen=True)
class _IngestContext:
    """Resolved ingest parameters shared by the ingest helper functions."""

    collection_name: str
    batch_size: int
    replace_source: bool
    sid: str
    source_id_key: str
    source_id_value: Any
    chunk_id_key: str
    metadata: dict[str, Any]
    source_id_type: str = "keyword"


async def _ensure_ingest_indexes(
    client: AsyncQdrantClient, context: _IngestContext, settings: Settings
) -> None:
    """Ensure required payload indexes before ingest writes begin."""
    try:
        await ensure_payload_indexes(client, context.collection_name, settings=settings)
        if context.replace_source:
            # Prune scrolls filter on source_id; strict-mode collections
            # reject that filter unless the field is indexed.
            await ensure_source_id_index(
                client,
                context.collection_name,
                settings=settings,
                field_name=context.source_id_key,
                field_schema=_source_id_index_schema(context.source_id_type),
            )
    except PayloadIndexTypeMismatch as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as e:
        raise HTTPException(502, f"Qdrant error: {e}") from e


async def _source_ids_before_ingest(
    client: AsyncQdrantClient, context: _IngestContext, settings: Settings
) -> set[str]:
    """Return existing source ids only when replace_source requires pruning."""
    if not context.replace_source:
        return set()
    return await _qdrant(
        lambda: _scroll_source_point_ids(
            client,
            context.collection_name,
            context.source_id_value,
            context.source_id_key,
            context.source_id_type,
        ),
        "scroll source ids",
        settings=settings,
    )


async def _embed_ingest_chunks(
    chunks: list[dict[str, Any]],
    context: _IngestContext,
    embedder: EmbedderClient,
    settings: Settings,
) -> list[EmbeddingResult] | IngestResponse:
    """Embed all chunks before any write and return a failure response on error."""
    embed_batch_size = min(
        context.batch_size, getattr(embedder, "max_batch_size", context.batch_size)
    )
    batch_starts = list(range(0, len(chunks), embed_batch_size))
    batch_texts = [
        [c["text"] for c in chunks[bs : bs + embed_batch_size]] for bs in batch_starts
    ]
    embed_semaphore = asyncio.Semaphore(settings.smart_embedder_embed_concurrency)
    embed_aborted = False

    async def _embed_batch(bs: int, texts: list[str]):
        nonlocal embed_aborted
        async with embed_semaphore:
            if embed_aborted:
                return bs, texts, None
            try:
                return bs, texts, await embedder.embed(texts)
            except Exception:
                embed_aborted = True
                raise

    embed_results = await asyncio.gather(
        *(_embed_batch(bs, texts) for bs, texts in zip(batch_starts, batch_texts)),
        return_exceptions=True,
    )

    all_embeddings: list[EmbeddingResult] = []
    for bs, result in zip(batch_starts, embed_results):
        if isinstance(result, Exception):
            error_msg = (
                str(result.detail) if isinstance(result, HTTPException) else str(result)
            )
            return IngestResponse(
                ok=False,
                ingested=0,
                collection_name=context.collection_name,
                failed_at_chunk=bs,
                error=error_msg,
            )
        _, texts, embeddings = result
        if embeddings is None:
            continue
        if len(embeddings) != len(texts):
            return IngestResponse(
                ok=False,
                ingested=0,
                collection_name=context.collection_name,
                failed_at_chunk=bs,
                error="Embedding count mismatch",
            )
        all_embeddings.extend(embeddings)
    return all_embeddings


def _build_ingest_batches(
    chunks: list[dict[str, Any]],
    embeddings: list[EmbeddingResult],
    context: _IngestContext,
    vector_names: ActiveVectorNames,
) -> tuple[list[str], list[tuple[int, list[PointStruct]]]]:
    """Build deterministic Qdrant points and group them into upload batches."""
    new_ids: list[str] = []
    batches: list[tuple[int, list[PointStruct]]] = []
    for batch_start in range(0, len(chunks), context.batch_size):
        batch = chunks[batch_start : batch_start + context.batch_size]
        batch_embeddings = embeddings[batch_start : batch_start + context.batch_size]
        points: list[PointStruct] = []
        for chunk, emb in zip(batch, batch_embeddings):
            payload = {
                k: v
                for k, v in chunk.items()
                if k not in CHUNK_EXCLUDE_FIELDS and k != CHUNK_ORDINAL_FIELD
            }
            payload.update(
                {
                    k: v
                    for k, v in context.metadata.items()
                    if k not in PROTECTED_PAYLOAD_FIELDS
                }
            )
            # Both service-owned keys are written last so caller metadata cannot
            # shadow them. The ordinal is relocated rather than copied, so a
            # chunk_id_key other than "id" leaves no duplicate behind.
            payload[context.chunk_id_key] = chunk[CHUNK_ORDINAL_FIELD]
            payload[context.source_id_key] = context.source_id_value
            point_id = make_point_id(context.sid, chunk[CHUNK_ORDINAL_FIELD])
            new_ids.append(point_id)
            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        vector_names.dense: emb.dense,
                        vector_names.sparse: emb.sparse,
                    },
                    payload=payload,
                )
            )
        batches.append((batch_start, points))
    return new_ids, batches


async def _upsert_ingest_batches(
    batches: list[tuple[int, list[PointStruct]]],
    client: AsyncQdrantClient,
    context: _IngestContext,
    settings: Settings,
) -> tuple[int, IngestResponse | None]:
    """Upload point batches with bounded concurrency and report partial failure."""
    semaphore = asyncio.Semaphore(settings.ingest_upload_concurrency)

    async def _upsert(points: list[PointStruct]) -> None:
        async with semaphore:
            await retry_async(
                lambda: client.upsert(
                    collection_name=context.collection_name,
                    points=points,
                    wait=True,
                ),
                should_retry=is_transient_qdrant,
                attempts=settings.retry_max_attempts,
                base_delay=settings.retry_base_delay,
                description="ingest upsert",
            )

    results = await asyncio.gather(
        *(_upsert(points) for _, points in batches),
        return_exceptions=True,
    )

    ingested = 0
    failures: list[tuple[int, Exception]] = []
    for (batch_start, points), result in zip(batches, results):
        if isinstance(result, Exception):
            failures.append((batch_start, result))
            continue
        ingested += len(points)
    if failures:
        failed_at_chunk, error = min(failures, key=lambda item: item[0])
        return ingested, IngestResponse(
            ok=False,
            ingested=ingested,
            collection_name=context.collection_name,
            failed_at_chunk=failed_at_chunk,
            error=str(error),
        )
    return ingested, None


async def _prune_stale_source_points(
    old_ids: set[str],
    new_ids: list[str],
    client: AsyncQdrantClient,
    context: _IngestContext,
    settings: Settings,
) -> None:
    """Delete stale points for replace_source after new writes are durable."""
    if not context.replace_source:
        return
    stale = sorted(old_ids - set(new_ids))
    delete_chunks = [
        stale[start : start + _STALE_DELETE_CHUNK]
        for start in range(0, len(stale), _STALE_DELETE_CHUNK)
    ]
    delete_semaphore = asyncio.Semaphore(settings.ingest_upload_concurrency)

    async def _delete_chunk(ids: list[str]) -> None:
        async with delete_semaphore:
            await retry_async(
                lambda: client.delete(
                    collection_name=context.collection_name,
                    points_selector=PointIdsList(points=ids),
                    wait=True,
                ),
                should_retry=is_transient_qdrant,
                attempts=settings.retry_max_attempts,
                base_delay=settings.retry_base_delay,
                description="prune stale points",
            )

    results = await asyncio.gather(
        *(_delete_chunk(ids) for ids in delete_chunks),
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, Exception):
            raise HTTPException(502, f"Qdrant error: {result}") from result


async def _run_ingest(
    *,
    chunks: list[dict[str, Any]],
    collection_name: str,
    batch_size: int,
    replace_source: bool,
    sid: str,
    source_id_key: str = "source_id",
    source_id_type: str = "keyword",
    chunk_id_key: str = CHUNK_ORDINAL_FIELD,
    metadata: dict[str, Any],
    embedder: EmbedderClient,
    client: AsyncQdrantClient,
    settings: Settings,
    coordinator: MutationCoordinator,
    manager: VectorReindexManager,
) -> IngestResponse:
    """Embed validated chunks, upsert them, then optionally prune.

    Embeds all chunks first (so a failed embed touches no data), upserts new
    points with bounded parallelism, and — when ``replace_source`` is set —
    prunes stale points only after every wait=True upload is durable.
    """
    try:
        source_id_value = _coerce_source_id_value(sid, source_id_type)
        _source_id_index_schema(source_id_type)
    except ValueError as exc:
        return IngestResponse(
            ok=False,
            ingested=0,
            collection_name=collection_name,
            error=f"source_id {exc}",
        )
    context = _IngestContext(
        collection_name=collection_name,
        batch_size=batch_size,
        replace_source=replace_source,
        sid=sid,
        source_id_key=source_id_key,
        source_id_value=source_id_value,
        source_id_type=source_id_type,
        chunk_id_key=chunk_id_key,
        metadata=metadata,
    )
    embeddings = await _embed_ingest_chunks(chunks, context, embedder, settings)
    if isinstance(embeddings, IngestResponse):
        return embeddings
    async with mutation_scope(client, collection_name, settings, coordinator, manager):
        vector_names = await guard_vector_operation(
            client,
            context.collection_name,
            embeddings,
            settings,
        )
        await _ensure_ingest_indexes(client, context, settings)
        old_ids = await _source_ids_before_ingest(client, context, settings)
        new_ids, batches = _build_ingest_batches(
            chunks,
            embeddings,
            context,
            vector_names,
        )
        ingested, failure = await _upsert_ingest_batches(
            batches, client, context, settings
        )
        if failure is not None:
            return failure
        await _prune_stale_source_points(old_ids, new_ids, client, context, settings)
    return IngestResponse(ok=True, ingested=ingested, collection_name=collection_name)


@router.post("/count", response_model=CountPointsResponse)
async def count_points(
    request: CountPointsRequest,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
):
    """Count matching points without retrieving them.

    A read-only preflight for filters: unlike the count baked into
    ``/points/delete-by-filter`` it never sits next to a destructive action.
    An absent or empty filter counts the whole collection. ``exact=false``
    returns the fast approximate estimate.
    """
    collection_name = resolve_collection_name(request.collection_name, settings)
    http_request.state.log_collection = collection_name
    try:
        count_filter = build_qdrant_filter(request.filters, settings=settings)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    result = await _qdrant(
        lambda: client.count(
            collection_name=collection_name,
            count_filter=count_filter,
            exact=request.exact,
        ),
        "count points",
        settings,
    )
    http_request.state.log_points = result.count
    return CountPointsResponse(
        ok=True,
        collection_name=collection_name,
        count=result.count,
        exact=request.exact,
    )


@router.post("/facet", response_model=FacetPointsResponse)
async def facet_points(
    request: FacetPointsRequest,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
):
    """Return distinct values of one indexed payload field and their counts.

    Requires a payload index on ``key`` (Qdrant rejects an unindexed field);
    the 400 it raises is passed through as-is rather than guessed at here.
    """
    collection_name = resolve_collection_name(request.collection_name, settings)
    http_request.state.log_collection = collection_name
    try:
        facet_filter = build_qdrant_filter(request.filters, settings=settings)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    try:
        result = await retry_async(
            lambda: client.facet(
                collection_name=collection_name,
                key=request.key,
                facet_filter=facet_filter,
                limit=request.limit,
                exact=request.exact,
            ),
            attempts=settings.retry_max_attempts,
            base_delay=settings.retry_base_delay,
            should_retry=is_transient_qdrant,
            description="facet points",
        )
    except UnexpectedResponse as exc:
        # An unindexed key is a client mistake (Qdrant rejects it outright),
        # not a Qdrant failure — surface it as 400 with Qdrant's own reason
        # ("No appropriate index for faceting: ...") rather than the generic
        # 502 or the uninformative HTTP reason phrase.
        if exc.status_code in (400, 422):
            reason = exc.reason_phrase
            with suppress(TypeError, ValueError, KeyError):
                reason = json.loads(exc.content)["status"]["error"] or reason
            raise HTTPException(
                400, f"key '{request.key}' cannot be faceted: {reason}"
            ) from exc
        raise HTTPException(502, f"Qdrant facet failed: {exc}") from exc
    return FacetPointsResponse(
        ok=True,
        collection_name=collection_name,
        key=request.key,
        exact=request.exact,
        hits=[FacetHit(value=hit.value, count=hit.count) for hit in result.hits],
    )


def _payload_selection(value: Union[bool, list[str]]) -> Union[bool, list[str]]:
    """Translate a request's ``with_payload`` into Qdrant's payload selector."""
    return qdrant_payload_selector(value) if isinstance(value, list) else value


@router.post("/scroll", response_model=ScrollPointsResponse)
async def scroll_points(
    request: ScrollPointsRequest,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
):
    """Return a page of filtered points without embedding or vector search."""
    collection_name = resolve_collection_name(request.collection_name, settings)
    http_request.state.log_collection = collection_name
    try:
        scroll_filter = build_qdrant_filter(request.filters, settings=settings)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    with_payload = _payload_selection(request.with_payload)
    # Qdrant rejects offset+order_by outright; say so here instead of forwarding
    # a request that can only come back as a raw 400.
    if request.order_by is not None and request.offset is not None:
        raise HTTPException(
            422,
            "offset cannot be combined with order_by — page an ordered scroll "
            "with order_by.start_from instead",
        )
    # Qdrant types start_from after the ordered field: it takes a number or a
    # datetime, never a free string, and the client model enforces that. Catch the
    # rejection here so a mistyped value is a 422 with a usable message instead of
    # an unhandled validation error surfacing as 500.
    order_by = None
    if request.order_by is not None:
        try:
            order_by = models.OrderBy(
                key=request.order_by.key,
                direction=request.order_by.direction,
                start_from=request.order_by.start_from,
            )
        except PydanticValidationError as e:
            raise HTTPException(
                422,
                "order_by.start_from must be a number or a datetime matching the "
                f"ordered field, got {request.order_by.start_from!r}",
            ) from e

    scroll_operation = _qdrant(
        lambda: client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=request.limit,
            offset=request.offset,
            order_by=order_by,
            with_payload=with_payload,
            with_vectors=False,
        ),
        "scroll points",
        settings,
    )
    if request.include_total:
        (points, next_offset), count_result = await asyncio.gather(
            scroll_operation,
            _qdrant(
                lambda: client.count(
                    collection_name=collection_name,
                    count_filter=scroll_filter,
                    exact=True,
                ),
                "count filtered points",
                settings,
            ),
        )
        total = count_result.count
    else:
        points, next_offset = await scroll_operation
        total = None
    results = [
        ScrolledPoint(id=str(point.id), payload=point.payload or {}) for point in points
    ]
    http_request.state.log_points = len(results)
    return ScrollPointsResponse(
        collection_name=collection_name,
        returned=len(results),
        total=total,
        next_page_offset=next_offset,
        results=results,
    )


def _prepare_export_file(document, settings) -> tuple[str, Path, int]:
    """Write and fsync a temporary export without publishing its final name."""
    root = settings.export_data_root
    root.mkdir(parents=True, exist_ok=True)
    payload = document.text.encode("utf-8")
    target = export_target(root, document.filename)
    handle, temp_path = tempfile.mkstemp(dir=str(root), suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            # Flush user-space buffers, then force the bytes onto stable storage
            # before the rename publishes the file under its final name.
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # Leaving a .tmp behind would look like a partial export to anyone
        # listing the directory.
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise
    return temp_path, target, len(payload)


def _sync_export_directory(root: Path) -> None:
    """Make the published directory entry durable on POSIX filesystems."""
    if os.name == "posix":
        directory_fd = os.open(str(root), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


async def _await_export_io(task):
    """Drain a native filesystem worker before releasing the export lease."""
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Cancelling an await cannot stop a running thread. Repeated cancellation
        # must not abandon the worker or release capacity while it still runs.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()
        raise


async def _write_export_atomically(document, settings) -> tuple[str, int]:
    """Prepare off-loop and publish only while the request is still active."""
    prepared = asyncio.create_task(
        asyncio.to_thread(_prepare_export_file, document, settings)
    )
    try:
        temp_path, target, size = await _await_export_io(prepared)
        # No await separates successful preparation from publication. A cancelled
        # preparation is drained and cleaned up in finally, never published later.
        os.replace(temp_path, target)
        await _await_export_io(
            asyncio.create_task(
                asyncio.to_thread(_sync_export_directory, settings.export_data_root)
            )
        )
        return str(target), size
    finally:
        # Even when the request was cancelled, the drained preparation may have
        # succeeded. Retrieve its path so that no unpublished temporary survives.
        if (
            prepared.done()
            and not prepared.cancelled()
            and prepared.exception() is None
        ):
            Path(prepared.result()[0]).unlink(missing_ok=True)


@router.post(
    "/export",
    dependencies=[Depends(reserve_export_slot)],
    response_model=None,
    responses={
        200: {
            "model": ExportCollectionResponse,
            "description": (
                "With destination='file', the written path and counts. With "
                "destination='body', the export document itself, served as an "
                "attachment instead of this model."
            ),
        },
        413: {"description": "Export exceeds EXPORT_MAX_POINTS or EXPORT_MAX_BYTES."},
        503: {"description": "EXPORT_MAX_CONCURRENT exports are already active."},
    },
)
async def export_collection(
    request: ExportCollectionRequest,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
):
    """Export a whole collection's payload as one JSON document.

    The document is produced once and either written under the export root or
    returned as an attachment; both paths emit identical bytes.
    """
    collection_name = resolve_collection_name(request.collection_name, settings)
    http_request.state.log_collection = collection_name
    count_result = await _qdrant(
        lambda: client.count(collection_name=collection_name, exact=True),
        "count points for export",
        settings,
    )
    if count_result.count > settings.export_max_points:
        raise HTTPException(
            413,
            f"collection '{collection_name}' holds {count_result.count} points, "
            f"above EXPORT_MAX_POINTS={settings.export_max_points} — raise the "
            "cap together with the container memory limit, or snapshot the "
            "Qdrant volume instead",
        )

    probe = getattr(http_request.app.state, "health_probe", None)
    try:
        document = await build_export_document(
            client,
            collection_name,
            embedder_profile=getattr(probe, "embedder_profile", None),
            embedder_models=getattr(probe, "embedder_models", None),
            max_points=settings.export_max_points,
            max_bytes=settings.export_max_bytes,
            run=lambda operation, description: _qdrant(
                operation, description, settings
            ),
        )
    except ExportLimitExceeded as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    if request.destination == "body":
        http_request.state.log_points = document.n_chunks
        return Response(
            content=document.text,
            media_type="application/json",
            headers={"Content-Disposition": attachment_disposition(document.filename)},
        )

    if document.n_chunks == 0:
        http_request.state.log_points = 0
        return ExportCollectionResponse(
            ok=True,
            collection_name=collection_name,
            n_chunks=0,
            path=None,
            bytes=len(document.text.encode("utf-8")),
        )

    try:
        path, size = await _write_export_atomically(document, settings)
    except OSError as exc:
        # An unwritable export root (missing mount, full disk) is a dependency
        # failure like any other, and the documented contract for a failed
        # export is 502 with nothing written — not a bare 500.
        raise HTTPException(
            502, f"failed to write the export under {settings.export_data_root}: {exc}"
        ) from exc
    http_request.state.log_points = document.n_chunks
    return ExportCollectionResponse(
        ok=True,
        collection_name=collection_name,
        n_chunks=document.n_chunks,
        path=path,
        bytes=size,
    )


@router.post("", response_model=UpsertPointResponse)
async def upsert_point(
    request: UpsertPointRequest,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    coordinator: Annotated[MutationCoordinator, Depends(get_mutation_coordinator)],
    manager: Annotated[VectorReindexManager, Depends(get_vector_reindex_manager)],
    embedder: EmbedderClient = Depends(get_embedder_client),
):
    """Embed the given text and upsert it as a single point.

    ``update_mode`` other than ``upsert`` is resolved against a preflight
    retrieve before embedding, so a point Qdrant would skip anyway never costs
    a Smart Embedder call.
    """
    collection_name = resolve_collection_name(request.collection_name, settings)
    http_request.state.log_collection = collection_name
    pid = _normalize_point_id(request.point_id)
    if request.update_mode != "upsert":
        exists = pid in await _existing_point_ids(
            client, collection_name, [pid], settings
        )
        skip = (request.update_mode == "insert_only" and exists) or (
            request.update_mode == "update_only" and not exists
        )
        if skip:
            http_request.state.log_points = 0
            return UpsertPointResponse(
                ok=True,
                point_id=request.point_id,
                collection_name=collection_name,
                skipped=True,
            )

    try:
        embeddings = await embedder.embed([request.text])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Smart Embedder embedding failed: {e}") from e
    if len(embeddings) != 1:
        raise HTTPException(502, "Embedding count mismatch")
    async with mutation_scope(client, collection_name, settings, coordinator, manager):
        vector_names = await guard_vector_operation(
            client, collection_name, embeddings, settings
        )
        emb = embeddings[0]
        point = PointStruct(
            id=pid,
            vector={
                vector_names.dense: emb.dense,
                vector_names.sparse: emb.sparse,
            },
            payload=_upsert_payload(request.payload, request.text),
        )
        await _qdrant(
            lambda: client.upsert(
                collection_name=collection_name,
                points=[point],
                wait=True,
                update_mode=(
                    None
                    if request.update_mode == "upsert"
                    else models.UpdateMode(request.update_mode)
                ),
            ),
            "upsert point",
            settings=settings,
        )
    http_request.state.log_points = 1
    return UpsertPointResponse(
        ok=True,
        point_id=request.point_id,
        collection_name=collection_name,
    )


@router.post("/batch", response_model=BatchUpsertResponse)
async def batch_upsert_points(
    request: BatchUpsertRequest,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    coordinator: Annotated[MutationCoordinator, Depends(get_mutation_coordinator)],
    manager: Annotated[VectorReindexManager, Depends(get_vector_reindex_manager)],
    embedder: EmbedderClient = Depends(get_embedder_client),
):
    """Embed all texts in the batch and upsert them in one Qdrant call.

    ``update_mode`` other than ``upsert`` is resolved against a preflight
    retrieve before embedding, so points Qdrant would skip anyway never cost a
    Smart Embedder call.
    """
    collection_name = resolve_collection_name(request.collection_name, settings)
    http_request.state.log_collection = collection_name
    ids = [_normalize_point_id(item.point_id) for item in request.points]
    kept = list(zip(request.points, ids))
    skipped = 0
    if request.update_mode != "upsert":
        existing = await _existing_point_ids(client, collection_name, ids, settings)
        want_existing = request.update_mode == "update_only"
        kept = [(item, pid) for item, pid in kept if (pid in existing) == want_existing]
        skipped = len(request.points) - len(kept)

    if not kept:
        http_request.state.log_points = 0
        return BatchUpsertResponse(
            ok=True, upserted=0, collection_name=collection_name, skipped=skipped
        )

    texts = [item.text for item, _ in kept]
    try:
        embeddings = await embedder.embed(texts)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Smart Embedder embedding failed: {e}") from e
    if len(embeddings) != len(texts):
        raise HTTPException(502, "Embedding count mismatch")
    async with mutation_scope(client, collection_name, settings, coordinator, manager):
        vector_names = await guard_vector_operation(
            client, collection_name, embeddings, settings
        )
        points = [
            PointStruct(
                id=pid,
                vector={
                    vector_names.dense: emb.dense,
                    vector_names.sparse: emb.sparse,
                },
                payload=_upsert_payload(item.payload, item.text),
            )
            for (item, pid), emb in zip(kept, embeddings)
        ]
        await _qdrant(
            lambda: client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
                update_mode=(
                    None
                    if request.update_mode == "upsert"
                    else models.UpdateMode(request.update_mode)
                ),
            ),
            "batch upsert",
            settings=settings,
        )
    http_request.state.log_points = len(points)
    return BatchUpsertResponse(
        ok=True,
        upserted=len(points),
        collection_name=collection_name,
        skipped=skipped,
    )


class PointResponse(BaseModel):
    """Response carrying a single point's id and payload."""

    id: str
    payload: dict[str, Any]
    collection_name: str
    # Populated only when with_vectors=true: names/shapes, never raw numbers.
    vectors: Optional[dict[str, dict[str, Any]]] = None


@router.get("/{point_id}", response_model=PointResponse)
async def get_point(
    point_id: str,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    collection_name: Optional[str] = None,
    with_vectors: bool = False,
):
    """Retrieve a single point's payload by id (404 if not found).

    ``with_vectors`` adds a summary of the point's stored vectors — names and
    shapes, never the raw numbers — useful to confirm which embedding
    generation a point currently carries during a re-embed.
    """
    collection_name = resolve_collection_name(collection_name, settings)
    _pid, results = await _retrieve_point_or_404(
        client,
        collection_name,
        point_id,
        settings,
        with_payload=True,
        with_vectors=with_vectors,
    )
    r = results[0]
    return PointResponse(
        id=str(r.id),
        payload=r.payload or {},
        collection_name=collection_name,
        vectors=_summarize_vectors(r.vector) if with_vectors else None,
    )


class RetrievePointsResponse(BaseModel):
    """Response carrying the points found for a batch id retrieve."""

    collection_name: str
    returned: int
    results: list[ScrolledPoint]


@router.post("/retrieve", response_model=RetrievePointsResponse)
async def retrieve_points(
    request: RetrievePointsRequest,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
):
    """Retrieve many points by id in one call (missing ids are omitted)."""
    collection_name = resolve_collection_name(request.collection_name, settings)
    http_request.state.log_collection = collection_name
    ids = [_normalize_point_id(i) for i in request.ids]
    results = await _qdrant(
        lambda: client.retrieve(
            collection_name=collection_name,
            ids=ids,
            with_payload=_payload_selection(request.with_payload),
            with_vectors=request.with_vectors,
        ),
        "retrieve points",
        settings=settings,
    )
    http_request.state.log_points = len(results)
    return RetrievePointsResponse(
        collection_name=collection_name,
        returned=len(results),
        results=[
            ScrolledPoint(
                id=str(r.id),
                payload=r.payload or {},
                vectors=(
                    _summarize_vectors(r.vector) if request.with_vectors else None
                ),
            )
            for r in results
        ],
    )


class DeleteResponse(BaseModel):
    """Response confirming a single-point deletion.

    ``deleted`` is 1 when the id existed and 0 when it did not, mirroring
    :class:`BulkDeleteResponse` so a caller reads the same field either way.
    """

    ok: bool
    deleted: int
    collection_name: str


@router.delete("/{point_id}", response_model=DeleteResponse)
async def delete_point(
    point_id: str,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    coordinator: Annotated[MutationCoordinator, Depends(get_mutation_coordinator)],
    manager: Annotated[VectorReindexManager, Depends(get_vector_reindex_manager)],
    collection_name: Optional[str] = None,
):
    """Delete a single point by id.

    Deleting an id that does not exist is a no-op reported as ``deleted: 0``,
    not a 404: DELETE is idempotent, and the caller's intent ("this id must
    not be there") is already satisfied when it isn't. This also keeps the
    endpoint consistent with the bulk delete below, which the MCP tool routes
    to purely on list length — a 404 here made one id and two ids behave
    oppositely for identical intent. A malformed id is still a 400, since
    that is a request Qdrant could never satisfy rather than one already
    satisfied.
    """
    collection_name = resolve_collection_name(collection_name, settings)
    http_request.state.log_collection = collection_name
    async with mutation_scope(client, collection_name, settings, coordinator, manager):
        pid = _normalize_point_id(point_id)
        existing = await _existing_point_ids(client, collection_name, [pid], settings)
        await _qdrant(
            lambda: client.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=[pid]),
                wait=True,
            ),
            "delete point",
            settings=settings,
        )
    deleted = 1 if pid in existing else 0
    http_request.state.log_points = deleted
    return DeleteResponse(ok=True, deleted=deleted, collection_name=collection_name)


class BulkDeleteResponse(BaseModel):
    """Response reporting how many ids were submitted for deletion."""

    ok: bool
    deleted: int
    collection_name: str


@router.delete("", response_model=BulkDeleteResponse)
async def bulk_delete_points(
    request: BulkDeleteRequest,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    coordinator: Annotated[MutationCoordinator, Depends(get_mutation_coordinator)],
    manager: Annotated[VectorReindexManager, Depends(get_vector_reindex_manager)],
):
    """Delete many points by id in a single Qdrant call.

    ``deleted`` reflects how many of the submitted ids actually existed, not
    how many were submitted: Qdrant's own delete call is silent about which
    ids were real, so a resolved-existence pass runs first (the same
    ``_existing_point_ids`` helper ``insert_only``/``update_only`` already
    use) and duplicate ids collapse to one. Without this, a bulk call with
    stale or duplicate ids over-reports success in a way the single-id
    endpoint's 404-on-miss never would for the same selector='ids' delete.
    """
    collection_name = resolve_collection_name(request.collection_name, settings)
    http_request.state.log_collection = collection_name
    async with mutation_scope(client, collection_name, settings, coordinator, manager):
        ids = [_normalize_point_id(i) for i in request.ids]
        existing = await _existing_point_ids(client, collection_name, ids, settings)
        await _qdrant(
            lambda: client.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=ids),
                wait=True,
            ),
            "bulk delete",
            settings=settings,
        )
        deleted = len({i for i in ids if i in existing})
    http_request.state.log_points = deleted
    return BulkDeleteResponse(ok=True, deleted=deleted, collection_name=collection_name)


@router.post("/delete-by-filter", response_model=BulkDeleteResponse)
async def delete_points_by_filter(
    request: DeleteByFilterRequest,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    coordinator: Annotated[MutationCoordinator, Depends(get_mutation_coordinator)],
    manager: Annotated[VectorReindexManager, Depends(get_vector_reindex_manager)],
):
    """Delete every point matching a payload/id filter.

    The filter must resolve to at least one condition — an empty filter would
    match (and delete) the whole collection, so it is rejected. The matching
    count is taken before the delete so the response can report how many points
    were removed.
    """
    collection_name = resolve_collection_name(request.collection_name, settings)
    http_request.state.log_collection = collection_name
    async with mutation_scope(client, collection_name, settings, coordinator, manager):
        try:
            scroll_filter = build_qdrant_filter(request.filters, settings=settings)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        if scroll_filter is None:
            raise HTTPException(400, "filters must contain at least one condition")
        count_result = await _qdrant(
            lambda: client.count(
                collection_name=collection_name,
                count_filter=scroll_filter,
                exact=True,
            ),
            "count filtered points",
            settings=settings,
        )
        matched = count_result.count
        await _qdrant(
            lambda: client.delete(
                collection_name=collection_name,
                points_selector=FilterSelector(filter=scroll_filter),
                wait=True,
            ),
            "delete by filter",
            settings=settings,
        )
    http_request.state.log_points = matched
    return BulkDeleteResponse(ok=True, deleted=matched, collection_name=collection_name)


class PayloadUpdateResponse(BaseModel):
    """Response confirming a payload patch or replace."""

    ok: bool
    point_id: str
    collection_name: str


@router.patch("/{point_id}/payload", response_model=PayloadUpdateResponse)
async def patch_payload(
    point_id: str,
    request: PayloadUpdateRequest,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    coordinator: Annotated[MutationCoordinator, Depends(get_mutation_coordinator)],
    manager: Annotated[VectorReindexManager, Depends(get_vector_reindex_manager)],
):
    """Merge the given fields into an existing point's payload (404 if missing)."""
    _reject_payload_text(request.payload)
    collection_name = resolve_collection_name(request.collection_name, settings)
    http_request.state.log_collection = collection_name
    async with mutation_scope(client, collection_name, settings, coordinator, manager):
        pid, _existing = await _retrieve_point_or_404(
            client, collection_name, point_id, settings, with_payload=False
        )
        await _qdrant(
            lambda: client.set_payload(
                collection_name=collection_name,
                payload=request.payload,
                points=[pid],
                wait=True,
            ),
            "set payload",
            settings=settings,
        )
    return PayloadUpdateResponse(
        ok=True, point_id=point_id, collection_name=collection_name
    )


@router.put("/{point_id}/payload", response_model=PayloadUpdateResponse)
async def put_payload(
    point_id: str,
    request: PayloadUpdateRequest,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    coordinator: Annotated[MutationCoordinator, Depends(get_mutation_coordinator)],
    manager: Annotated[VectorReindexManager, Depends(get_vector_reindex_manager)],
):
    """Replace an existing point's payload, preserving protected service fields.

    Service-owned fields (``text``, ``source_id``) the caller omits are carried
    forward so a full replace cannot silently break retrieval. 404 if missing.
    """
    _reject_payload_text(request.payload)
    collection_name = resolve_collection_name(request.collection_name, settings)
    http_request.state.log_collection = collection_name
    async with mutation_scope(client, collection_name, settings, coordinator, manager):
        pid, existing = await _retrieve_point_or_404(
            client, collection_name, point_id, settings, with_payload=True
        )
        current_payload = existing[0].payload or {}
        merged = dict(request.payload)
        for field in PROTECTED_PAYLOAD_FIELDS:
            if field not in merged and field in current_payload:
                merged[field] = current_payload[field]
        await _qdrant(
            lambda: client.overwrite_payload(
                collection_name=collection_name,
                payload=merged,
                points=[pid],
                wait=True,
            ),
            "overwrite payload",
            settings=settings,
        )
    return PayloadUpdateResponse(
        ok=True, point_id=point_id, collection_name=collection_name
    )


@router.post("/{point_id}/payload/delete-keys", response_model=PayloadUpdateResponse)
async def delete_payload_keys(
    point_id: str,
    request: PayloadKeysDeleteRequest,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    coordinator: Annotated[MutationCoordinator, Depends(get_mutation_coordinator)],
    manager: Annotated[VectorReindexManager, Depends(get_vector_reindex_manager)],
):
    """Remove named keys from a point's payload (404 if the point is missing).

    Service-owned fields (``text``, ``source_id``) are protected: deleting them
    would break retrieval and reranking, so the request is rejected before any
    write.
    """
    try:
        _validate_payload_delete_keys(request.keys)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    collection_name = resolve_collection_name(request.collection_name, settings)
    http_request.state.log_collection = collection_name
    async with mutation_scope(client, collection_name, settings, coordinator, manager):
        pid, _existing = await _retrieve_point_or_404(
            client, collection_name, point_id, settings, with_payload=False
        )
        await _qdrant(
            lambda: client.delete_payload(
                collection_name=collection_name,
                keys=request.keys,
                points=[pid],
                wait=True,
            ),
            "delete payload keys",
            settings=settings,
        )
    return PayloadUpdateResponse(
        ok=True, point_id=point_id, collection_name=collection_name
    )


@router.post("/{point_id}/payload/clear", response_model=PayloadUpdateResponse)
async def clear_payload(
    point_id: str,
    request: ClearPayloadRequest,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    coordinator: Annotated[MutationCoordinator, Depends(get_mutation_coordinator)],
    manager: Annotated[VectorReindexManager, Depends(get_vector_reindex_manager)],
):
    """Clear a point's payload, keeping only protected service fields.

    A raw Qdrant clear wipes ``text`` and ``source_id`` too, which breaks
    retrieval; instead the payload is overwritten with just the protected
    fields that were present. 404 if the point does not exist.
    """
    collection_name = resolve_collection_name(request.collection_name, settings)
    http_request.state.log_collection = collection_name
    async with mutation_scope(client, collection_name, settings, coordinator, manager):
        pid, existing = await _retrieve_point_or_404(
            client, collection_name, point_id, settings, with_payload=True
        )
        current_payload = existing[0].payload or {}
        preserved = {
            field: current_payload[field]
            for field in PROTECTED_PAYLOAD_FIELDS
            if field in current_payload
        }
        await _qdrant(
            lambda: client.overwrite_payload(
                collection_name=collection_name,
                payload=preserved,
                points=[pid],
                wait=True,
            ),
            "clear payload",
            settings=settings,
        )
    return PayloadUpdateResponse(
        ok=True, point_id=point_id, collection_name=collection_name
    )


async def _resolve_payload_filter(
    filters: dict[str, Any],
    collection_name: str,
    client: AsyncQdrantClient,
    settings: Settings,
) -> tuple[Filter, int]:
    """Compile a filter for a bulk payload write and count what it matches.

    Shared by the three filtered payload endpoints: refuses a filter that
    resolves to no condition (it would rewrite every
    point), and returns the matched count so the response can report scope.
    """
    try:
        payload_filter = build_qdrant_filter(filters, settings=settings)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    if payload_filter is None:
        raise HTTPException(400, "filters must contain at least one condition")
    count_result = await _qdrant(
        lambda: client.count(
            collection_name=collection_name,
            count_filter=payload_filter,
            exact=True,
        ),
        "count filtered points",
        settings=settings,
    )
    return payload_filter, count_result.count


@router.patch("/payload", response_model=PayloadFilterUpdateResponse)
async def patch_payload_by_filter(
    request: PayloadFilterUpdateRequest,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    coordinator: Annotated[MutationCoordinator, Depends(get_mutation_coordinator)],
    manager: Annotated[VectorReindexManager, Depends(get_vector_reindex_manager)],
):
    """Merge a payload into every point matching a filter.

    The bulk counterpart of ``PATCH /points/{point_id}/payload``: same merge
    semantics, applied server-side in one call. There is deliberately no
    filtered REPLACE — a full replace preserves each point's own ``text`` and
    ``source_id``, which a single write cannot do across many points.
    """
    _reject_payload_text(request.payload)
    collection_name = resolve_collection_name(request.collection_name, settings)
    http_request.state.log_collection = collection_name
    async with mutation_scope(client, collection_name, settings, coordinator, manager):
        payload_filter, matched = await _resolve_payload_filter(
            request.filters, collection_name, client, settings
        )
        await _qdrant(
            lambda: client.set_payload(
                collection_name=collection_name,
                payload=request.payload,
                points=FilterSelector(filter=payload_filter),
                wait=True,
            ),
            "set payload by filter",
            settings=settings,
        )
    http_request.state.log_points = matched
    return PayloadFilterUpdateResponse(
        ok=True, matched=matched, collection_name=collection_name
    )


@router.post("/payload/delete-keys", response_model=PayloadFilterUpdateResponse)
async def delete_payload_keys_by_filter(
    request: PayloadFilterKeysDeleteRequest,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    coordinator: Annotated[MutationCoordinator, Depends(get_mutation_coordinator)],
    manager: Annotated[VectorReindexManager, Depends(get_vector_reindex_manager)],
):
    """Remove named keys from every point matching a filter.

    Service-owned fields (``text``, ``source_id``) stay protected exactly as in
    the single-point endpoint: the request is rejected before any write.
    """
    try:
        _validate_payload_delete_keys(request.keys)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    collection_name = resolve_collection_name(request.collection_name, settings)
    http_request.state.log_collection = collection_name
    async with mutation_scope(client, collection_name, settings, coordinator, manager):
        payload_filter, matched = await _resolve_payload_filter(
            request.filters, collection_name, client, settings
        )
        await _qdrant(
            lambda: client.delete_payload(
                collection_name=collection_name,
                keys=request.keys,
                points=FilterSelector(filter=payload_filter),
                wait=True,
            ),
            "delete payload keys by filter",
            settings=settings,
        )
    http_request.state.log_points = matched
    return PayloadFilterUpdateResponse(
        ok=True, matched=matched, collection_name=collection_name
    )


@router.post("/payload/clear", response_model=PayloadFilterUpdateResponse)
async def clear_payload_by_filter(
    request: PayloadFilterClearRequest,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    coordinator: Annotated[MutationCoordinator, Depends(get_mutation_coordinator)],
    manager: Annotated[VectorReindexManager, Depends(get_vector_reindex_manager)],
):
    """Clear the payload of every point matching a filter, keeping protected fields.

    A raw Qdrant clear would wipe ``text`` and ``source_id`` too, so instead the
    non-protected keys observed on the collection are deleted by filter. Keys are
    taken from a payload sample, so a field absent from the sample survives; the
    response reports how many points matched.
    """
    collection_name = resolve_collection_name(request.collection_name, settings)
    http_request.state.log_collection = collection_name
    async with mutation_scope(client, collection_name, settings, coordinator, manager):
        payload_filter, matched = await _resolve_payload_filter(
            request.filters, collection_name, client, settings
        )
        sampled_keys, _sampled = await sample_payload_keys(
            client,
            collection_name,
            _CLEAR_PAYLOAD_KEY_SAMPLE,
            settings=settings,
        )
        keys = [key for key in sampled_keys if key not in PROTECTED_PAYLOAD_FIELDS]
        if keys:
            await _qdrant(
                lambda: client.delete_payload(
                    collection_name=collection_name,
                    keys=keys,
                    points=FilterSelector(filter=payload_filter),
                    wait=True,
                ),
                "clear payload by filter",
                settings=settings,
            )
    http_request.state.log_points = matched
    return PayloadFilterUpdateResponse(
        ok=True, matched=matched, collection_name=collection_name
    )

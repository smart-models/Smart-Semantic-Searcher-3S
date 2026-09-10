"""Pydantic schemas returned by collection management endpoints."""

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from qdrant_middleware.core.config import Settings
from qdrant_middleware.schemas.embedding_contract import EmbeddingCompatibility
from qdrant_middleware.services.payload_paths import canonical_payload_index_field


class SchemaFinding(BaseModel):
    """Single schema validation mismatch detected on a live collection."""

    category: str
    field: str
    expected: Any = None
    actual: Any = None
    severity: Literal["critical", "high", "medium", "low"]
    message: str


class SuggestedFix(BaseModel):
    """Human-readable remediation option for schema drift."""

    type: str
    destructive: bool
    auto_fixable: bool
    impact: str
    description: str
    missing_indexes: list[str] | None = None
    warning: str | None = None
    endpoint: str | None = None
    method: str | None = None
    path_params: dict[str, Any] | None = None
    body: dict[str, Any] | None = None


@dataclass(frozen=True)
class CollectionSchema:
    """Per-request overrides for the optional collection schema toggles.

    Use :meth:`from_settings` to construct with the app's runtime defaults;
    pass explicit values to override.
    """

    dense_on_disk: bool | None = None
    quantization: bool | None = None
    sparse_idf: bool | None = None
    sparse_on_disk: bool | None = None
    on_disk_payload: bool | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "CollectionSchema":
        """Create schema expectations from runtime settings."""
        return cls(
            dense_on_disk=settings.dense_on_disk,
            quantization=settings.quantization_enabled,
            sparse_idf=settings.sparse_idf,
            sparse_on_disk=settings.sparse_on_disk,
            on_disk_payload=settings.on_disk_payload,
        )


class CollectionReport(BaseModel):
    """Collection existence and schema compatibility report."""

    collection_name: str
    exists: bool
    # None when the report was produced with validate=false: no schema check ran.
    schema_ok: bool | None
    errors: list[str] = Field(default_factory=list)
    findings: list[SchemaFinding] = Field(default_factory=list)
    suggested_fixes: list[SuggestedFix] = Field(default_factory=list)
    status: str | None = None
    points_count: int | None = None
    vectors_count: int | None = None
    dense: dict[str, Any] | None = None
    sparse: dict[str, Any] | None = None
    quantization: str
    shards: int | None = None
    replicas: int | None = None
    on_disk_payload: bool | None = None
    embedding_compatibility: EmbeddingCompatibility | None = None


PayloadIndexSchemaName = Literal[
    "keyword", "text", "integer", "float", "bool", "geo", "datetime", "uuid"
]


class PayloadIndexRequest(BaseModel):
    """Single payload index requested by API clients."""

    field_name: str
    field_schema: PayloadIndexSchemaName
    phrase_matching: bool = False

    @field_validator("field_name")
    @classmethod
    def validate_field_name(cls, value: str) -> str:
        """Strip whitespace and reject empty payload field names."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("field_name cannot be empty")
        return stripped

    @model_validator(mode="after")
    def validate_phrase_matching(self) -> "PayloadIndexRequest":
        """phrase_matching is a text-index feature only."""
        if self.phrase_matching and self.field_schema != "text":
            raise ValueError("phrase_matching requires field_schema 'text'")
        return self


class PayloadIndexCreateRequest(BaseModel):
    """Request body for creating one or more payload indexes."""

    indexes: list[PayloadIndexRequest]

    @field_validator("indexes")
    @classmethod
    def validate_indexes(
        cls, value: list[PayloadIndexRequest]
    ) -> list[PayloadIndexRequest]:
        """Reject empty index lists and duplicate field names."""
        if not value:
            raise ValueError("indexes cannot be empty")
        seen: set[str] = set()
        for index in value:
            normalized = canonical_payload_index_field(index.field_name).casefold()
            if normalized in seen:
                raise ValueError(f"duplicate payload index field: {index.field_name}")
            seen.add(normalized)
        return value


class PayloadIndexCreateResponse(BaseModel):
    """Response returned after user-managed payload index creation.

    ``updated`` lists fields whose existing index had a different schema type
    and was dropped and recreated with the requested one.
    """

    ok: bool = True
    collection_name: str
    created: list[str] = Field(default_factory=list)
    updated: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)


class PayloadIndexInfo(BaseModel):
    """Single payload index reported by Qdrant."""

    field_name: str
    field_schema: str
    phrase_matching: bool | None = None


class PayloadIndexListResponse(BaseModel):
    """Response returned when listing payload indexes for a collection."""

    ok: bool = True
    collection_name: str
    indexes: list[PayloadIndexInfo] = Field(default_factory=list)
    count: int


class PayloadIndexDeleteResponse(BaseModel):
    """Response returned after deleting a user-managed payload index."""

    ok: bool = True
    collection_name: str
    field_name: str
    deleted: bool = True


class PayloadKeysResponse(BaseModel):
    """Payload keys observed on a sample of a collection's points."""

    ok: bool = True
    collection_name: str
    keys: list[str] = Field(default_factory=list)
    count: int
    sampled_points: int

"""Request and response models for collection vector regeneration."""

from pydantic import BaseModel, Field, field_validator

from qdrant_middleware.schemas.embedding_contract import (
    ActiveVectorNames,
    EmbeddingContract,
    ReindexJobState,
)


class ReindexPreflightResponse(BaseModel):
    """Read-only validation result produced before a reindex job starts."""

    collection_name: str
    can_start: bool
    qdrant_version: str | None = None
    source_contract: EmbeddingContract | None = None
    target_contract: EmbeddingContract
    source_vectors: ActiveVectorNames
    total_points: int = Field(ge=0)
    eligible_points: int = Field(ge=0)
    invalid_points: int = Field(ge=0)
    invalid_point_ids: list[str] = Field(default_factory=list)
    payload_field: str = "text"
    blockers: list[str] = Field(default_factory=list)


class ReindexStartRequest(BaseModel):
    """Exact-name confirmation required before starting regeneration."""

    confirm_collection: str

    @field_validator("confirm_collection")
    @classmethod
    def validate_confirmation(cls, value: str) -> str:
        """Reject blank confirmations while preserving exact comparison."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("confirm_collection cannot be empty")
        return value


class ReindexActionResponse(BaseModel):
    """Current persisted state after a reindex control action.

    ``job`` is null only after a cancel that had to reset an unreadable
    metadata namespace: there is no job state left to report.
    """

    ok: bool = True
    collection_name: str
    job: ReindexJobState | None = None
    task_running: bool = False
    metadata_invalid: bool = False


class ReindexStatusResponse(BaseModel):
    """Current persisted and local execution state for a collection."""

    collection_name: str
    job: ReindexJobState | None = None
    task_running: bool = False
    # True when the metadata namespace cannot be parsed: no job can be reported
    # and a cancel is the way to clear it.
    metadata_invalid: bool = False

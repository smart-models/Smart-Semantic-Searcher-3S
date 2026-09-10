"""Embedding-space compatibility models shared by collection operations."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DenseEmbeddingContract(BaseModel):
    """Dense-vector properties that define collection compatibility."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    revision: str | None = None
    dimension: int = Field(gt=0)
    normalization: str = Field(min_length=1)


class SparseEmbeddingContract(BaseModel):
    """Sparse-vector properties that define collection compatibility."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    revision: str | None = None
    encoding: str = Field(min_length=1)


class EmbeddingContract(BaseModel):
    """Canonical embedding contract emitted by Smart Embedder."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(gt=0)
    profile: str = Field(min_length=1)
    dense: DenseEmbeddingContract
    sparse: SparseEmbeddingContract
    fingerprint: str

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        """Require the versioned SHA-256 representation used by the server."""
        prefix = "sha256:"
        digest = value.removeprefix(prefix)
        if not value.startswith(prefix) or len(digest) != 64:
            raise ValueError("fingerprint must be sha256:<64 lowercase hex chars>")
        if any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("fingerprint digest must be lowercase hexadecimal")
        return value


class EmbedderContractInfo(BaseModel):
    """Runtime contract plus non-blocking server diagnostics."""

    contract: EmbeddingContract
    server_version: str | None = None
    reranker: str | None = None


class ActiveVectorNames(BaseModel):
    """Dense and sparse names used by the current embedding generation."""

    model_config = ConfigDict(extra="forbid")

    dense: str = Field(min_length=1)
    sparse: str = Field(min_length=1)

    @field_validator("dense", "sparse")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Reject whitespace-only names and normalize surrounding whitespace."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("vector name cannot be empty")
        return stripped

    @model_validator(mode="after")
    def validate_distinct_names(self) -> "ActiveVectorNames":
        """Dense and sparse vector names must be different in Qdrant."""
        if self.dense == self.sparse:
            raise ValueError("dense and sparse vector names must be different")
        return self


class ReindexJobState(BaseModel):
    """Persisted, resumable state for one collection re-embedding job."""

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1)
    status: Literal[
        "running",
        "paused",
        "interrupted",
        "finalizing",
        "completed",
        "failed",
        "cancelled",
    ]
    source_contract: EmbeddingContract | None = None
    target_contract: EmbeddingContract
    source_vectors: ActiveVectorNames
    target_vectors: ActiveVectorNames
    payload_field: str = "text"
    total_points: int = Field(ge=0)
    processed_points: int = Field(ge=0)
    last_offset: int | str | None = None
    started_at: str
    updated_at: str
    error: str | None = None

    @model_validator(mode="after")
    def validate_progress(self) -> "ReindexJobState":
        """Reject impossible progress counters."""
        if self.processed_points > self.total_points:
            raise ValueError("processed_points cannot exceed total_points")
        return self


class EmbeddingCompatibility(BaseModel):
    """Comparison between a collection contract and the active embedder."""

    status: Literal["compatible", "incompatible"]
    stored: EmbeddingContract | None = None
    active: EmbeddingContract | None = None
    active_vectors: ActiveVectorNames | None = None
    reindex_job: ReindexJobState | None = None
    message: str

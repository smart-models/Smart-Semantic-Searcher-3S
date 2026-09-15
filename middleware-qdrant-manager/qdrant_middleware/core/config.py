"""Typed application settings loaded from environment and .env."""

import ipaddress
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field, PrivateAttr, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
QDRANT_ROOT = PACKAGE_ROOT
WORKSPACE_ROOT = QDRANT_ROOT.parent
QDRANT_ENV_FILE = WORKSPACE_ROOT / ".env"

# Deliberately module constants rather than Settings fields. pydantic-settings
# populates any field lacking an explicit alias from the environment under its
# own upper-cased name, so as fields these two would be operator-tunable:
#   - LOOPBACK_HOSTS classifies a QDRANT_API_HOST value as loopback or not. As
#     a field, LOOPBACK_HOSTS='["0.0.0.0"]' would let any future bind-host rule
#     built on it disable itself.
#   - INGEST_DATA_ROOT is the sandbox root that file-based ingest is confined
#     to; the whole point of the sandbox is that it cannot be relocated.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
INGEST_DATA_ROOT = QDRANT_ROOT / "data"
# The only non-loopback addresses MIDDLEWARE_BIND_HOST may publish on (RFC 1918),
# a constant for the same reason as LOOPBACK_HOSTS.
PRIVATE_IPV4_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
)

LABEL_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
RESERVED_LABEL_PREFIX = "@"
SERVICE_TOKEN_LABEL = "@mcp"
REMOVED_ENV_NAMES = {
    "MIDDLEWARE_API_TOKEN": (
        "MIDDLEWARE_API_TOKENS (inbound list) or MIDDLEWARE_CLIENT_TOKEN "
        "(the credential the MCP server presents)"
    ),
    "SMART_EMBEDDER_API_TOKEN": "SMART_EMBEDDER_CLIENT_TOKEN",
    "ALLOW_UNAUTHENTICATED_LOCAL": "MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL",
}


def parse_api_tokens(raw: str, variable: str) -> dict[str, str]:
    """Parse ``token:label,token:label`` into a mapping of token to label.

    Args:
        raw: The raw environment value.
        variable: Name of the variable being parsed, used in error messages.

    Returns:
        Every token mapped to its audit label, in declaration order.

    Raises:
        ValueError: On a malformed entry, an out-of-charset label, a reserved
            label prefix, or a duplicate token. Never echoes the token value.
    """
    tokens: dict[str, str] = {}
    for entry in (part.strip() for part in raw.split(",")):
        if not entry:
            continue
        token, separator, label = (piece.strip() for piece in entry.partition(":"))
        if not separator or not token or not label:
            raise ValueError(f"{variable} has an entry that is not token:label")
        if label.startswith(RESERVED_LABEL_PREFIX):
            raise ValueError(
                f"{variable} label {label!r} must not start with "
                f"{RESERVED_LABEL_PREFIX!r}; that prefix is reserved for the "
                "credentials services present to each other"
            )
        if not LABEL_PATTERN.match(label):
            raise ValueError(
                f"{variable} label {label!r} must match [A-Za-z0-9._-]+. A label "
                "containing ':' usually means the token itself contains ':', "
                "which this format cannot express"
            )
        if token in tokens:
            raise ValueError(f"{variable} contains a duplicate token")
        tokens[token] = label
    return tokens


def _names_in_env_file(env_file: Path) -> set[str]:
    """Return the removed variable names assigned in ``env_file``."""
    if not env_file.is_file():
        return set()
    found: set[str] = set()
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()
        name, separator, value = stripped.partition("=")
        # An empty assignment is ignored, so `MIDDLEWARE_API_TOKEN=` left
        # behind in a migrated .env is not an error. check_token_config in
        # scripts/verify_token_config.py tests the value the same way; the
        # two must agree or the preflight and the boot disagree on one file.
        if separator and value.strip() and name.strip() in REMOVED_ENV_NAMES:
            found.add(name.strip())
    return found


def reject_removed_env_names(env_file: Path | None) -> None:
    """Refuse to start when a variable removed by the multi-token change is set.

    Args:
        env_file: The ``.env`` file this process would read, or ``None`` to scan
            the process environment only.

    Raises:
        ValueError: If any removed name is present, naming its replacement.
    """
    found = {name for name in REMOVED_ENV_NAMES if os.environ.get(name, "").strip()}
    if env_file is not None:
        found |= _names_in_env_file(env_file)
    if found:
        details = "; ".join(
            f"{name} was removed -- use {REMOVED_ENV_NAMES[name]}"
            for name in sorted(found)
        )
        raise ValueError(details)


class Settings(BaseSettings):
    """Runtime configuration for the Qdrant middleware FastAPI app."""

    # populate_by_name keeps field-name keyword construction working now that
    # every environment-settable field carries an explicit alias; tests build
    # Settings both ways (MIDDLEWARE_API_TOKEN=... alongside sparse_idf=...).
    model_config = SettingsConfigDict(
        env_file=QDRANT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
        case_sensitive=False,
        hide_input_in_errors=True,
    )

    qdrant_url: AnyHttpUrl = Field("http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: SecretStr | None = Field(None, alias="QDRANT_API_KEY")
    collection_name: str = Field("", alias="QDRANT_COLLECTION")
    client_timeout: int = Field(30, alias="QDRANT_TIMEOUT", ge=1, le=3600)

    api_host: str = Field("127.0.0.1", alias="QDRANT_API_HOST")
    api_port: int = Field(6336, alias="MIDDLEWARE_PORT", ge=1, le=65535)
    # The host address Compose publishes the port on. It never reaches a
    # socket (QDRANT_API_HOST does); it decides whether serving without
    # authentication is allowed.
    bind_host: str = Field("127.0.0.1", alias="MIDDLEWARE_BIND_HOST")
    api_tokens_raw: str = Field("", alias="MIDDLEWARE_API_TOKENS")
    # The credential the MCP server presents, delivered by compose so the
    # operator sets one value that works at both ends. Never listed in
    # MIDDLEWARE_API_TOKENS.
    service_token: SecretStr | None = Field(None, alias="SERVICE_TOKEN")
    allow_unauthenticated_local: bool = Field(
        False, alias="MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL"
    )
    expose_frontend: bool = Field(True, alias="EXPOSE_FRONTEND")
    expose_middleware_swagger: bool = Field(True, alias="EXPOSE_MIDDLEWARE_SWAGGER")
    request_timeout: int = Field(30, alias="MIDDLEWARE_REQUEST_TIMEOUT", ge=1, le=3600)
    request_body_max_bytes: int = Field(
        26_214_400, alias="REQUEST_BODY_MAX_BYTES", ge=1, le=1_000_000_000
    )
    rate_limit_requests_per_minute: int = Field(
        300,
        alias="MIDDLEWARE_RATE_LIMIT_REQUESTS_PER_MINUTE",
        ge=1,
        le=1_000_000,
    )
    rate_limit_burst_size: int = Field(
        180,
        alias="MIDDLEWARE_RATE_LIMIT_BURST_SIZE",
        ge=1,
        le=1_000_000,
    )

    smart_embedder_url: AnyHttpUrl = Field(
        "http://localhost:8010", alias="SMART_EMBEDDER_URL"
    )
    smart_embedder_client_token: SecretStr | None = Field(
        None, alias="SMART_EMBEDDER_CLIENT_TOKEN"
    )
    smart_embedder_timeout: int = Field(
        60, alias="SMART_EMBEDDER_TIMEOUT", ge=1, le=3600
    )
    smart_embedder_max_batch_size: int = Field(
        32, alias="SMART_EMBEDDER_MAX_BATCH_SIZE", ge=1, le=128
    )
    smart_embedder_rerank_batch_size: int = Field(
        32, alias="SMART_EMBEDDER_RERANK_BATCH_SIZE", ge=1, le=128
    )
    smart_embedder_embed_concurrency: int = Field(
        4, alias="SMART_EMBEDDER_EMBED_CONCURRENCY", ge=1, le=16
    )

    query_default_limit: int = Field(10, alias="QUERY_DEFAULT_LIMIT", ge=1, le=1000)
    query_max_limit: int = Field(50, alias="QUERY_MAX_LIMIT", ge=1, le=1000)
    query_default_candidate_limit: int = Field(
        50, alias="QUERY_DEFAULT_CANDIDATE_LIMIT", ge=1, le=10000
    )
    query_max_candidate_limit: int = Field(
        200, alias="QUERY_MAX_CANDIDATE_LIMIT", ge=1, le=10000
    )
    query_default_rerank: bool = Field(True, alias="QUERY_DEFAULT_RERANK")
    query_max_rerank_top_k: int = Field(
        200, alias="QUERY_MAX_RERANK_TOP_K", ge=1, le=1024
    )
    query_default_rerank_top_k: int = Field(
        20, alias="QUERY_DEFAULT_RERANK_TOP_K", ge=1
    )
    query_rerank_max_chars: int = Field(
        1000, alias="QUERY_RERANK_MAX_CHARS", ge=0, le=200000
    )
    query_default_fusion: Literal["rrf", "dbsf"] = Field(
        "rrf", alias="QUERY_DEFAULT_FUSION"
    )
    query_indexed_only: bool = Field(False, alias="QUERY_INDEXED_ONLY")
    query_hnsw_ef: int | None = Field(None, alias="QUERY_HNSW_EF", ge=1)
    query_quantization_rescore: bool = Field(True, alias="QUERY_QUANTIZATION_RESCORE")
    query_quantization_oversampling: float = Field(
        2.0, alias="QUERY_QUANTIZATION_OVERSAMPLING", ge=1.0, le=10.0
    )
    query_filter_field_allowlist_raw: str = Field(
        "", alias="QUERY_FILTER_FIELD_ALLOWLIST"
    )
    query_filter_max_depth: int = Field(
        20, alias="QUERY_FILTER_MAX_DEPTH", ge=1, le=100
    )

    # The effective inbound ACL, computed once in validate_startup so every
    # request handler reads a settled mapping rather than re-parsing a string.
    _api_tokens: dict[str, str] = PrivateAttr(default_factory=dict)
    query_filter_max_conditions: int = Field(
        100, alias="QUERY_FILTER_MAX_CONDITIONS", ge=1, le=1000
    )

    # Defaults applied when the middleware creates a collection. Every one of
    # these was already environment-settable under its upper-cased field name,
    # so the aliases below name existing behaviour rather than opening anything
    # new -- they make it explicit and reviewable. Documented in .env.example
    # section 20.
    dense_vector_name: str = Field("dense", alias="DENSE_VECTOR_NAME", min_length=1)
    sparse_vector_name: str = Field("sparse", alias="SPARSE_VECTOR_NAME", min_length=1)
    dense_dim: int = Field(1024, alias="DENSE_DIM", ge=1)
    hnsw_m: int = Field(16, alias="HNSW_M", ge=0)
    hnsw_ef_construct: int = Field(200, alias="HNSW_EF_CONSTRUCT", ge=1)
    hnsw_full_scan_threshold: int = Field(
        10_000, alias="HNSW_FULL_SCAN_THRESHOLD", ge=0
    )
    sparse_on_disk: bool = Field(True, alias="SPARSE_ON_DISK")
    sparse_full_scan_threshold: int = Field(
        20_000, alias="SPARSE_FULL_SCAN_THRESHOLD", ge=0
    )
    sparse_idf: bool = Field(True, alias="SPARSE_IDF")
    dense_on_disk: bool = Field(True, alias="DENSE_ON_DISK")
    on_disk_payload: bool = Field(True, alias="QDRANT_ON_DISK_PAYLOAD")
    quantization_enabled: bool = Field(True, alias="QUANTIZATION_ENABLED")
    quantization_always_ram: bool = Field(True, alias="QUANTIZATION_ALWAYS_RAM")
    indexing_threshold: int = Field(20_000, alias="INDEXING_THRESHOLD", ge=0)
    memmap_threshold: int = Field(20_000, alias="MEMMAP_THRESHOLD", ge=0)
    shard_number: int = Field(1, alias="SHARD_NUMBER", ge=1)
    replication_factor: int = Field(1, alias="REPLICATION_FACTOR", ge=1)
    hnsw_inline_storage: bool = Field(True, alias="HNSW_INLINE_STORAGE")
    dense_float16: bool = Field(True, alias="DENSE_FLOAT16")
    prevent_unoptimized: bool = Field(True, alias="PREVENT_UNOPTIMIZED")
    sparse_float16: bool = Field(True, alias="QDRANT_SPARSE_FLOAT16")

    ingest_max_file_bytes: int = Field(
        67_108_864, alias="INGEST_MAX_FILE_BYTES", ge=1, le=10_000_000_000
    )
    # The bound that actually protects the process. Ingest embeds every chunk
    # before its first write, so one request holds every dense and sparse
    # vector in memory at once -- roughly 52 KB per point at DENSE_DIM=1024,
    # because a Python list[float] stores boxed floats rather than a packed
    # array. Chunk count, not file size, is what that peak scales with: a
    # generous byte limit full of small chunks exhausts the container long
    # before the byte limit is reached. 10000 matches the max_length the
    # sibling /points/batch endpoint already enforces on its own point list.
    ingest_max_chunks: int = Field(
        10_000, alias="INGEST_MAX_CHUNKS", ge=1, le=1_000_000
    )
    ingest_default_batch_size: int = Field(
        64, alias="INGEST_DEFAULT_BATCH_SIZE", ge=1, le=256
    )
    ingest_max_batch_size: int = Field(
        256, alias="INGEST_MAX_BATCH_SIZE", ge=1, le=1024
    )
    ingest_upload_concurrency: int = Field(
        4, alias="INGEST_UPLOAD_CONCURRENCY", ge=1, le=16
    )

    export_data_root: Path = QDRANT_ROOT / "exports"
    export_max_points: int = Field(50_000, alias="EXPORT_MAX_POINTS", ge=1)
    export_max_bytes: int = Field(67_108_864, alias="EXPORT_MAX_BYTES", ge=1)
    export_max_concurrent: int = Field(1, alias="EXPORT_MAX_CONCURRENT", ge=1)

    health_probe_interval: float = Field(
        15.0, alias="HEALTH_PROBE_INTERVAL", ge=1.0, le=3600.0
    )
    embedder_probe_timeout: float = Field(
        3.0, alias="EMBEDDER_PROBE_TIMEOUT", ge=0.5, le=60.0
    )
    retry_max_attempts: int = Field(3, alias="RETRY_MAX_ATTEMPTS", ge=1, le=10)
    retry_base_delay: float = Field(0.1, alias="RETRY_BASE_DELAY", ge=0.0, le=10.0)

    @model_validator(mode="after")
    def validate_invariants(self) -> "Settings":
        """Enforce cross-field invariants after the model is populated."""
        if self.query_default_limit > self.query_max_limit:
            raise ValueError("QUERY_DEFAULT_LIMIT must be <= QUERY_MAX_LIMIT")
        if self.query_default_candidate_limit > self.query_max_candidate_limit:
            raise ValueError(
                "QUERY_DEFAULT_CANDIDATE_LIMIT must be <= QUERY_MAX_CANDIDATE_LIMIT"
            )
        if self.query_default_candidate_limit < self.query_default_limit:
            raise ValueError(
                "QUERY_DEFAULT_CANDIDATE_LIMIT must be >= QUERY_DEFAULT_LIMIT"
            )
        if self.query_max_candidate_limit < self.query_max_limit:
            raise ValueError("QUERY_MAX_CANDIDATE_LIMIT must be >= QUERY_MAX_LIMIT")
        if self.query_default_rerank_top_k > self.query_max_rerank_top_k:
            raise ValueError(
                "QUERY_DEFAULT_RERANK_TOP_K must be <= QUERY_MAX_RERANK_TOP_K"
            )
        if self.ingest_default_batch_size > self.ingest_max_batch_size:
            raise ValueError(
                "INGEST_DEFAULT_BATCH_SIZE must be <= INGEST_MAX_BATCH_SIZE"
            )
        try:
            bind_address = ipaddress.ip_address(self.bind_host.strip())
        except ValueError:
            bind_address = None
        if bind_address is None or not (
            bind_address.is_loopback
            or any(bind_address in network for network in PRIVATE_IPV4_NETWORKS)
        ):
            raise ValueError(
                "MIDDLEWARE_BIND_HOST must be a loopback or private RFC 1918 IPv4 "
                "literal; wildcard and public addresses and hostnames are refused"
            )
        tokens = parse_api_tokens(self.api_tokens_raw, "MIDDLEWARE_API_TOKENS")
        service = self.service_token.get_secret_value() if self.service_token else ""
        if service:
            if service in tokens:
                raise ValueError(
                    "SERVICE_TOKEN duplicates an entry in MIDDLEWARE_API_TOKENS"
                )
            tokens[service] = SERVICE_TOKEN_LABEL
        if not tokens and not self.allow_unauthenticated_local:
            raise ValueError(
                "MIDDLEWARE_API_TOKENS must define at least one token:label entry "
                "unless MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=true"
            )
        if not tokens and not bind_address.is_loopback:
            raise ValueError(
                "MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=true requires a loopback "
                "MIDDLEWARE_BIND_HOST; refusing to serve unauthenticated on "
                f"{self.bind_host}"
            )
        self._api_tokens = tokens
        return self

    def is_loopback_host(self, host: str) -> bool:
        """Return whether ``host`` resolves to a loopback address."""
        normalized = host.strip().lower()
        return normalized in LOOPBACK_HOSTS or normalized.startswith("127.")

    @property
    def ingest_data_root(self) -> Path:
        """Return the fixed sandbox root that file-based ingest may not escape."""
        return INGEST_DATA_ROOT

    @property
    def api_tokens(self) -> dict[str, str]:
        """Return the effective inbound ACL: operator entries plus the service token."""
        return dict(self._api_tokens)

    @property
    def qdrant_api_key_value(self) -> str | None:
        """Return the plaintext Qdrant API key, or ``None`` when unset."""
        return self.qdrant_api_key.get_secret_value() if self.qdrant_api_key else None

    @property
    def smart_embedder_client_token_value(self) -> str | None:
        """Return the credential presented to Smart Embedder, or ``None``."""
        return (
            self.smart_embedder_client_token.get_secret_value()
            if self.smart_embedder_client_token
            else None
        )

    @property
    def query_filter_field_allowlist(self) -> frozenset[str]:
        """Return configured payload fields allowed in query filters."""
        fields = {
            field.strip()
            for field in self.query_filter_field_allowlist_raw.split(",")
            if field.strip()
        }
        return frozenset(fields)

    @property
    def public_auth_paths(self) -> frozenset[str]:
        """Return paths exempt from bearer auth (docs included when public).

        Only liveness stays public on a private publish: the other probes report
        collections, models and counters, which only a loopback publish keeps local.
        """
        paths = {"/livez"}
        if ipaddress.ip_address(self.bind_host.strip()).is_loopback:
            paths.update({"/health", "/readyz", "/stats", "/metrics"})
        if self.expose_middleware_swagger:
            paths.update({"/docs", "/redoc", "/openapi.json"})
        return frozenset(paths)

    @property
    def public_auth_prefixes(self) -> tuple[str, ...]:
        """Return path prefixes exempt from bearer auth when docs are public."""
        return ("/docs/", "/redoc/") if self.expose_middleware_swagger else ()

    @property
    def timeout_exempt_paths(self) -> frozenset[str]:
        """Return paths excluded from the per-request timeout middleware.

        Ingest and export are both unbounded-by-design walks: ingest embeds
        and upserts every chunk, export scrolls every page. `EXPORT_MAX_POINTS`
        is what actually bounds the export, not the request clock.
        """
        return frozenset(
            {
                "/health",
                "/stats",
                "/metrics",
                "/points/ingest",
                "/points/ingest-chunks",
                "/points/export",
            }
        )


@lru_cache
def get_settings() -> Settings:
    """Return cached process-default settings.

    The removed-name scan runs before any Settings instance exists: a stale
    variable must abort the boot rather than be quietly ignored by the
    ``extra="ignore"`` policy.
    """
    reject_removed_env_names(QDRANT_ENV_FILE)
    return Settings()

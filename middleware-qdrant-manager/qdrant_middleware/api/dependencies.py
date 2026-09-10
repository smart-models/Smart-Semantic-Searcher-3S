"""FastAPI dependencies for app state, clients, and bearer authentication."""

import logging
import secrets
import threading
from collections.abc import AsyncIterator, Mapping
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from qdrant_client import AsyncQdrantClient

from qdrant_middleware.clients.embedder import EmbedderClient
from qdrant_middleware.core.config import Settings
from qdrant_middleware.core.mutation_coordinator import MutationCoordinator
from qdrant_middleware.middleware.metrics import record_rejection

logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)


def match_bearer_token(candidate: str, tokens: Mapping[str, str]) -> str | None:
    """Return the audit label of the entry matching ``candidate``, if any.

    Every entry is compared even after a hit, so the time taken does not reveal
    an entry's position in the list.

    ``secrets.compare_digest`` refuses ``str`` arguments that are not
    ASCII-only, raising ``TypeError`` rather than returning False. The
    candidate arrives from an unauthenticated HTTP header (ASGI decodes
    headers latin-1, so any byte >= 0x80 lands here as a non-ASCII character),
    so comparing the strings directly turns a plain authentication failure
    into a 500. Encoding both sides to UTF-8 keeps the comparison
    constant-time -- the bytes overload has no such restriction -- while
    making every possible input a verdict instead of a crash.

    Args:
        candidate: The token presented in the ``Authorization`` header.
        tokens: The effective inbound ACL, token to label.

    Returns:
        The matched label, or ``None`` when no entry matches.
    """
    presented = candidate.encode("utf-8")
    matched: str | None = None
    for known_token, label in tokens.items():
        if secrets.compare_digest(presented, known_token.encode("utf-8")):
            matched = label
    return matched


def get_request_settings(request: Request) -> Settings:
    """Return the per-app settings object stored by ``create_app``."""
    return request.app.state.settings


async def reserve_export_slot(request: Request) -> AsyncIterator[None]:
    """Admit an export without queuing another document in memory."""
    runtime = request.app.state.runtime
    if runtime.active_exports >= runtime.settings.export_max_concurrent:
        raise HTTPException(
            503,
            "Export capacity exhausted; retry after the active export finishes",
            headers={"Retry-After": "1"},
        )
    # No await separates the check and increment on this app's event loop.
    runtime.active_exports += 1
    try:
        yield
    finally:
        runtime.active_exports -= 1


def get_qdrant_client(request: Request) -> AsyncQdrantClient:
    """Return the lifespan-owned Qdrant client for the current app."""
    return request.app.state.qdrant_client


def get_mutation_coordinator(request: Request) -> MutationCoordinator:
    """Return the per-app coordinator that serializes collection mutations."""
    return request.app.state.runtime.mutations


def get_vector_reindex_manager(request: Request):
    """Return the app's reindex manager, creating its test fallback if needed."""
    from qdrant_middleware.services.vector_reindex import VectorReindexManager

    manager = getattr(request.app.state, "vector_reindex_manager", None)
    if manager is None:
        manager = VectorReindexManager(get_mutation_coordinator(request))
        request.app.state.vector_reindex_manager = manager
    return manager


def get_embedder_client(request: Request) -> EmbedderClient:
    """Return the lazily created embedder client bound to app settings."""
    state = request.app.state
    embedder = getattr(state, "embedder", None)
    if embedder is not None:
        return embedder

    lock = getattr(state, "embedder_lock", None)
    if lock is None:
        lock = threading.Lock()
        state.embedder_lock = lock

    with lock:
        embedder = getattr(state, "embedder", None)
        if embedder is None:
            embedder = EmbedderClient(state.settings)
            state.embedder = embedder
        return embedder


async def require_bearer_auth(
    request: Request,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Security(bearer_scheme)
    ],
    settings: Annotated[Settings, Depends(get_request_settings)],
) -> None:
    """Validate bearer auth unless the route is configured as public."""
    if request.url.path in settings.public_auth_paths:
        return
    if any(
        request.url.path.startswith(prefix) for prefix in settings.public_auth_prefixes
    ):
        return
    tokens = settings.api_tokens
    if not tokens:
        return
    label = (
        match_bearer_token(credentials.credentials, tokens)
        if credentials is not None
        else None
    )
    if label is None:
        runtime = getattr(request.app.state, "runtime", None)
        if runtime is not None:
            runtime.stats.rejected_requests += 1
        record_rejection("auth")
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    request.state.auth_label = label
    logger.info("request authenticated: %s", label)


def require_operational_readiness(request: Request) -> None:
    """Allow data operations only while Qdrant and Smart Embedder are reachable.

    The cached probe is populated before the application starts serving in the
    production lifespan. A missing probe is tolerated for isolated router
    tests that intentionally construct a bare application.
    """
    probe = getattr(request.app.state, "health_probe", None)
    if probe is None:
        return

    qdrant_reachable = bool(getattr(probe, "reachable", False))
    embedder_reachable = bool(getattr(probe, "embedder_reachable", False))
    if qdrant_reachable and embedder_reachable:
        return

    unavailable = []
    if not qdrant_reachable:
        unavailable.append("Qdrant")
    if not embedder_reachable:
        unavailable.append("Smart Embedder")
    raise HTTPException(
        status_code=503,
        detail={
            "code": "dependencies_unavailable",
            "message": (
                "Operations are disabled until Qdrant and Smart Embedder "
                "are both reachable."
            ),
            "unavailable": unavailable,
        },
    )

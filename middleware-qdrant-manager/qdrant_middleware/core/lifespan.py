"""FastAPI lifespan wiring for clients, probes, and shutdown cleanup."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from qdrant_middleware.clients.embedder import EmbedderClient
from qdrant_middleware.middleware.metrics import server_info
from qdrant_middleware.repositories.qdrant_collections import (
    align_strict_mode_configs,
)
from qdrant_middleware.services.health_probe import HealthProbe
from qdrant_middleware.services.qdrant_client import create_qdrant_client

log = logging.getLogger(__name__)


async def _align_existing_strict_mode(client, settings) -> None:
    """Best-effort strict-mode alignment: startup must survive Qdrant being down."""
    try:
        updated = await align_strict_mode_configs(client, settings)
    except Exception as e:
        log.warning("Strict-mode alignment skipped: %s", e)
        return
    if updated:
        log.info("Strict-mode config updated for: %s", ", ".join(updated))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create runtime clients on startup and close them on shutdown."""
    runtime = app.state.runtime
    settings = app.state.settings
    runtime.is_shutting_down = False
    # Log the effective inbound ACL once, by label only, so an operator can see
    # that the auto-accepted service credential is present without any token
    # value ever reaching a log line.
    labels = sorted(settings.api_tokens.values())
    log.info(
        "inbound auth: %s",
        ", ".join(labels) if labels else "disabled (no tokens configured)",
    )
    app.state.qdrant_client = create_qdrant_client(settings)
    await _align_existing_strict_mode(app.state.qdrant_client, settings)
    if getattr(app.state, "embedder", None) is None:
        app.state.embedder = EmbedderClient(settings)
    runtime.rate_limiter.start_cleanup_task()
    server_info.info({"version": "1.0.0", "collection": settings.collection_name})
    app.state.health_probe = HealthProbe(
        app.state.qdrant_client,
        settings.health_probe_interval,
        embedder_url=str(settings.smart_embedder_url),
        probe_timeout=settings.embedder_probe_timeout,
        embedder_client_token=settings.smart_embedder_client_token_value,
    )
    await app.state.health_probe.probe_once()
    app.state.health_probe.start()
    log.info("Qdrant manager ready.")
    try:
        yield
    finally:
        runtime.is_shutting_down = True
        probe = getattr(app.state, "health_probe", None)
        if probe is not None:
            await probe.stop()
        reindex_manager = getattr(app.state, "vector_reindex_manager", None)
        if reindex_manager is not None:
            await reindex_manager.stop()
        await runtime.rate_limiter.stop_cleanup_task()
        app_embedder = getattr(app.state, "embedder", None)
        if app_embedder is not None:
            await app_embedder.close()
            app.state.embedder = None
        client = getattr(app.state, "qdrant_client", None)
        if client is not None:
            await client.close()
        app.state.qdrant_client = None

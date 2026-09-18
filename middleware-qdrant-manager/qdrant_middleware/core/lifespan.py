"""FastAPI lifespan wiring for clients, probes, and shutdown cleanup."""

import logging
from contextlib import AsyncExitStack, asynccontextmanager

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

    async def close_client(attribute, method):
        """Reset state even when a resource fails during cleanup."""
        resource = getattr(app.state, attribute, None)
        try:
            if resource is not None:
                await getattr(resource, method)()
        finally:
            setattr(app.state, attribute, None)

    async with AsyncExitStack() as cleanup:
        try:
            app.state.qdrant_client = create_qdrant_client(settings)
            cleanup.push_async_callback(close_client, "qdrant_client", "close")
            await _align_existing_strict_mode(app.state.qdrant_client, settings)
            if getattr(app.state, "embedder", None) is None:
                app.state.embedder = EmbedderClient(settings)
            cleanup.push_async_callback(close_client, "embedder", "close")
            cleanup.push_async_callback(runtime.rate_limiter.stop_cleanup_task)
            runtime.rate_limiter.start_cleanup_task()
            manager = getattr(app.state, "vector_reindex_manager", None)
            if manager is not None:
                cleanup.push_async_callback(manager.stop)
            server_info.info(
                {"version": "1.0.0", "collection": settings.collection_name}
            )
            app.state.health_probe = HealthProbe(
                app.state.qdrant_client,
                settings.health_probe_interval,
                embedder_url=str(settings.smart_embedder_url),
                probe_timeout=settings.embedder_probe_timeout,
                embedder_client_token=settings.smart_embedder_client_token_value,
            )
            cleanup.push_async_callback(app.state.health_probe.stop)
            await app.state.health_probe.probe_once()
            app.state.health_probe.start()
            log.info("Qdrant manager ready.")
            yield
        finally:
            runtime.is_shutting_down = True

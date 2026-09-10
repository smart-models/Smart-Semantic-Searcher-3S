"""Collection mutation scope shared by point and reindex workflows."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import HTTPException
from qdrant_client import AsyncQdrantClient

from qdrant_middleware.core.config import Settings
from qdrant_middleware.core.mutation_coordinator import MutationCoordinator
from qdrant_middleware.services.embedding_guard import guard_collection_mutation
from qdrant_middleware.services.vector_reindex import VectorReindexManager


@asynccontextmanager
async def mutation_scope(
    client: AsyncQdrantClient,
    collection_name: str,
    settings: Settings,
    coordinator: MutationCoordinator,
    manager: VectorReindexManager,
    *,
    allow_missing: bool = False,
    allow_invalid_metadata: bool = False,
) -> AsyncIterator[None]:
    """Lease a collection, then reject live or persisted reindex activity."""
    async with coordinator.lease(collection_name):
        if manager.is_running(collection_name):
            raise HTTPException(
                409,
                detail={
                    "code": "collection_mutation_blocked",
                    "status": "reindex_in_progress",
                    "message": "Collection vectors are being regenerated.",
                    "collection_name": collection_name,
                },
            )
        try:
            await guard_collection_mutation(
                client,
                collection_name,
                settings,
                allow_invalid_metadata=allow_invalid_metadata,
            )
        except HTTPException as exc:
            if not (allow_missing and exc.status_code == 404):
                raise
        yield

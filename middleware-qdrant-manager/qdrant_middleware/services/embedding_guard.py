"""Guards that keep Qdrant collections in one embedding vector space."""

from fastapi import HTTPException
from qdrant_client import AsyncQdrantClient

from qdrant_middleware.clients.embedder import EmbeddingResult
from qdrant_middleware.core.config import Settings
from qdrant_middleware.repositories.qdrant_collections import (
    EmbeddingContractViolation,
    PayloadIndexCollectionNotFound,
    require_collection_mutable,
    require_embedding_fingerprint,
)
from qdrant_middleware.schemas.embedding_contract import ActiveVectorNames


def response_embedding_fingerprint(
    embeddings: list[EmbeddingResult],
) -> str | None:
    """Return one response fingerprint, rejecting mixed or partial batches."""
    fingerprints = [
        getattr(embedding, "contract_fingerprint", None) for embedding in embeddings
    ]
    present = {value for value in fingerprints if value}
    if not present:
        # Current Smart Embedder responses always carry a fingerprint. Keeping
        # the absence compatible supports existing third-party test doubles.
        return None
    if len(present) != 1 or any(value is None for value in fingerprints):
        raise HTTPException(
            502, "Smart Embedder returned inconsistent contract fingerprints"
        )
    return present.pop()


async def guard_vector_operation(
    client: AsyncQdrantClient,
    collection_name: str,
    embeddings: list[EmbeddingResult],
    settings: Settings,
    *,
    mutation: bool = True,
) -> ActiveVectorNames:
    """Validate the response and return the collection's active vector names."""
    fingerprint = response_embedding_fingerprint(embeddings)
    if fingerprint is None:
        if mutation:
            await guard_collection_mutation(client, collection_name, settings)
        return ActiveVectorNames(
            dense=settings.dense_vector_name,
            sparse=settings.sparse_vector_name,
        )
    try:
        return await require_embedding_fingerprint(
            client,
            collection_name,
            fingerprint,
            settings=settings,
            allow_during_reindex=not mutation,
        )
    except EmbeddingContractViolation as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "embedding_contract_mismatch",
                "status": exc.status,
                "message": exc.message,
                "collection_name": collection_name,
            },
        ) from exc
    except PayloadIndexCollectionNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail=f"collection '{collection_name}' not found",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            502, f"Qdrant embedding contract verification failed: {exc}"
        ) from exc


async def guard_collection_mutation(
    client: AsyncQdrantClient,
    collection_name: str,
    settings: Settings,
    *,
    allow_invalid_metadata: bool = False,
) -> None:
    """Map the persisted reindex write lock to a stable HTTP 409 response."""
    try:
        await require_collection_mutable(
            client,
            collection_name,
            settings=settings,
            allow_invalid_metadata=allow_invalid_metadata,
        )
    except EmbeddingContractViolation as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "collection_mutation_blocked",
                "status": exc.status,
                "message": exc.message,
                "collection_name": collection_name,
            },
        ) from exc
    except PayloadIndexCollectionNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail=f"collection '{collection_name}' not found",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            502, f"Qdrant collection mutation verification failed: {exc}"
        ) from exc

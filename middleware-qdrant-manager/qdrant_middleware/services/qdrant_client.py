"""Factory for the lifespan-owned AsyncQdrantClient."""

from qdrant_client import AsyncQdrantClient

from qdrant_middleware.core.config import Settings


def create_qdrant_client(settings: Settings) -> AsyncQdrantClient:
    """Create a Qdrant client from runtime settings."""
    return AsyncQdrantClient(
        url=str(settings.qdrant_url).rstrip("/"),
        api_key=settings.qdrant_api_key_value,
        timeout=settings.client_timeout,
    )

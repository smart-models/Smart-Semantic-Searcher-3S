"""Small shared helpers for FastAPI route modules."""

from fastapi import HTTPException

from qdrant_middleware.core.config import Settings


def resolve_collection_name(value: str | None, settings: Settings) -> str:
    """Resolve an optional collection name from request input or settings."""
    name = value if value is not None else settings.collection_name
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="collection_name cannot be empty")
    return name.strip()

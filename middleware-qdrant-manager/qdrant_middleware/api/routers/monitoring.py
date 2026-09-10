"""Operational health, statistics, and Prometheus endpoints."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter(tags=["MONITORING"])


def _dependencies_ready(probe) -> bool:
    """Return whether both external services needed for operations are ready."""
    return bool(
        getattr(probe, "reachable", False)
        and getattr(probe, "embedder_reachable", False)
    )


@router.get("/livez", status_code=200)
async def livez():
    """Return a liveness response when the process can serve requests."""
    return {"status": "alive"}


@router.get("/readyz")
async def readyz(request: Request):
    """Return readiness only when Qdrant and Smart Embedder are reachable."""
    probe = request.app.state.health_probe
    if _dependencies_ready(probe):
        return {
            "status": "ready",
            "qdrant": "reachable",
            "collections": probe.collections,
            **_embedder_fields(probe),
        }
    return JSONResponse(
        status_code=503,
        content={
            "status": "not_ready",
            "qdrant": "reachable"
            if getattr(probe, "reachable", False)
            else "unreachable",
            **_embedder_fields(probe),
        },
    )


def _embedder_fields(probe) -> dict:
    """Return cached embedder fields, tolerating probes without them."""
    reachable = getattr(probe, "embedder_reachable", None)
    if reachable is None:
        return {"embedder": "unknown"}
    fields = {"embedder": "reachable" if reachable else "unreachable"}
    profile = getattr(probe, "embedder_profile", None)
    if profile:
        fields["embedder_profile"] = profile
    models = getattr(probe, "embedder_models", None)
    if models:
        fields["embedder_models"] = models
    return fields


@router.get("/health")
async def health_check(request: Request):
    """Return combined service, Qdrant, and embedder health information."""
    probe = request.app.state.health_probe
    settings = request.app.state.settings
    if _dependencies_ready(probe):
        return {
            "status": "healthy",
            "qdrant": "reachable",
            "collections": probe.collections,
            "default_collection": settings.collection_name,
            **_embedder_fields(probe),
        }
    qdrant_reachable = bool(getattr(probe, "reachable", False))
    embedder_reachable = bool(getattr(probe, "embedder_reachable", False))
    unavailable = []
    if not qdrant_reachable:
        unavailable.append("Qdrant")
    if not embedder_reachable:
        unavailable.append("Smart Embedder")
    return JSONResponse(
        status_code=503,
        content={
            "status": "unhealthy",
            "qdrant": "reachable" if qdrant_reachable else "unreachable",
            "detail": f"{' and '.join(unavailable)} unreachable.",
            **_embedder_fields(probe),
        },
    )


@router.get("/stats", status_code=200)
async def server_stats(request: Request):
    """Return in-memory request counters and uptime."""
    return request.app.state.runtime.stats.get_stats()


@router.get("/metrics")
async def metrics():
    """Expose Prometheus metrics in the standard text format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

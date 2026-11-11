"""HTTP routing for the query endpoint.

This module is a thin FastAPI layer: it wires ``POST /query/search`` to the
logic in :mod:`qdrant_middleware.services.query_executor` and maps errors to
responses. All retrieval, fusion, and rerank logic lives in that service, not
in this router.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from qdrant_client import AsyncQdrantClient

from qdrant_middleware.api.dependencies import (
    get_embedder_client,
    get_qdrant_client,
    get_request_settings,
)
from qdrant_middleware.clients.embedder import EmbedderClient
from qdrant_middleware.core.config import Settings
from qdrant_middleware.services.embedding_guard import guard_vector_operation
from qdrant_middleware.services.query_executor import (
    QueryRequest,
    QueryResponse,
    _build_fusion_query,
    _build_response,
    _required_prefetch_depth,
    _rerank_hits,
    run_hybrid_query,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["QUERY"])


@router.post("/search", response_model=QueryResponse)
async def search(
    request: QueryRequest,
    http_request: Request,
    client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    settings: Annotated[Settings, Depends(get_request_settings)],
    embedder: EmbedderClient = Depends(get_embedder_client),
):
    """Run a single hybrid (dense + sparse) query with optional reranking.

    Embeds the query, runs server-side RRF/DBSF fusion in Qdrant, and — when
    ``rerank`` is set — reorders the top candidates via the Smart Embedder reranker.
    """
    try:
        request = request.apply_defaults(settings)
        fusion_query = _build_fusion_query(request)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    http_request.state.log_collection = request.collection_name

    try:
        embeddings = await embedder.embed([request.query], input_type="query")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Smart Embedder embedding failed: {e}") from e
    if len(embeddings) != 1:
        raise HTTPException(502, "Smart Embedder embedding count mismatch")
    vector_names = await guard_vector_operation(
        client,
        request.collection_name,
        embeddings,
        settings,
        mutation=False,
    )

    # Fetch a pool deep enough to serve this page from position 0: run_hybrid_query
    # no longer applies offset itself (see its docstring), so the depth requested
    # here must already include it — the same offset+window depth apply_defaults
    # already validated against candidate_limit, kept in sync via the same helper.
    result_limit = _required_prefetch_depth(
        request.offset,
        request.limit,
        request.rerank_top_k if request.rerank else None,
    )
    hits = await run_hybrid_query(
        request,
        embeddings[0],
        client,
        settings,
        result_limit=result_limit,
        fusion_query=fusion_query,
        vector_names=vector_names,
    )
    if not request.rerank or not hits:
        response = _build_response(request, hits, reranked=False, candidates=len(hits))
    else:
        response = await _rerank_hits(request, hits, embedder, settings)
    http_request.state.log_points = response.returned
    return response

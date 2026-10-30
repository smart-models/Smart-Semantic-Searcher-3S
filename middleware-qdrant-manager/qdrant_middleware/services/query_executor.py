"""Core retrieval logic for the hybrid query endpoint.

Holds the request/response models, validation, and execution for
``POST /query/search`` — a single hybrid query with server-side RRF/DBSF fusion
and optional cross-encoder reranking. The thin FastAPI wiring lives in
:mod:`qdrant_middleware.api.routers.query`.
"""

import json
import logging
from typing import Any, Literal, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator
from qdrant_client import models

from qdrant_middleware.clients.embedder import EmbedderClient
from qdrant_middleware.clients.retry import retry_async
from qdrant_middleware.core.config import Settings
from qdrant_middleware.schemas.embedding_contract import ActiveVectorNames
from qdrant_middleware.repositories.qdrant_collections import (
    collection_has_quantization,
    is_transient_qdrant,
)
from qdrant_middleware.services.filter_builder import build_qdrant_filter
from qdrant_middleware.services.payload_paths import qdrant_payload_selector

log = logging.getLogger(__name__)


def _resolved_collection_name(value: str | None, settings: Settings) -> str:
    """Resolve and validate the target collection for a query request."""
    collection_name = value if value is not None else settings.collection_name
    if not collection_name or not collection_name.strip():
        raise ValueError("collection_name cannot be empty")
    return collection_name.strip()


def _resolved_limit(value: int | None, settings: Settings) -> int:
    """Resolve and validate the response limit."""
    limit = settings.query_default_limit if value is None else value
    if not (1 <= limit <= settings.query_max_limit):
        raise ValueError(f"limit must be between 1 and {settings.query_max_limit}")
    return limit


def _resolved_candidate_limit(value: int | None, settings: Settings) -> int:
    """Resolve and validate the candidate limit."""
    candidate_limit = settings.query_default_candidate_limit if value is None else value
    if not (1 <= candidate_limit <= settings.query_max_candidate_limit):
        raise ValueError(
            "candidate_limit must be between 1 and "
            f"{settings.query_max_candidate_limit}"
        )
    return candidate_limit


def _required_prefetch_depth(
    offset: int | None, limit: int, rerank_top_k: int | None
) -> int:
    """Candidates each prefetch must return to serve this page.

    Fusion can only rank a point that appeared in one of the two candidate lists,
    so a page starting at ``offset`` requires both branches to retrieve
    ``offset + window`` candidates. The window is the rerank window when
    reranking (the whole window is fetched, then reordered) or ``limit`` otherwise.
    """
    window = max(limit, rerank_top_k) if rerank_top_k else limit
    return (offset or 0) + window


def _resolved_rerank_top_k(
    value: int | None, candidate_limit: int, settings: Settings
) -> int:
    """Resolve and validate the rerank window size."""
    if value is None:
        return min(
            settings.query_default_rerank_top_k,
            candidate_limit,
            settings.query_max_rerank_top_k,
        )
    if value > settings.query_max_rerank_top_k:
        raise ValueError(
            f"rerank_top_k must be between 1 and {settings.query_max_rerank_top_k}"
        )
    if value > candidate_limit:
        raise ValueError("rerank_top_k cannot exceed candidate_limit")
    return value


class QueryRequest(BaseModel):
    """Validated request body for ``POST /query/search``.

    Settings-derived fields default to ``None`` and are resolved from the
    app's runtime :class:`Settings` via :meth:`apply_defaults` in the route
    handler, so ``create_app(Settings(...))`` actually controls defaults.
    """

    query: str = Field(min_length=1)
    collection_name: Optional[str] = None
    limit: Optional[int] = None
    candidate_limit: Optional[int] = None
    fusion: Optional[Literal["rrf", "dbsf"]] = None
    rrf_k: Optional[int] = Field(default=None, ge=1, le=10000)
    rrf_weights: Optional[list[float]] = None
    rerank: Optional[bool] = None
    rerank_top_k: Optional[int] = Field(default=None, ge=1, le=10000)
    rerank_normalize: bool = True
    filters: Optional[dict[str, Any]] = None
    with_payload: Optional[list[str]] = None
    # Both are applied by Qdrant on the FUSED result, i.e. before any rerank:
    # the threshold compares against the fusion score (rank-based for RRF,
    # normalized for DBSF), never against the reranker score.
    offset: Optional[int] = Field(default=None, ge=0)
    score_threshold: Optional[float] = None
    # Per-request overrides of the prefetch search params. None means "use the
    # process defaults from Settings", so an omitted field keeps the previous
    # behaviour exactly.
    hnsw_ef: Optional[int] = Field(default=None, ge=1, le=100000)
    exact: Optional[bool] = None
    rescore: Optional[bool] = None
    oversampling: Optional[float] = Field(default=None, ge=1.0, le=100.0)
    # Upper bound for the Qdrant call itself; without it a pathological query can
    # hold a worker for the configured Qdrant client timeout.
    timeout: Optional[int] = Field(default=None, ge=1, le=3600)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        """Strip whitespace and reject blank query values."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("query cannot be blank")
        return stripped

    @field_validator("collection_name")
    @classmethod
    def strip_collection_name(cls, value: Optional[str]) -> Optional[str]:
        """Strip whitespace and reject blank collection values (None is OK)."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("collection_name cannot be blank")
        return stripped

    @field_validator("rrf_weights")
    @classmethod
    def validate_rrf_weights(
        cls, value: Optional[list[float]]
    ) -> Optional[list[float]]:
        """Require exactly two non-negative weights (dense, sparse) if provided."""
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError(
                "rrf_weights must contain exactly two weights: dense and sparse"
            )
        if any(weight < 0 for weight in value):
            raise ValueError("rrf_weights must be non-negative")
        return value

    @field_validator("filters")
    @classmethod
    def validate_filters(
        cls, value: Optional[dict[str, Any]]
    ) -> Optional[dict[str, Any]]:
        """Require filters to be a JSON object when provided."""
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("filters must be an object")
        return value

    def apply_defaults(self, settings: Settings) -> "QueryRequest":
        """Resolve ``None`` fields from runtime settings and cross-validate.

        Returns a new :class:`QueryRequest` with all settings-derived defaults
        filled in and constraints enforced.  Must be called by the route handler
        before the request is used.
        """
        data = self.model_dump()
        data["collection_name"] = _resolved_collection_name(
            data["collection_name"], settings
        )
        data["limit"] = _resolved_limit(data["limit"], settings)
        data["candidate_limit"] = _resolved_candidate_limit(
            data["candidate_limit"], settings
        )
        if data["candidate_limit"] < data["limit"]:
            raise ValueError("candidate_limit must be >= limit")
        data["fusion"] = data["fusion"] or settings.query_default_fusion
        data["rerank"] = (
            settings.query_default_rerank if data["rerank"] is None else data["rerank"]
        )
        data["rerank_top_k"] = _resolved_rerank_top_k(
            data["rerank_top_k"], data["candidate_limit"], settings
        )
        # Default to the middleware's own HTTP budget: letting Qdrant work longer
        # than the request that is waiting for it only ties up a worker.
        data["timeout"] = (
            settings.request_timeout if data["timeout"] is None else data["timeout"]
        )
        # Paging depth must fit inside candidate_limit. A fused hit at position N
        # exists only if it was in one of the two candidate lists, so a page at
        # `offset` is well-founded only when both prefetches already reach
        # offset+window. Requiring it (instead of quietly deepening the prefetch)
        # keeps the candidate pool fixed for a given candidate_limit, which is what
        # makes consecutive pages tile instead of shifting under the caller.
        required_depth = _required_prefetch_depth(
            data["offset"],
            data["limit"],
            data["rerank_top_k"] if data["rerank"] else None,
        )
        if required_depth > data["candidate_limit"]:
            raise ValueError(
                f"offset {data['offset']} needs candidate_limit >= {required_depth} "
                f"to be well-founded (currently {data['candidate_limit']}): fusion "
                "can only rank a point that one of the two branches retrieved. "
                f"Raise candidate_limit (max {settings.query_max_candidate_limit}) "
                "or lower the offset."
            )
        build_qdrant_filter(data["filters"], settings=settings)
        return QueryRequest.model_validate(data)


class QueryHit(BaseModel):
    """A single scored result, with an optional rerank score and its payload."""

    id: str
    score: float
    rerank_score: Optional[float] = None
    payload: dict[str, Any]


class QueryResponse(BaseModel):
    """Response body for ``POST /query/search``."""

    query: str
    collection_name: str
    fusion: str
    reranked: bool
    # Fused hits the middleware received for this page, i.e. the size of the
    # window Qdrant returned after offset and limit — not the candidate pool.
    candidates: int
    returned: int
    skipped_without_text: int = 0
    # Echoed so a caller can tell "end of the ranking" from "nothing matched" and
    # reproduce a run from its response alone.
    offset: int = 0
    score_threshold: Optional[float] = None
    results: list[QueryHit]


def _build_fusion_query(request: QueryRequest) -> models.FusionQuery | models.RrfQuery:
    """Build the Qdrant fusion query (RRF or DBSF) from the request.

    RRF tuning options (``rrf_k``/``rrf_weights``) are valid only with
    ``fusion='rrf'``; combining them with DBSF raises HTTP 400.
    """
    has_rrf_options = request.rrf_k is not None or request.rrf_weights is not None
    if request.fusion == "rrf":
        if has_rrf_options:
            # rrf_weights map POSITIONALLY to the prefetch list, which is built in
            # the order [dense, sparse] in run_hybrid_query. Keep these two in sync:
            # reordering the prefetches would silently re-map the weights.
            return models.RrfQuery(
                rrf=models.Rrf(k=request.rrf_k, weights=request.rrf_weights)
            )
        return models.FusionQuery(fusion=models.Fusion.RRF)

    if has_rrf_options:
        raise HTTPException(
            400, "rrf_k and rrf_weights are supported only with fusion='rrf'"
        )
    return models.FusionQuery(fusion=models.Fusion.DBSF)


def _build_search_params(
    quantization_active: bool,
    settings: Settings,
    request: Optional["QueryRequest"] = None,
) -> Optional[models.SearchParams]:
    """Build prefetch search params (HNSW ef, exact, indexed-only, quantization).

    Each value comes from the request when the caller set it and from
    :class:`Settings` otherwise, so an omitted field keeps the process default.
    Quantization rescore/oversampling are only attached when the target collection
    is actually quantized (``quantization_active``), so unquantized collections
    never receive ignored quantization params. Returns None when nothing but
    Qdrant's own defaults would be sent.
    """
    hnsw_ef = settings.query_hnsw_ef
    exact = None
    rescore = settings.query_quantization_rescore
    oversampling = settings.query_quantization_oversampling
    if request is not None:
        if request.hnsw_ef is not None:
            hnsw_ef = request.hnsw_ef
        exact = request.exact
        if request.rescore is not None:
            rescore = request.rescore
        if request.oversampling is not None:
            oversampling = request.oversampling

    quantization = (
        models.QuantizationSearchParams(rescore=rescore, oversampling=oversampling)
        if quantization_active
        else None
    )
    if (
        hnsw_ef is None
        and exact is None
        and not settings.query_indexed_only
        and quantization is None
    ):
        return None
    return models.SearchParams(
        hnsw_ef=hnsw_ef,
        exact=exact,
        indexed_only=settings.query_indexed_only,
        quantization=quantization,
    )


def _qdrant_query_failure(exc: Exception) -> HTTPException:
    """Map a Qdrant query failure to 400 when it rejected the input, else 502.

    A filter on a field with no matching index (or any other input Qdrant
    itself refuses) is a client mistake, not a backend failure — surfacing it
    as 502 miscounts it as a Qdrant outage and hides the actual reason behind
    the query executor's generic error message. Qdrant's raw response body is
    never forwarded as-is: it carries server internals unusable by a client.
    """
    status_code = getattr(exc, "status_code", None)
    if status_code not in {400, 422}:
        return HTTPException(502, f"Qdrant query failed: {exc}")
    reason = "rejected by Qdrant"
    content = getattr(exc, "content", None)
    if content:
        try:
            parsed = json.loads(content)
            reason = str(parsed["status"]["error"]) or reason
        except (TypeError, ValueError, KeyError):
            pass
    return HTTPException(400, f"Qdrant query failed: {reason}")


def _point_to_hit(point: models.ScoredPoint) -> QueryHit:
    """Convert a Qdrant ``ScoredPoint`` into a :class:`QueryHit`."""
    return QueryHit(
        id=str(point.id),
        score=float(point.score),
        payload=point.payload or {},
    )


def _without_text(payload: dict[str, Any]) -> dict[str, Any]:
    """Return ``payload`` minus its ``text`` key."""
    return {key: value for key, value in payload.items() if key != "text"}


def _build_response(
    request: QueryRequest,
    hits: list[QueryHit],
    reranked: bool,
    candidates: int,
    skipped_without_text: int = 0,
) -> QueryResponse:
    """Slice ``hits[offset:offset+limit]`` and assemble the search response.

    ``hits`` is always the full, stably-sorted (and, when reranking, fully
    reranked) pool from position 0 — ``run_hybrid_query`` no longer applies
    ``offset`` itself. Applying it exactly once, here, after any reranking
    has already reordered the whole pool, is what keeps adjacent pages from
    each reranking a different, overlapping window and handing back
    duplicate or skipped ids across pages.
    """
    start = request.offset or 0
    final = hits[start : start + request.limit]
    if request.with_payload is not None and "text" not in request.with_payload:
        # Fetched for the reranker only (see run_hybrid_query); not selected.
        final = [
            hit.model_copy(update={"payload": _without_text(hit.payload)})
            for hit in final
        ]
    return QueryResponse(
        query=request.query,
        collection_name=request.collection_name,
        fusion=request.fusion,
        reranked=reranked,
        candidates=candidates,
        returned=len(final),
        skipped_without_text=skipped_without_text,
        offset=request.offset or 0,
        score_threshold=request.score_threshold,
        results=final,
    )


def _text_for_rerank(hit: QueryHit, settings: Settings) -> Optional[str]:
    """Return a hit's non-blank ``text`` payload, or None if absent/empty."""
    text = hit.payload.get("text")
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None
    if (
        settings.query_rerank_max_chars > 0
        and len(stripped) > settings.query_rerank_max_chars
    ):
        return stripped[: settings.query_rerank_max_chars]
    return stripped


async def _rerank_hits(
    request: QueryRequest,
    hits: list[QueryHit],
    embedder: EmbedderClient,
    settings: Settings,
) -> QueryResponse:
    """Rerank the top window of hits and return the reordered response.

    Only the first ``rerank_top_k`` hits are reranked (the tail is appended
    unchanged). Hits without non-blank ``text`` are skipped and counted in
    ``skipped_without_text``.
    """
    candidates = len(hits)
    rerank_top_k = request.rerank_top_k or min(
        request.candidate_limit, settings.query_max_rerank_top_k
    )
    head = hits[:rerank_top_k]
    tail = hits[rerank_top_k:]

    skipped_without_text = 0
    rerank_inputs: list[tuple[QueryHit, str]] = []
    unreranked_head: list[QueryHit] = []
    for hit in head:
        text = _text_for_rerank(hit, settings)
        if text is None:
            skipped_without_text += 1
            unreranked_head.append(hit)
            continue
        rerank_inputs.append((hit, text))

    if not rerank_inputs:
        return _build_response(
            request,
            hits,
            reranked=False,
            candidates=candidates,
            skipped_without_text=skipped_without_text,
        )

    passages = [text for _, text in rerank_inputs]
    try:
        reranked = await embedder.rerank(
            query=request.query,
            passages=passages,
            normalize=request.rerank_normalize,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Smart Embedder rerank failed: {e}") from e

    reordered: list[QueryHit] = []
    seen: set[int] = set()
    for result in reranked:
        if (
            result.index < 0
            or result.index >= len(rerank_inputs)
            or result.index in seen
        ):
            continue
        seen.add(result.index)
        hit = rerank_inputs[result.index][0]
        reordered.append(hit.model_copy(update={"rerank_score": result.score}))

    for index, (hit, _) in enumerate(rerank_inputs):
        if index not in seen:
            unreranked_head.append(hit)

    return _build_response(
        request,
        reordered + unreranked_head + tail,
        reranked=True,
        candidates=candidates,
        skipped_without_text=skipped_without_text,
    )


async def run_hybrid_query(
    request: QueryRequest,
    embedding: Any,
    client: Any,
    settings: Settings,
    result_limit: Optional[int] = None,
    fusion_query: Any = None,
    vector_names: ActiveVectorNames | None = None,
) -> list[QueryHit]:
    """Run one hybrid (dense + sparse) Qdrant query and return fused hits.

    Issues a single Query API call with two prefetches (named ``dense`` and
    ``sparse``) fused server-side. ``result_limit`` truncates the fused output
    to that many hits *from position 0*; when omitted it defaults to
    ``candidate_limit``. ``request.offset`` is deliberately **not** applied
    here — the caller is responsible for slicing ``[offset:offset+limit]``
    from the returned pool (see ``_build_response``), once, after any
    reranking has had a chance to reorder the whole pool. Applying offset at
    this layer instead used to hand each page a different, overlapping
    sub-pool to rerank independently, which could return the same id on two
    adjacent pages or drop one entirely. ``fusion_query`` lets the caller
    pass a pre-built fusion query to avoid reconstructing it. Qdrant
    rejecting the input (e.g. a filter on a field with no matching index)
    becomes 400; any other Qdrant failure becomes 502.
    """
    # Fetch the whole fused candidate pool, then apply a deterministic tie-break
    # and truncate locally. Asking Qdrant for separate offset windows lets equal-score
    # RRF hits move between calls, producing duplicates and gaps across pages.
    page_limit = result_limit if result_limit is not None else request.candidate_limit
    fetch_limit = request.candidate_limit
    if fusion_query is None:
        fusion_query = _build_fusion_query(request)
    vector_names = vector_names or ActiveVectorNames(
        dense=settings.dense_vector_name,
        sparse=settings.sparse_vector_name,
    )
    qdrant_filter = build_qdrant_filter(request.filters, settings=settings)
    quantization_active = await collection_has_quantization(
        client, request.collection_name, settings
    )
    search_params = _build_search_params(quantization_active, settings, request)
    # No explicit selection means the whole payload: a hardcoded shortlist would
    # silently drop every other field a caller stored.
    payload_fields = True
    if request.with_payload is not None:
        fields = request.with_payload
        # The reranker scores `text`: fetch it even when the caller did not
        # select it, or every hit is skipped. _build_response strips it again.
        if request.rerank and "text" not in fields:
            fields = [*fields, "text"]
        payload_fields = qdrant_payload_selector(fields)
    # Fixed by candidate_limit alone: apply_defaults has already refused any offset
    # that would reach past it, so the pool stays identical across the pages of one
    # query instead of deepening as the caller advances.
    prefetch_limit = request.candidate_limit

    async def _do():
        return await client.query_points(
            collection_name=request.collection_name,
            prefetch=[
                models.Prefetch(
                    query=embedding.dense,
                    using=vector_names.dense,
                    limit=prefetch_limit,
                    filter=qdrant_filter,
                    params=search_params,
                ),
                models.Prefetch(
                    query=embedding.sparse,
                    using=vector_names.sparse,
                    limit=prefetch_limit,
                    filter=qdrant_filter,
                    params=search_params,
                ),
            ],
            # Outer query is a fusion query: hnsw/quant SearchParams apply only to the
            # prefetches above, so none is passed here.
            query=fusion_query,
            with_payload=payload_fields,
            with_vectors=False,
            limit=fetch_limit,
            # Threshold applies to the fused score. Paging happens locally after
            # the stable score/id ordering, so Qdrant must see no offset here.
            offset=None,
            score_threshold=request.score_threshold,
            timeout=request.timeout,
        )

    try:
        result = await retry_async(
            _do,
            attempts=settings.retry_max_attempts,
            base_delay=settings.retry_base_delay,
            should_retry=is_transient_qdrant,
            description="Qdrant hybrid query",
        )
    except Exception as e:
        log.warning("Qdrant hybrid query failed: %s", e)
        raise _qdrant_query_failure(e) from e

    hits = sorted(
        (_point_to_hit(point) for point in result.points),
        key=lambda hit: (-hit.score, hit.id),
    )
    return hits[:page_limit]

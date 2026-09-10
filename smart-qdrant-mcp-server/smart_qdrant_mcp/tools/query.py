"""Hybrid search over a collection."""

from __future__ import annotations

from typing import Any, Literal, Optional

from mcp.server import MCPServer
from pydantic import BaseModel, ConfigDict, Field, field_validator

from smart_qdrant_mcp.context import ToolContext
from smart_qdrant_mcp.errors import error_result
from smart_qdrant_mcp.formatting import (
    ResponseFormat,
    StructuredToolResult,
    bullet,
    paginate,
    render,
)

RANKING_KEYS = (
    "query",
    "collection_name",
    "fusion",
    "reranked",
    "candidates",
    "returned",
    "skipped_without_text",
    "score_threshold",
)


class SearchInput(BaseModel):
    """Input model for hybrid search.

    Every option defaults to None so the middleware applies its own configured
    default; sending an explicit value overrides it for this call only.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language search text (e.g. 'how do I rotate the token')",
    )
    collection_name: Optional[str] = Field(
        default=None,
        description="Target collection; omit to use the server default (e.g. 'docs')",
    )
    limit: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum hits to return in this page (e.g. 20)",
    )
    candidate_limit: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Candidates fetched per retriever before fusion; raise it to improve "
            "recall at the cost of latency (e.g. 100)"
        ),
    )
    fusion: Optional[Literal["rrf", "dbsf"]] = Field(
        default=None,
        description=(
            "Fusion strategy: 'rrf' (rank-based, robust default) or 'dbsf' "
            "(score-based, needs comparable score scales)"
        ),
    )
    rrf_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=10000,
        description="RRF smoothing constant; only valid with fusion='rrf' (e.g. 60)",
    )
    rrf_weights: Optional[list[float]] = Field(
        default=None,
        description=(
            "Per-retriever weights for RRF, dense first (e.g. [1.0, 0.7]); only "
            "valid with fusion='rrf'"
        ),
    )
    rerank: Optional[bool] = Field(
        default=None,
        description="Re-score fused hits with the cross-encoder reranker (e.g. true)",
    )
    rerank_top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=10000,
        description="How many fused hits to hand to the reranker (e.g. 25)",
    )
    rerank_normalize: bool = Field(
        default=True,
        description="Normalise reranker scores to 0-1 before sorting (e.g. true)",
    )
    filters: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Payload filter, same syntax as Qdrant (e.g. "
            '{"must": [{"key": "lang", "match": {"value": "it"}}]}). '
            "See the resource docs://reference/filters"
        ),
    )
    with_payload: Optional[list[str]] = Field(
        default=None,
        description=(
            "Payload keys to return; omit for all, which is far more verbose "
            "(e.g. ['text', 'source_id'])"
        ),
    )
    offset: Optional[int] = Field(
        default=None,
        ge=0,
        description="Hits to skip, for paging through a ranking (e.g. 20)",
    )
    score_threshold: Optional[float] = Field(
        default=None,
        description=(
            "Drop hits scoring below this on the FUSED score, before reranking "
            "(e.g. 0.2)"
        ),
    )
    hnsw_ef: Optional[int] = Field(
        default=None,
        ge=1,
        le=100000,
        description="HNSW search breadth for the prefetch stage (e.g. 128)",
    )
    exact: Optional[bool] = Field(
        default=None,
        description="Bypass the HNSW index and scan exactly; slow (e.g. false)",
    )
    rescore: Optional[bool] = Field(
        default=None,
        description="Re-score quantized candidates against full vectors (e.g. true)",
    )
    oversampling: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=100.0,
        description="Quantization oversampling factor (e.g. 2.0)",
    )
    timeout: Optional[int] = Field(
        default=None,
        ge=1,
        le=3600,
        description="Upper bound in seconds for the Qdrant call (e.g. 30)",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (default) or 'json'",
    )

    @field_validator("query")
    @classmethod
    def reject_blank_query(cls, value: str) -> str:
        """Reject a whitespace-only query before it reaches the middleware."""
        if not value.strip():
            raise ValueError("query cannot be blank")
        return value


def _pagination_footer(payload: dict[str, Any]) -> str:
    """Render the closing line: the offset for the next page, or the end."""
    if payload.get("has_more"):
        return bullet(
            "More pages",
            f"yes — pass `{payload.get('next_offset')}` as offset to fetch "
            "the next page",
        )
    return bullet("More pages", "no — the result set is complete")


def _rows(payload: dict[str, Any]) -> list[str]:
    """Render search results as a markdown hit list."""
    items = payload.get("items") or []
    if not items:
        return ["", "No results matched this query."]
    total = payload.get("total")
    header = [
        bullet("Collection", payload.get("collection_name")),
        bullet("Fusion", payload.get("fusion")),
        bullet("Reranked", payload.get("reranked")),
        bullet("Candidates", payload.get("candidates")),
        bullet(
            "Returned",
            f"{payload.get('count')} of {total if total is not None else 'unknown'} "
            f"(offset {payload.get('offset')})",
        ),
        bullet("Skipped without text", payload.get("skipped_without_text")),
        bullet("Score threshold", payload.get("score_threshold")),
        "",
    ]
    lines: list[str] = []
    for index, hit in enumerate(items, start=1):
        rerank = hit.get("rerank_score")
        score = f"score {hit.get('score')}"
        if rerank is not None:
            score += f", rerank {rerank}"
        lines.append(f"## {index}. {hit.get('id')} ({score})")
        for key, value in (hit.get("payload") or {}).items():
            lines.append(bullet(key, value))
        lines.append("")
    lines.append(_pagination_footer(payload))
    return [*header, *lines]


def register(mcp: MCPServer, ctx: ToolContext) -> None:
    """Register the search tool on the server."""

    @mcp.tool(
        name="qdrant_search",
        annotations={
            "title": "Hybrid search a Qdrant collection",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def qdrant_search(params: SearchInput) -> StructuredToolResult:
        """Search a collection with dense + sparse retrieval, fusion and reranking.

        This is the main retrieval entry point. The query text is embedded by the
        Smart Embedder, both retrievers run, their rankings are fused (RRF by
        default), and the top hits are optionally re-scored by a cross-encoder.
        Every tuning option is optional: omitted fields keep the server's
        configured defaults, so a bare query is a valid call.

        Args:
            params (SearchInput): Validated input containing:
                - query (str): natural-language search text, required.
                - collection_name (str | None): target collection.
                - limit (int | None): hits per page.
                - candidate_limit (int | None): candidates per retriever pre-fusion.
                - fusion ('rrf' | 'dbsf' | None): fusion strategy.
                - rrf_k (int | None), rrf_weights (list[float] | None): RRF tuning.
                - rerank (bool | None), rerank_top_k (int | None),
                  rerank_normalize (bool): reranking controls.
                - filters (dict | None): Qdrant payload filter.
                - with_payload (list[str] | None): payload keys to return.
                - offset (int | None): hits to skip.
                - score_threshold (float | None): minimum fused score.
                - hnsw_ef, exact, rescore, oversampling, timeout: Qdrant search
                  params, per call.
                - response_format (ResponseFormat): 'markdown' or 'json'.

        Returns:
            str: Markdown hit list, or JSON with this schema:

            {
                "query": str,
                "collection_name": str,
                "fusion": str,               # "rrf" or "dbsf"
                "reranked": bool,
                "candidates": int,           # fused hits in this window
                "returned": int,
                "skipped_without_text": int, # hits the reranker could not score
                "score_threshold": float | None,
                "total": int | None,         # None: search does not count matches
                "count": int,
                "offset": int,
                "has_more": bool,
                "next_offset": int | None,
                # When `limit` was omitted, the middleware resolves its own
                # page size and never echoes it, so this server cannot tell a
                # full page from a genuinely short one; `has_more` is then
                # reported conservatively (true whenever the page is
                # non-empty) and only an empty page reliably means the end.
                "items": [
                    {
                        "id": str,
                        "score": float,          # fused score
                        "rerank_score": float | None,
                        "payload": {...}
                    }
                ]
            }

        Examples:
            - Use when: answering a question from stored documents.
            - Use when: checking whether a topic is covered — search, then read
              the payloads.
            - Don't use when: you already know the point ids — use
              qdrant_read_points, which skips embedding entirely.
            - Don't use when: you want to enumerate or filter without ranking —
              use qdrant_scroll_points, which is cheaper and ordered.
            - Don't use when: you only need a number — use qdrant_count_points.

        Error Handling:
            Input validation is handled by SearchInput before any HTTP call.
            Returns "Error: invalid request — <field>: <reason>" on 400,
            "Error: not found ..." when the collection does not exist, and an
            embedding-contract message pointing at
            docs://concepts/embedding-contract on 409.
        """
        body = params.model_dump(exclude_none=True, exclude={"response_format"})
        try:
            raw = await ctx.client.request(
                "POST", "/query/search", json=body, retryable=True
            )
        except Exception as exc:  # -> agent message
            return error_result(exc)

        data = raw if isinstance(raw, dict) else {}
        results = data.get("results") or []
        offset = data.get("offset", 0)
        if params.limit is not None:
            # The caller told us the page size, so a short page reliably means
            # the end and a full page means "there may be more".
            page = paginate(results, total=None, offset=offset, limit=params.limit)
        else:
            # The middleware resolved its own page size and QueryResponse never
            # echoes it, so this server cannot know whether `results` is a full
            # page. Never claim completeness it cannot verify: a non-empty page
            # always reports more (an extra call that comes back empty is
            # cheap), and only a genuinely empty page is the end.
            count = len(results)
            next_offset = offset + count if count and isinstance(offset, int) else None
            page = paginate(
                results, total=None, offset=offset, limit=0, next_offset=next_offset
            )
        payload = {**{key: data.get(key) for key in RANKING_KEYS}, **page}
        return render(
            payload,
            params.response_format,
            title="Search results",
            rows=_rows,
        )

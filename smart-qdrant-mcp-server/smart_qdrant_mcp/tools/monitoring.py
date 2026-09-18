"""Health and readiness reporting."""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer
from pydantic import BaseModel, ConfigDict, Field

from smart_qdrant_mcp.context import ToolContext
from smart_qdrant_mcp.errors import MiddlewareError, error_result
from smart_qdrant_mcp.formatting import (
    ResponseFormat,
    StructuredToolResult,
    bullet,
    render,
)


class CheckHealthInput(BaseModel):
    """Input model for the aggregated health check."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description=(
            "Output format: 'markdown' for a compact verdict, 'json' for the "
            "full structured payload (e.g. 'json')"
        ),
    )


def _rows(payload: dict[str, Any]) -> list[str]:
    """Render the health verdict as markdown fact lines.

    The middleware's /health and /stats are flat: `qdrant` and `embedder` are
    status strings such as "reachable", not nested objects, and the counters
    are named uptime_seconds/total_requests/rejected_requests. Reading them as
    nested dicts raised AttributeError against the live service.
    """
    stats = payload.get("stats") or {}
    models = payload.get("embedder_models") or {}
    return [
        bullet("Ready", "yes" if payload.get("ready") else "no"),
        bullet("Alive", "yes" if payload.get("alive") else "no"),
        bullet("Status", payload.get("status")),
        bullet("Qdrant", payload.get("qdrant")),
        bullet("Collections", payload.get("collections")),
        bullet("Embedder", payload.get("embedder")),
        bullet("Embedder profile", payload.get("embedder_profile")),
        bullet(
            "Embedder models", ", ".join(f"{k}={v}" for k, v in models.items() if v)
        ),
        bullet("Uptime (s)", stats.get("uptime_seconds")),
        bullet("Requests served", stats.get("total_requests")),
        bullet("Requests rejected", stats.get("rejected_requests")),
    ]


def register(mcp: MCPServer, ctx: ToolContext) -> None:
    """Register the monitoring tool on the server."""

    @mcp.tool(
        name="qdrant_check_health",
        annotations={
            "title": "Check Qdrant stack health",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def qdrant_check_health(params: CheckHealthInput) -> StructuredToolResult:
        """Report whether the Qdrant stack can serve requests right now.

        Aggregates the middleware's four monitoring endpoints into one verdict:
        liveness (the process is up), readiness (Qdrant and the embedder are
        both reachable), dependency detail, and in-memory request counters. Read
        this before concluding that a failing tool call is a request problem —
        an unready dependency looks identical from a single failure.

        Args:
            params (CheckHealthInput): Validated input containing:
                - response_format (ResponseFormat): 'markdown' (default) or 'json'.

        Returns:
            str: Markdown verdict, or JSON with this schema:

            {
                "ready": bool,              # Qdrant and embedder both reachable
                "alive": bool,              # the middleware process responds
                "status": str,              # e.g. "healthy"
                "qdrant": str,              # e.g. "reachable"
                "collections": int,         # collections currently in Qdrant
                "default_collection": str,  # "" when none is configured
                "embedder": str,            # e.g. "reachable"
                "embedder_profile": str,    # "basic" or "advanced"
                "embedder_models": {        # active model ids; null where the
                    "dense": str,           # profile provides none (advanced
                    "sparse": str,          # has no colbert model)
                    "colbert": str | None,
                    "reranker": str
                },
                "stats": {                  # in-memory counters
                    "uptime": str,
                    "uptime_seconds": int,
                    "total_requests": int,
                    "rejected_requests": int,
                    "collection": str
                }
            }

        Examples:
            - Use when: a call failed with 503 and you need to know which
              dependency is down.
            - Use when: confirming the stack is usable before a long ingest.
            - Don't use when: you need Prometheus metrics — read the resource
              qhm://metrics instead.
            - Don't use when: you want to know whether a collection is valid —
              use qdrant_verify_collection.

        Error Handling:
            Never raises for an unhealthy stack: readiness failures are reported
            as "ready": false. Returns an "Error: ..." string only when the
            middleware itself is unreachable.
        """
        try:
            health = await ctx.client.request(
                "GET", "/health", accepted_statuses=frozenset({503})
            )
            alive = await _probe(ctx, "/livez")
            ready = await _probe(ctx, "/readyz")
            stats = await ctx.client.request("GET", "/stats")
        except Exception as exc:  # -> agent message
            return error_result(exc)

        health_payload = health if isinstance(health, dict) else {}
        payload = {
            "ready": ready,
            "alive": alive,
            "status": health_payload.get("status"),
            "qdrant": health_payload.get("qdrant"),
            "collections": health_payload.get("collections"),
            "default_collection": health_payload.get("default_collection"),
            "embedder": health_payload.get("embedder"),
            "embedder_profile": health_payload.get("embedder_profile"),
            "embedder_models": health_payload.get("embedder_models") or {},
            "stats": stats if isinstance(stats, dict) else {},
        }
        return render(
            payload,
            params.response_format,
            title="Qdrant stack health",
            rows=_rows,
        )


async def _probe(ctx: ToolContext, path: str) -> bool:
    """Return True when a liveness/readiness probe answers success."""
    try:
        await ctx.client.request("GET", path)
    except MiddlewareError:
        return False
    return True

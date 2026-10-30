"""Live state proxied from the middleware, exposed as MCP resources.

These are resources rather than tools because they take no arguments and are
read for context, not called to act. There is no per-resource cache control
in ``mcp==2.2.0`` — cache hints are configured once per JSON-RPC method (e.g.
all of ``resources/read``), and that method is shared with any documentation
resources added later, so a short TTL for live state cannot coexist with a
long TTL for static docs under the same hint. This module leaves
``resources/read`` uncached: correctness (fresh health/metrics) matters more
than shaving a request for these three cheap proxy calls.

Unlike a tool call, a resource read has no ``is_error`` flag on the wire —
``ReadResourceResult`` is just content. Catching the upstream failure here
and returning ``describe_error(exc)``'s text used to mean a client reading
``qhm://health`` while the middleware is down received that sentence *as the
health document itself*, indistinguishable from a real, healthy body. None
of these three functions catches anything any more: letting the exception
propagate lets the SDK's own ``read_resource`` wrapper
(``mcp/server/mcpserver/server.py``) turn it into a genuine protocol-level
error (``ResourceError`` -> ``MCPError``), which is the only way for a
resource read to signal failure at all.
"""

from __future__ import annotations

import json

from mcp.server import MCPServer

from smart_qdrant_mcp.context import ToolContext


def register(mcp: MCPServer, ctx: ToolContext) -> None:
    """Register the live state resources."""

    @mcp.resource(
        "qhm://collections",
        title="Current Qdrant collections",
        mime_type="application/json",
    )
    async def collections_resource() -> str:
        """List the collections currently present in Qdrant."""
        payload = await ctx.client.request("GET", "/collections")
        return json.dumps(payload, indent=2, ensure_ascii=False, default=str)

    @mcp.resource(
        "qhm://health",
        title="Middleware dependency health",
        mime_type="application/json",
    )
    async def health_resource() -> str:
        """Report the middleware's combined dependency health."""
        payload = await ctx.client.request("GET", "/health")
        return json.dumps(payload, indent=2, ensure_ascii=False, default=str)

    @mcp.resource("qhm://metrics", title="Middleware Prometheus metrics")
    async def metrics_resource() -> str:
        """Expose the middleware's Prometheus metrics in text format.

        No explicit ``mime_type`` here: this is Prometheus exposition text,
        not JSON or Markdown, and the SDK's own default (``text/plain``,
        ``mcp/server/mcpserver/resources/types.py``) already describes it
        correctly.
        """
        return await ctx.client.text("/metrics")

"""Whole-collection export.

The counterpart to the ingest tools, and the only tool on this server whose
call results in a file being written. The write happens in the middleware, to a
directory the middleware owns, under a filename the middleware derives: no
path, filename or content from the caller reaches the filesystem. See
`docs/superpowers/specs/2026-08-07-collection-export-endpoint-design.md` §6.1
for how that squares with this server exposing no filesystem tools.

The endpoint can also return the document in the response body; this tool never
asks for that. A whole collection in a tool result is exactly the context cost
the endpoint was built to remove.
"""

from __future__ import annotations

from typing import Any, Optional

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


class ExportCollectionInput(BaseModel):
    """Input model for exporting a whole collection to a file."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    collection_name: Optional[str] = Field(
        default=None,
        description=(
            "Collection to export; omit to use the middleware's configured "
            "default (e.g. 'my_docs')"
        ),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description=(
            "Output format: 'markdown' for a compact summary, 'json' for the "
            "full structured payload (e.g. 'json')"
        ),
    )


def _rows(payload: dict[str, Any]) -> list[str]:
    """Render the export outcome as markdown fact lines."""
    return [
        bullet("Collection", payload.get("collection_name")),
        bullet("Chunks exported", payload.get("n_chunks")),
        bullet("Path (middleware container)", payload.get("path")),
        bullet("Bytes", payload.get("bytes")),
    ]


def register(mcp: MCPServer, ctx: ToolContext) -> None:
    """Register the export tool on the server."""

    @mcp.tool(
        name="qdrant_export_collection",
        annotations={
            "title": "Export a collection to a file",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    )
    async def qdrant_export_collection(
        params: ExportCollectionInput,
    ) -> StructuredToolResult:
        """Export every point's payload of one collection to a JSON file.

        The export is the whole collection, deliberately: there is no filter
        parameter and no vector parameter, and neither can be added per call.
        Export a subset by scrolling with qdrant_scroll_points and a filter
        instead; vectors are not retrievable through any export or scroll path.

        One call replaces a paginated qdrant_scroll_points walk: the middleware
        does the walk and writes the file, so the collection never passes
        through your context window. The document is
        {"chunks": [...], "metadata": {...}} — the same shape the web console
        downloads. Each chunk is the stored payload verbatim plus a trailing
        "point_id" holding the Qdrant point id as a UUID string; it is its own
        key so it cannot collide with a chunk's integer "id", which ingest
        currently keeps in the payload. Strip "point_id" before re-ingesting.

        The reported path is inside the middleware container
        (/app/middleware-qdrant-manager/exports/...), which on the host is
        ./middleware-qdrant-manager/exports/... . This tool reports a path; it
        does not hand you a file, and no tool on this server can read it back.

        Args:
            params (ExportCollectionInput): Validated input containing:
                - collection_name (str | None): collection to export, or None
                  for the middleware default.
                - response_format (ResponseFormat): 'markdown' (default) or 'json'.

        Returns:
            str: Markdown summary, or JSON with this schema:

            {
                "ok": bool,
                "collection_name": str,
                "n_chunks": int,          # 0 means nothing was written
                "path": str | None,       # null for an empty collection
                "bytes": int
            }

        Examples:
            - Use when: the user wants a backup or a copy of a collection's
              payload on disk.
            - Use when: you would otherwise loop qdrant_scroll_points to page
              through a whole collection.
            - Don't use when: you need the data in the conversation to reason
              over — scroll a filtered slice instead.
            - Don't use when: you need the vectors. The export carries payload
              only; re-ingesting it re-embeds under the active profile.

        Error Handling:
            A collection larger than EXPORT_MAX_POINTS is refused with 413
            before the walk starts; nothing is written. A failure part-way
            through the walk returns 502 and writes nothing — there is no such
            thing as a partial export file.
        """
        body: dict[str, Any] = {"destination": "file"}
        if params.collection_name is not None:
            body["collection_name"] = params.collection_name
        try:
            raw = await ctx.client.request("POST", "/points/export", json=body)
        except Exception as exc:  # -> agent message
            return error_result(exc)
        if (
            not isinstance(raw, dict)
            or raw.get("ok") is not True
            or not isinstance(raw.get("collection_name"), str)
            or type(raw.get("n_chunks")) is not int
            or raw["n_chunks"] < 0
            or type(raw.get("bytes")) is not int
            or raw["bytes"] < 0
            or "path" not in raw
            or (raw["path"] is not None and not isinstance(raw["path"], str))
            or (raw["n_chunks"] > 0 and not raw["path"])
        ):
            return error_result(
                MiddlewareError(502, "Invalid export response", "/points/export")
            )
        payload = raw
        return render(
            payload,
            params.response_format,
            title="Collection export",
            rows=_rows,
        )

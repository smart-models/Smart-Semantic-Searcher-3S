"""Point read and write tools: retrieve, scroll, count, facet, upsert, delete.

Task 15 added the write tools (``qdrant_upsert_points``,
``qdrant_delete_points``) to this module and extended ``register`` to include
them. Payload-only edits (set/clear/delete-keys) are a separate, later tool
surface, not part of this module.

Two design decisions carried over from the middleware's own
``qdrant_middleware/api/routers/points.py`` shape this file:

``qdrant_read_points`` always calls the batch endpoint (``POST
/points/retrieve``), never the single-id ``GET /points/{id}`` — even for one
id. The batch endpoint returns whatever subset of the requested ids it
found; the single-id endpoint 404s on a miss instead. Rather than surface
that 404, the tool reconstructs the signal itself: it returns ``found`` and
``not_found`` lists, so one missing id among five does not fail the call,
and an agent gets a usable answer either way.

``qdrant_scroll_points`` forwards the middleware's own pagination cursor
rather than inventing one. ``paginate()`` (``smart_qdrant_mcp.formatting``)
guarantees ``has_more`` is true if and only if ``next_offset`` is not
``None``, and it cannot fabricate a cursor for an opaque offset — the real
one from the response must be passed through. The middleware's
``ScrollPointsResponse`` names that field ``next_page_offset``
(points.py:396, always present in the response body, ``null`` included);
this tool reads exactly that key and no other. Paging an *ordered* scroll
(``order_by`` set) cannot use ``offset`` at all — Qdrant rejects the
combination outright — so ``order_by.start_from`` is the only way to page
one; the tool does not duplicate that check locally and instead lets the
middleware's 422 flow through ``describe_error``.

``qdrant_upsert_points`` and ``qdrant_delete_points`` each collapse two
middleware endpoints into one tool, picking between them by point count:
exactly one point/id uses the flat single-item endpoint, more than one uses
the batch endpoint. ``qdrant_delete_points`` additionally branches on
``selector`` -- ``"ids"`` deletes by id (single or bulk), ``"filter"``
always calls ``POST /points/delete-by-filter``. Deletion is irreversible and
this is the most dangerous tool in the server, so a ``selector="filter"``
call **without** ``confirm`` never deletes anything: it calls ``POST
/points/count`` with the same filter instead and reports the match count, a
free rehearsal for the caller to size the blast radius before opting in with
``confirm=<collection_name>``. ``selector="ids"`` has no such rehearsal -- a
missing or mismatched ``confirm`` is refused outright, with no HTTP call
made either way.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union
from urllib.parse import quote

from mcp.server import MCPServer
from pydantic import BaseModel, ConfigDict, Field

from smart_qdrant_mcp.context import ToolContext
from smart_qdrant_mcp.errors import error_result
from smart_qdrant_mcp.formatting import (
    ResponseFormat,
    StructuredToolResult,
    bullet,
    paginate,
    render,
)
from smart_qdrant_mcp.guardrails import (
    count_matching,
    refuse_if,
    require_confirm,
    require_non_blank_ids,
    require_selector_args,
)
from smart_qdrant_mcp.tools.collections import ExactString


class OrderBySpec(BaseModel):
    """Ordered-scroll spec: sort points by an indexed payload field.

    Mirrors ``OrderBySpec`` in the middleware's own
    ``qdrant_middleware/api/routers/points.py:279-298`` (kept in lockstep by
    inspection, not import, since the two services ship as independent
    images). Qdrant refuses ``offset`` together with ``order_by``; paging an
    ordered scroll goes through ``start_from`` instead — resume at the first
    ordered value not yet seen, not a row count.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    key: str = Field(
        ...,
        min_length=1,
        description="Indexed payload field to order by (e.g. 'created_at')",
    )
    direction: Literal["asc", "desc"] = Field(
        default="asc", description="Sort direction (e.g. 'desc')"
    )
    start_from: Optional[Union[int, float, str]] = Field(
        default=None,
        description=(
            "Resume value for paging an ordered scroll: the first ordered "
            "value not yet seen, matching the ordered field's type (a "
            "number or a datetime string) — not a row offset (e.g. 1700000000)"
        ),
    )


class RetrievePointsInput(BaseModel):
    """Input model for retrieving many points by id in one batch call."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description=(
            "Point ids to retrieve — UUIDs or unsigned integers as strings "
            "(e.g. ['1', '2', 'b3b3...-uuid'])"
        ),
    )
    collection_name: Optional[str] = Field(
        default=None,
        description=(
            "Collection to read from; omit to use the server default (e.g. 'docs')"
        ),
    )
    with_vectors: bool = Field(
        default=False,
        description=(
            "Include a summary of each found point's stored vectors — names "
            "and shapes only, never the raw numbers (e.g. true)"
        ),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (default) or 'json' (e.g. 'json')",
    )


class ScrollPointsInput(BaseModel):
    """Input model for paging through points by filter, without ranking."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    filters: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Payload filter, same syntax as Qdrant (e.g. "
            '{"must": [{"key": "lang", "match": {"value": "it"}}]}); omit '
            "to scroll the whole collection. See the resource "
            "docs://reference/filters"
        ),
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=1000,
        description=(
            "Maximum points per page (e.g. 50). 20 is this server's "
            "deliberate default, overriding the middleware's own default "
            "of 10, so agents get fewer round trips when paging"
        ),
    )
    offset: Optional[Union[int, str]] = Field(
        default=None,
        description=(
            "Page cursor: an integer offset, or the next_offset value this "
            "tool returned from a previous call — treat it as opaque either "
            "way. Cannot be combined with order_by (e.g. 'cursor-2')"
        ),
    )
    order_by: Optional[OrderBySpec] = Field(
        default=None,
        description=(
            "Sort by an indexed payload field instead of storage order; "
            "mutually exclusive with offset — page this with "
            "order_by.start_from instead (e.g. "
            "{'key': 'created_at', 'direction': 'desc'})"
        ),
    )
    with_payload: Union[bool, list[str]] = Field(
        default=True,
        description=(
            "true for every payload field (verbose), false for none, or a "
            "list of keys to return only those (e.g. ['text', 'source_id'])"
        ),
    )
    include_total: bool = Field(
        default=True,
        description=(
            "Also count total matches for this filter; set false to skip "
            "that extra count call, cheaper on large collections (e.g. false)"
        ),
    )
    collection_name: Optional[str] = Field(
        default=None,
        description=(
            "Collection to scroll; omit to use the server default (e.g. 'docs')"
        ),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (default) or 'json' (e.g. 'json')",
    )


class CountPointsInput(BaseModel):
    """Input model for counting points matching a filter, without fetching them."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    filters: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Payload filter, same syntax as Qdrant; omit to count the whole "
            'collection (e.g. {"must": [{"key": "lang", "match": '
            '{"value": "it"}}]}). See the resource docs://reference/filters'
        ),
    )
    exact: bool = Field(
        default=True,
        description=(
            "true for an exact count, false for a fast approximate estimate "
            "(e.g. false)"
        ),
    )
    collection_name: Optional[str] = Field(
        default=None,
        description=(
            "Collection to count in; omit to use the server default (e.g. 'docs')"
        ),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (default) or 'json' (e.g. 'json')",
    )


class FacetPointsInput(BaseModel):
    """Input model for the distinct values of one payload field and their counts."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    key: str = Field(
        ..., min_length=1, description="Indexed payload field to facet on (e.g. 'lang')"
    )
    filters: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Payload filter restricting which points are faceted, same "
            'syntax as Qdrant (e.g. {"must": [{"key": "source_id", "match": '
            '{"value": "manual"}}]}); omit to facet the whole collection. '
            "See the resource docs://reference/filters"
        ),
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=1000,
        description="Maximum distinct values to return (e.g. 50)",
    )
    exact: bool = Field(
        default=True,
        description=(
            "true for exact counts, false for a faster approximation (e.g. false)"
        ),
    )
    collection_name: Optional[str] = Field(
        default=None,
        description=(
            "Collection to facet in; omit to use the server default (e.g. 'docs')"
        ),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (default) or 'json' (e.g. 'json')",
    )


class PointSpec(BaseModel):
    """One point to upsert: text to embed, plus an id and extra payload.

    Mirrors ``BatchUpsertItem`` in the middleware's own
    ``qdrant_middleware/api/routers/points.py:633-651`` (kept in lockstep by
    inspection, not import, since the two services ship as independent
    images). ``point_id`` format (UUID or unsigned integer) is not validated
    locally, the same choice ``qdrant_read_points`` makes for the same
    reason: that rule belongs to the middleware, and duplicating it here
    risks the two drifting apart with nothing to catch it.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    point_id: str = Field(
        ...,
        min_length=1,
        description=(
            "A UUID or an unsigned integer, given as a string (e.g. "
            "'b3b3f7b0-6e2b-4c1a-9c0a-1234567890ab' or '42')"
        ),
    )
    text: str = Field(
        ...,
        min_length=1,
        description=(
            "Text to embed server-side and store on this point; blank or "
            "whitespace-only text is rejected (e.g. 'hello world')"
        ),
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extra payload fields stored alongside the embedded text (e.g. "
            "{'lang': 'it', 'version': 3})"
        ),
    )


class UpsertPointsInput(BaseModel):
    """Input model for upserting 1 to 10000 points from raw text in one call."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    points: list[PointSpec] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description=(
            "Points to upsert, 1 to 10000 entries and no duplicate "
            "point_id. Exactly one point uses the flat single-point "
            "endpoint; more than one uses the batch endpoint (e.g. "
            "[{'point_id': '1', 'text': 'hello'}])"
        ),
    )
    collection_name: Optional[str] = Field(
        default=None,
        description=(
            "Collection to upsert into; omit to use the server default (e.g. 'docs')"
        ),
    )
    update_mode: Literal["upsert", "insert_only", "update_only"] = Field(
        default="upsert",
        description=(
            "'upsert' (default) always writes each point; 'insert_only' "
            "skips ids that already exist; 'update_only' skips ids that do "
            "not exist yet. Skipped points are counted, never silently "
            "dropped (e.g. 'insert_only')"
        ),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (default) or 'json' (e.g. 'json')",
    )

    # Duplicate point_id rejection ("a duplicate id in one call would
    # silently overwrite itself in Qdrant") is enforced in the
    # qdrant_upsert_points tool body (as a ToolError), not here -- see the
    # module docstring in tools/payload.py for why a cross-field business
    # rule like this one is raised from the tool body rather than a
    # model_validator.


class DeletePointsInput(BaseModel):
    """Input model for deleting points by id or by filter.

    Two independent hazards this model guards against. An empty
    ``filters`` object would match, and — once confirmed — delete, the
    whole collection, so it is refused outright regardless of ``confirm``.
    And a ``selector="filter"`` delete run without ``confirm`` never
    deletes anything at all: it becomes a free dry-run count instead, since
    there is otherwise no way to preview a filter's blast radius before it
    is too late to undo.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    selector: Literal["ids", "filter"] = Field(
        ...,
        description=(
            "Delete specific points by id ('ids') or everything matching a "
            "filter ('filter') (e.g. 'ids')"
        ),
    )
    collection_name: str = Field(
        ...,
        min_length=1,
        description=(
            "Target collection; required so confirm can be checked "
            "against it and so the dry-run count has somewhere to count "
            "in (e.g. 'docs')"
        ),
    )
    ids: Optional[list[str]] = Field(
        default=None,
        max_length=10000,
        description=(
            "Point ids to delete when selector='ids'; required and "
            "non-empty in that case. One id uses the single-point "
            "endpoint, more than one uses the bulk endpoint (e.g. "
            "['p1', 'p2'])"
        ),
    )
    filters: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Payload filter when selector='filter'; required and must not "
            "be empty in that case, since an empty filter would match "
            'every point, same syntax as Qdrant (e.g. {"must": [{"key": '
            '"lang", "match": {"value": "it"}}]}). See the resource '
            "docs://reference/filters"
        ),
    )
    confirm: Optional[ExactString] = Field(
        default=None,
        description=(
            "Must equal collection_name exactly (not whitespace- or "
            "case-normalised) for the delete to actually run. With "
            "selector='filter', omitting confirm performs a free dry-run "
            "count instead of deleting anything -- the safe way to check "
            "a filter's blast radius first (e.g. 'docs')"
        ),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (default) or 'json' (e.g. 'json')",
    )

    # selector/argument pairing is enforced in the qdrant_delete_points tool
    # body via guardrails.require_selector_args (as a ToolError), not here --
    # see the module docstring in tools/payload.py for why.


def _retrieve_rows(payload: dict[str, Any]) -> list[str]:
    """Render a batch retrieve result, naming missing ids up front."""
    found = payload.get("found") or []
    not_found = payload.get("not_found") or []
    rows = [
        bullet("Collection", payload.get("collection_name")),
        bullet("Requested", payload.get("requested")),
        bullet("Found", len(found)),
    ]
    if not_found:
        rows.append(bullet("Not found", ", ".join(str(pid) for pid in not_found)))
    rows.append("")
    for hit in found:
        rows.append(f"## {hit.get('id')}")
        for key, value in (hit.get("payload") or {}).items():
            rows.append(bullet(key, value))
        for name, shape in (hit.get("vectors") or {}).items():
            rows.append(bullet(f"vector:{name}", shape))
        rows.append("")
    return rows


def _pagination_footer(payload: dict[str, Any]) -> str:
    """Render the closing line: the offset for the next page, or the end.

    ``next_offset`` may be an opaque cursor string rather than an integer;
    it is rendered as-is, in backticks, ready to pass back as ``offset``.
    """
    if payload.get("has_more"):
        return bullet(
            "More pages",
            f"yes — pass `{payload.get('next_offset')}` as offset to fetch "
            "the next page",
        )
    return bullet("More pages", "no — the result set is complete")


def _scroll_rows(payload: dict[str, Any]) -> list[str]:
    """Render a scroll page as a markdown point list, in returned order."""
    items = payload.get("items") or []
    total = payload.get("total")
    header = [
        bullet("Collection", payload.get("collection_name")),
        bullet(
            "Returned",
            f"{payload.get('count')} of {total if total is not None else 'unknown'} "
            f"(offset {payload.get('offset')})",
        ),
        "",
    ]
    if not items:
        return [*header, "No points matched this filter."]
    lines: list[str] = []
    for point in items:
        lines.append(f"## {point.get('id')}")
        for key, value in (point.get("payload") or {}).items():
            lines.append(bullet(key, value))
        lines.append("")
    lines.append(_pagination_footer(payload))
    return [*header, *lines]


def _count_rows(payload: dict[str, Any]) -> list[str]:
    """Render a point count result."""
    return [
        bullet("Collection", payload.get("collection_name")),
        bullet("Count", payload.get("count")),
        bullet("Exact", payload.get("exact")),
    ]


def _facet_rows(payload: dict[str, Any]) -> list[str]:
    """Render facet hits as a markdown value/count list."""
    hits = payload.get("hits") or []
    rows = [
        bullet("Collection", payload.get("collection_name")),
        bullet("Key", payload.get("key")),
        bullet("Exact", payload.get("exact")),
        "",
    ]
    if not hits:
        rows.append("No values found for this key.")
        return rows
    rows.extend(f"- **{hit.get('value')}**: {hit.get('count')}" for hit in hits)
    return rows


def _upsert_rows(payload: dict[str, Any]) -> list[str]:
    """Render an upsert outcome, whether it addressed one point or a batch."""
    return [
        bullet("Collection", payload.get("collection_name")),
        bullet("Upserted", payload.get("upserted")),
        bullet("Skipped", payload.get("skipped")),
    ]


def _delete_rows(payload: dict[str, Any]) -> list[str]:
    """Render a delete outcome, normalised across all three endpoints."""
    return [
        bullet("Collection", payload.get("collection_name")),
        bullet("Deleted", payload.get("deleted")),
        bullet("Selector", payload.get("selector")),
    ]


async def _dispatch_upsert(ctx: ToolContext, params: UpsertPointsInput) -> object:
    """Route an upsert to the flat single-point endpoint or the batch one.

    Exactly one point uses ``POST /points`` with a flat body
    (``UpsertPointRequest``, points.py:600-620); more than one uses ``POST
    /points/batch`` with a ``points`` list (``BatchUpsertRequest``,
    points.py:654-668). ``collection_name`` is omitted from the body
    entirely when unset, the same convention every read tool in this module
    uses to let the middleware fall back to its own configured default.
    """
    if len(params.points) == 1:
        point = params.points[0]
        body: dict[str, Any] = {
            "point_id": point.point_id,
            "text": point.text,
            "payload": point.payload,
            "update_mode": params.update_mode,
        }
        if params.collection_name is not None:
            body["collection_name"] = params.collection_name
        return await ctx.client.request("POST", "/points", json=body)

    body = {
        "points": [point.model_dump() for point in params.points],
        "update_mode": params.update_mode,
    }
    if params.collection_name is not None:
        body["collection_name"] = params.collection_name
    return await ctx.client.request("POST", "/points/batch", json=body)


async def _dispatch_delete(ctx: ToolContext, params: DeletePointsInput) -> object:
    """Route a delete to the endpoint matching its selector and cardinality.

    ``selector="filter"`` always calls ``POST /points/delete-by-filter``.
    ``selector="ids"`` with exactly one id calls ``DELETE
    /points/{point_id}`` (points.py:1787-1809): that handler's
    ``collection_name`` parameter sits after two ``Depends(...)``
    parameters and is not part of a Pydantic request body, so FastAPI
    treats it as a query parameter, not a body field -- confirmed by
    reading the handler signature directly rather than assumed. More than
    one id calls ``DELETE /points`` (``BulkDeleteRequest``,
    points.py:682-694) with a JSON body instead.
    """
    if params.selector == "filter":
        return await ctx.client.request(
            "POST",
            "/points/delete-by-filter",
            json={
                "collection_name": params.collection_name,
                "filters": params.filters or {},
            },
        )
    ids = params.ids or []
    if len(ids) == 1:
        return await ctx.client.request(
            "DELETE",
            f"/points/{quote(ids[0], safe='')}",
            params={"collection_name": params.collection_name},
        )
    return await ctx.client.request(
        "DELETE",
        "/points",
        json={"collection_name": params.collection_name, "ids": ids},
    )


def register(mcp: MCPServer, ctx: ToolContext) -> None:
    """Register the point read and write tools on the server."""

    @mcp.tool(
        name="qdrant_read_points",
        annotations={
            "title": "Retrieve points by id",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def qdrant_read_points(params: RetrievePointsInput) -> StructuredToolResult:
        """Retrieve up to 1000 points by id, reporting misses instead of failing.

        Always calls the batch endpoint (``POST /points/retrieve``), never
        the single-id ``GET /points/{id}`` — even when ``ids`` has one
        entry. The batch endpoint returns whatever subset of the requested
        ids it found; the single-id endpoint would 404 on a miss instead.
        Rather than surface that 404, this tool reconstructs the signal
        itself: the response splits ids into ``found`` (with payload, and
        vectors when requested) and ``not_found`` (bare ids), so one missing
        id among five never fails the whole call — the agent gets a usable
        answer either way and can decide what the miss means.

        ``with_vectors=true`` adds a summary of each found point's stored
        vectors — names and shapes only, **never the raw numbers** — which
        is how you confirm which embedding generation a point currently
        carries (e.g. mid re-embed) without paying for the full vector data.

        Args:
            params (RetrievePointsInput): Validated input containing:
                - ids (list[str]): point ids to retrieve, 1 to 1000 entries,
                  required.
                - collection_name (str | None): collection to read from;
                  omit for the server default.
                - with_vectors (bool): include vector name/shape summaries;
                  default false.
                - response_format (ResponseFormat): 'markdown' or 'json'.

        Returns:
            str: Markdown fact sheet, or JSON with this schema:

            {
                "collection_name": str,
                "requested": int,
                "found": [
                    {
                        "id": str,
                        "payload": {...},
                        "vectors": {"<name>": {"type": str, ...}} | null
                    }
                ],
                "not_found": [str, ...]   # requested ids that returned nothing
            }

        Examples:
            - Use when: you already know the point ids and want their
              payload directly, with no embedding or search cost.
            - Use when: confirming whether specific ids still exist after a
              delete or a reindex.
            - Don't use when: you are searching by meaning rather than by
              known id -> use qdrant_search instead.
            - Don't use when: you want to enumerate points by filter rather
              than by known id -> use qdrant_scroll_points instead.

        Error Handling:
            Input validation (1-1000 ids) is handled by Pydantic before any
            HTTP call. Missing ids never raise — they are reported in
            not_found. A malformed id (not a UUID or unsigned integer) is a
            different failure mode: it is not caught locally (id format is
            the middleware's own rule, not duplicated here to avoid the two
            drifting), so it surfaces as "Error: invalid request ..." from
            the middleware's 422 rather than appearing in not_found — that
            distinction matters when deciding whether to retry (fix the id
            and resend) or treat it as a genuine miss. Returns "Error: ..."
            (see describe_error) for every other middleware failure (bad
            collection name, unreachable middleware, etc.).
        """
        body = params.model_dump(exclude_none=True, exclude={"response_format"})
        try:
            raw = await ctx.client.request(
                "POST", "/points/retrieve", json=body, retryable=True
            )
        except Exception as exc:  # -> agent message
            return error_result(exc)

        data = raw if isinstance(raw, dict) else {}
        found = data.get("results") or []
        returned_ids = {str(hit.get("id")) for hit in found}
        payload = {
            "collection_name": data.get("collection_name"),
            "requested": len(params.ids),
            "found": found,
            "not_found": [pid for pid in params.ids if pid not in returned_ids],
        }
        return render(
            payload,
            params.response_format,
            title="Point retrieve result",
            rows=_retrieve_rows,
        )

    @mcp.tool(
        name="qdrant_scroll_points",
        annotations={
            "title": "Page through points by filter",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def qdrant_scroll_points(params: ScrollPointsInput) -> StructuredToolResult:
        """Page through points matching a filter, ordered but never ranked.

        This is enumeration, not retrieval by relevance: there is no query
        text, no embedding, and no score. Use it to walk a collection, or a
        filtered subset of one, page by page. For "which points best match
        this text" use qdrant_search instead — scroll has no notion of
        relevance at all.

        Paging follows the cursor the middleware returns, not offset
        arithmetic: pass the previous page's ``next_offset`` back as
        ``offset`` to continue (it may be an opaque string on some
        collections, a plain integer on others — treat it as opaque either
        way). ``offset`` and ``order_by`` cannot be combined — Qdrant rejects
        that outright, because an ordered scroll's cursor is the last-seen
        value of the ordered field, not a row count — so page an ordered
        scroll through ``order_by.start_from`` instead. ``include_total=false``
        skips the extra count call the default performs alongside the
        scroll, which is cheaper on large collections when the exact total
        is not actually needed.

        Args:
            params (ScrollPointsInput): Validated input containing:
                - filters (dict | None): Qdrant payload filter; omit to
                  scroll the whole collection.
                - limit (int): points per page, 1-1000, default 20.
                - offset (int | str | None): page cursor — an integer offset
                  or a previous page's next_offset value; mutually exclusive
                  with order_by.
                - order_by (OrderBySpec | None): sort by an indexed payload
                  field (key, direction, start_from) instead of storage
                  order; mutually exclusive with offset.
                - with_payload (bool | list[str]): true for every field,
                  false for none, or specific keys; default true.
                - include_total (bool): also count total matches; default
                  true.
                - collection_name (str | None): collection to scroll; omit
                  for the server default.
                - response_format (ResponseFormat): 'markdown' or 'json'.

        Returns:
            str: Markdown point list, or JSON with this schema:

            {
                "collection_name": str,
                "total": int | null,      # null when include_total=false
                "count": int,
                "offset": int | str | null,
                "has_more": bool,
                "next_offset": int | str | null,
                "items": [{"id": str, "payload": {...}}, ...]
            }

        Examples:
            - Use when: exporting or auditing every point in a collection, a
              page at a time.
            - Use when: listing points matching a payload filter with no
              interest in ranking them.
            - Use when: paging through a large result set in a stable,
              indexed order (order_by).
            - Don't use when: relevance to a query matters -> use
              qdrant_search instead.
            - Don't use when: you only need a number, not the points
              themselves -> use qdrant_count_points instead.

        Error Handling:
            Returns "Error: invalid request ..." (422) if offset and
            order_by are combined, or if order_by.start_from does not match
            the ordered field's type. Returns "Error: ..." (see
            describe_error) for every other middleware failure.
        """
        body = params.model_dump(exclude_none=True, exclude={"response_format"})
        try:
            raw = await ctx.client.request(
                "POST", "/points/scroll", json=body, retryable=True
            )
        except Exception as exc:  # -> agent message
            return error_result(exc)

        data = raw if isinstance(raw, dict) else {}
        # ScrollPointsResponse always carries this field under the name
        # next_page_offset (middleware points.py:396), null included when
        # there is no next page — read exactly that key, never invent one.
        page = paginate(
            data.get("results") or [],
            total=data.get("total"),
            offset=params.offset if params.offset is not None else 0,
            limit=params.limit,
            next_offset=data.get("next_page_offset"),
        )
        payload = {"collection_name": data.get("collection_name"), **page}
        return render(
            payload,
            params.response_format,
            title="Scrolled points",
            rows=_scroll_rows,
        )

    @mcp.tool(
        name="qdrant_count_points",
        annotations={
            "title": "Count points matching a filter",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def qdrant_count_points(params: CountPointsInput) -> StructuredToolResult:
        """Count points matching a filter without retrieving any of them.

        An absent or empty filter counts the whole collection. With the
        default ``exact=true`` Qdrant scans to produce an exact count;
        ``exact=false`` returns a fast approximate estimate instead, much
        cheaper on a large collection when the precise number does not
        matter.

        Args:
            params (CountPointsInput): Validated input containing:
                - filters (dict | None): Qdrant payload filter; omit to
                  count the whole collection.
                - exact (bool): exact count vs a cheap estimate; default
                  true.
                - collection_name (str | None): collection to count in;
                  omit for the server default.
                - response_format (ResponseFormat): 'markdown' or 'json'.

        Returns:
            str: Markdown fact sheet, or JSON with this schema:

            {
                "collection_name": str,
                "count": int,
                "exact": bool
            }

        Examples:
            - Use when: sizing a bulk delete before running it, with the
              same filter, so the blast radius is known first.
            - Use when: checking whether a filter matches anything before
              scrolling or searching it.
            - Don't use when: you need the points themselves, not just a
              number -> use qdrant_scroll_points instead.

        Error Handling:
            Returns "Error: invalid request ..." (422) for a malformed
            filter, and describe_error's standard mapping for every other
            middleware failure.
        """
        body = params.model_dump(exclude_none=True, exclude={"response_format"})
        try:
            raw = await ctx.client.request(
                "POST", "/points/count", json=body, retryable=True
            )
        except Exception as exc:  # -> agent message
            return error_result(exc)

        payload = raw if isinstance(raw, dict) else {}
        return render(
            payload,
            params.response_format,
            title="Point count",
            rows=_count_rows,
        )

    @mcp.tool(
        name="qdrant_facet_points",
        annotations={
            "title": "Facet a payload field",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def qdrant_facet_points(params: FacetPointsInput) -> StructuredToolResult:
        """Return the distinct values of one payload field and each value's count.

        ``key`` must already carry a payload index — Qdrant refuses to facet
        an unindexed field outright, and faceting one without an index would
        be slow anyway, scanning every point instead of using one. If this
        call fails saying the key cannot be faceted, create an index for it
        with qdrant_create_payload_index first, then retry.

        Args:
            params (FacetPointsInput): Validated input containing:
                - key (str): indexed payload field to facet on, required.
                - filters (dict | None): Qdrant payload filter restricting
                  which points are faceted; omit to facet the whole
                  collection.
                - limit (int): maximum distinct values to return, 1-1000,
                  default 20.
                - exact (bool): exact counts vs a faster approximation;
                  default true.
                - collection_name (str | None): collection to facet in;
                  omit for the server default.
                - response_format (ResponseFormat): 'markdown' or 'json'.

        Returns:
            str: Markdown value/count list, or JSON with this schema:

            {
                "collection_name": str,
                "key": str,
                "exact": bool,
                "hits": [{"value": str | int | bool, "count": int}, ...]
            }

        Examples:
            - Use when: building a filter UI, or checking what values a
              field actually takes before writing a filter against it.
            - Use when: sanity-checking that a payload field was indexed and
              populated as expected after an ingest.
            - Don't use when: the field has no payload index yet -> call
              qdrant_create_payload_index first.
            - Don't use when: you want the points themselves rather than a
              value/count summary -> use qdrant_scroll_points instead.

        Error Handling:
            Returns "Error: invalid request — key '<key>' cannot be faceted:
            ..." (400) when the field is not indexed, and describe_error's
            standard mapping for every other middleware failure.
        """
        body = params.model_dump(exclude_none=True, exclude={"response_format"})
        try:
            raw = await ctx.client.request(
                "POST", "/points/facet", json=body, retryable=True
            )
        except Exception as exc:  # -> agent message
            return error_result(exc)

        payload = raw if isinstance(raw, dict) else {}
        return render(
            payload,
            params.response_format,
            title="Facet result",
            rows=_facet_rows,
        )

    @mcp.tool(
        name="qdrant_upsert_points",
        annotations={
            "title": "Upsert points from text",
            "readOnlyHint": False,
            # Existing point IDs can be overwritten.
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def qdrant_upsert_points(params: UpsertPointsInput) -> StructuredToolResult:
        """Embed text server-side and write it as one point, or a batch of points.

        Each entry in ``points`` needs a ``point_id`` (a UUID or an unsigned
        integer, given as a string) and non-blank ``text``; ``text`` is
        embedded by the server, never by the caller. Exactly one point calls
        the flat single-point endpoint; more than one (up to 10000 in a
        single call) calls the batch endpoint instead — this tool picks
        automatically, so the caller never has to choose. A duplicate
        ``point_id`` within the same call is rejected before any HTTP
        request, since it would silently overwrite itself in Qdrant with no
        way to tell which write actually won.

        ``update_mode`` controls what happens to ids that already exist (or
        do not): the default ``"upsert"`` always writes; ``"insert_only"``
        skips ids that already exist, leaving them untouched; ``"update_only"``
        skips ids that do not exist yet, never creating new points. Either
        way, skipped points are counted in the response rather than silently
        dropped — a ``skipped`` count greater than zero is not a failure, it
        is ``update_mode`` doing exactly what it was asked to do.

        Args:
            params (UpsertPointsInput): Validated input containing:
                - points (list[PointSpec]): 1 to 10000 points, no duplicate
                  point_id, required. Each has point_id (str, UUID or
                  unsigned integer), text (str, non-blank), and payload
                  (dict, default {}).
                - collection_name (str | None): collection to upsert into;
                  omit for the server default.
                - update_mode ("upsert" | "insert_only" | "update_only"):
                  write policy for existing/missing ids; default "upsert".
                - response_format (ResponseFormat): 'markdown' or 'json'.

        Returns:
            str: Markdown fact sheet, or JSON with this schema:

            {
                "collection_name": str,
                "upserted": int,   # points actually written
                "skipped": int     # points update_mode declined to write
            }

        Examples:
            - Use when: writing new content, or overwriting known point ids
              with fresh text and payload, from 1 up to 10000 points in one
              call.
            - Use when: only adding points that do not already exist, or
              only refreshing ones that do -> set update_mode accordingly.
            - Don't use when: the content lives in a file on the middleware
              host or needs source-level chunk/id management -> use
              qdrant_ingest_file or qdrant_ingest_chunks instead, which
              handle chunk-to-point id derivation and source replacement.

        Error Handling:
            A blank/whitespace-only text or more than 10000 points are
            rejected by Pydantic before any HTTP call is made. A duplicate
            point_id across the batch is refused the same way but as a
            ToolError raised from this tool's own body, so it returns the
            same "Error: ..." (see describe_error) shape as every other
            guardrail refusal instead of the SDK's raw validation-error
            envelope. A malformed point_id (not a UUID or unsigned integer)
            is not caught locally — that is the middleware's own rule — and
            surfaces as "Error: invalid request ..." (422) instead. Returns
            "Error: ..." for every other middleware failure.
        """
        try:
            ids = [point.point_id for point in params.points]
            refuse_if(
                len(set(ids)) != len(ids),
                "points contains duplicate point_id values",
            )
            raw = await _dispatch_upsert(ctx, params)
        except Exception as exc:  # -> agent message
            return error_result(exc)

        data = raw if isinstance(raw, dict) else {}
        # UpsertPointResponse (single point, points.py:623-630) and
        # BatchUpsertResponse (batch, points.py:671-679) report the same
        # facts under different shapes: the single response's `skipped` is
        # a bool describing the one point addressed; the batch response's
        # `skipped` is already a count. Normalise both into one
        # upserted/skipped count pair.
        if "point_id" in data:
            was_skipped = bool(data.get("skipped"))
            upserted = 0 if was_skipped else 1
            skipped = 1 if was_skipped else 0
        else:
            upserted = data.get("upserted", 0)
            skipped = data.get("skipped", 0)
        payload = {
            "collection_name": data.get("collection_name"),
            "upserted": upserted,
            "skipped": skipped,
        }
        return render(
            payload,
            params.response_format,
            title="Upsert result",
            rows=_upsert_rows,
        )

    @mcp.tool(
        name="qdrant_delete_points",
        annotations={
            "title": "Delete points by id or filter",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def qdrant_delete_points(params: DeletePointsInput) -> StructuredToolResult:
        """Delete points by id or by filter. Deletion is irreversible.

        This is the most dangerous tool in this server. ``selector="ids"``
        deletes specific points: one id calls the single-point endpoint,
        several call the bulk endpoint; either way a missing or mismatched
        ``confirm`` is refused outright, with no HTTP call made at all —
        there is no dry run for an id-based delete, since the caller already
        named exactly what it wants gone.

        ``selector="filter"`` is the one that needs a rehearsal:
        **omitting ``confirm`` runs a free dry-run count instead of
        deleting anything.** It calls the same count endpoint the filter
        would otherwise be deleted through, and reports how many points it
        matches, so the caller can see the blast radius before opting in
        with ``confirm=<collection_name>``. Only when ``confirm`` is present
        and matches ``collection_name`` exactly does the filter delete
        actually run. An empty ``filters`` object is refused before any
        HTTP call either way, confirmed or not, since it would match every
        point in the collection.

        Args:
            params (DeletePointsInput): Validated input containing:
                - selector ("ids" | "filter"): which addressing mode to
                  use, required.
                - collection_name (str): target collection, required.
                - ids (list[str] | None): point ids to delete; required and
                  non-empty when selector="ids".
                - filters (dict | None): payload filter; required and
                  non-empty when selector="filter".
                - confirm (str | None): must equal collection_name exactly;
                  omitting it with selector="filter" triggers the dry run
                  instead of a refusal.
                - response_format (ResponseFormat): 'markdown' or 'json'
                  (ignored for the dry-run message, which is always plain
                  text).

        Returns:
            str: For a filter dry run, a plain-text sentence naming the
            match count and the exact confirm value to resend. Otherwise,
            a markdown fact sheet or JSON with this schema:

            {
                "collection_name": str,
                "deleted": int,
                "selector": "ids" | "filter"
            }

        Examples:
            - Use when: removing specific points whose ids are already
              known -> selector="ids".
            - Use when: removing every point matching a payload condition,
              after first checking the dry-run count that a bare call
              (no confirm) returns -> selector="filter".
            - Don't use when: unsure what a filter matches -> call
              qdrant_count_points or this tool without confirm first; both
              are free and change nothing.
            - Don't use when: the whole collection should go, not just its
              points -> use qdrant_delete_collection instead.

        Error Handling:
            A missing/empty ids or filters for the chosen selector is
            refused before any HTTP call -- as a ToolError raised from this
            tool's own body, so it returns the same "Error: ..." (see
            describe_error) shape as every other guardrail refusal instead
            of the SDK's raw validation-error envelope. A refused
            confirmation (selector="ids" with a missing or mismatched
            confirm) also makes no HTTP call. The count-based dry run
            (`count_matching`) raises rather than reporting zero if the
            count endpoint's response cannot be unambiguously read, since a
            fabricated zero would make a mass delete look safe when it is
            not. Returns "Error: ..." (see describe_error) for every other
            middleware failure.
        """
        try:
            require_selector_args(params.selector, params.ids, params.filters)
            require_non_blank_ids(params.ids)
            if params.selector == "filter" and params.confirm is None:
                matched = await count_matching(
                    ctx, params.collection_name, params.filters or {}
                )
                dry_run = {
                    "collection_name": params.collection_name,
                    "matched": matched,
                    "deleted": 0,
                    "dry_run": True,
                }
                return render(
                    dry_run,
                    params.response_format,
                    title="Delete points dry run",
                    rows=lambda payload: [
                        (
                            f"Dry run: this filter matches {payload['matched']} "
                            f"point(s) in {payload['collection_name']!r} right "
                            "now. Nothing was deleted. Call again with "
                            f"confirm={payload['collection_name']!r} to delete "
                            "them — that is a second call, so the set it "
                            "removes is whatever matches then, not this count."
                        )
                    ],
                )
            require_confirm(params.confirm, params.collection_name)
            raw = await _dispatch_delete(ctx, params)
        except Exception as exc:  # -> agent message
            return error_result(exc)

        data = raw if isinstance(raw, dict) else {}
        # All three delete endpoints now report `deleted` — the single-point
        # one counts 1 or 0 (an absent id is an idempotent no-op there, not a
        # 404), the bulk and by-filter ones count the ids that actually
        # existed. The fallback below only covers a middleware predating that
        # change; assuming 1 is wrong whenever the id was already gone, which
        # is exactly why the endpoint reports the count itself now.
        deleted = data.get("deleted")
        if deleted is None:
            deleted = 1
        payload = {
            "collection_name": data.get("collection_name"),
            "deleted": deleted,
            "selector": params.selector,
        }
        return render(
            payload,
            params.response_format,
            title="Delete result",
            rows=_delete_rows,
        )

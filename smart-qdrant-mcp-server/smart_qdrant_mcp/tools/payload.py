"""Payload-only writes: set, delete-keys, and clear, by id or by filter.

Three tools, seven middleware endpoints
(``qdrant_middleware/api/routers/points.py:1897-2178``), one shared shape:
``selector="ids"`` addresses specific points, ``selector="filter"`` addresses
everything a payload filter matches.

The two selectors are not the same operation wearing a costume. ``selector="filter"``
calls one of the three by-filter endpoints (``PATCH /points/payload``, ``POST
/points/payload/delete-keys``, ``POST /points/payload/clear``) and gets back a
single ``matched`` count from one Qdrant-side write. ``selector="ids"`` cannot do
that: every one of the per-point endpoints
(``PATCH``/``PUT /points/{point_id}/payload``, ``POST
/points/{point_id}/payload/delete-keys``, ``POST
/points/{point_id}/payload/clear``) takes exactly one point id in its path, so N
ids mean N sequential HTTP requests, each able to fail independently of the
others. ``_fan_out`` is deliberately not atomic: it keeps going after a
failure rather than stopping the batch, and returns which ids succeeded and
which failed with why, so a caller never has to guess whether "some ids
returned failures" left the successful writes in place — they always did. An
agent that mistakes a partial failure here for a total one will needlessly
re-send already-applied writes; one that mistakes it for a total success will
walk away thinking untouched ids were updated. The markdown rendering leads
with an explicit "not atomic" sentence whenever ``failed`` is non-empty, for
the same reason ``qdrant_create_payload_index`` leads with its own partial-failure
sentence rather than burying it under a clean-looking bullet list.

``qdrant_set_payload(selector="filter", mode="replace")`` is refused before any
HTTP call: the middleware has ``PATCH /points/payload`` (a filtered merge) but
no filtered ``PUT`` — a full replace has to decide, per point, which existing
fields to carry forward (``PROTECTED_PAYLOAD_FIELDS``, points.py:220), which a
single filtered write cannot do across many points with different existing
payloads. The refusal names both workarounds: merge instead, or resolve the
filter to explicit ids first with ``qdrant_scroll_points`` and call again with
``selector="ids"``.

That refusal is raised as a ``ToolError`` in the tool body, not as a Pydantic
``model_validator``. A ``model_validator`` failure surfaces to an MCP client
as the SDK's own validation-error envelope ("Error executing tool
qdrant_set_payload: 1 validation error ...", with a pydantic.dev URL) rather
than the clean ``describe_error``-shaped sentence every other guardrail in
this service produces. Raising ``ToolError`` from the tool body instead routes
this refusal through the same ``describe_error`` path as every other
confirmation gate, while still making zero HTTP calls.

Every other cross-field business rule in this module -- the selector/argument
pairing (shared as ``guardrails.require_selector_args``), the explicit
``collection_name`` requirement for ``mode="replace"``, and the blank-key
rejection on ``qdrant_delete_payload_keys`` -- is raised the same way, from
the owning tool's body, for the same reason. Only shape and type validation
(``min_length``/``max_length``, ``Literal`` membership, required-field
presence) stays on the Pydantic models themselves: the framework envelope is
the right answer there, since none of it carries an agent-facing sentence
worth protecting from that envelope's noise.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Optional
from urllib.parse import quote

from mcp.server import MCPServer
from pydantic import BaseModel, ConfigDict, Field, field_validator

from smart_qdrant_mcp.context import ToolContext
from smart_qdrant_mcp.errors import (
    MiddlewareError,
    ToolError,
    error_result,
    redact_detail,
)
from smart_qdrant_mcp.formatting import (
    ResponseFormat,
    StructuredToolResult,
    bullet,
    render,
)
from smart_qdrant_mcp.guardrails import (
    refuse_if,
    require_confirm,
    require_fan_out_size,
    require_non_blank_ids,
    require_selector_args,
)
from smart_qdrant_mcp.tools.collections import ExactString


# Raised as a ToolError from the qdrant_set_payload tool body -- not from a
# Pydantic model_validator -- so it flows through describe_error like every
# other guardrail refusal in this service, rather than surfacing as the
# SDK's own validation-error envelope. See the module docstring for why.
REPLACE_BY_FILTER_MESSAGE = (
    "replace is only supported for explicit ids: the middleware has no "
    "replace-by-filter endpoint. Use mode='merge', or collect the ids with "
    "qdrant_scroll_points and call again with selector='ids'"
)


class SetPayloadInput(BaseModel):
    """Input model for merging or replacing payload fields on points.

    ``mode="replace"`` combined with ``selector="filter"`` is rejected in the
    ``qdrant_set_payload`` tool body (as a ``ToolError``), not here — see the
    module docstring for why.

    ``mode="replace"`` discards every unprotected field on the addressed
    points — the same destruction ``qdrant_clear_payload`` performs — so it
    is gated the same way: ``confirm`` is required and must equal
    ``collection_name`` exactly, and ``collection_name`` itself must be given
    explicitly rather than defaulted, since a confirmation token has to be
    checked against a name the caller actually gave. ``mode="merge"`` (the
    default) needs neither: every field it can change is reversible in
    place, and nothing existing is ever discarded.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    selector: Literal["ids", "filter"] = Field(
        ...,
        description=(
            "Address specific points by id ('ids') or every point matching "
            "a filter ('filter') (e.g. 'ids')"
        ),
    )
    mode: Literal["merge", "replace"] = Field(
        default="merge",
        description=(
            "'merge' (default) adds/overwrites the given fields, keeping "
            "everything else, no confirm required; 'replace' discards every "
            "existing field except protected ones (text, source_id), is "
            "only valid with selector='ids', and requires collection_name "
            "and a matching confirm (e.g. 'merge')"
        ),
    )
    payload: dict[str, Any] = Field(
        ...,
        min_length=1,
        description=(
            "Non-empty non-text fields to write (e.g. {'lang': 'it', "
            "'version': 3}). Use qdrant_upsert_points to change text and "
            "regenerate its vector."
        ),
    )
    ids: Optional[list[str]] = Field(
        default=None,
        max_length=10000,
        description=(
            "Point ids to address when selector='ids'; required and "
            "non-empty in that case, and capped at 1000 because each id "
            "costs its own request -- see the tool description for the "
            "non-atomicity that implies. For a larger set use "
            "selector='filter', which writes them in one request "
            "(e.g. ['p1', 'p2'])"
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
    collection_name: Optional[str] = Field(
        default=None,
        description=(
            "Target collection; omit to use the server default when "
            "mode='merge'. Required (not defaulted) when mode='replace', "
            "since confirm has to be checked against a name the caller "
            "actually gave (e.g. 'docs')"
        ),
    )
    confirm: Optional[ExactString] = Field(
        default=None,
        description=(
            "Required only when mode='replace': must equal collection_name "
            "exactly (not whitespace- or case-normalised), e.g. "
            "confirm='docs' for collection_name='docs'"
        ),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (default) or 'json' (e.g. 'json')",
    )

    # The selector/argument pairing and the explicit-collection_name
    # requirement for mode="replace" are both enforced in the
    # qdrant_set_payload tool body (as ToolError), not here -- see the
    # module docstring for why.


class DeletePayloadKeysInput(BaseModel):
    """Input model for removing named payload keys from points."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    selector: Literal["ids", "filter"] = Field(
        ...,
        description=(
            "Address specific points by id ('ids') or every point matching "
            "a filter ('filter') (e.g. 'ids')"
        ),
    )
    keys: list[str] = Field(
        ...,
        min_length=1,
        description="Payload keys to remove, none blank (e.g. ['lang', 'draft'])",
    )
    ids: Optional[list[str]] = Field(
        default=None,
        max_length=10000,
        description=(
            "Point ids to address when selector='ids'; required and "
            "non-empty in that case, and capped at 1000 because each id "
            "costs its own request -- see the tool description for the "
            "non-atomicity that implies. For a larger set use "
            "selector='filter', which writes them in one request "
            "(e.g. ['p1', 'p2'])"
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
    collection_name: Optional[str] = Field(
        default=None,
        description="Target collection; omit to use the server default (e.g. 'docs')",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (default) or 'json' (e.g. 'json')",
    )

    @field_validator("keys")
    @classmethod
    def strip_keys(cls, value: list[str]) -> list[str]:
        """Strip whitespace from each key.

        Rejecting a key that turns out blank *after* stripping is a
        cross-field-style business rule ("matching the middleware's own
        rule") rather than shape validation, so it is enforced in the
        qdrant_delete_payload_keys tool body (as a ToolError) instead of
        here -- see the module docstring for why. Only the normalisation
        (stripping) stays on the model, since every field on this call still
        needs it applied before that tool-body check runs.
        """
        return [key.strip() for key in value]

    # The selector/argument pairing is enforced in the
    # qdrant_delete_payload_keys tool body, not here -- see the module
    # docstring for why.


class ClearPayloadInput(BaseModel):
    """Input model for clearing points' payloads, keeping protected fields.

    ``confirm`` is required regardless of selector: clearing removes every
    unprotected key from every addressed point, with no dry run offered (there
    is nothing to size the way a delete's match count sizes a blast radius --
    every match is fully cleared either way). ``collection_name`` is required
    too, unlike the other two tools in this module: a confirmation token has
    to be checked against something the caller actually named, and a target
    that defaulted to the server's own collection could never be genuinely
    confirmed against. ``qdrant_create_collection`` (``recreate=true``) and
    ``qdrant_ingest_file``/``qdrant_ingest_chunks`` (``replace_source=true``)
    apply the same rule for the same reason.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    selector: Literal["ids", "filter"] = Field(
        ...,
        description=(
            "Address specific points by id ('ids') or every point matching "
            "a filter ('filter') (e.g. 'ids')"
        ),
    )
    ids: Optional[list[str]] = Field(
        default=None,
        max_length=10000,
        description=(
            "Point ids to address when selector='ids'; required and "
            "non-empty in that case, and capped at 1000 because each id "
            "costs its own request -- see the tool description for the "
            "non-atomicity that implies. For a larger set use "
            "selector='filter', which writes them in one request "
            "(e.g. ['p1', 'p2'])"
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
    collection_name: str = Field(
        ...,
        min_length=1,
        description=(
            "Target collection, required (not defaulted): confirm has to be "
            "checked against a name the caller actually gave (e.g. 'docs')"
        ),
    )
    confirm: Optional[ExactString] = Field(
        default=None,
        description=(
            "Required: must equal collection_name exactly (not whitespace- "
            "or case-normalised) to confirm the clear, e.g. confirm='docs' "
            "for collection_name='docs'"
        ),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (default) or 'json' (e.g. 'json')",
    )

    # The selector/argument pairing is enforced in the qdrant_clear_payload
    # tool body, not here -- see the module docstring for why.


async def _fan_out(
    ctx: ToolContext,
    *,
    method: str,
    path_for: Callable[[str], str],
    body: dict[str, Any],
    ids: list[str],
) -> dict[str, Any]:
    """Apply a per-point payload write to each id in turn.

    The middleware's per-point payload endpoints take a single ``point_id``
    in the path, so N ids mean N requests. This is deliberately not atomic:
    a failure part-way through leaves earlier writes applied, which the
    report makes explicit rather than hiding behind a single error. Only
    ``MiddlewareError`` (a non-success HTTP response) is caught per id --
    anything else (a bug, a cancellation) propagates and aborts the whole
    fan-out, since only an HTTP-level failure is the "this one point had a
    problem" case the report is designed to describe.
    """
    updated: list[str] = []
    failed: list[dict[str, str]] = []
    collection_name = None
    for point_id in ids:
        try:
            raw = await ctx.client.request(method, path_for(point_id), json=body)
        except MiddlewareError as exc:
            # redact_detail, not str(exc.detail): this is the one place in the
            # service where unredacted upstream text could otherwise reach the
            # client -- every other failure path returns through
            # describe_error, which redacts on every branch.
            failed.append({"point_id": point_id, "reason": redact_detail(exc.detail)})
        else:
            updated.append(point_id)
            if collection_name is None and isinstance(raw, dict):
                collection_name = raw.get("collection_name")
    return {
        # Every id failing leaves no response to read the name from.
        "collection_name": collection_name or body.get("collection_name"),
        "updated": updated,
        "failed": failed,
        "atomic": False,
    }


def _fan_out_rows(payload: dict[str, Any]) -> list[str]:
    """Render a per-id write report, leading with non-atomicity on any failure."""
    rows = [
        bullet("Collection", payload.get("collection_name")),
        bullet("Updated", len(payload.get("updated") or [])),
        bullet("Failed", len(payload.get("failed") or [])),
    ]
    failed = payload.get("failed") or []
    if failed:
        rows.append("")
        rows.extend(
            f"- {failure['point_id']}: {failure['reason']}" for failure in failed
        )
        rows.extend(
            [
                "",
                "This operation is not atomic: the updates listed above were "
                "applied, the failures were not. Fix the causes and re-run "
                "for the failed ids only.",
            ]
        )
    return rows


def _filter_write_rows(payload: dict[str, Any]) -> list[str]:
    """Render a by-filter payload write's matched count."""
    return [
        bullet("Collection", payload.get("collection_name")),
        bullet("Matched", payload.get("matched")),
    ]


def _point_payload_path(point_id: str, suffix: str = "") -> str:
    """Build a per-point payload path, URL-encoding the id as one segment."""
    return f"/points/{quote(point_id, safe='')}/payload{suffix}"


def register(mcp: MCPServer, ctx: ToolContext) -> None:
    """Register the three payload write tools on the server."""

    @mcp.tool(
        name="qdrant_set_payload",
        annotations={
            "title": "Set point payload (merge or replace)",
            "readOnlyHint": False,
            # mode="replace" discards every unprotected field on the
            # addressed points -- the same worst case qdrant_clear_payload
            # gates. The hint describes that worst case, not the common
            # mode="merge" call, the same reasoning qdrant_create_collection
            # applies to recreate=true: a false hint here would stop clients
            # from asking for confirmation on the call that actually needs it.
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def qdrant_set_payload(params: SetPayloadInput) -> StructuredToolResult:
        """Merge or replace payload fields on points addressed by id or filter.

        ``mode="merge"`` (the default) adds and overwrites the given non-text fields,
        leaving everything else on the point untouched -- no confirmation
        needed. ``mode="replace"`` discards every existing field except the
        protected service fields (``text``, ``source_id``, carried forward
        automatically) and is only valid with ``selector="ids"`` --
        ``mode="replace"`` with ``selector="filter"`` is refused before any
        HTTP call, because a filtered replace would need to decide, per
        point, which fields to keep, and one write cannot do that across many
        differing payloads. Use ``mode="merge"`` instead, or resolve the
        filter to explicit ids first with ``qdrant_scroll_points`` and call
        again with ``selector="ids"``.

        ``mode="replace"`` is confirm-gated the same way ``qdrant_clear_payload``
        is: ``collection_name`` must be given explicitly (not defaulted) and
        ``confirm`` must equal it exactly, or the call is refused before any
        HTTP request is made.

        ``selector="ids"`` is **not atomic**: the middleware's per-point
        payload endpoints take one id each, so N ids mean N sequential
        requests. A failure partway through leaves the earlier writes
        applied -- the response names exactly which ids succeeded and which
        failed with why, rather than failing the whole call. Sending 2+ ids
        that include a duplicate is not itself an error; each occurrence is
        written independently.

        ``selector="filter"`` is one call against the by-filter endpoint and
        reports how many points it matched; there is no per-point detail to
        report for that path.

        Args:
            params (SetPayloadInput): Validated input containing:
                - selector ("ids" | "filter"): addressing mode, required.
                - mode ("merge" | "replace"): write semantics; default
                  "merge"; "replace" requires selector="ids", an explicit
                  collection_name, and a matching confirm.
                - payload (dict): non-empty non-text fields to write, required.
                - ids (list[str] | None): point ids; required and non-empty
                  when selector="ids".
                - filters (dict | None): payload filter; required and
                  non-empty when selector="filter".
                - collection_name (str | None): target collection; omit for
                  the server default when mode="merge"; required when
                  mode="replace".
                - confirm (str | None): required only when mode="replace";
                  must equal collection_name exactly.
                - response_format (ResponseFormat): 'markdown' or 'json'.

        Returns:
            str: Markdown fact sheet, or JSON. For selector="ids":

            {
                "collection_name": str | null,
                "updated": [str, ...],    # ids that were written
                "failed": [{"point_id": str, "reason": str}, ...],
                "atomic": false
            }

            For selector="filter":

            {
                "collection_name": str,
                "matched": int    # points the filter matched and rewrote
            }

        Examples:
            - Use when: adding or overwriting a few payload fields on known
              point ids -> selector="ids", mode="merge" (the common case).
            - Use when: rewriting a field's value across every point that
              matches a filter in one call -> selector="filter".
            - Use when: fully replacing a point's payload down to only its
              protected fields -> selector="ids", mode="replace",
              collection_name=<name>, confirm=<same name>.
            - Don't use when: mode="replace" is wanted across many points at
              once -> not supported; merge instead, or resolve the filter to
              ids with qdrant_scroll_points first.
            - Don't use when: only removing specific keys, not setting new
              ones -> use qdrant_delete_payload_keys instead.
            - Don't use when: wiping the payload down to nothing but
              protected fields -> use qdrant_clear_payload instead.

        Error Handling:
            An empty payload is rejected by Pydantic before any HTTP call. A
            missing/empty ids or filters for the chosen selector,
            mode="replace" with selector="filter", and mode="replace"
            without an explicit collection_name are all refused before any
            HTTP call too, but each as a ToolError raised from this tool's
            own body rather than a Pydantic validator, so every one of them
            returns the same "Error: ..." (see describe_error) shape as
            every other guardrail refusal instead of the SDK's raw
            validation-error envelope. mode="replace" with a missing or
            mismatched confirm is refused the same way as
            qdrant_clear_payload -- zero HTTP calls. For selector="ids", an
            individual id's failure (e.g. 404 because it does not exist) is
            captured per id in the failed list rather than raised -- the call
            itself only returns "Error: ..." for a failure outside the
            fan-out (an unreachable middleware, for example). For
            selector="filter", any middleware failure returns "Error: ..."
            directly, since there the whole write is one call.
        """
        try:
            require_selector_args(params.selector, params.ids, params.filters)
            require_non_blank_ids(params.ids)
            require_fan_out_size(params.ids)
            if params.mode == "replace":
                refuse_if(
                    not params.collection_name,
                    "mode='replace' requires an explicit collection_name so "
                    "that confirm can be checked against it",
                )
            if params.mode == "replace" and params.selector == "filter":
                raise ToolError(REPLACE_BY_FILTER_MESSAGE)
            if params.mode == "replace":
                require_confirm(params.confirm, params.collection_name)

            if params.selector == "filter":
                body: dict[str, Any] = {
                    "filters": params.filters,
                    "payload": params.payload,
                }
                if params.collection_name is not None:
                    body["collection_name"] = params.collection_name
                raw = await ctx.client.request("PATCH", "/points/payload", json=body)
                data = raw if isinstance(raw, dict) else {}
                payload = {
                    "collection_name": data.get("collection_name"),
                    "matched": data.get("matched"),
                }
                return render(
                    payload,
                    params.response_format,
                    title="Set payload result",
                    rows=_filter_write_rows,
                )

            method = "PATCH" if params.mode == "merge" else "PUT"
            fan_body: dict[str, Any] = {"payload": params.payload}
            if params.collection_name is not None:
                fan_body["collection_name"] = params.collection_name
            result = await _fan_out(
                ctx,
                method=method,
                path_for=_point_payload_path,
                body=fan_body,
                ids=params.ids or [],
            )
        except Exception as exc:  # -> agent message
            return error_result(exc)

        return render(
            result,
            params.response_format,
            title="Set payload result",
            rows=_fan_out_rows,
        )

    @mcp.tool(
        name="qdrant_delete_payload_keys",
        annotations={
            "title": "Delete payload keys",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def qdrant_delete_payload_keys(
        params: DeletePayloadKeysInput,
    ) -> StructuredToolResult:
        """Remove named payload keys from points addressed by id or filter.

        The protected service fields (``text``, ``source_id``) can never be
        removed this way -- the middleware rejects the request outright
        (400) if a key resolves to either root, including a quoted, nested or
        array path, before touching any
        point, since removing either would break retrieval and reranking.

        ``selector="ids"`` is **not atomic**: the middleware's per-point
        endpoint takes one id each, so N ids mean N sequential requests. A
        failure partway through leaves earlier removals applied -- the
        response names exactly which ids succeeded and which failed with
        why, rather than failing the whole call.

        ``selector="filter"`` is one call against the by-filter endpoint and
        reports how many points it matched.

        Args:
            params (DeletePayloadKeysInput): Validated input containing:
                - selector ("ids" | "filter"): addressing mode, required.
                - keys (list[str]): payload paths to remove, non-blank,
                  required.
                - ids (list[str] | None): point ids; required and non-empty
                  when selector="ids".
                - filters (dict | None): payload filter; required and
                  non-empty when selector="filter".
                - collection_name (str | None): target collection; omit for
                  the server default.
                - response_format (ResponseFormat): 'markdown' or 'json'.

        Returns:
            str: Markdown fact sheet, or JSON. For selector="ids":

            {
                "collection_name": str | null,
                "updated": [str, ...],    # ids the keys were removed from
                "failed": [{"point_id": str, "reason": str}, ...],
                "atomic": false
            }

            For selector="filter":

            {
                "collection_name": str,
                "matched": int    # points the filter matched and rewrote
            }

        Examples:
            - Use when: removing one or more specific payload keys from
              known point ids -> selector="ids".
            - Use when: removing a stale field from every point a filter
              matches in one call -> selector="filter".
            - Don't use when: keys is not known in advance -> call
              qdrant_list_payload_keys or qdrant_read_points first to see
              what exists.
            - Don't use when: the goal is to add or overwrite fields, not
              remove them -> use qdrant_set_payload instead.
            - Don't use when: wiping everything but protected fields -> use
              qdrant_clear_payload instead, which needs no key list.

        Error Handling:
            A missing/empty ids or filters for the chosen selector, and
            blank entries in keys (after whitespace is stripped), are
            refused before any HTTP call -- each as a ToolError raised from
            this tool's own body, returning the same "Error: ..." (see
            describe_error) shape as every other guardrail refusal instead
            of the SDK's raw validation-error envelope. A request naming a
            protected key (text, source_id) is rejected by the middleware
            with 400 before any write, for selector="ids" per failing id
            (captured in failed) and for selector="filter" for the whole
            call. For selector="ids", an individual id's other failures
            (e.g. 404) are likewise captured per id in the failed list
            rather than raised. Returns "Error: ..." (see describe_error)
            for a failure outside the fan-out, and directly for any
            selector="filter" failure.
        """
        try:
            refuse_if(
                any(not key for key in params.keys),
                "payload keys cannot be blank",
            )
            require_selector_args(params.selector, params.ids, params.filters)
            require_non_blank_ids(params.ids)
            require_fan_out_size(params.ids)
            if params.selector == "filter":
                body: dict[str, Any] = {"filters": params.filters, "keys": params.keys}
                if params.collection_name is not None:
                    body["collection_name"] = params.collection_name
                raw = await ctx.client.request(
                    "POST", "/points/payload/delete-keys", json=body
                )
                data = raw if isinstance(raw, dict) else {}
                payload = {
                    "collection_name": data.get("collection_name"),
                    "matched": data.get("matched"),
                }
                return render(
                    payload,
                    params.response_format,
                    title="Delete payload keys result",
                    rows=_filter_write_rows,
                )

            fan_body: dict[str, Any] = {"keys": params.keys}
            if params.collection_name is not None:
                fan_body["collection_name"] = params.collection_name
            result = await _fan_out(
                ctx,
                method="POST",
                path_for=lambda pid: _point_payload_path(pid, "/delete-keys"),
                body=fan_body,
                ids=params.ids or [],
            )
        except Exception as exc:  # -> agent message
            return error_result(exc)

        return render(
            result,
            params.response_format,
            title="Delete payload keys result",
            rows=_fan_out_rows,
        )

    @mcp.tool(
        name="qdrant_clear_payload",
        annotations={
            "title": "Clear point payload",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def qdrant_clear_payload(params: ClearPayloadInput) -> StructuredToolResult:
        """Clear points' payloads down to only protected fields. Confirm required.

        Every payload key except the protected service fields (``text``,
        ``source_id``) is removed from each addressed point; those two
        survive automatically so the point stays retrievable. There is no
        partial clear -- ``confirm`` must equal ``collection_name`` exactly
        (not whitespace- or case-normalised, and never empty) for either
        selector, or the call is refused before any HTTP request is made.
        ``collection_name`` itself is required, not defaulted, for the same
        reason ``qdrant_create_collection``'s ``recreate=true`` and
        ``qdrant_ingest_file``/``qdrant_ingest_chunks``'s
        ``replace_source=true`` both require their own confirmation target
        explicitly: a confirmation token has to be checked against a name
        the caller actually gave, never one the server defaulted to on its
        behalf. Unlike qdrant_delete_points there is no dry-run count for
        ``selector="filter"``: a clear has nothing proportional to preview,
        every match is fully cleared either way, so ``confirm`` is required
        outright rather than offered as an opt-in after a rehearsal.

        ``selector="ids"`` is **not atomic**: the middleware's per-point
        endpoint takes one id each, so N ids mean N sequential requests. A
        failure partway through leaves earlier clears applied -- the
        response names exactly which ids succeeded and which failed with
        why, rather than failing the whole call.

        ``selector="filter"`` is one call against the by-filter endpoint and
        reports how many points it matched.

        Args:
            params (ClearPayloadInput): Validated input containing:
                - selector ("ids" | "filter"): addressing mode, required.
                - ids (list[str] | None): point ids; required and non-empty
                  when selector="ids".
                - filters (dict | None): payload filter; required and
                  non-empty when selector="filter".
                - collection_name (str): target collection, required (not
                  defaulted) -- confirm has to be checked against a name the
                  caller actually gave.
                - confirm (str | None): required; must equal collection_name
                  exactly.
                - response_format (ResponseFormat): 'markdown' or 'json'.

        Returns:
            str: Markdown fact sheet, or JSON. For selector="ids":

            {
                "collection_name": str,
                "updated": [str, ...],    # ids that were cleared
                "failed": [{"point_id": str, "reason": str}, ...],
                "atomic": false
            }

            For selector="filter":

            {
                "collection_name": str,
                "matched": int    # points the filter matched and cleared
            }

        Examples:
            - Use when: resetting a point (or every point a filter matches)
              back to just its embedded text, discarding all other metadata.
            - Don't use when: only some keys should go, not all of them ->
              use qdrant_delete_payload_keys instead, which takes a key list.
            - Don't use when: unsure what a filter matches -> call
              qdrant_count_points with the same filter first; clearing has
              no built-in dry run.

        Error Handling:
            A missing collection_name is rejected by Pydantic before any
            HTTP call. A missing/empty ids or filters for the chosen
            selector is refused the same way but as a ToolError raised from
            this tool's own body, returning the same "Error: ..." (see
            describe_error) shape as every other guardrail refusal instead
            of the SDK's raw validation-error envelope. A missing or
            mismatched confirm -- including an empty string, which is
            refused outright rather than compared -- is refused before any
            HTTP call is made, for either selector. For selector="ids", an
            individual id's failure (e.g. 404) is captured per id in the
            failed list rather than raised. Returns "Error: ..." (see
            describe_error) for a failure outside the fan-out, and directly
            for any selector="filter" failure.
        """
        try:
            require_selector_args(params.selector, params.ids, params.filters)
            require_non_blank_ids(params.ids)
            require_fan_out_size(params.ids)
            require_confirm(params.confirm, params.collection_name)

            if params.selector == "filter":
                body: dict[str, Any] = {
                    "collection_name": params.collection_name,
                    "filters": params.filters,
                }
                raw = await ctx.client.request(
                    "POST", "/points/payload/clear", json=body
                )
                data = raw if isinstance(raw, dict) else {}
                payload = {
                    "collection_name": data.get("collection_name"),
                    "matched": data.get("matched"),
                }
                return render(
                    payload,
                    params.response_format,
                    title="Clear payload result",
                    rows=_filter_write_rows,
                )

            result = await _fan_out(
                ctx,
                method="POST",
                path_for=lambda pid: _point_payload_path(pid, "/clear"),
                body={"collection_name": params.collection_name},
                ids=params.ids or [],
            )
        except Exception as exc:  # -> agent message
            return error_result(exc)

        return render(
            result,
            params.response_format,
            title="Clear payload result",
            rows=_fan_out_rows,
        )

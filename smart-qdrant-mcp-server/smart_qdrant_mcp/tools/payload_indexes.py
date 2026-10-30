"""Payload key discovery and payload index lifecycle."""

from __future__ import annotations

import json
from typing import Any, Literal, Optional
from urllib.parse import quote

from mcp.server import MCPServer
from pydantic import BaseModel, ConfigDict, Field

from smart_qdrant_mcp.context import ToolContext
from smart_qdrant_mcp.errors import error_result
from smart_qdrant_mcp.formatting import (
    ResponseFormat,
    StructuredToolResult,
    bullet,
    render,
)
from smart_qdrant_mcp.guardrails import refuse_if, require_confirm
from smart_qdrant_mcp.tools.collections import ExactString

# Copied verbatim from PayloadIndexSchemaName in
# middleware-qdrant-manager/qdrant_middleware/schemas/collections.py:87-89. A
# drifted member here produces a tool call that Pydantic accepts locally but
# the middleware then rejects (or, worse, one the middleware accepts but this
# Literal never allowed) — keep the two in lockstep by inspection, not import,
# since the two services ship as independent images.
PayloadIndexSchemaName = Literal[
    "keyword", "text", "integer", "float", "bool", "geo", "datetime", "uuid"
]


class ListPayloadKeysInput(BaseModel):
    """Input model for sampling a collection's observed payload keys."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    collection_name: str = Field(
        ..., min_length=1, description="Collection to sample (e.g. 'docs')"
    )
    sample_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description=(
            "Points to sample when discovering keys; higher catches rarer "
            "keys at the cost of one larger scroll call (e.g. 500)"
        ),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (default) or 'json' (e.g. 'json')",
    )


class ListPayloadIndexesInput(BaseModel):
    """Input model for listing a collection's configured payload indexes."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    collection_name: str = Field(
        ..., min_length=1, description="Collection to inspect (e.g. 'docs')"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (default) or 'json' (e.g. 'json')",
    )


class PayloadIndexSpec(BaseModel):
    """One payload index to create."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    field_name: str = Field(
        ..., min_length=1, description="Payload key to index (e.g. 'source_id')"
    )
    field_schema: PayloadIndexSchemaName = Field(
        ..., description="Qdrant index type for the field (e.g. 'keyword')"
    )
    phrase_matching: bool = Field(
        default=False,
        description=(
            "Enable phrase queries; only valid with field_schema='text' (e.g. false)"
        ),
    )

    # phrase_matching=true paired with a non-'text' field_schema is rejected
    # in the qdrant_create_payload_index tool body (as a ToolError), not
    # here: this is a nested model, so the check there loops over the
    # submitted specs rather than living on a per-item model_validator --
    # see the module docstring in tools/payload.py for why a cross-field
    # business rule like this one is raised from the tool body rather than
    # a validator.


class CreatePayloadIndexInput(BaseModel):
    """Input model for creating (or type-repairing) one or more payload indexes."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    collection_name: str = Field(
        ..., min_length=1, description="Collection to index (e.g. 'docs')"
    )
    indexes: list[PayloadIndexSpec] = Field(
        ...,
        min_length=1,
        description=(
            "Indexes to create, e.g. "
            "[{'field_name': 'lang', 'field_schema': 'keyword'}]"
        ),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (default) or 'json' (e.g. 'json')",
    )

    # Duplicate field_name rejection ("matching the middleware's own rule")
    # is enforced in the qdrant_create_payload_index tool body, not here --
    # see the note on PayloadIndexSpec above.


class DeletePayloadIndexInput(BaseModel):
    """Input model for permanently deleting a user-managed payload index."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    collection_name: str = Field(
        ..., min_length=1, description="Collection the index belongs to (e.g. 'docs')"
    )
    field_name: str = Field(
        ...,
        min_length=1,
        description="Payload field whose index to delete (e.g. 'lang')",
    )
    confirm: Optional[ExactString] = Field(
        default=None,
        description=(
            "Required: must equal collection_name exactly (not whitespace- "
            "or case-normalised) to confirm deletion, e.g. confirm='docs' "
            "for collection_name='docs'"
        ),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (default) or 'json' (e.g. 'json')",
    )


def _keys_rows(payload: dict[str, Any]) -> list[str]:
    """Render a sampled payload-keys report."""
    rows = [
        bullet("Collection", payload.get("collection_name")),
        bullet("Sampled points", payload.get("sampled_points")),
        bullet("Key count", payload.get("count")),
    ]
    keys = payload.get("keys") or []
    if keys:
        rows.extend(["", "## Keys", *(f"- {key}" for key in keys)])
    else:
        rows.append(bullet("Keys", None))
    return rows


def _indexes_rows(payload: dict[str, Any]) -> list[str]:
    """Render the configured payload indexes.

    Only the three fields PayloadIndexInfo actually carries (field_name,
    field_schema, phrase_matching) are read here; there is no on-disk size
    or point count per index to show.
    """
    rows = [
        bullet("Collection", payload.get("collection_name")),
        bullet("Index count", payload.get("count")),
    ]
    indexes = payload.get("indexes") or []
    if not indexes:
        rows.append(bullet("Indexes", None))
        return rows
    rows.extend(["", "## Indexes"])
    for index in indexes:
        field = index.get("field_name")
        schema = index.get("field_schema")
        phrase = index.get("phrase_matching")
        suffix = f", phrase_matching {phrase}" if phrase is not None else ""
        rows.append(f"- **{field}**: {schema}{suffix}")
    return rows


def _create_index_rows(payload: dict[str, Any]) -> list[str]:
    """Render the four outcome buckets a create call reports.

    A 200 response does not mean every requested index was created: Qdrant
    reports success and failure per index rather than failing the whole
    call, so ``failed`` can be non-empty on an otherwise-ok response. When
    it is, a plain-language summary leads the output — before the
    per-bucket bullets — so a skim cannot mistake a partial failure for a
    clean success.
    """

    def _joined(key: str) -> Optional[str]:
        items = payload.get(key) or []
        return ", ".join(items) if items else None

    failed = payload.get("failed") or []
    rows: list[str] = []
    if failed:
        plural = "index" if len(failed) == 1 else "indexes"
        rows.append(f"**{len(failed)} {plural} failed to create: {', '.join(failed)}**")
        rows.append("")
    rows.extend(
        [
            bullet("Collection", payload.get("collection_name")),
            bullet("Created", _joined("created")),
            bullet("Updated", _joined("updated")),
            bullet("Skipped", _joined("skipped")),
            bullet("Failed", _joined("failed")),
        ]
    )
    return rows


def _delete_index_rows(payload: dict[str, Any]) -> list[str]:
    """Render a delete outcome: whether it deleted, and which field/collection."""
    return [
        bullet("Deleted", payload.get("deleted")),
        bullet("Collection", payload.get("collection_name")),
        bullet("Field", payload.get("field_name")),
    ]


def register(mcp: MCPServer, ctx: ToolContext) -> None:
    """Register the four payload-index tools on the server."""

    @mcp.tool(
        name="qdrant_list_payload_keys",
        annotations={
            "title": "List a collection's observed payload keys",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def qdrant_list_payload_keys(
        params: ListPayloadKeysInput,
    ) -> StructuredToolResult:
        """Discover payload keys by sampling a collection's points.

        This is discovery, not configuration: the keys returned are whatever
        was observed on ``sample_size`` sampled points, not a schema. A key
        that exists on only a handful of points in a large collection can be
        missed entirely at a low ``sample_size`` — the result is a lower
        bound on the real key set, never an exhaustive one. Raise
        ``sample_size`` (up to 1000) if a key you expect to exist is absent
        from the result.

        Args:
            params (ListPayloadKeysInput): Validated input containing:
                - collection_name (str): collection to sample, required.
                - sample_size (int): points to sample, 1-1000, default 100.
                - response_format (ResponseFormat): 'markdown' or 'json'.

        Returns:
            str: Markdown fact sheet, or JSON with this schema:

            {
                "ok": true,
                "collection_name": str,
                "keys": [str, ...],
                "count": int,
                "sampled_points": int
            }

        Examples:
            - Use when: you don't know what payload fields a collection has
              before writing a filter for qdrant_search.
            - Use when: spot-checking that ingest actually wrote a field you
              expect, e.g. 'source_id'.
            - Don't use when: you want to know which keys are actually
              *indexed* (and therefore fast to filter on) → use
              qdrant_list_payload_indexes instead; a key can appear here and
              still have no index, or vice versa be indexed but rare enough
              to miss the sample.

        Error Handling:
            Returns "Error: not found ..." pointing at qdrant_list_collections
            when the collection does not exist (404), and describe_error's
            standard mapping for every other failure.
        """
        encoded = quote(params.collection_name, safe="")
        try:
            raw = await ctx.client.request(
                "GET",
                f"/collections/{encoded}/payload-keys",
                params={"sample_size": params.sample_size},
            )
        except Exception as exc:  # -> agent message
            return error_result(exc)

        payload = raw if isinstance(raw, dict) else {}
        return render(
            payload,
            params.response_format,
            title="Payload keys",
            rows=_keys_rows,
        )

    @mcp.tool(
        name="qdrant_list_payload_indexes",
        annotations={
            "title": "List a collection's payload indexes",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def qdrant_list_payload_indexes(
        params: ListPayloadIndexesInput,
    ) -> StructuredToolResult:
        """List the payload indexes actually configured on a collection.

        Unlike qdrant_list_payload_keys this is authoritative, not sampled:
        it reports exactly what Qdrant has indexed, so every entry here is
        genuinely fast to filter on and nothing is missed regardless of
        collection size.

        Args:
            params (ListPayloadIndexesInput): Validated input containing:
                - collection_name (str): collection to inspect, required.
                - response_format (ResponseFormat): 'markdown' or 'json'.

        Returns:
            str: Markdown fact sheet, or JSON with this schema:

            {
                "ok": true,
                "collection_name": str,
                "indexes": [
                    {"field_name": str, "field_schema": str,
                     "phrase_matching": bool | null},
                    ...
                ],
                "count": int
            }

        Examples:
            - Use when: checking whether a field you plan to filter on is
              indexed before relying on it being fast.
            - Use when: auditing a collection's indexes before deciding what
              to pass to qdrant_create_payload_index.
            - Don't use when: you want to discover what payload fields exist
              at all, indexed or not → use qdrant_list_payload_keys instead.

        Error Handling:
            Returns "Error: not found ..." pointing at qdrant_list_collections
            when the collection does not exist (404), and describe_error's
            standard mapping for every other failure.
        """
        encoded = quote(params.collection_name, safe="")
        try:
            raw = await ctx.client.request(
                "GET", f"/collections/{encoded}/payload-indexes"
            )
        except Exception as exc:  # -> agent message
            return error_result(exc)

        payload = raw if isinstance(raw, dict) else {}
        return render(
            payload,
            params.response_format,
            title="Payload indexes",
            rows=_indexes_rows,
        )

    @mcp.tool(
        name="qdrant_create_payload_index",
        annotations={
            "title": "Create payload indexes on a collection",
            "readOnlyHint": False,
            # A type repair drops the old index before recreating it.
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def qdrant_create_payload_index(
        params: CreatePayloadIndexInput,
    ) -> StructuredToolResult:
        """Create one or more user-managed payload indexes.

        Every collection runs Qdrant strict mode with unindexed filtering
        refused, so a key must carry an index of a fitting type before any
        search, count, scroll, facet or filter-addressed write can filter on
        it — without one the call fails with "'<key>' has no payload index
        that can serve this filter". See the resource docs://reference/filters.

        Creating an index for a field that is already indexed with the same
        ``field_schema`` is a no-op reported in the ``skipped`` bucket
        (safe to retry). If the field is already indexed with a *different*
        type, the existing index is dropped and recreated with the requested
        type — reported in ``updated`` — which is why this tool is
        idempotent but not itself marked destructive: the old index's type
        is lost, but no payload data or points are touched.

        Duplicate ``field_name`` entries within one call, and
        ``phrase_matching=true`` paired with a non-'text' ``field_schema``,
        are both rejected before any request reaches the middleware.

        Args:
            params (CreatePayloadIndexInput): Validated input containing:
                - collection_name (str): collection to index, required.
                - indexes (list[PayloadIndexSpec]): at least one index to
                  create, no duplicate field_name; each has field_name,
                  field_schema (one of 'keyword', 'text', 'integer', 'float',
                  'bool', 'geo', 'datetime', 'uuid'), and phrase_matching
                  (only valid with field_schema='text').
                - response_format (ResponseFormat): 'markdown' or 'json'.

        Returns:
            str: Markdown fact sheet, or JSON with this schema:

            {
                "ok": true,
                "collection_name": str,
                "created": [str, ...],
                "updated": [str, ...],
                "skipped": [str, ...],
                "failed": [str, ...]
            }

        Examples:
            - Use when: a field you plan to filter on regularly has no index
              yet (confirm first with qdrant_list_payload_indexes).
            - Use when: repairing an index that was accidentally created with
              the wrong field_schema.
            - Don't use when: you only want to discover what fields exist,
              not index anything → use qdrant_list_payload_keys instead.

        Error Handling:
            phrase_matching=true paired with a non-'text' field_schema, and a
            duplicate field_name across the submitted indexes, are both
            refused before any HTTP call -- each as a ToolError raised from
            this tool's own body (checked in that order, per index, so the
            first offending spec's message is the one returned), giving the
            same "Error: ..." (see describe_error) shape as every other
            guardrail refusal instead of the SDK's raw validation-error
            envelope. Returns "Error: ..." for any middleware failure. A 409
            means either an embedding-contract mutation block or a
            payload-index type mismatch reported by the middleware itself;
            the message states which and the fields involved. A 200
            response is NOT proof every requested index was created: Qdrant
            reports success and failure per index rather than failing the
            whole call, so ``failed`` can be non-empty on an otherwise-ok
            response. Check the ``failed`` list in JSON mode, or the leading
            "N index(es) failed to create" summary line in markdown mode,
            before assuming the requested indexes exist.
        """
        try:
            for spec in params.indexes:
                refuse_if(
                    spec.phrase_matching and spec.field_schema != "text",
                    "phrase_matching requires field_schema 'text'",
                )
            seen: set[str] = set()
            for spec in params.indexes:
                key = spec.field_name.strip()
                if key.startswith('"'):
                    try:
                        literal = json.loads(key)
                    except json.JSONDecodeError:
                        pass  # Let the middleware validate full path syntax.
                    else:
                        if isinstance(literal, str):
                            key = (
                                json.dumps(literal)
                                if any(char in literal for char in '.[]"')
                                else literal
                            )
                key = key.casefold()
                refuse_if(
                    key in seen, f"duplicate payload index field: {spec.field_name}"
                )
                seen.add(key)

            body = {
                "indexes": [spec.model_dump() for spec in params.indexes],
            }
            encoded = quote(params.collection_name, safe="")
            raw = await ctx.client.request(
                "POST",
                f"/collections/{encoded}/payload-indexes",
                json=body,
            )
        except Exception as exc:  # -> agent message
            return error_result(exc)

        payload = raw if isinstance(raw, dict) else {}
        return render(
            payload,
            params.response_format,
            title="Payload indexes created",
            rows=_create_index_rows,
        )

    @mcp.tool(
        name="qdrant_delete_payload_index",
        annotations={
            "title": "Delete a payload index",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def qdrant_delete_payload_index(
        params: DeletePayloadIndexInput,
    ) -> StructuredToolResult:
        """Permanently delete a user-managed payload index.

        This does not delete the payload data itself, only the index on it:
        the field remains readable and its values remain in every point, but
        every collection runs Qdrant strict mode, so any search, count,
        scroll, facet or filter-addressed write that filters on this key is
        refused afterwards until it is indexed again — see the resource
        docs://reference/filters. ``confirm`` must equal ``collection_name``
        exactly; a missing or mismatched ``confirm`` is rejected before any
        request reaches the middleware.

        Args:
            params (DeletePayloadIndexInput): Validated input containing:
                - collection_name (str): collection the index belongs to,
                  required.
                - field_name (str): payload field whose index to delete,
                  required.
                - confirm (str | None): required; must equal collection_name
                  exactly.
                - response_format (ResponseFormat): 'markdown' or 'json'.

        Returns:
            str: Markdown fact sheet, or JSON with this schema:

            {
                "ok": true,
                "collection_name": str,
                "field_name": str,
                "deleted": true
            }

        Examples:
            - Use when: retiring a filter field you no longer query on, to
              free the index's memory/disk footprint.
            - Use when: about to recreate an index with a different
              field_schema (though qdrant_create_payload_index does the drop
              and recreate for you in one idempotent call when the type
              differs — prefer that unless you specifically want the field
              left unindexed).
            - Don't use when: you want to change the index's type rather
              than remove it entirely → use qdrant_create_payload_index with
              the new field_schema instead.

        Error Handling:
            Returns "Error: ..." (see describe_error) on any middleware
            failure or on a refused confirmation gate — in the latter case no
            HTTP call is made at all. A 404 means either the collection or
            the field's index did not exist; call qdrant_list_payload_indexes
            to see what's actually configured.
        """
        try:
            require_confirm(params.confirm, params.collection_name)
            raw = await ctx.client.request(
                "DELETE",
                "/collections/"
                f"{quote(params.collection_name, safe='')}/payload-indexes/"
                f"{quote(params.field_name, safe='')}",
            )
        except Exception as exc:  # -> agent message
            return error_result(exc)

        payload = raw if isinstance(raw, dict) else {}
        return render(
            payload,
            params.response_format,
            title="Payload index deleted",
            rows=_delete_index_rows,
        )

"""Collection inspection and lifecycle."""

from __future__ import annotations

from typing import Annotated, Any, Optional
from urllib.parse import quote

from mcp.server import MCPServer
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from smart_qdrant_mcp.context import ToolContext
from smart_qdrant_mcp.errors import error_result
from smart_qdrant_mcp.formatting import (
    ResponseFormat,
    StructuredToolResult,
    bullet,
    paginate,
    render,
)
from smart_qdrant_mcp.guardrails import refuse_if, require_confirm

# The model-level str_strip_whitespace=True strips every string field, but a
# confirmation token must be compared exactly as typed: " docs" and "docs "
# must not silently pass require_confirm's check. This annotation overrides
# the model config for the one field where leniency would be wrong. The
# max_length cap mirrors the middleware's own confirmation-token limit
# (confirm_delete in api/routers/collections.py is Query(max_length=255)), so
# an over-long token is refused locally instead of reaching the middleware.
ExactString = Annotated[str, StringConstraints(strip_whitespace=False, max_length=255)]


class ListCollectionsInput(BaseModel):
    """Input model for listing every collection the middleware knows about."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (default) or 'json' (e.g. 'json')",
    )


class GetCollectionInput(BaseModel):
    """Input model for a single collection's schema report."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    collection_name: str = Field(
        ...,
        min_length=1,
        description="Collection to inspect (e.g. 'docs')",
    )
    # Named ``validate_schema`` in Python -- a field literally named
    # ``validate`` shadows ``BaseModel.validate`` (pydantic v2's deprecated
    # v1-compat classmethod) and pydantic warns ``UserWarning`` about that
    # shadowing on every import. ``alias="validate"`` keeps the wire name
    # (the query parameter the middleware receives, and what every existing
    # test calls this field) unchanged -- only the Python attribute moved.
    validate_schema: bool = Field(
        default=True,
        alias="validate",
        description=(
            "Run schema/embedding-contract checks; set false for an info-only "
            "report with no validation (e.g. false)"
        ),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (default) or 'json' (e.g. 'json')",
    )


class VerifyCollectionInput(BaseModel):
    """Input model for verifying a collection against expected schema toggles.

    Every toggle defaults to None so the middleware checks against its own
    configured expectation; sending an explicit value overrides it for this
    call only.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    collection_name: Optional[str] = Field(
        default=None,
        description=(
            "Collection to verify; omit to use the server default (e.g. 'docs')"
        ),
    )
    dense_on_disk: Optional[bool] = Field(
        default=None,
        description="Expected toggle: dense vectors stored on disk (e.g. true)",
    )
    quantization: Optional[bool] = Field(
        default=None,
        description="Expected toggle: quantization enabled (e.g. true)",
    )
    sparse_idf: Optional[bool] = Field(
        default=None,
        description="Expected toggle: sparse IDF modifier enabled (e.g. true)",
    )
    sparse_on_disk: Optional[bool] = Field(
        default=None,
        description="Expected toggle: sparse vectors stored on disk (e.g. false)",
    )
    on_disk_payload: Optional[bool] = Field(
        default=None,
        description="Expected toggle: payload stored on disk (e.g. true)",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (default) or 'json' (e.g. 'json')",
    )


class CreateCollectionInput(BaseModel):
    """Input model for creating (or destructively recreating) a collection.

    The four toggle fields have no default because the middleware itself
    requires them on every create call — there is no server-side fallback to
    guess from, unlike ``qdrant_update_schema_toggles`` where each toggle is
    optional and only the ones supplied are changed.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    collection_name: Optional[str] = Field(
        default=None,
        description=(
            "Collection to create; omit to use the middleware's configured "
            "default. Required when recreate=true, since the confirmation "
            "token is checked against this name (e.g. 'docs')"
        ),
    )
    dense_on_disk: bool = Field(
        ...,
        description="Store dense vectors on disk rather than in memory (e.g. false)",
    )
    quantization: bool = Field(
        ..., description="Enable vector quantization for this collection (e.g. true)"
    )
    sparse_idf: bool = Field(
        ..., description="Enable the sparse-vector IDF modifier (e.g. true)"
    )
    sparse_on_disk: bool = Field(
        ...,
        description="Store sparse vectors on disk rather than in memory (e.g. false)",
    )
    on_disk_payload: Optional[bool] = Field(
        default=None,
        description=(
            "Store point payload on disk; omit to fall back to the "
            "middleware's configured default (e.g. true)"
        ),
    )
    recreate: bool = Field(
        default=False,
        description=(
            "Drop the collection first if it already exists, then create it "
            "fresh — DESTROYS ALL POINTS in the existing collection "
            "(e.g. false)"
        ),
    )
    confirm: Optional[ExactString] = Field(
        default=None,
        description=(
            "Required only when recreate=true: must equal collection_name "
            "exactly (not whitespace- or case-normalised), e.g. "
            "confirm='docs' for collection_name='docs'"
        ),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (default) or 'json' (e.g. 'json')",
    )

    # recreate=true requiring an explicit collection_name is enforced in the
    # qdrant_create_collection tool body (as a ToolError), not here -- see
    # the module docstring in tools/payload.py for why a cross-field business
    # rule like this one is raised from the tool body rather than a
    # model_validator.


class UpdateSchemaTogglesInput(BaseModel):
    """Input model for changing mutable schema toggles on an existing collection.

    This never recreates the collection and carries no confirmation gate:
    every toggle it can change is reversible in place, unlike
    ``qdrant_create_collection``'s ``recreate=true``.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    collection_name: str = Field(
        ..., min_length=1, description="Collection to update (e.g. 'docs')"
    )
    dense_on_disk: Optional[bool] = Field(
        default=None,
        description="New value for dense-vector on-disk storage (e.g. true)",
    )
    quantization: Optional[bool] = Field(
        default=None, description="New value for quantization enablement (e.g. true)"
    )
    sparse_idf: Optional[bool] = Field(
        default=None, description="New value for the sparse IDF modifier (e.g. false)"
    )
    sparse_on_disk: Optional[bool] = Field(
        default=None,
        description="New value for sparse-vector on-disk storage (e.g. false)",
    )
    on_disk_payload: Optional[bool] = Field(
        default=None, description="New value for payload on-disk storage (e.g. true)"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (default) or 'json' (e.g. 'json')",
    )

    # "at least one toggle must be provided" (mirroring the middleware's own
    # rule) is enforced in the qdrant_update_schema_toggles tool body, not
    # here -- see the note on CreateCollectionInput above.


class DeleteCollectionInput(BaseModel):
    """Input model for permanently deleting a collection and all its points."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    collection_name: str = Field(
        ..., min_length=1, description="Collection to permanently delete (e.g. 'docs')"
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


def _list_rows(payload: dict[str, Any]) -> list[str]:
    """Render collection names as a markdown bullet list."""
    items = payload.get("items") or []
    if not items:
        return ["", "No collections exist yet."]
    return [f"- {name}" for name in items]


def _report_rows(payload: dict[str, Any]) -> list[str]:
    """Render a collection report, listing schema findings last.

    Every key read here exists on CollectionReport
    (schemas/collections.py:65-84). There is no "vectors" or "toggles"
    grouping: dense and sparse are separate objects, and quantization is a
    string naming the configured mode, not a boolean.
    """
    dense = payload.get("dense") or {}
    sparse = payload.get("sparse") or {}
    dense_line = (
        f"size {dense.get('size')}, on_disk {dense.get('on_disk')}" if dense else None
    )
    sparse_line = (
        f"idf {sparse.get('idf')}, on_disk {sparse.get('on_disk')}" if sparse else None
    )
    rows = [
        bullet("Collection", payload.get("collection_name")),
        bullet("Exists", payload.get("exists")),
        bullet("Status", payload.get("status")),
        bullet("Points", payload.get("points_count")),
        bullet("Vectors stored", payload.get("vectors_count")),
        bullet("Dense", dense_line),
        bullet("Sparse", sparse_line),
        bullet("Quantization", payload.get("quantization")),
        bullet("Payload on disk", payload.get("on_disk_payload")),
        bullet(
            "Shards / replicas",
            f"{payload.get('shards')} / {payload.get('replicas')}",
        ),
    ]
    schema_ok = payload.get("schema_ok")
    rows.append(
        bullet(
            "Schema validated",
            "not checked (validate=false)" if schema_ok is None else schema_ok,
        )
    )
    # embedding_compatibility (EmbeddingCompatibility,
    # schemas/embedding_contract.py:121-129) is the key diagnostic for the
    # most common failure — a contract mismatch — so it renders ahead of the
    # errors/findings lists: the verdict first, then the stored/active
    # profiles and the message explaining any mismatch, mirroring what the
    # frontend's collection report surfaces. Absent (validate=false or an
    # unreachable embedder), only the verdict line shows, as an em dash.
    compatibility = payload.get("embedding_compatibility")
    if compatibility:
        stored = compatibility.get("stored") or {}
        active = compatibility.get("active") or {}
        rows.extend(
            [
                bullet("Embedding compatibility", compatibility.get("status")),
                bullet("Stored profile", stored.get("profile")),
                bullet("Active profile", active.get("profile")),
                bullet("Compatibility detail", compatibility.get("message")),
            ]
        )
    else:
        rows.append(bullet("Embedding compatibility", None))
    errors = payload.get("errors") or []
    if errors:
        rows.extend(["", "## Errors", *(f"- {error}" for error in errors)])
    findings = payload.get("findings") or []
    if findings:
        rows.extend(["", "## Findings", *(f"- {finding}" for finding in findings)])
    fixes = payload.get("suggested_fixes") or []
    if fixes:
        rows.extend(["", "## Suggested fixes", *(f"- {fix}" for fix in fixes)])
    return rows


def _operation_rows(payload: dict[str, Any]) -> list[str]:
    """Render a create/toggle-update outcome: whether it created, then the report.

    Reuses ``_report_rows`` for the nested ``report`` object rather than
    duplicating its field list.
    """
    rows = [bullet("Created", payload.get("created"))]
    rows.extend(_report_rows(payload.get("report") or {}))
    return rows


def _delete_rows(payload: dict[str, Any]) -> list[str]:
    """Render a delete outcome: whether it deleted, and which collection."""
    return [
        bullet("Deleted", payload.get("deleted")),
        bullet("Collection", payload.get("collection_name")),
    ]


def register(mcp: MCPServer, ctx: ToolContext) -> None:
    """Register the collection read tools on the server.

    Task 10 adds the write tools (create, verify-and-repair variants, delete,
    ...) to this same module and calls their registration from here too.
    """

    @mcp.tool(
        name="qdrant_list_collections",
        annotations={
            "title": "List Qdrant collections",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def qdrant_list_collections(
        params: ListCollectionsInput = ListCollectionsInput(),
    ) -> StructuredToolResult:
        """List the names of every collection the middleware knows about.

        Returns the complete list in a single response; there is no server-side
        paging for this endpoint, so the response envelope always reports the
        end of results.

        Args:
            params (ListCollectionsInput): Validated input containing:
                - response_format (ResponseFormat): 'markdown' (default) or 'json'.

        Returns:
            str: Markdown bullet list of names, or JSON with this schema:

            {
                "total": int,
                "count": int,
                "offset": 0,
                "has_more": false,
                "next_offset": null,
                "items": [str, ...]   # collection names
            }

        Examples:
            - Use when: discovering what collections exist before targeting one.
            - Use when: confirming a collection name before calling qdrant_search.
            - Don't use when: you need schema, point counts or vector config for
              a specific collection → use qdrant_get_collection instead.

        Error Handling:
            Returns "Error: ..." (see describe_error) on any middleware failure;
            never raises.
        """
        try:
            raw = await ctx.client.request("GET", "/collections")
        except Exception as exc:  # -> agent message
            return error_result(exc)

        data = raw if isinstance(raw, dict) else {}
        items = data.get("collections") or []
        page = paginate(items, total=len(items), offset=0, limit=len(items) or 1)
        return render(
            page,
            params.response_format,
            title="Collections",
            rows=_list_rows,
        )

    @mcp.tool(
        name="qdrant_get_collection",
        annotations={
            "title": "Get a Qdrant collection's schema report",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def qdrant_get_collection(params: GetCollectionInput) -> StructuredToolResult:
        """Return existence, point counts, vector config and schema status.

        With the default ``validate=true`` this also checks the collection's
        vector layout against the currently active embedding contract, so a
        non-empty ``errors`` list means the collection cannot serve the current
        Smart Embedder profile as-is. Set ``validate=false`` to skip that check
        and get an info-only report (counts, vector config) faster — the right
        choice when all you need is a point count or existence check.

        Args:
            params (GetCollectionInput): Validated input containing:
                - collection_name (str): collection to inspect, required.
                - validate (bool): run schema/contract checks; default true.
                - response_format (ResponseFormat): 'markdown' or 'json'.

        Returns:
            str: Markdown fact sheet, or JSON with this schema:

            {
                "collection_name": str,
                "exists": bool,
                "schema_ok": bool | null,   # null only when validate=false
                "errors": [str, ...],
                "findings": [{...}, ...],
                "suggested_fixes": [{...}, ...],
                "status": str | null,
                "points_count": int | null,
                "vectors_count": int | null,
                "dense": {"size": int, "distance": str, "on_disk": bool} | null,
                "sparse": {"idf": bool, "on_disk": bool} | null,
                "quantization": str,        # configured mode, not a boolean
                "shards": int | null,
                "replicas": int | null,
                "on_disk_payload": bool | null,
                "embedding_compatibility": {...} | null
            }

        Examples:
            - Use when: checking a collection's point count or vector size.
            - Use when: diagnosing a 409 embedding-contract error from
              qdrant_search or an ingest call.
            - Don't use when: you don't know the collection's name yet → use
              qdrant_list_collections instead.
            - Don't use when: you want to check against specific expected
              toggles rather than the collection's own config → use
              qdrant_verify_collection.

        Error Handling:
            Returns "Error: not found ..." pointing at qdrant_list_collections
            when the collection does not exist (404), and describe_error's
            standard mapping for every other failure.
        """
        encoded = quote(params.collection_name, safe="")
        try:
            raw = await ctx.client.request(
                "GET",
                f"/collections/{encoded}",
                params={"validate": params.validate_schema},
            )
        except Exception as exc:  # -> agent message
            return error_result(exc)

        payload = raw if isinstance(raw, dict) else {}
        return render(
            payload,
            params.response_format,
            title="Collection report",
            rows=_report_rows,
        )

    @mcp.tool(
        name="qdrant_verify_collection",
        annotations={
            "title": "Verify a Qdrant collection against expected toggles",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def qdrant_verify_collection(
        params: VerifyCollectionInput = VerifyCollectionInput(),
    ) -> StructuredToolResult:
        """Check a collection against expected schema toggles without changing it.

        This never repairs or mutates the collection — it only reports drift.
        A non-empty ``errors`` list means the collection cannot serve the
        currently active embedding profile as-is; read the resource
        docs://concepts/embedding-contract for what the fields mean, then
        decide whether to recreate the collection or adopt its contract.

        Args:
            params (VerifyCollectionInput): Validated input containing:
                - collection_name (str | None): collection to verify; omit for
                  the server default.
                - dense_on_disk, quantization, sparse_idf, sparse_on_disk,
                  on_disk_payload (bool | None): expected toggle values;
                  omitted fields are checked against the middleware's own
                  configured expectation instead of a guessed default.
                - response_format (ResponseFormat): 'markdown' or 'json'.

        Returns:
            str: Markdown fact sheet, or JSON with the same CollectionReport
            schema documented on qdrant_get_collection.

        Examples:
            - Use when: confirming a collection matches a specific set of
              schema toggles before relying on it.
            - Use when: a mutation was rejected with a 409 embedding-contract
              error and you need to see exactly what mismatched.
            - Don't use when: you just want the collection's current state
              without checking against expected toggles → use
              qdrant_get_collection instead.

        Error Handling:
            Unlike qdrant_get_collection, this never 404s for a missing
            collection: verification always answers with a normal report.
            A missing collection comes back as ``exists: false`` with the
            reason recorded in ``errors`` (and ``schema_ok: false``) — check
            ``exists``/``errors``, not the string "Error:", to detect
            absence. describe_error's standard mapping still applies to
            actual request failures (bad input, unreachable middleware, ...).
        """
        body = params.model_dump(exclude_none=True, exclude={"response_format"})
        try:
            raw = await ctx.client.request(
                "POST", "/collections/verify", json=body, retryable=True
            )
        except Exception as exc:  # -> agent message
            return error_result(exc)

        payload = raw if isinstance(raw, dict) else {}
        return render(
            payload,
            params.response_format,
            title="Collection verification",
            rows=_report_rows,
        )

    @mcp.tool(
        name="qdrant_create_collection",
        annotations={
            "title": "Create or recreate a collection",
            "readOnlyHint": False,
            # recreate=True drops the collection first, destroying every
            # point in it: the hint describes that worst case, not the
            # common plain-create call, so clients treat every call as
            # needing confirmation before recreate is actually requested.
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def qdrant_create_collection(
        params: CreateCollectionInput,
    ) -> StructuredToolResult:
        """Create a collection, or repair indexes if it already exists.

        With ``recreate=false`` (the default) an existing collection is left
        in place and only missing payload indexes are repaired; the four
        toggle fields (``dense_on_disk``, ``quantization``, ``sparse_idf``,
        ``sparse_on_disk``) are still required on every call because the
        middleware has no server-side default to fall back to for them.

        With ``recreate=true`` the existing collection is dropped and
        recreated from scratch — **this destroys every point currently
        stored in it**, irreversibly. ``recreate=true`` requires
        ``collection_name`` to be given explicitly (a defaulted name cannot
        be checked against a confirmation token) and requires ``confirm`` to
        equal ``collection_name`` exactly; a missing or mismatched ``confirm``
        is rejected before any request reaches the middleware.

        Vector dimension, distance metric and the vector names are NOT
        settable here and have no parameter: the schema follows the embedding
        profile the Smart Embedder is running (dense dimension 1024 under both
        the basic and advanced profiles). A collection that needs a different
        dimension needs a different embedder, not a different argument.

        Args:
            params (CreateCollectionInput): Validated input containing:
                - collection_name (str | None): collection to create; omit
                  for the middleware's default, except when recreate=true
                  where it is required.
                - dense_on_disk, quantization, sparse_idf, sparse_on_disk
                  (bool): required schema toggles, no default.
                - on_disk_payload (bool | None): optional; omit to use the
                  middleware's configured default.
                - recreate (bool): drop and recreate if the collection
                  already exists; default false.
                - confirm (str | None): required only when recreate=true;
                  must equal collection_name exactly.
                - response_format (ResponseFormat): 'markdown' or 'json'.

        Returns:
            str: Markdown fact sheet, or JSON with this schema:

            {
                "created": bool,
                "collection_name": str,
                "exists": bool,
                "schema_ok": bool | null,
                "errors": [str, ...],
                "findings": [{...}, ...],
                "suggested_fixes": [{...}, ...],
                "status": str | null,
                "points_count": int | null,
                "vectors_count": int | null,
                "dense": {"size": int, "distance": str, "on_disk": bool} | null,
                "sparse": {"idf": bool, "on_disk": bool} | null,
                "quantization": str,
                "shards": int | null,
                "replicas": int | null,
                "on_disk_payload": bool | null,
                "embedding_compatibility": {...} | null
            }

        Examples:
            - Use when: setting up a new collection before the first ingest.
            - Use when: repairing missing payload indexes on an existing,
              schema-compatible collection (recreate=false).
            - Use when: wiping a collection and starting over with the same
              name (recreate=true, confirm=collection_name).
            - Don't use when: you only want to flip a toggle on an existing
              collection without losing its points → use
              qdrant_update_schema_toggles instead.

        Error Handling:
            Returns "Error: ..." (see describe_error) on any middleware
            failure or on a refused confirmation gate — in the latter case no
            HTTP call is made at all. A 409 means the collection exists but
            its schema or embedding contract does not match; read the
            message for whether to recreate it or adopt its contract.
        """
        try:
            if params.recreate:
                refuse_if(
                    not params.collection_name,
                    "recreate=true requires an explicit collection_name so "
                    "that confirm can be checked against it",
                )
                require_confirm(params.confirm, params.collection_name)
            body = params.model_dump(
                exclude_none=True, exclude={"response_format", "confirm"}
            )
            if params.recreate:
                body["confirm_recreate"] = params.confirm
            raw = await ctx.client.request("POST", "/collections", json=body)
        except Exception as exc:  # -> agent message
            return error_result(exc)

        payload = raw if isinstance(raw, dict) else {}
        return render(
            payload,
            params.response_format,
            title="Collection created",
            rows=_operation_rows,
        )

    @mcp.tool(
        name="qdrant_update_schema_toggles",
        annotations={
            "title": "Update collection schema toggles",
            "readOnlyHint": False,
            # Toggles can be disabled as well as enabled, so this is not an
            # additive-only update even though every point is preserved.
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def qdrant_update_schema_toggles(
        params: UpdateSchemaTogglesInput,
    ) -> StructuredToolResult:
        """Change mutable schema toggles on an existing collection in place.

        This never recreates the collection and never drops points; it only
        flips the toggles supplied (at least one is required by the
        middleware). Always sent as ``PATCH``: the middleware registers
        ``PUT`` and ``PATCH`` on the same handler, so there is no
        behavioural difference and no mode to choose between.

        Args:
            params (UpdateSchemaTogglesInput): Validated input containing:
                - collection_name (str): collection to update, required.
                - dense_on_disk, quantization, sparse_idf, sparse_on_disk,
                  on_disk_payload (bool | None): new value for each toggle;
                  omitted fields are left unchanged. At least one must be
                  given.
                - response_format (ResponseFormat): 'markdown' or 'json'.

        Returns:
            str: Markdown fact sheet, or JSON with the same schema documented
            on qdrant_create_collection.

        Examples:
            - Use when: moving a collection's vectors on-disk to save memory
              without losing its points.
            - Use when: enabling quantization on an existing collection.
            - Don't use when: the collection does not exist yet → use
              qdrant_create_collection instead.
            - Don't use when: you want to drop and rebuild the collection
              → use qdrant_create_collection with recreate=true instead.

        Error Handling:
            A call with every toggle omitted is refused before any HTTP
            call. Returns "Error: ..." (see describe_error) on any
            middleware failure. A 409 means the requested toggle(s) did not
            take effect as asked — the collection's actual state is
            unchanged from what you expected; call qdrant_get_collection to
            see what happened.
        """
        encoded = quote(params.collection_name, safe="")
        body = params.model_dump(
            exclude_none=True, exclude={"collection_name", "response_format"}
        )
        try:
            refuse_if(not body, "at least one schema toggle must be provided")
            raw = await ctx.client.request(
                "PATCH", f"/collections/{encoded}/schema-toggles", json=body
            )
        except Exception as exc:  # -> agent message
            return error_result(exc)

        payload = raw if isinstance(raw, dict) else {}
        return render(
            payload,
            params.response_format,
            title="Schema toggles updated",
            rows=_operation_rows,
        )

    @mcp.tool(
        name="qdrant_delete_collection",
        annotations={
            "title": "Delete a Qdrant collection",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def qdrant_delete_collection(
        params: DeleteCollectionInput,
    ) -> StructuredToolResult:
        """Permanently delete a collection and every point stored in it.

        This is irreversible: there is no undo, backup or soft-delete.
        ``confirm`` must equal ``collection_name`` exactly; a missing or
        mismatched ``confirm`` is rejected before any request reaches the
        middleware.

        Args:
            params (DeleteCollectionInput): Validated input containing:
                - collection_name (str): collection to delete, required.
                - confirm (str | None): required; must equal collection_name
                  exactly.
                - response_format (ResponseFormat): 'markdown' or 'json'.

        Returns:
            str: Markdown fact sheet, or JSON with this schema:

            {
                "ok": bool,
                "deleted": bool,
                "collection_name": str
            }

        Examples:
            - Use when: permanently removing a collection you no longer need.
            - Use when: clearing a collection before recreating it with a
              different schema (though qdrant_create_collection with
              recreate=true does both steps in one confirmed call).
            - Don't use when: you only want to change a schema toggle without
              losing data → use qdrant_update_schema_toggles instead.
            - Don't use when: you want to drop and immediately recreate the
              same collection in one call → use qdrant_create_collection
              with recreate=true instead.

        Error Handling:
            Returns "Error: ..." (see describe_error) on any middleware
            failure or on a refused confirmation gate — in the latter case no
            HTTP call is made at all. A 404 means the collection did not
            exist; call qdrant_list_collections to see what does.
        """
        try:
            require_confirm(params.confirm, params.collection_name)
            raw = await ctx.client.request(
                "DELETE",
                f"/collections/{quote(params.collection_name, safe='')}",
                params={"confirm_delete": params.confirm},
            )
        except Exception as exc:  # -> agent message
            return error_result(exc)

        payload = raw if isinstance(raw, dict) else {}
        return render(
            payload,
            params.response_format,
            title="Deleted collection",
            rows=_delete_rows,
        )

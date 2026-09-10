"""File-based and inline chunk ingestion.

Both tools share one contract with the middleware: point ids are derived
deterministically as ``UUID5(source_id:id)`` from each chunk's integer
``id`` — the chunker's per-source ordinal, not the point id itself. That
ordinal is ordinary payload: it is written under ``chunk_id_key`` (default
``"id"``) and *relocated* rather than copied, so renaming it leaves no
duplicate. The chunker-internal ``overflow_details``,
``overlap_sentences_count`` and ``split_metadata`` used to be stripped from the
payload before writing; the middleware currently keeps them. Re-ingesting the
same
source **overwrites** its points instead of duplicating them — that is why
both tools are ``idempotentHint: true``. The corollary an agent will get
wrong unless told explicitly: a source that has *shrunk* since the last
ingest leaves its removed chunks' points behind as orphans, because their
ids simply never appear again in a later call. ``replace_source=true`` is
the remedy — it deletes every existing point for that ``source_id`` first,
after the new points are durably written, so a shrunk source ends up with
exactly the points its latest content produced.

``replace_source=true`` is gated by ``confirm``, the same shape
``qdrant_create_collection`` uses for ``recreate=true``: ``confirm`` must
equal the ``source_id`` being replaced, exactly (see ``ExactString`` in
``smart_qdrant_mcp.tools.collections``). ``old_ids`` — the points a replace will
delete — is scoped purely by string equality on ``source_id``
(``_scroll_source_point_ids`` in the middleware), not by any relationship to
the *content* being ingested now. A mistyped or reused ``source_id`` that
happens to name a real, unrelated source therefore deletes that unrelated
source's points wholesale, with no overlap to save any of them — a
materially larger blast radius than the ordinary "shrunk source" case, and
one a plain shrunk-source framing does not warn against. Both tools are
therefore ``destructiveHint: true``: the annotation describes the
worst case a call can reach, not the common plain-ingest call, exactly as
``qdrant_create_collection`` is marked destructive for its ``recreate=true``
path even though most calls never set it.

A partial failure is reported inside an HTTP 200: ``ok: false`` with
``failed_at_chunk`` naming the position embedding or upsert failed at.
Chunks before that position were already written. ``_ingest_rows`` leads the
rendered output with that failure, before the ordinary counts, so a skim
can never mistake it for a clean success — the same convention
``qdrant_create_payload_index`` uses for its ``failed`` bucket.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

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

# Copied verbatim from SourceIdType in
# middleware-qdrant-manager/qdrant_middleware/api/routers/points.py:222-224. A
# drifted member here produces a tool call that Pydantic accepts locally but
# the middleware then rejects (or, worse, one the middleware accepts but this
# Literal never allowed) — keep the two in lockstep by inspection, not import,
# since the two services ship as independent images.
SourceIdType = Literal[
    "keyword", "text", "integer", "float", "bool", "datetime", "uuid", "geo", "untyped"
]

_SOURCE_ID_TYPE_DESCRIPTION = (
    "Qdrant payload type to coerce source_id into: 'keyword', 'text', "
    "'integer', 'float', 'bool', 'datetime', 'uuid', 'geo', or 'untyped' "
    "(e.g. 'keyword')"
)


class IngestFileInput(BaseModel):
    """Input model for ingesting chunks from a JSON file under the data root."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    file_path: str = Field(
        ...,
        min_length=1,
        description=(
            "Path to a JSON file, relative to the middleware's own data/ "
            "root (e.g. 'corpus/manual.json'). The file lives on the "
            "middleware host, not on the machine running this tool. "
            "Absolute paths and '..' segments are rejected upstream with "
            "400."
        ),
    )
    collection_name: Optional[str] = Field(
        default=None,
        description=(
            "Collection to ingest into (e.g. 'docs'); defaults to the "
            "middleware's configured default collection"
        ),
    )
    batch_size: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Chunks per embed/upsert batch (e.g. 64); defaults to the "
            "middleware's configured ingest batch size"
        ),
    )
    replace_source: bool = Field(
        default=False,
        description=(
            "Delete every existing point for this source_id after the new "
            "points are written, so a shrunk source leaves no orphan "
            "points behind. Deletion is scoped by source_id string equality "
            "alone, so a mistyped or reused source_id can wipe an unrelated "
            "source instead — requires an explicit source_id and a matching "
            "confirm (e.g. true)"
        ),
    )
    source_id: Optional[str] = Field(
        default=None,
        description=(
            "Source id to tag every chunk's payload with (e.g. "
            "'manual-v2'); defaults to the file's name without extension. "
            "Required, not defaulted, when replace_source=true, since "
            "confirm is checked against this value"
        ),
    )
    source_id_key: str = Field(
        default="source_id",
        min_length=1,
        description="Payload key that holds the source id (e.g. 'source_id')",
    )
    source_id_type: SourceIdType = Field(
        default="keyword", description=_SOURCE_ID_TYPE_DESCRIPTION
    )
    chunk_id_key: str = Field(
        default="id",
        min_length=1,
        description=(
            "Payload key that holds each chunk's ordinal (e.g. 'chunk_id'). "
            "The ordinal is relocated, not copied, so a value other than 'id' "
            "leaves no 'id' behind in the payload. Must differ from "
            "source_id_key and cannot be 'text', 'overflow_details', "
            "'overlap_sentences_count' or 'split_metadata'"
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extra payload fields merged into every chunk (e.g. "
            "{'lang': 'it', 'version': 3})"
        ),
    )
    confirm: Optional[ExactString] = Field(
        default=None,
        description=(
            "Required only when replace_source=true: must equal source_id "
            "exactly (not whitespace- or case-normalised), e.g. "
            "confirm='manual-v2' for source_id='manual-v2'"
        ),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (default) or 'json' (e.g. 'json')",
    )

    # replace_source=true requiring an explicit source_id (not the
    # file-name default) is enforced in the qdrant_ingest_file tool body (as
    # a ToolError), not here -- see the module docstring in
    # tools/payload.py for why a cross-field business rule like this one is
    # raised from the tool body rather than a model_validator.


class IngestChunksInput(BaseModel):
    """Input model for ingesting chunks supplied inline, with no file on disk."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    chunks: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description=(
            "Chunks to ingest. Each is an object with an integer 'id', "
            "unique within the call, and a non-empty 'text'; every key, "
            "'id' and 'text' included, becomes payload (e.g. "
            "[{'id': 1, 'text': 'hello', 'lang': 'it'}])"
        ),
    )
    source_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Source id to tag every chunk's payload with; required here "
            "since there is no file name to fall back to (e.g. 'manual')"
        ),
    )
    collection_name: Optional[str] = Field(
        default=None,
        description=(
            "Collection to ingest into (e.g. 'docs'); defaults to the "
            "middleware's configured default collection"
        ),
    )
    batch_size: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Chunks per embed/upsert batch (e.g. 64); defaults to the "
            "middleware's configured ingest batch size"
        ),
    )
    replace_source: bool = Field(
        default=False,
        description=(
            "Delete every existing point for this source_id after the new "
            "points are written, so a shrunk source leaves no orphan "
            "points behind. Deletion is scoped by source_id string equality "
            "alone, so a mistyped or reused source_id can wipe an unrelated "
            "source instead — requires a matching confirm (e.g. true)"
        ),
    )
    source_id_key: str = Field(
        default="source_id",
        min_length=1,
        description="Payload key that holds the source id (e.g. 'source_id')",
    )
    source_id_type: SourceIdType = Field(
        default="keyword", description=_SOURCE_ID_TYPE_DESCRIPTION
    )
    chunk_id_key: str = Field(
        default="id",
        min_length=1,
        description=(
            "Payload key that holds each chunk's ordinal (e.g. 'chunk_id'). "
            "The ordinal is relocated, not copied, so a value other than 'id' "
            "leaves no 'id' behind in the payload. Must differ from "
            "source_id_key and cannot be 'text', 'overflow_details', "
            "'overlap_sentences_count' or 'split_metadata'"
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extra payload fields merged into every chunk (e.g. "
            "{'lang': 'it', 'version': 3})"
        ),
    )
    confirm: Optional[ExactString] = Field(
        default=None,
        description=(
            "Required only when replace_source=true: must equal source_id "
            "exactly (not whitespace- or case-normalised), e.g. "
            "confirm='manual' for source_id='manual'"
        ),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (default) or 'json' (e.g. 'json')",
    )


def _ingest_rows(payload: dict[str, Any]) -> list[str]:
    """Render an ingest outcome, leading with the failing chunk when there is one.

    A partial failure still returns HTTP 200 (``ok: false``,
    ``failed_at_chunk`` naming the position); chunks before that position
    were already written. The failure line is emitted before the ordinary
    counts — mirroring how ``qdrant_create_payload_index`` leads with its
    failure summary ahead of the per-bucket bullets — so a skim of the
    output can never mistake a partial failure for a clean success.
    """
    failed_at = payload.get("failed_at_chunk")
    rows: list[str] = []
    if failed_at is not None:
        rows.append(
            f"**Ingest failed at chunk {failed_at}.** Chunks before it were "
            "written; re-run from that position after fixing the input."
        )
        rows.append("")
    rows.extend(
        [
            bullet("Collection", payload.get("collection_name")),
            bullet("Ingested", payload.get("ingested")),
            bullet("Skipped", payload.get("skipped")),
        ]
    )
    return rows


def register(mcp: MCPServer, ctx: ToolContext) -> None:
    """Register the file-based and inline ingest tools on the server."""

    @mcp.tool(
        name="qdrant_ingest_file",
        annotations={
            "title": "Ingest chunks from a file on the middleware host",
            "readOnlyHint": False,
            # replace_source=True deletes every point sharing source_id —
            # scoped by string equality alone, so a mistyped or reused
            # source_id wipes an unrelated source, not just the shrunk one.
            # The hint describes that worst case, not the common
            # plain-ingest call, exactly as qdrant_create_collection is
            # marked destructive for its recreate=true path.
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def qdrant_ingest_file(params: IngestFileInput) -> StructuredToolResult:
        """Embed and upsert chunks read from a JSON file on the middleware host.

        ``file_path`` is resolved **relative to the middleware's own
        ``data/`` root** — it is not a path on the machine running this
        tool, and the file must already exist there. Absolute paths and
        ``..`` segments are rejected upstream with HTTP 400 before any file
        is touched. The file is a JSON object with a top-level ``chunks``
        array, each entry an object with an integer ``id`` unique within the
        call and a non-empty ``text``; every key becomes payload. ``id`` is
        the chunker's ordinal and is stored under ``chunk_id_key`` (default
        ``"id"``) — set that to keep the ordinal under a name of your own,
        with no duplicate left behind. Point ids are derived
        deterministically from ``source_id`` and each chunk's integer ``id``,
        so re-ingesting the
        same file overwrites its previous points rather than duplicating
        them. If the file has shrunk since the last ingest, the chunks that
        no longer appear leave orphan points behind — pass
        ``replace_source=true`` to delete that source's existing points
        first, so only the current content's points remain afterward.

        ``replace_source=true`` carries a second, distinct hazard beyond the
        shrunk-source case: the points it deletes are selected by string
        equality on ``source_id`` alone, with no relationship to the file
        being ingested now. A mistyped or reused ``source_id`` that happens
        to name a real, unrelated source deletes that source's points
        wholesale — none of its chunk ids will collide with the new file's,
        so nothing is spared. For that reason ``replace_source=true``
        requires an explicit ``source_id`` (the file-name default is
        refused — a defaulted value cannot be safely confirmed) and a
        ``confirm`` equal to that ``source_id`` exactly; a missing or
        mismatched ``confirm`` is rejected before any request reaches the
        middleware.

        Args:
            params (IngestFileInput): Validated input containing:
                - file_path (str): JSON file path relative to the
                  middleware's data/ root, required.
                - collection_name (str | None): target collection; defaults
                  to the middleware's configured default.
                - batch_size (int | None): chunks per embed/upsert batch,
                  >= 1; defaults to the middleware's configured batch size.
                - replace_source (bool): delete the source's existing
                  points first, default False; requires source_id and
                  confirm when true.
                - source_id (str | None): id tagged onto every chunk;
                  defaults to the file's name without extension, except
                  when replace_source=true where it is required.
                - source_id_key (str): payload key for the source id,
                  default 'source_id'.
                - source_id_type (SourceIdType): payload type to coerce
                  source_id into, default 'keyword'.
                - chunk_id_key (str): payload key for each chunk's ordinal,
                  default 'id'; the ordinal is relocated there, so a
                  different value leaves no 'id' in the payload.
                - metadata (dict[str, Any]): extra fields merged into every
                  chunk's payload, default {}.
                - confirm (str | None): required only when
                  replace_source=true; must equal source_id exactly.
                - response_format (ResponseFormat): 'markdown' or 'json'.

        Returns:
            str: Markdown fact sheet, or JSON with this schema:

            {
                "ok": bool,
                "ingested": int,
                "skipped": int,
                "collection_name": str,
                "failed_at_chunk": int | null,
                "error": str | null
            }

        Examples:
            - Use when: a corpus file already sitting under the
              middleware's data/ directory needs to be embedded and stored.
            - Use when: a source file was edited and re-ingesting should
              overwrite the old points sharing its source_id.
            - Use when: a source file shrank and the removed chunks' points
              must not linger -> pass replace_source=true, source_id, and
              confirm=source_id.
            - Don't use when: the content is in this conversation rather
              than a file on the middleware host -> use
              qdrant_ingest_chunks instead.

        Error Handling:
            Returns "Error: invalid request ..." (see describe_error, 400)
            when the path escapes the data/ root, does not exist, or is too
            large. Returns "Error: ..." for every other middleware failure,
            or on a refused confirmation gate or a replace_source=true call
            missing an explicit source_id — in both of the latter cases no
            HTTP call is made at all. A 200 response with ok: false and a
            populated failed_at_chunk is a partial success, not an
            exception: chunks before that position were written and are
            visible in the rendered output before the counts.
        """
        try:
            if params.replace_source:
                refuse_if(
                    not params.source_id,
                    "replace_source=true requires an explicit source_id (not "
                    "the file-name default) so that confirm can be checked "
                    "against it",
                )
                require_confirm(params.confirm, params.source_id, noun="source_id")
            body = params.model_dump(
                exclude_none=True, exclude={"confirm", "response_format"}
            )
            raw = await ctx.client.request("POST", "/points/ingest", json=body)
        except Exception as exc:  # -> agent message
            return error_result(exc)

        payload = raw if isinstance(raw, dict) else {}
        return render(
            payload,
            params.response_format,
            title="Ingest result",
            rows=_ingest_rows,
        )

    @mcp.tool(
        name="qdrant_ingest_chunks",
        annotations={
            "title": "Ingest chunks supplied inline",
            "readOnlyHint": False,
            # replace_source=True deletes every point sharing source_id —
            # scoped by string equality alone, so a mistyped or reused
            # source_id wipes an unrelated source, not just the shrunk one.
            # The hint describes that worst case, not the common
            # plain-ingest call, exactly as qdrant_create_collection is
            # marked destructive for its recreate=true path.
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def qdrant_ingest_chunks(params: IngestChunksInput) -> StructuredToolResult:
        """Embed and upsert chunks supplied directly in the request, no file needed.

        Each entry in ``chunks`` needs an integer ``id``, unique within the
        call, and a non-empty ``text``; every key becomes payload. The ``id``
        is the chunker's ordinal that the point id derives from together with
        ``source_id``, and it is stored under ``chunk_id_key`` (default
        ``"id"``) — set that to keep the ordinal under a name of your own,
        with no duplicate left behind. ``source_id`` is required here —
        unlike qdrant_ingest_file there is no file name to fall back to.
        Point ids are deterministic (``source_id`` + ``id``), so
        re-sending the same chunks overwrites their previous points rather
        than duplicating them; if a later call sends fewer chunks than
        before, the chunks that are now missing leave orphan points behind
        unless ``replace_source=true`` deletes that source's existing
        points first. Read the resource docs://workflows/ingest for the
        full embed/upsert/prune contract shared with qdrant_ingest_file.

        ``replace_source=true`` carries a second, distinct hazard beyond the
        shrunk-source case: the points it deletes are selected by string
        equality on ``source_id`` alone, with no relationship to the chunks
        supplied now. A mistyped or reused ``source_id`` that happens to
        name a real, unrelated source deletes that source's points
        wholesale — none of its chunk ids will collide with the new
        chunks', so nothing is spared. For that reason ``replace_source=true``
        requires ``confirm`` to equal ``source_id`` exactly; a missing or
        mismatched ``confirm`` is rejected before any request reaches the
        middleware.

        Args:
            params (IngestChunksInput): Validated input containing:
                - chunks (list[dict[str, Any]]): at least one chunk object,
                  required.
                - source_id (str): id tagged onto every chunk, required.
                - collection_name (str | None): target collection; defaults
                  to the middleware's configured default.
                - batch_size (int | None): chunks per embed/upsert batch,
                  >= 1; defaults to the middleware's configured batch size.
                - replace_source (bool): delete the source's existing
                  points first, default False; requires confirm when true.
                - source_id_key (str): payload key for the source id,
                  default 'source_id'.
                - source_id_type (SourceIdType): payload type to coerce
                  source_id into, default 'keyword'.
                - chunk_id_key (str): payload key for each chunk's ordinal,
                  default 'id'; the ordinal is relocated there, so a
                  different value leaves no 'id' in the payload.
                - metadata (dict[str, Any]): extra fields merged into every
                  chunk's payload, default {}.
                - confirm (str | None): required only when
                  replace_source=true; must equal source_id exactly.
                - response_format (ResponseFormat): 'markdown' or 'json'.

        Returns:
            str: Markdown fact sheet, or JSON with this schema:

            {
                "ok": bool,
                "ingested": int,
                "skipped": int,
                "collection_name": str,
                "failed_at_chunk": int | null,
                "error": str | null
            }

        Examples:
            - Use when: chunk text and metadata are already available in
              this conversation, with no shared filesystem with the
              middleware.
            - Use when: re-sending an edited set of chunks should overwrite
              the previous points for the same source_id.
            - Use when: fewer chunks are sent than last time and the
              missing ones must not linger as orphan points -> pass
              replace_source=true and confirm=source_id.
            - Don't use when: the content already exists as a JSON file on
              the middleware host -> use qdrant_ingest_file instead, which
              does not require inlining the whole file into the call.

        Error Handling:
            Missing source_id, an empty chunks list, or a refused
            confirmation gate (replace_source=true with a missing or
            mismatched confirm) are all rejected before any request is
            made — no HTTP call occurs in any of these cases. Returns
            "Error: ..." (see describe_error) for every middleware failure.
            A 200 response with ok: false and a populated failed_at_chunk is
            a partial success, not an exception: chunks before that
            position were written and are visible in the rendered output
            before the counts.
        """
        try:
            if params.replace_source:
                require_confirm(params.confirm, params.source_id, noun="source_id")
            body = params.model_dump(
                exclude_none=True, exclude={"confirm", "response_format"}
            )
            raw = await ctx.client.request("POST", "/points/ingest-chunks", json=body)
        except Exception as exc:  # -> agent message
            return error_result(exc)

        payload = raw if isinstance(raw, dict) else {}
        return render(
            payload,
            params.response_format,
            title="Ingest result",
            rows=_ingest_rows,
        )

"""Vector reindex supervision: preflight, start, status, and control.

A reindex regenerates every point's vectors in the background against a new
embedding contract, with durable batch-boundary checkpoints so it survives a
restart. While a job is ``running``, ``finalizing`` or ``paused`` the
middleware rejects collection and point mutations with 409 to protect it —
that is intended, not a bug, and every docstring below says so. These four
tools are how an agent supervises that job: check readiness, start it,
poll it, and pause/resume/cancel it.
"""

from __future__ import annotations

import math
from asyncio import sleep
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
from smart_qdrant_mcp.guardrails import require_confirm
from smart_qdrant_mcp.tools.collections import ExactString

# Maps the validated `action` literal to its endpoint sub-path. Currently the
# identity mapping, but kept as a table rather than an if/elif chain so the
# full set of controllable actions is visible in one place and the sub-path
# is never computed by string-munging the action name.
CONTROL_PATHS: dict[str, str] = {
    "pause": "pause",
    "resume": "resume",
    "cancel": "cancel",
}

# How often a waiting status call re-reads the job. The middleware answers
# from collection metadata, so a read every two seconds costs next to nothing.
STATUS_POLL_SECONDS = 2


class ReindexPreflightInput(BaseModel):
    """Input model for a read-only reindex readiness check."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    collection_name: str = Field(
        ..., min_length=1, description="Collection to check (e.g. 'docs')"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (default) or 'json' (e.g. 'json')",
    )


class StartReindexInput(BaseModel):
    """Input model for starting a background vector reindex job."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    collection_name: str = Field(
        ..., min_length=1, description="Collection to reindex (e.g. 'docs')"
    )
    confirm: Optional[ExactString] = Field(
        default=None,
        description=(
            "Required: must equal collection_name exactly (not whitespace- "
            "or case-normalised) to confirm starting the job, e.g. "
            "confirm='docs' for collection_name='docs'"
        ),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (default) or 'json' (e.g. 'json')",
    )


class ReindexStatusInput(BaseModel):
    """Input model for reading a collection's reindex job status."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    collection_name: str = Field(
        ..., min_length=1, description="Collection to check (e.g. 'docs')"
    )
    wait_seconds: int = Field(
        default=0,
        ge=0,
        le=30,
        description=(
            "Seconds to wait for a running job to change state before "
            "answering: returns as soon as its status or task_running "
            "changes, else after this long; 0 (default) answers at once "
            "(e.g. 30)"
        ),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (default) or 'json' (e.g. 'json')",
    )


class ControlReindexInput(BaseModel):
    """Input model for pausing, resuming, or cancelling a reindex job."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    collection_name: str = Field(
        ..., min_length=1, description="Collection whose job to control (e.g. 'docs')"
    )
    action: Literal["pause", "resume", "cancel"] = Field(
        ...,
        description=(
            "Control action to apply: 'pause' (keep checkpoint), 'resume' "
            "(continue from checkpoint), or 'cancel' (abandon progress), "
            "e.g. 'pause'"
        ),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (default) or 'json' (e.g. 'json')",
    )


def _preflight_rows(payload: dict[str, Any]) -> list[str]:
    """Render a preflight report, leading with blockers when it cannot start.

    A ``can_start: false`` response must not read like an ordinary summary:
    the blockers are what an agent needs first, ahead of the point counts,
    mirroring how qdrant_create_payload_index leads with its failure summary
    before the per-bucket bullets.
    """
    blockers = payload.get("blockers") or []
    rows: list[str] = []
    if not payload.get("can_start") and blockers:
        rows.append(f"**Cannot start: {'; '.join(blockers)}**")
        rows.append("")
    rows.extend(
        [
            bullet("Collection", payload.get("collection_name")),
            bullet("Can start", payload.get("can_start")),
            bullet("Qdrant version", payload.get("qdrant_version")),
            bullet("Total points", payload.get("total_points")),
            bullet("Eligible points", payload.get("eligible_points")),
            bullet("Invalid points", payload.get("invalid_points")),
            bullet("Payload field", payload.get("payload_field")),
        ]
    )
    invalid_ids = payload.get("invalid_point_ids") or []
    if invalid_ids:
        rows.extend(["", "## Invalid point IDs", *(f"- {pid}" for pid in invalid_ids)])
    if blockers:
        rows.extend(["", "## Blockers", *(f"- {blocker}" for blocker in blockers)])
    return rows


def _job_rows(payload: dict[str, Any]) -> list[str]:
    """Render the three job states a status/action response can report.

    "No job exists" and "a job finished" must never render identically, nor
    must either be confused with "the metadata namespace is unreadable" —
    each has a different remedy (start one, do nothing, or cancel to clear
    it). Every key read here exists on ReindexJobState
    (schemas/embedding_contract.py:86-111): ``status`` (not ``state``) and
    ``processed_points``/``total_points`` (not ``processed``/``total``).
    """
    job = payload.get("job")
    if payload.get("metadata_invalid"):
        return [
            bullet("Collection", payload.get("collection_name")),
            "",
            "Reindex metadata is unreadable: the job state was reset. Run "
            "qdrant_reindex_preflight before starting a new job.",
        ]
    if not job:
        return [
            bullet("Collection", payload.get("collection_name")),
            bullet("Running", payload.get("task_running")),
            "",
            "There is no reindex job for this collection.",
        ]
    return [
        bullet("Collection", payload.get("collection_name")),
        bullet("Job ID", job.get("job_id")),
        bullet("State", job.get("status")),
        bullet("Running", payload.get("task_running")),
        bullet(
            "Processed",
            f"{job.get('processed_points')} of {job.get('total_points')}",
        ),
    ]


def _status_rows(payload: dict[str, Any]) -> list[str]:
    """Render reindex status. See _job_rows for the three-state logic."""
    return _job_rows(payload)


def _job_phase(payload: dict[str, Any]) -> tuple[object, object]:
    """Return what a waiting status call watches: state and task_running.

    Processed counts are left out on purpose: they move every batch, and a
    wait that ended on them would end after one read.
    """
    return (payload.get("job") or {}).get("status"), payload.get("task_running")


def _action_rows(payload: dict[str, Any]) -> list[str]:
    """Render a start/pause/resume/cancel outcome: the ok flag plus job state.

    ``ok`` is the one field ReindexActionResponse has that
    ReindexStatusResponse does not (schemas/reindex.py:44-56 vs 58-66); the
    rest of the state block is identical, so it is shared via _job_rows.
    """
    return [bullet("Ok", payload.get("ok")), *_job_rows(payload)]


def register(mcp: MCPServer, ctx: ToolContext) -> None:
    """Register the four reindex supervision tools on the server."""

    @mcp.tool(
        name="qdrant_reindex_preflight",
        annotations={
            "title": "Check reindex readiness for a collection",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def qdrant_reindex_preflight(
        params: ReindexPreflightInput,
    ) -> StructuredToolResult:
        """Check whether a reindex can start, without changing anything.

        This validates the complete source collection and reports how many
        points are eligible for regeneration and how many are not (with
        their IDs), plus any blockers that would prevent a start (for
        example, a job already running). It makes no changes to the
        collection or its points. Call this before every
        qdrant_start_reindex — a ``can_start: false`` response tells you
        why without having to attempt (and have rejected) the real start.

        Args:
            params (ReindexPreflightInput): Validated input containing:
                - collection_name (str): collection to check, required.
                - response_format (ResponseFormat): 'markdown' or 'json'.

        Returns:
            str: Markdown fact sheet, or JSON with this schema:

            {
                "collection_name": str,
                "can_start": bool,
                "qdrant_version": str | null,
                "source_contract": {...} | null,   # opaque; embedding-contract
                "target_contract": {...},          # resource has the shape
                "source_vectors": {...},
                "total_points": int,
                "eligible_points": int,
                "invalid_points": int,
                "invalid_point_ids": [str, ...],
                "payload_field": str,
                "blockers": [str, ...]
            }

        Examples:
            - Use when: about to call qdrant_start_reindex and you need to
              know first whether it will be accepted.
            - Use when: diagnosing why points are missing from search after
              a contract change — invalid_point_ids names exactly which
              points would be skipped.
            - Don't use when: you already know a job is running and want its
              live progress → use qdrant_get_reindex_status instead; this
              tool reports readiness, not an in-progress job's state.

        Error Handling:
            Returns "Error: not found ..." pointing at qdrant_list_collections
            when the collection does not exist (404), and describe_error's
            standard mapping (including the embedding-contract branch on
            409) for every other failure.
        """
        try:
            raw = await ctx.client.request(
                "POST",
                f"/collections/{quote(params.collection_name, safe='')}"
                "/reindex/preflight",
                retryable=True,
            )
        except Exception as exc:  # -> agent message
            return error_result(exc)

        payload = raw if isinstance(raw, dict) else {}
        return render(
            payload,
            params.response_format,
            title="Reindex preflight",
            rows=_preflight_rows,
        )

    @mcp.tool(
        name="qdrant_start_reindex",
        annotations={
            "title": "Start a background vector reindex",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    )
    async def qdrant_start_reindex(params: StartReindexInput) -> StructuredToolResult:
        """Start regenerating every point's vectors in the background.

        The job runs after this call returns, checkpointing at durable batch
        boundaries so it can resume from where it left off after an
        interruption (see qdrant_control_reindex for resume). While it is
        ``running``, ``finalizing`` or ``paused``, **the middleware rejects
        collection and point mutations on this collection with 409** — this
        is intentional protection against a mutation racing the
        regeneration, not a bug. ``confirm`` must equal ``collection_name``
        exactly; a missing or mismatched ``confirm`` is rejected before any
        request reaches the middleware, so starting twice without checking
        status first is never silently harmless. Read the resource
        docs://workflows/reindex for the full lifecycle before using this.

        Args:
            params (StartReindexInput): Validated input containing:
                - collection_name (str): collection to reindex, required.
                - confirm (str | None): required; must equal collection_name
                  exactly.
                - response_format (ResponseFormat): 'markdown' or 'json'.

        Returns:
            str: Markdown fact sheet, or JSON with this schema:

            {
                "ok": bool,
                "collection_name": str,
                "job": {
                    "job_id": str, "status": str, "total_points": int,
                    "processed_points": int, "started_at": str,
                    "updated_at": str, "error": str | null, ...
                } | null,
                "task_running": bool,
                "metadata_invalid": bool
            }

        Examples:
            - Use when: qdrant_reindex_preflight reported can_start=true and
              you are ready to regenerate the collection's vectors.
            - Use when: recovering an interrupted job by starting fresh after
              qdrant_control_reindex(action='cancel').
            - Don't use when: a job is already running or paused for this
              collection → use qdrant_get_reindex_status first; starting
              again is not a no-op and is not what you want mid-job.
            - Don't use when: you only want to check whether a start would
              succeed → use qdrant_reindex_preflight instead, which makes no
              changes.

        Error Handling:
            Returns "Error: ..." (see describe_error) on any middleware
            failure or on a refused confirmation gate — in the latter case no
            HTTP call is made at all. A 409 means a job is already active (or
            the embedding contract is incompatible); call
            qdrant_get_reindex_status to see which.
        """
        try:
            require_confirm(params.confirm, params.collection_name)
            raw = await ctx.client.request(
                "POST",
                f"/collections/{quote(params.collection_name, safe='')}/reindex",
                json={"confirm_collection": params.confirm},
            )
        except Exception as exc:  # -> agent message
            return error_result(exc)

        payload = raw if isinstance(raw, dict) else {}
        return render(
            payload,
            params.response_format,
            title="Reindex started",
            rows=_action_rows,
        )

    @mcp.tool(
        name="qdrant_get_reindex_status",
        annotations={
            "title": "Get a collection's reindex job status",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def qdrant_get_reindex_status(
        params: ReindexStatusInput,
    ) -> StructuredToolResult:
        """Report a reindex job's progress, distinguishing three states.

        These three look alike to a casual read but have different remedies,
        so this tool never lets them render the same way:

        1. No job exists (never started, or the last one finished and was
           cleared) — reported as "There is no reindex job for this
           collection."
        2. A job exists, with its state (running, paused, interrupted,
           finalizing, completed, failed, or cancelled) and processed/total
           point counts.
        3. The metadata namespace is unreadable — no job state can be
           reported at all, and the only way to clear it is
           qdrant_control_reindex(action='cancel'), after which
           qdrant_reindex_preflight can run again.

        With ``wait_seconds`` the call itself does the waiting: it re-reads
        the job every few seconds and answers as soon as its status or
        ``task_running`` changes (a pause taking effect, a job finishing or
        failing), or once the time is up. A job with no task running cannot
        change on its own, so that answer comes at once.

        Args:
            params (ReindexStatusInput): Validated input containing:
                - collection_name (str): collection to check, required.
                - wait_seconds (int): 0-30, default 0 (answer at once).
                - response_format (ResponseFormat): 'markdown' or 'json'.

        Returns:
            str: Markdown fact sheet, or JSON with this schema:

            {
                "collection_name": str,
                "job": {
                    "job_id": str, "status": str, "total_points": int,
                    "processed_points": int, "started_at": str,
                    "updated_at": str, "error": str | null, ...
                } | null,
                "task_running": bool,
                "metadata_invalid": bool
            }

        Examples:
            - Use when: waiting for an in-progress reindex to complete or
              fail before relying on the collection's vectors — pass
              wait_seconds=30 rather than calling again at once.
            - Use when: a collection mutation was rejected with a 409
              mentioning reindex, to see whether it is safe to wait it out.
            - Don't use when: you want to check whether a reindex *can*
              start, not whether one is running → use
              qdrant_reindex_preflight instead.

        Error Handling:
            Returns "Error: not found ..." pointing at qdrant_list_collections
            when the collection does not exist (404), and describe_error's
            standard mapping for every other failure. This call itself never
            409s for a corrupt metadata namespace — that case is reported as
            metadata_invalid: true in a normal 200 response instead, so
            status-checking always works even when the state is corrupt.
        """
        path = f"/collections/{quote(params.collection_name, safe='')}/reindex"
        try:
            raw = await ctx.client.request("GET", path)
            payload = raw if isinstance(raw, dict) else {}
            phase = _job_phase(payload)
            polls = math.ceil(params.wait_seconds / STATUS_POLL_SECONDS)
            while polls and phase[1] and _job_phase(payload) == phase:
                polls -= 1
                await sleep(STATUS_POLL_SECONDS)
                raw = await ctx.client.request("GET", path)
                payload = raw if isinstance(raw, dict) else {}
        except Exception as exc:  # -> agent message
            return error_result(exc)

        return render(
            payload,
            params.response_format,
            title="Reindex status",
            rows=_status_rows,
        )

    @mcp.tool(
        name="qdrant_control_reindex",
        annotations={
            "title": "Pause, resume, or cancel a reindex job",
            "readOnlyHint": False,
            # cancel irreversibly abandons the durable checkpoint.
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    )
    async def qdrant_control_reindex(
        params: ControlReindexInput,
    ) -> StructuredToolResult:
        """Pause, resume, or cancel a collection's reindex job.

        ``pause`` stops the job at the next durable batch boundary and keeps
        its checkpoint, so a later ``resume`` continues from the last offset
        committed rather than restarting. That boundary is why ``pause``
        returns before the work has actually stopped: the status flips to
        ``paused`` while the in-flight batch finishes, and a ``resume`` sent
        in that window is refused with "a reindex task is already running".
        Call qdrant_get_reindex_status with ``wait_seconds`` until
        ``task_running`` is false before resuming. ``cancel`` abandons the job's
        progress entirely — the source generation is left active and
        untouched, but the checkpoint is gone, so a fresh
        qdrant_start_reindex begins from zero. **A paused job still blocks
        collection and point mutations with 409**, exactly like a running
        one; only ``cancel`` (or letting it run to completion) lifts that
        block. Cancel is also the only way to clear an unreadable metadata
        namespace (metadata_invalid: true) reported by
        qdrant_get_reindex_status.

        Args:
            params (ControlReindexInput): Validated input containing:
                - collection_name (str): collection whose job to control,
                  required.
                - action ('pause' | 'resume' | 'cancel'): control action to
                  apply, required.
                - response_format (ResponseFormat): 'markdown' or 'json'.

        Returns:
            str: Markdown fact sheet, or JSON with this schema:

            {
                "ok": bool,
                "collection_name": str,
                "job": {
                    "job_id": str, "status": str, "total_points": int,
                    "processed_points": int, "started_at": str,
                    "updated_at": str, "error": str | null, ...
                } | null,     # null only after a cancel that reset an
                              # unreadable metadata namespace
                "task_running": bool,
                "metadata_invalid": bool
            }

        Examples:
            - Use when: pausing a running job before a planned maintenance
              window, then resuming it afterward.
            - Use when: abandoning a reindex started against the wrong
              embedding contract, to start a fresh one.
            - Use when: clearing a stuck collection reported as
              metadata_invalid by qdrant_get_reindex_status (action='cancel').
            - Don't use when: no job is running or paused for this
              collection → check qdrant_get_reindex_status first; pause and
              resume are not no-ops.

        Error Handling:
            Returns "Error: ..." (see describe_error) on any middleware
            failure. An unrecognized action is rejected by input validation
            before any request is made — no HTTP call occurs. A 404 means
            the collection does not exist; call qdrant_list_collections to
            see what does.
        """
        try:
            raw = await ctx.client.request(
                "POST",
                f"/collections/{quote(params.collection_name, safe='')}"
                f"/reindex/{CONTROL_PATHS[params.action]}",
            )
        except Exception as exc:  # -> agent message
            return error_result(exc)

        payload = raw if isinstance(raw, dict) else {}
        return render(
            payload,
            params.response_format,
            title="Reindex control",
            rows=_action_rows,
        )

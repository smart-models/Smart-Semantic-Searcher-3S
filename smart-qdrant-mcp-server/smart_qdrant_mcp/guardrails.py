"""Confirmation gates and cross-field business rules for tool bodies.

Every check here used to live in a Pydantic ``model_validator`` or
``field_validator`` and was moved into the tool body instead, so its
refusal is raised as a ``ToolError`` and flows through ``describe_error``
like every other guardrail -- rather than surfacing to the client as the
SDK's own validation-error envelope (``"Error executing tool ...: 1
validation error ..."``, with a pydantic.dev URL). See the module docstring
in ``tools/payload.py`` for the fuller rationale; this module holds the
checks reused across more than one tool.
"""

from __future__ import annotations

from typing import Any, Optional

from smart_qdrant_mcp.context import ToolContext
from smart_qdrant_mcp.errors import ToolError


def refuse_if(condition: bool, message: str) -> None:
    """Raise ``ToolError(message)`` when ``condition`` is true.

    A thin wrapper so a tool body reads as a direct translation of its
    business rule ("if this bad state holds, refuse") rather than repeating
    ``if ...: raise ToolError(...)`` inline at every call site that used to
    be a one-off Pydantic validator.
    """
    if condition:
        raise ToolError(message)


def require_selector_args(
    selector: str, ids: Optional[list[str]], filters: Optional[dict[str, Any]]
) -> None:
    """Raise ``ToolError`` unless the chosen selector's own argument is present.

    ``selector="ids"`` needs a non-empty ``ids`` list; ``selector="filter"``
    needs a non-empty ``filters`` object, since an empty filter would match
    every point. Shared by every tool that offers this id-or-filter
    addressing choice: ``qdrant_set_payload``, ``qdrant_delete_payload_keys``,
    ``qdrant_clear_payload`` (``tools/payload.py``) and
    ``qdrant_delete_points`` (``tools/points.py``), rather than reimplementing
    the same two-line check four times.
    """
    if selector == "ids" and not ids:
        raise ToolError("selector='ids' requires a non-empty ids list")
    if selector == "filter" and not filters:
        raise ToolError(
            "selector='filter' requires a non-empty filters object; an "
            "empty filter would match every point"
        )


def require_non_blank_ids(ids: Optional[list[str]]) -> None:
    """Raise ``ToolError`` if any point id is blank or whitespace-only.

    A blank id is meaningless upstream and dangerous in the one place it is
    interpolated into a path: the single-id endpoints build ``/points/{id}``,
    so ``""`` produces ``/points/`` — which the middleware answers with a 307
    to ``/points``, the *bulk* delete route. Nothing follows that redirect
    today (``httpx`` does not by default), so the call currently fails with a
    confusing 404/307 rather than deleting anything; refusing the id here
    keeps that from ever depending on a client flag.

    Applied by the tools that interpolate an id into a path — the writes:
    ``qdrant_delete_points`` and the three payload writes' per-id fan-out.
    ``qdrant_read_points`` deliberately does not use it: it addresses points
    through ``POST /points/retrieve`` with no path interpolation, and its
    docstring already delegates id *validity* to the middleware so the two do
    not drift. Mirrors the ``keys`` check the payload tools already perform.
    """
    if ids and any(not str(point_id).strip() for point_id in ids):
        raise ToolError("point ids cannot be blank")


# One request per id is the only shape the middleware offers for a payload
# write addressed by ids: there is no bulk-by-ids payload endpoint, only
# per-id and by-filter. So this limit is not about request size, it is about
# what a single tool call can actually finish and report on. A thousand
# sequential round trips is already minutes; beyond that the call outlives any
# client timeout, and a cancellation takes the per-id report of what was
# already written down with it.
FAN_OUT_ID_LIMIT = 1000


def require_fan_out_size(ids: Optional[list[str]]) -> None:
    """Raise ``ToolError`` when an ids-addressed payload write is too large.

    The message names the way out rather than only the limit: ``selector=
    "filter"`` performs the same write in one request, whatever the number of
    points it matches. ``qdrant_scroll_points`` is how a caller turns a list of
    ids it already holds into a filter it can express.
    """
    if ids is not None and len(ids) > FAN_OUT_ID_LIMIT:
        raise ToolError(
            f"selector='ids' writes one request per id and is capped at "
            f"{FAN_OUT_ID_LIMIT} ({len(ids)} given): a larger set cannot "
            "finish or report reliably inside one call. Use "
            "selector='filter', which writes them in a single request, or "
            "split the ids across several calls."
        )


def require_confirm(
    confirm: str | None, target: str, *, noun: str = "collection name"
) -> None:
    """Raise unless ``confirm`` exactly matches ``target``.

    ``noun`` names what ``target`` actually is in the resulting message
    (default ``"collection name"``, matching every existing caller in
    ``collections.py``, ``payload_indexes.py`` and ``reindex.py``, all of
    which confirm against a collection name). A caller confirming against a
    different identifier — ``ingest.py`` confirms against ``source_id``,
    which is independent of ``collection_name`` — must pass the matching
    noun, or the message misdirects an agent into retrying with the wrong
    value.

    A falsy ``confirm`` (``None`` or ``""``) is refused outright, before it
    is ever compared to ``target``: without this, a caller whose ``target``
    itself normalises to ``""`` (an unset value coerced to an empty string)
    could satisfy the check with an equally empty ``confirm`` and never
    actually confirm anything. Every current call site passes a non-empty
    ``target``, so this changes nothing for them.
    """
    if not confirm or confirm != target:
        raise ToolError(
            f"this operation destroys data: set confirm to the exact {noun} "
            f"{target!r} to proceed"
        )


_COUNT_UNVERIFIED = (
    "the count endpoint returned a response the count could not be "
    "determined from; the delete must not proceed on an unverified count"
)


async def count_matching(
    ctx: ToolContext, collection: str, filters: dict[str, Any]
) -> int:
    """Count the points a filter matches, for a dry run before deleting them.

    Any response shape that is not unambiguously a count raises ``ToolError``
    rather than defaulting to zero: zero is exactly the answer that makes a
    caller comfortable proceeding with a mass delete, so a malformed body
    must never be able to impersonate an empty result set. A ``MiddlewareError``
    from the request itself is not caught here and propagates unchanged, so
    ``describe_error`` maps it to its own branch instead of the generic one.
    """
    payload = await ctx.client.request(
        "POST",
        "/points/count",
        json={"collection_name": collection, "filters": filters},
        retryable=True,
    )
    if not isinstance(payload, dict) or "count" not in payload:
        raise ToolError(_COUNT_UNVERIFIED)
    count = payload["count"]
    if isinstance(count, bool) or not isinstance(count, int):
        raise ToolError(_COUNT_UNVERIFIED)
    return count

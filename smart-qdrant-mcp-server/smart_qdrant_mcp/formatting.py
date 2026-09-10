"""Output rendering. Every tool that returns text goes through here.

Markdown is the default because it costs fewer context tokens than JSON for the
same information; JSON exists for callers that post-process the result.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from enum import Enum
from typing import Annotated, Any

from mcp_types import CallToolResult, TextContent

StructuredToolResult = Annotated[CallToolResult, dict[str, Any]]
"""Tool result with a generic object output schema and dual text/JSON content."""


class ResponseFormat(str, Enum):
    """Output format for tool responses."""

    MARKDOWN = "markdown"
    JSON = "json"


def bullet(label: str, value: object) -> str:
    """Render one markdown fact line, showing an em dash for missing values."""
    shown = "—" if value is None or value == "" else value
    return f"- **{label}**: {shown}"


def render(
    payload: dict[str, Any],
    fmt: ResponseFormat,
    *,
    title: str,
    rows: Callable[[dict[str, Any]], list[str]],
) -> CallToolResult:
    """Return text for the model plus the same payload as structured content."""
    if fmt is ResponseFormat.JSON:
        text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    else:
        lines = [f"# {title}", *rows(payload)]
        text = "\n".join(lines).rstrip() + "\n" if len(lines) > 1 else f"# {title}"
    return CallToolResult(
        content=[TextContent(text=text)],
        structured_content=payload,
        is_error=False,
    )


def paginate(
    items: list[Any],
    *,
    total: int | None,
    offset: int | str | None,
    limit: int,
    next_offset: int | str | None = None,
) -> dict[str, Any]:
    """Wrap a page of results in the envelope every listing tool returns.

    ``total`` is None when the upstream endpoint does not count matches; in that
    case a full page is treated as "there may be more", but only when ``offset``
    is an integer this function can advance itself. For cursor-paginated
    endpoints (a non-integer ``offset``, e.g. a scroll point ID) there is no way
    to derive a continuation cursor here: the caller must pass the real one via
    ``next_offset``. Without that, or when ``limit`` is not positive, this
    reports the page as the end of results rather than "more, but nothing to
    page with".

    The invariant ``has_more == (next_offset is not None)`` holds for every
    return value.
    """
    count = len(items)
    if next_offset is not None:
        has_more = True
    elif total is not None:
        consumed = (offset if isinstance(offset, int) else 0) + count
        has_more = consumed < total
        next_offset = consumed if has_more else None
    elif limit <= 0 or not isinstance(offset, int):
        has_more = False
    else:
        has_more = count >= limit
        if has_more:
            next_offset = offset + count
    return {
        "total": total,
        "count": count,
        "offset": offset,
        "has_more": has_more,
        "next_offset": next_offset if has_more else None,
        "items": items,
    }

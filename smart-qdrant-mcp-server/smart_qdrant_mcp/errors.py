"""Translate upstream failures into next-step guidance for an agent.

Two rules govern this module. Every message says what to do next, and no
message carries upstream internals (paths, tracebacks, hostnames) to the
client — those belong in the server log only.
"""

from __future__ import annotations

import re

import httpx
from mcp_types import CallToolResult, TextContent

# The literal ``reindex``/``re-index`` markers never appear in the real
# messages the middleware raises for an active job blocking a mutation —
# every one of those instead says the vectors are being regenerated and the
# operation is temporarily blocked (verified against every
# ``EmbeddingContractViolation("reindex_in_progress", ...)`` call site in
# middleware-qdrant-manager/qdrant_middleware/repositories/qdrant_collections.py:
# the payload-index, vector-operation, collection-mutation, and
# schema-toggle paths). Kept for forward compatibility with any future
# message that does use the word, but "regenerated" and "temporarily
# blocked" are what actually fires today.
REINDEX_MARKERS = ("reindex", "re-index", "regenerated", "temporarily blocked")
# Real embedding-contract-mismatch wording, verified against every
# ``EmbeddingContractViolation("incompatible", ...)`` /
# ``EmbeddingContractViolation("schema_mismatch", ...)`` call site and the
# plain-string 409 raised directly by the create-collection route in
# middleware-qdrant-manager/qdrant_middleware/repositories/qdrant_collections.py
# and .../api/routers/collections.py: none of these contain the literal
# phrase "embedding contract" alone, so the original two markers below never
# matched any message the middleware actually produces.
CONTRACT_MARKERS = (
    "embedding contract",
    "vector name",
    "vectors do not match",
    "different vector spaces",
    "smart embedder contract",
    "no dense vector",
    "no sparse vector",
)

_WINDOWS_PATH_RE = re.compile(r"^[A-Za-z]:\\")
# A bare "host:port" or "ip:port" fragment, e.g. "10.0.4.22:6333" or
# "internal-qdrant.svc.cluster.local:6333[/path]". The lookahead requires a
# letter or a dot in the host part so ordinary "12:30"-style tokens are left
# alone.
_HOST_PORT_RE = re.compile(
    r"^(?=[A-Za-z0-9._\-]*[A-Za-z.])[A-Za-z0-9._\-]+:[0-9]+(?:/\S*)?$"
)
# Punctuation a sentence commonly wraps a fragment in -- never "/", which is
# part of the Unix-path fragment itself, so a leading or trailing slash must
# survive stripping.
_LEADING_PUNCT_RE = re.compile(r"^[\"'(\[{<]+")
_TRAILING_PUNCT_RE = re.compile(r"[\"')\]}>.,;:!?]+$")


class MiddlewareError(Exception):
    """A non-success response from the middleware."""

    def __init__(self, status: int | None, detail: object, path: str) -> None:
        """Capture the failing response so a tool can render it for the agent.

        Args:
            status: HTTP status code, or ``None`` when the request never got a
                response (connection error, timeout).
            detail: Decoded response body, kept unnormalised; the renderers
                decide how much of it reaches the agent's context.
            path: Request path the failure came from.
        """
        super().__init__(f"{status} from {path}")
        self.status = status
        self.detail = detail
        self.path = path


class ToolError(Exception):
    """A guardrail or precondition failure raised by a tool itself."""


# Upstream text reaches the agent's context verbatim, so it is bounded twice:
# by how many validation items are worth reading, and by total characters. A
# body that validates a list item by item (ids, points, chunks) answers with one
# entry per element, so an unbounded join turns a single bad request into
# thousands of near-identical lines. The head is what an agent acts on; the rest
# is reported as a count so the message never pretends to be complete.
_MAX_VALIDATION_ITEMS = 10
_MAX_DETAIL_CHARS = 1000


def _truncate(text: str, limit: int = _MAX_DETAIL_CHARS) -> str:
    """Cut ``text`` to ``limit`` characters, marking that it was cut."""
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}… (truncated)"


def _flatten_validation_detail(detail: object) -> str:
    """Render FastAPI's structured validation detail as field: message pairs."""
    if not isinstance(detail, list):
        return _as_text(detail)
    parts: list[str] = []
    for item in detail:
        if not isinstance(item, dict):
            continue
        location = item.get("loc") or []
        field = ".".join(str(piece) for piece in location if piece != "body")
        parts.append(f"{field or 'body'}: {item.get('msg', 'invalid value')}")
    if not parts:
        return "invalid request body"
    shown = parts[:_MAX_VALIDATION_ITEMS]
    remaining = len(parts) - len(shown)
    joined = "; ".join(shown)
    if remaining:
        joined = f"{joined}; (+{remaining} more validation errors)"
    return joined


def _as_text(detail: object) -> str:
    """Coerce a detail payload to a short single-line string."""
    if isinstance(detail, dict):
        message = detail.get("message") or detail.get("detail") or ""
        return str(message)
    return str(detail or "")


def _strip_punctuation(token: str) -> tuple[str, str, str]:
    """Split ``token`` into (leading punctuation, core, trailing punctuation).

    A sentence commonly wraps a sensitive fragment in "(...)", a trailing
    "." or ",", or similar -- which defeats the anchored checks below
    (a bare host:port, a leading Unix path, a Windows drive path) unless
    that wrapping is peeled off first. Only bracket/quote/sentence
    punctuation is stripped, never "/", since a leading or trailing slash
    is part of the path fragment itself, not decoration around it.
    """
    leading_match = _LEADING_PUNCT_RE.match(token)
    leading = leading_match.group(0) if leading_match else ""
    rest = token[len(leading) :]
    trailing_match = _TRAILING_PUNCT_RE.search(rest)
    trailing = trailing_match.group(0) if trailing_match else ""
    core = rest[: len(rest) - len(trailing)] if trailing else rest
    return leading, core, trailing


def _redact_token(token: str) -> str:
    """Redact one whitespace-delimited token if it exposes internals.

    A token is replaced with ``[redacted]`` when it looks like a path, a
    traceback marker, a URL or a ``host:port`` pair; ordinary words are left
    untouched.

    The traceback/URL/``.py``/``/app/`` checks below are substring checks on
    purpose, not anchored to the token's start: ``"url=http://q:6333/x"`` and
    ``"(main.py)"`` must be caught the same as a bare fragment. The
    remaining checks (a leading Unix path, a Windows drive path, a bare
    host:port) *are* anchored, because they need the fragment's shape, not
    just its presence -- so they run against the token with any wrapping
    punctuation peeled off, via ``_strip_punctuation``.
    """
    lowered = token.lower()
    if "traceback" in lowered:
        return "[redacted]"
    if "http://" in lowered or "https://" in lowered:
        return "[redacted]"
    if ".py" in lowered:
        return "[redacted]"
    if "/app/" in lowered:
        return "[redacted]"

    leading, core, trailing = _strip_punctuation(token)
    if (
        _WINDOWS_PATH_RE.match(core)
        or (core.startswith("/") and len(core) > 1)
        or _HOST_PORT_RE.match(core)
    ):
        return f"{leading}[redacted]{trailing}"
    return token


def redact(text: str) -> str:
    """Redact internal implementation fragments from upstream text.

    This is targeted redaction, not suppression: only the offending
    fragments (paths, tracebacks, URLs, host:port pairs) are replaced with
    ``[redacted]``, so the surrounding words — such as a validation field
    name and its reason — remain and keep the message actionable.

    The single entry point for this: every call site that needs to put
    upstream text in front of a client goes through here (or through
    ``redact_detail`` below, for a raw ``detail`` payload rather than
    already-flattened text) rather than re-implementing the fragment
    patterns. Being that single point, it is also where the length ceiling
    belongs: redaction bounds *what* upstream text can reach an agent, and
    ``_truncate`` bounds *how much* of it does, so a 20 KB HTML error page or
    a per-item validation dump cannot land in the context window whole. The
    cut happens after redaction so a fragment split across the boundary
    cannot survive by being half-printed.
    """
    return _truncate(" ".join(_redact_token(token) for token in text.split()))


def redact_detail(detail: object) -> str:
    """Render an upstream ``detail`` payload as safe-to-return text.

    Shares the exact flattening ``describe_error`` uses for a structured
    (list-shaped) validation detail, then redacts the result -- so a call
    site outside ``describe_error`` (``payload.py``'s per-id fan-out, which
    reports an upstream failure's reason directly rather than through
    ``describe_error``) gets the same redaction guarantee instead of a
    second, divergent implementation.
    """
    if isinstance(detail, list):
        return redact(_flatten_validation_detail(detail))
    return redact(_as_text(detail))


def describe_error(exc: Exception) -> str:
    """Return the message a tool should return to the client for this failure."""
    if isinstance(exc, ToolError):
        return f"Error: {exc}"
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return (
            "Error: the middleware did not respond (transient network failure). "
            "Reads were already retried; a write is sent once and may still "
            "have landed, so check before re-sending it. Call "
            "qdrant_check_health if it keeps failing."
        )
    if not isinstance(exc, MiddlewareError):
        return f"Error: unexpected failure of type {type(exc).__name__}."

    text = _as_text(exc.detail)
    if exc.status == 400:
        safe_detail = redact(_flatten_validation_detail(exc.detail))
        return f"Error: invalid request — {safe_detail}"
    if exc.status == 422:
        safe_detail = redact(_flatten_validation_detail(exc.detail))
        return f"Error: invalid request — {safe_detail}"
    if exc.status == 401 or exc.status == 403:
        return (
            "Error: the MCP server is not authorized against the middleware. "
            "This is a server configuration problem, not a request problem."
        )
    if exc.status == 404:
        safe_text = redact(text) if text else ""
        return (
            f"Error: not found — {safe_text or 'no such collection or point'}. "
            "Call qdrant_list_collections to see what exists, or "
            "qdrant_read_points to confirm a point id."
        )
    if exc.status == 409:
        lowered = text.lower()
        if lowered in {
            "no controllable reindex job for this action",
            "no paused, interrupted or failed reindex job to resume",
        }:
            return (
                f"Error: reindex state conflict — {redact(text)}. "
                "Call qdrant_get_reindex_status to inspect the current job "
                "before choosing another action."
            )
        # The reindex check runs first deliberately: a blocked mutation is
        # the actionable condition even when the text also mentions the
        # embedding contract.
        if any(marker in lowered for marker in REINDEX_MARKERS):
            return (
                f"Error: blocked by an active reindex job — {redact(text)}. "
                "A paused job blocks mutations too: wait for it to finish "
                "(qdrant_get_reindex_status shows progress), or ask the user "
                "before cancelling it with qdrant_control_reindex, which "
                "abandons its progress."
            )
        if any(marker in lowered for marker in CONTRACT_MARKERS):
            return (
                f"Error: embedding contract mismatch — {redact(text)}. Read "
                "the resource docs://concepts/embedding-contract, then use "
                "qdrant_verify_collection to see the expected vector layout."
            )
        return (
            f"Error: conflict — {redact(text)}. Call qdrant_get_collection "
            "to inspect the collection's current state before retrying."
        )
    if exc.status == 413:
        error_code = exc.detail.get("error") if isinstance(exc.detail, dict) else None
        if isinstance(exc.detail, dict) and exc.detail.get("code") == (
            "request_budget_exceeded"
        ):
            return (
                "Error: inference admission budget exceeded. Reduce the batch "
                "size, input length or requested vector output before retrying."
            )
        if error_code == "rerank_capacity_exceeded":
            safe_text = redact(text) if text else ""
            detail_suffix = f" — {safe_text}" if safe_text else ""
            return (
                f"Error: rerank capacity exceeded{detail_suffix}. Retrying "
                "will not help — the lever is passage length, not passage "
                "count. Shorten passages (QUERY_RERANK_MAX_CHARS) or lower "
                "QWEN_RERANK_MAX_LENGTH on the Smart Embedder."
            )
        lowered = text.lower()
        export_limit = next(
            (
                name
                for name in ("EXPORT_MAX_POINTS", "EXPORT_MAX_BYTES")
                if name.lower() in lowered
            ),
            None,
        )
        if export_limit:
            return (
                f"Error: collection too large to export — {redact(text)}. "
                "No file was published. This tool has no filter parameter; "
                f"ask the operator to review {export_limit} and memory capacity."
            )
        return (
            "Error: request body too large. Send smaller batches — split the "
            "points or chunks across several calls."
        )
    if exc.status == 429:
        return (
            "Error: rate limited by the middleware. Wait a few seconds before "
            "retrying, and reduce concurrent calls."
        )
    if exc.status == 503:
        if text.lower().startswith("export capacity exhausted"):
            return (
                "Error: export capacity is occupied. Retry after the active "
                "export finishes."
            )
        return (
            "Error: a dependency is not ready (Qdrant or the embedder). Call "
            "qdrant_check_health to see which one, then retry."
        )
    if (
        exc.status == 502
        and "request_budget_busy" in text
        and ("Smart Embedder server error 503" in text)
    ):
        return (
            "Error: inference capacity remained occupied after retries. "
            "Retry after active work drains, or reduce concurrent calls."
        )
    safe = redact(text) if text else ""
    suffix = f" — {safe}" if safe else ""
    return (
        f"Error: the middleware failed with status {exc.status}{suffix}. "
        "Call qdrant_check_health to check dependency status, then retry."
    )


def error_result(exc: Exception) -> CallToolResult:
    """Build the ``CallToolResult`` a tool's ``except`` block should return for ``exc``.

    Carries exactly ``describe_error(exc)``'s text, with ``is_error`` set to
    ``True``. This is the fix for the defect that motivates this module's
    helper split: a tool that instead ``return``ed that text as a plain
    ``str`` produced a successful-looking result (``is_error=False``), so a
    client reading the flag rather than the words could not tell a failure
    from a real answer. Raising instead is *not* the fix -- the SDK
    (``mcp/server/mcpserver/tools/base.py``) catches any exception a tool
    raises and re-wraps it as ``ToolError(f"Error executing tool {name}: {e}")``,
    reintroducing exactly the framework envelope this service's error
    messages are designed to avoid. Returning this ``CallToolResult`` directly
    is the one path that gets both the correct flag and the unwrapped text:
    ``Tool.run``'s ``convert_result`` passes an already-built
    ``CallToolResult`` through unchanged rather than re-serialising it (see
    ``FuncMetadata.convert_result`` in
    ``mcp/server/mcpserver/utilities/func_metadata.py``).
    """
    return CallToolResult(
        content=[TextContent(text=describe_error(exc))],
        structured_content={},
        is_error=True,
    )

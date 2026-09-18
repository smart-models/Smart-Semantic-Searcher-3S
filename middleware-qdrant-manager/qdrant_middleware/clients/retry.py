"""Generic async retry helper with bounded exponential backoff.

Shared by the Qdrant and Smart Embedder clients to retry transient failures on
idempotent operations only. Callers explicitly opt in to which exceptions count
as transient via the ``should_retry`` predicate, and to reading a server's own
``Retry-After`` hint via the ``retry_after`` extractor.
"""

import asyncio
import logging
import math
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Awaitable, Callable, Optional, TypeVar

from qdrant_middleware.core.config import get_settings

log = logging.getLogger(__name__)

T = TypeVar("T")

# HTTP statuses treated as transient by both the Qdrant and Smart Embedder
# clients: rate limiting plus upstream/server errors worth one more attempt.
RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})

# A server's Retry-After is honored only up to this many seconds. The cap is
# deliberately smaller than the MCP client's 30 s: this service answers under
# MIDDLEWARE_REQUEST_TIMEOUT (30 s by default), so parking a request for the
# 60 s Smart Embedder suggests on a 429 would guarantee a 504 instead of
# surfacing the 502 the caller can act on. Five seconds still covers the case
# that matters -- the Retry-After: 2 sent with a 503 rerank_vram_pressure --
# and leaves two full waits inside a 30 s budget.
RETRY_AFTER_CAP_SECONDS = 5.0


def parse_retry_after(raw: str | None, *, now: datetime | None = None) -> float | None:
    """Return the wait a ``Retry-After`` header value asks for, in seconds.

    Both RFC 9110 forms are honored: non-negative integer delta-seconds and an
    HTTP-date. Anything unparsable, negative or non-finite returns ``None`` so
    the caller falls back to computed backoff rather than trusting a malformed
    header. The result is capped at ``RETRY_AFTER_CAP_SECONDS``.

    Args:
        raw: The raw header value, or ``None`` when the response carried none.
        now: Reference instant for the HTTP-date form; defaults to the current
            UTC time and exists so tests can pin it.

    Returns:
        The capped wait in seconds, or ``None`` when there is nothing usable.
    """
    if raw is None:
        return None
    value = raw.strip()
    if value.isascii() and value.isdigit():
        seconds = float(value)
    else:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        reference = now or datetime.now(timezone.utc)
        seconds = max(0.0, (retry_at - reference).total_seconds())
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return min(seconds, RETRY_AFTER_CAP_SECONDS)


def backoff_delay(attempt: int, base_delay: float) -> float:
    """Return the exponential backoff for ``attempt``, jittered to [0.5x, 1x].

    ``attempt`` is zero-based, so the first wait is ``base_delay``. The uniform
    jitter desynchronizes concurrent callers that failed together, which would
    otherwise retry in lockstep and rebuild the burst that caused the failure;
    the lower bound keeps the wait near the pre-jitter policy rather than
    collapsing to zero. ``random`` is used at module level so a test can fix
    the sequence with ``random.seed``.

    Args:
        attempt: Zero-based index of the attempt that just failed.
        base_delay: The configured initial delay, in seconds.

    Returns:
        The number of seconds to wait before the next attempt.
    """
    return base_delay * (2**attempt) * random.uniform(0.5, 1.0)


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int | None = None,
    base_delay: float | None = None,
    should_retry: Optional[Callable[[Exception], bool]] = None,
    retry_after: Optional[Callable[[Exception], float | None]] = None,
    description: str = "operation",
) -> T:
    """Run an idempotent async operation with bounded exponential backoff.

    Retries only when ``should_retry(exc)`` is True (default: never retry — callers
    must opt in to the exceptions they consider transient). The final attempt's
    exception propagates unchanged.

    ``retry_after`` lets a caller surface the wait the server itself asked for.
    When it returns a number, that number replaces the computed backoff: a
    server that says "not before N seconds" knows something this side does not,
    and guessing 0.1 s against it wastes the attempt and adds load to whatever
    is already struggling. Returning ``None`` (the default, and the answer for
    a response that carried no hint) keeps the jittered exponential backoff.
    """
    if attempts is None or base_delay is None:
        settings = get_settings()
        if attempts is None:
            attempts = settings.retry_max_attempts
        if base_delay is None:
            base_delay = settings.retry_base_delay

    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            return await operation()
        except Exception as exc:  # noqa: PERF203 - failure belongs to this attempt
            last_exc = exc
            retryable = should_retry(exc) if should_retry is not None else False
            if not retryable or attempt == attempts - 1:
                raise
            suggested = retry_after(exc) if retry_after is not None else None
            delay = (
                suggested
                if suggested is not None
                else backoff_delay(attempt, base_delay)
            )
            log.warning(
                "Retrying %s after transient error in %.2fs (attempt %d/%d): %s",
                description,
                delay,
                attempt + 1,
                attempts,
                exc,
            )
            await asyncio.sleep(delay)
    assert last_exc is not None  # unreachable: loop either returns or raises
    raise last_exc

"""Shared admission estimates and reservations for native inference work.

These bound accepted token units and estimated JSON output, not physical memory.
UTF-8 length plus special-token allowance avoids tokenizer work before admission.
"""

from dataclasses import dataclass
from threading import Lock
from unicodedata import normalize


@dataclass(frozen=True)
class RequestCost:
    """Independent token-work and estimated output-byte reservations."""

    tokens: int
    output_bytes: int


class BudgetExceeded(Exception):
    """A request exceeds the configured limit or current available capacity."""

    def __init__(self, cost: RequestCost, limits: RequestCost, *, oversized: bool):
        """Keep refusal facts independent of the HTTP surface."""
        super().__init__("Inference admission budget exceeded")
        self.cost = cost
        self.limits = limits
        self.oversized = oversized


class RequestBudget:
    """Atomically reserve both resources across embedding and reranking."""

    def __init__(self, tokens: int, output_bytes: int, on_change=None):
        """Create a positive budget with an optional metrics callback."""
        if tokens < 1 or output_bytes < 1:
            raise ValueError("Request budget limits must be positive")
        self.limits = RequestCost(tokens, output_bytes)
        self._used = RequestCost(0, 0)
        self._lock = Lock()
        self._on_change = on_change

    @property
    def used(self) -> RequestCost:
        """Return a consistent snapshot for stats and controlled tests."""
        with self._lock:
            return self._used

    def reserve(self, cost: RequestCost):
        """Refuse immediately, or return one caller-owned reservation."""
        if cost.tokens < 0 or cost.output_bytes < 0:
            raise ValueError("Request estimates must be non-negative")
        with self._lock:
            oversized = (
                cost.tokens > self.limits.tokens
                or cost.output_bytes > self.limits.output_bytes
            )
            if (
                oversized
                or self._used.tokens + cost.tokens > self.limits.tokens
                or self._used.output_bytes + cost.output_bytes
                > self.limits.output_bytes
            ):
                raise BudgetExceeded(cost, self.limits, oversized=oversized)
            self._change(cost.tokens, cost.output_bytes)
            return Reservation(self, cost)

    def _change(self, tokens: int, output_bytes: int) -> None:
        self._used = RequestCost(
            self._used.tokens + tokens, self._used.output_bytes + output_bytes
        )
        if self._on_change is not None:
            self._on_change(self._used)


class Reservation:
    """Release once, after both the caller and its physical workers finish."""

    def __init__(self, budget: RequestBudget, cost: RequestCost):
        """Create a lease; only RequestBudget.reserve should allocate one."""
        self._budget = budget
        self._cost = cost
        self._caller_done = False
        self._workers = 0
        self._released = False

    def __enter__(self):
        """Keep ownership through an endpoint's response construction."""
        return self

    def __exit__(self, *_args):
        """Close caller ownership on success, failure or cancellation."""
        self.release()

    def track(self, future) -> None:
        """Attach to the concurrent future, never a cancelled asyncio wrapper."""
        with self._budget._lock:
            if self._caller_done:
                raise RuntimeError("Cannot submit work after releasing admission")
            self._workers += 1
        future.add_done_callback(self._worker_done)

    def release(self) -> None:
        """Close caller ownership idempotently; native work retains capacity."""
        with self._budget._lock:
            self._caller_done = True
            self._release_if_done()

    def _worker_done(self, _future) -> None:
        with self._budget._lock:
            self._workers -= 1
            self._release_if_done()

    def _release_if_done(self) -> None:
        if self._caller_done and not self._workers and not self._released:
            self._released = True
            self._budget._change(-self._cost.tokens, -self._cost.output_bytes)


def estimate_embeddings(
    sentences,
    *,
    advanced: bool,
    dense: bool,
    sparse: bool,
    colbert: bool,
    include_text: bool,
    dense_limit: int,
    sparse_limit: int,
    english_vocab: int = 30522,
    source_view: bool = True,
    sparse_top_k: int = 0,
) -> RequestCost:
    """Estimate fixed-profile output shapes without invoking a tokenizer.

    BASIC shares one forward pass and emits at most one sparse key per token.
    MILCO can activate its entire English vocabulary plus input source keys.
    Both fixed dense and BASIC ColBERT vectors have 1024 coordinates. Allow
    32 bytes per dense coordinate, 48 per sparse entry, and 6 per echoed UTF-8
    byte for escaping, plus per-item and envelope metadata allowances.
    """
    tokens, output = 0, 4096
    for sentence in sentences:
        byte_length = len(sentence.encode("utf-8"))
        token_bytes = max(byte_length, len(normalize("NFKC", sentence).encode("utf-8")))
        dense_tokens = min(token_bytes + 8, dense_limit)
        sparse_tokens = min(token_bytes + 8, sparse_limit)
        if advanced:
            tokens += (dense_tokens if dense else 0) + (sparse_tokens if sparse else 0)
        elif dense or sparse or colbert:
            tokens += dense_tokens
        output += 1024 + (6 * byte_length if include_text else 0)
        if dense:
            output += 1024 * 32
        if sparse:
            entries = (
                english_vocab + (sparse_tokens if source_view else 0)
                if advanced
                else sparse_tokens
            )
            if advanced and sparse_top_k:
                entries = min(entries, sparse_top_k)
            output += entries * 48
        if colbert:
            output += dense_tokens * (1024 * 32 + 4)
    return RequestCost(tokens, output)


def estimate_rerank(query: str, passages, *, pair_limit: int) -> RequestCost:
    """Reserve each repeated query/passage pair and its echoed result text."""
    query_bytes = max(
        len(query.encode("utf-8")), len(normalize("NFKC", query).encode("utf-8"))
    )
    tokens, output = 0, 4096
    for passage in passages:
        passage_bytes = len(passage.encode("utf-8"))
        token_bytes = max(
            passage_bytes, len(normalize("NFKC", passage).encode("utf-8"))
        )
        # Include ample space for the fixed Qwen instruction/prefix/suffix;
        # both backends still enforce their own final pair truncation limit.
        tokens += min(query_bytes + token_bytes + 1024, pair_limit)
        output += passage_bytes * 6 + 1024
    return RequestCost(tokens, output)

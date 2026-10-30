"""Shared inference admission and native-worker lease regressions."""

import re
import sys
import unittest
from concurrent.futures import Future
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from tests.book_cases import book_case  # noqa: E402
from request_budget import (  # noqa: E402
    BudgetExceeded,
    RequestBudget,
    RequestCost,
    estimate_embeddings,
    estimate_rerank,
)


class RequestBudgetTests(unittest.TestCase):
    """Pin TB-PER-001 for both resource dimensions and physical completion."""

    @book_case("TB-PER-001")
    def test_exact_capacity_refusal_release_and_retry(self):
        for dimension in ("tokens", "output_bytes"):
            with self.subTest(dimension=dimension):
                budget = RequestBudget(4096, 4096)

                def cost(n):
                    return (
                        RequestCost(n, 0)
                        if dimension == "tokens"
                        else RequestCost(0, n)
                    )

                first, second = budget.reserve(cost(2048)), budget.reserve(cost(2048))
                with self.assertRaises(BudgetExceeded) as caught:
                    budget.reserve(cost(1))
                self.assertFalse(caught.exception.oversized)
                first.release()
                first.release()
                third = budget.reserve(cost(1))
                second.release()
                third.release()
                self.assertEqual(budget.used, RequestCost(0, 0))
                with self.assertRaises(BudgetExceeded) as caught:
                    budget.reserve(cost(4097))
                self.assertTrue(caught.exception.oversized)

    @book_case("TB-PER-001")
    def test_abandoned_caller_retains_capacity_until_physical_completion(self):
        for completion in ("result", "error", "cancel"):
            with self.subTest(completion=completion):
                budget = RequestBudget(4096, 4096)
                lease = budget.reserve(RequestCost(2048, 2048))
                worker = Future()
                lease.track(worker)
                lease.release()
                lease.release()
                self.assertEqual(budget.used, RequestCost(2048, 2048))
                if completion == "result":
                    worker.set_result(None)
                elif completion == "error":
                    worker.set_exception(RuntimeError("native failure"))
                else:
                    worker.cancel()
                self.assertEqual(budget.used, RequestCost(0, 0))

    @book_case("TB-PER-001")
    def test_basic_colbert_and_advanced_sparse_have_distinct_estimates(self):
        options = dict(
            advanced=False,
            dense=True,
            sparse=True,
            colbert=False,
            include_text=False,
            dense_limit=8192,
            sparse_limit=8192,
        )
        basic = estimate_embeddings(["hello"], **options)
        colbert = estimate_embeddings(["hello"], **{**options, "colbert": True})
        advanced = estimate_embeddings(["hello"], **{**options, "advanced": True})
        self.assertEqual(colbert.tokens, basic.tokens)
        self.assertGreater(colbert.output_bytes, basic.output_bytes)
        self.assertEqual(advanced.tokens, basic.tokens * 2)
        self.assertGreater(advanced.output_bytes, basic.output_bytes)
        sparse_only = estimate_embeddings(
            ["hello"],
            **{**options, "advanced": True, "dense": False, "sparse_top_k": 16},
        )
        self.assertLess(sparse_only.output_bytes, advanced.output_bytes)

    def test_shipped_default_admits_a_full_rerank_batch(self):
        """The default must fit the largest rerank a client can send.

        Admission counts conservative UTF-8 bytes, not tokens, so a default
        below 32 pairs at QWEN_RERANK_MAX_LENGTH refuses an ordinary 20-chunk
        search with 413 request_budget_exceeded. The Compose fallback and the
        ``.env.example`` template an operator copies must both hold.
        """
        sources = {
            "compose.yml": (
                r"MAX_INFLIGHT_TOKENS: \$\{SMART_EMBEDDER_MAX_INFLIGHT_TOKENS:-(\d+)\}"
            ),
            ".env.example": r"(?m)^SMART_EMBEDDER_MAX_INFLIGHT_TOKENS=(\d+)$",
        }
        worst = estimate_rerank("q", ["x" * 32768] * 32, pair_limit=32768)
        for name, pattern in sources.items():
            text = (SERVER_DIR.parent / name).read_text(encoding="utf-8")
            with self.subTest(source=name):
                default = int(re.search(pattern, text).group(1))
                self.assertGreaterEqual(default, worst.tokens)

    def test_rerank_repeats_query_budget_and_caps_each_pair(self):
        single = estimate_rerank("query", ["passage"], pair_limit=2048)
        double = estimate_rerank("query", ["passage", "passage"], pair_limit=2048)
        self.assertEqual(double.tokens, single.tokens * 2)
        self.assertEqual(
            estimate_rerank("x" * 10000, ["y"] * 2, pair_limit=2048).tokens, 4096
        )

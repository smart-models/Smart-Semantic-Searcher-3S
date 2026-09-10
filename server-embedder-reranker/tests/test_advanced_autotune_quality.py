"""Tests for the advanced autotuner's retrieval-quality gate.

``advanced_autotune_quality`` scores a candidate against a fixed set of
documents and queries; a candidate that loses quality must not be accepted
however fast it is.
"""

import unittest

import advanced_autotune_quality as quality


class AdvancedQualityTests(unittest.TestCase):
    """The quality gate accepts equivalent output and rejects regressions."""

    def setUp(self):
        self.documents = (
            quality.QualityDocument("relevant", "relevant"),
            quality.QualityDocument("other", "other"),
        )
        self.queries = (quality.QualityQuery("query", "query", ("relevant",)),)

    def test_versioned_corpus_covers_cross_lingual_and_rare_entities(self):
        corpus = " ".join(item.text for item in (*quality.DOCUMENTS, *quality.QUERIES))

        self.assertEqual(quality.QUALITY_SUITE_VERSION, "advanced-retrieval-v1")
        self.assertIn("Łódź", corpus)
        self.assertIn("東京", corpus)
        self.assertIn("BRCA1", corpus)

    def test_identical_dense_reference_passes(self):
        vectors = {
            "query": [1.0, 0.0, 0.0],
            "relevant": [1.0, 0.0, 0.0],
            "other": [0.0, 1.0, 0.0],
        }

        result = quality.evaluate_dense(
            vectors,
            vectors,
            queries=self.queries,
            documents=self.documents,
            expected_dimension=3,
            k=2,
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.metrics["min_cosine"], 1.0)
        self.assertEqual(result.metrics["candidate_mrr"], 1.0)

    def test_dense_semantic_regression_is_rejected(self):
        reference = {
            "query": [1.0, 0.0, 0.0],
            "relevant": [1.0, 0.0, 0.0],
            "other": [0.0, 1.0, 0.0],
        }
        candidate = {**reference, "query": [-1.0, 0.0, 0.0]}

        result = quality.evaluate_dense(
            reference,
            candidate,
            queries=self.queries,
            documents=self.documents,
            expected_dimension=3,
            k=2,
        )

        self.assertFalse(result.passed)
        self.assertIn("dense cosine below threshold", result.failures)

    def test_identical_lexecho_sparse_reference_passes(self):
        vectors = {
            "query": {"5": 1.0, "35000": 0.5},
            "relevant": {"5": 0.9, "35000": 0.5},
            "other": {"9": 1.0},
        }

        result = quality.evaluate_sparse(
            vectors,
            vectors,
            reference_source_view=True,
            candidate_source_view=True,
            reference_top_k=0,
            candidate_top_k=0,
            queries=self.queries,
            documents=self.documents,
            k=2,
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.metrics["min_weighted_overlap"], 1.0)

    def test_sparse_gate_rejects_lexecho_or_top_k_changes(self):
        vectors = {
            "query": {"5": 1.0},
            "relevant": {"5": 0.9},
            "other": {"9": 1.0},
        }

        result = quality.evaluate_sparse(
            vectors,
            vectors,
            reference_source_view=True,
            candidate_source_view=False,
            reference_top_k=0,
            candidate_top_k=128,
            queries=self.queries,
            documents=self.documents,
            k=2,
        )

        self.assertFalse(result.passed)
        self.assertIn("MILCO LexEcho source_view changed", result.failures)
        self.assertIn("MILCO top_k changed", result.failures)

    def test_reranker_order_regression_is_rejected(self):
        reference = {"query": {"relevant": 0.9, "other": 0.1}}
        candidate = {"query": {"relevant": 0.1, "other": 0.9}}

        result = quality.evaluate_reranker(
            reference,
            candidate,
            queries=self.queries,
            k=1,
        )

        self.assertFalse(result.passed)
        self.assertLess(
            result.metrics["candidate_mrr"], result.metrics["reference_mrr"]
        )

    def test_identical_reranker_reference_passes(self):
        scores = {"query": {"relevant": 0.9, "other": 0.1}}

        result = quality.evaluate_reranker(
            scores,
            scores,
            queries=self.queries,
            k=2,
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.metrics["score_correlation"], 1.0)


if __name__ == "__main__":
    unittest.main()

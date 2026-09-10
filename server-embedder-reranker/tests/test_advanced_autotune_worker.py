"""Tests for the advanced autotuner's isolated benchmark worker.

The worker runs each candidate in its own process; these checks pin the
quality invariants it must report back to the runner.
"""

import io
import json
import os
import sys
import threading
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import advanced_autotune_worker as worker  # noqa: E402


class WorkerQualityInvariantTests(unittest.TestCase):
    """Invariants the isolated worker must report back to the runner."""

    def test_offline_runtime_import_has_private_auth_without_changing_environment(self):
        source = (
            "import os\n"
            "token, label = os.environ['API_TOKENS'].split(':')\n"
            "assert len(token) >= 32 and label == 'autotune'\n"
            "assert os.environ.get('SMART_EMBEDDER_ALLOW_UNAUTHENTICATED_LOCAL') "
            "!= 'true'\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "smart-embedder.py").write_text(source, encoding="utf-8")
            with patch.dict(os.environ, {"API_TOKENS": ""}, clear=True):
                first = worker._load_server(root)
                second = worker._load_server(root)
                self.assertNotEqual(first.token, second.token)
                self.assertEqual(dict(os.environ), {"API_TOKENS": ""})

    def test_failed_runtime_import_restores_missing_auth_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "smart-embedder.py").write_text(
                "raise RuntimeError('model load failed')\n", encoding="utf-8"
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "model load failed"):
                    worker._load_server(root)
                self.assertEqual(dict(os.environ), {})

    def test_dense_requires_1024_finite_dimensions(self):
        valid = np.ones((2, 1024), dtype=np.float32)
        invalid = np.ones((2, 512), dtype=np.float32)
        invalid[0, 0] = np.nan

        self.assertEqual(worker._validate_dense(valid, 2), [])
        self.assertGreaterEqual(len(worker._validate_dense(invalid, 2)), 2)

    def test_sparse_requires_positive_qdrant_values_and_lexecho_source_index(self):
        valid = [{"12": 0.5, "30523": 0.25}]
        no_source = [{"12": 0.5}]
        invalid = [{"-1": -0.5}]

        self.assertEqual(
            worker._validate_sparse(
                valid,
                expected_rows=1,
                source_view=True,
                english_vocab_size=30522,
            ),
            [],
        )
        self.assertIn(
            "LexEcho source_view produced no source-vocabulary index",
            worker._validate_sparse(
                no_source,
                expected_rows=1,
                source_view=True,
                english_vocab_size=30522,
            ),
        )
        self.assertTrue(
            worker._validate_sparse(
                invalid,
                expected_rows=1,
                source_view=False,
                english_vocab_size=30522,
            )
        )

    def test_reranker_scores_are_probabilities(self):
        self.assertEqual(worker._validate_scores([0.0, 0.5, 1.0], 3), [])
        self.assertTrue(worker._validate_scores([float("nan")], 1))
        self.assertTrue(worker._validate_scores([1.1], 1))

    def test_candidate_policy_is_applied_through_serving_environment(self):
        candidate = {
            "selected": {
                "advanced": {
                    "dense": {
                        "runtime": "torch",
                        "precision": "bf16",
                        "attention": "eager",
                        "torch_compile": True,
                        "torch_compile_mode": "reduce-overhead",
                        "pinned_memory": True,
                        "prefetch_batches": True,
                        "batch_policy": [{"max_tokens": 128, "batch_size": 8}],
                    },
                    "sparse": {
                        "compute_precision": "fp16",
                        "prefetch_batches": True,
                    },
                    "reranker": {"precision": "bf16"},
                    "schedule": {
                        "mode": "overlap",
                        "inference_lanes": 2,
                        "queue_batch": 4,
                        "flush_timeout_ms": 25,
                        "worker_spin": "passive",
                        "affinity": "compact",
                        "allow_oversubscription": True,
                        "intra_op_threads": 2,
                        "inter_op_threads": 1,
                    },
                }
            }
        }
        with patch.dict(os.environ, {}, clear=True):
            worker._apply_candidate_environment(candidate)

            self.assertEqual(os.environ["QWEN_EMBED_PRECISION"], "bf16")
            self.assertEqual(os.environ["QWEN_EMBED_RUNTIME"], "torch")
            self.assertEqual(os.environ["QWEN_EMBED_ATTENTION"], "eager")
            self.assertEqual(os.environ["QWEN_EMBED_TORCH_COMPILE"], "true")
            self.assertEqual(os.environ["QWEN_EMBED_PINNED_MEMORY"], "true")
            self.assertEqual(os.environ["QWEN_EMBED_PREFETCH_BATCHES"], "true")
            self.assertEqual(
                os.environ["QWEN_EMBED_BATCH_POLICY"],
                '[{"max_tokens":128,"batch_size":8}]',
            )
            self.assertEqual(os.environ["MILCO_COMPUTE_PRECISION"], "fp16")
            self.assertEqual(os.environ["MILCO_PREFETCH_BATCHES"], "true")
            self.assertEqual(os.environ["QWEN_RERANK_PRECISION"], "bf16")
            self.assertEqual(os.environ["ADVANCED_SCHEDULE_MODE"], "overlap")
            self.assertEqual(os.environ["ADVANCED_INFERENCE_LANES"], "2")
            self.assertEqual(os.environ["ADVANCED_QUEUE_BATCH"], "4")
            self.assertEqual(os.environ["ADVANCED_FLUSH_TIMEOUT_MS"], "25")
            self.assertEqual(os.environ["ADVANCED_CPU_SPIN_POLICY"], "passive")
            self.assertEqual(os.environ["ADVANCED_CPU_AFFINITY"], "compact")
            self.assertEqual(os.environ["ADVANCED_CPU_ALLOW_OVERSUBSCRIPTION"], "true")
            self.assertEqual(os.environ["OMP_WAIT_POLICY"], "PASSIVE")
            self.assertEqual(os.environ["KMP_BLOCKTIME"], "0")
            self.assertEqual(os.environ["EMBEDDER_NUM_THREADS"], "2")
            self.assertEqual(os.environ["EMBEDDER_INTEROP_THREADS"], "1")

    def test_latency_summary_reports_interpolated_p50_p95_p99(self):
        summary = worker._latency_summary([1.0, 3.0])

        self.assertEqual(summary["p50"], 2.0)
        self.assertAlmostEqual(summary["p95"], 2.9)
        self.assertAlmostEqual(summary["p99"], 2.98)
        self.assertAlmostEqual(summary["coefficient_of_variation"], 0.5)

    def test_mixed_samples_include_preparation_in_the_concurrent_worker(self):
        """Each sample prepares its own passages while embedding is in flight."""
        barrier = threading.Barrier(2)
        prepared = []
        caller_thread = threading.get_ident()

        def embed(texts, **kwargs):
            barrier.wait(timeout=2)
            return {"texts": texts}

        def prepare(query, passages):
            self.assertNotEqual(threading.get_ident(), caller_thread)
            barrier.wait(timeout=2)
            prepared.append(list(passages))
            return [[query, passage] for passage in passages], []

        def score(pairs, normalize):
            self.assertTrue(normalize)
            self.assertEqual([pair[1] for pair in pairs], prepared[-1])
            return [0.5] * len(pairs)

        with ThreadPoolExecutor(max_workers=2) as executor:
            server = SimpleNamespace(
                GPU_EXECUTOR=executor,
                embedding_service=SimpleNamespace(embed=embed),
                reranker=SimpleNamespace(
                    prepare_pairs_for_scoring=prepare, score=score
                ),
            )
            batches = [["short", "long passage " * 20], ["next sample"]]
            for passages in batches:
                embedding, scores = worker._run_mixed(server, passages)
                self.assertEqual(embedding, {"texts": passages})
                self.assertEqual(scores, [0.5] * len(passages))
        self.assertEqual(prepared, batches)

    def test_mixed_sample_propagates_preparation_failure(self):
        def fail_preparation(*args):
            raise ValueError("synthetic preparation failure")

        with ThreadPoolExecutor(max_workers=2) as executor:
            server = SimpleNamespace(
                GPU_EXECUTOR=executor,
                embedding_service=SimpleNamespace(embed=lambda *args, **kwargs: {}),
                reranker=SimpleNamespace(prepare_pairs_for_scoring=fail_preparation),
            )
            with self.assertRaisesRegex(ValueError, "synthetic preparation failure"):
                worker._run_mixed(server, ["sample"])

    def test_error_payload_preserves_candidate_name(self):
        candidate = json.dumps({"name": "dense-eager"})
        with (
            patch.object(
                worker, "run_candidate", side_effect=RuntimeError("probe failed")
            ),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            result = worker.main(["--candidate-json", candidate])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(result, 2)
        self.assertEqual(payload["candidate"], "dense-eager")
        self.assertEqual(payload["error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()

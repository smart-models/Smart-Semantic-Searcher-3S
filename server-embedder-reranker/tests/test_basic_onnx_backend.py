"""Tests for the BGE-M3 ONNX backend used by the basic profile.

Fake tokenizer and ``onnxruntime`` session objects stand in for the real
ones, so dense/sparse/ColBERT output handling is checked without an ONNX
export.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from basic_onnx_backend import BgeM3OnnxBackend  # noqa: E402


class FakeTokenizer:
    """Tokenizer stand-in with fixed special ids and predictable lengths."""

    cls_token_id = 101
    eos_token_id = 102
    pad_token_id = 0
    unk_token_id = 100

    def __call__(self, sentences, **_kwargs):
        rows = []
        for sentence in sentences:
            token_ids = [101, 5, 102]
            if "long" in sentence:
                token_ids = [101, 6, 7, 102]
            rows.append(token_ids)
        width = max(map(len, rows))
        input_ids = np.asarray(
            [row + [0] * (width - len(row)) for row in rows], dtype=np.int64
        )
        return {
            "input_ids": input_ids,
            "attention_mask": (input_ids != 0).astype(np.int64),
        }

    def token_lengths(self, sentences, _max_length):
        return [4 if "long" in sentence else 3 for sentence in sentences]


class FakeSessionOptions:
    """Stand-in for onnxruntime SessionOptions, recording what is set."""

    def __init__(self):
        self.entries = {}

    def add_session_config_entry(self, key, value):
        self.entries[key] = value


class FakeSession:
    """Stand-in ONNX session returning fixed dense, sparse and ColBERT output."""

    def __init__(self, _path, *, sess_options, providers):
        self.options = sess_options
        self.providers = providers

    def get_providers(self):
        return list(self.providers)

    def run(self, _outputs, inputs):
        input_ids = inputs["input_ids"]
        token_embeddings = np.stack(
            (
                input_ids / 100.0,
                np.ones_like(input_ids),
                (input_ids != 0).astype(np.float32),
            ),
            axis=-1,
        ).astype(np.float32)
        sentence_embedding = token_embeddings[:, 0]
        return token_embeddings, sentence_embedding


class FakeOrt:
    """Stand-in onnxruntime module exposing only what the backend uses."""

    SessionOptions = FakeSessionOptions
    InferenceSession = FakeSession

    class ExecutionMode:
        """The onnxruntime execution-mode constants the backend reads."""

        ORT_SEQUENTIAL = "sequential"
        ORT_PARALLEL = "parallel"

    class GraphOptimizationLevel:
        """The onnxruntime optimization-level constants the backend reads."""

        ORT_ENABLE_ALL = "all"


class OnnxBackendTests(unittest.TestCase):
    """BGE-M3 output reconstruction, policy application and session rebuilds."""

    def _backend(self, root: Path):
        (root / "onnx").mkdir()
        (root / "onnx" / "model.onnx").touch()
        torch.save(
            {"weight": torch.ones((1, 3)), "bias": torch.zeros(1)},
            root / "sparse_linear.pt",
        )
        torch.save(
            {"weight": torch.eye(3), "bias": torch.zeros(3)},
            root / "colbert_linear.pt",
        )
        return BgeM3OnnxBackend(
            snapshot_path=str(root),
            tokenizer=FakeTokenizer(),
            torch_module=torch,
            threads=4,
            ort_module=FakeOrt,
        )

    def test_reconstructs_all_three_bge_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            backend = self._backend(Path(temporary))

            result = backend.encode(
                ["short", "long"],
                batch_size=2,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True,
            )

            self.assertEqual(result["dense_vecs"].shape, (2, 3))
            self.assertEqual(set(result["lexical_weights"][0]), {"5"})
            self.assertEqual(set(result["lexical_weights"][1]), {"6", "7"})
            self.assertEqual(result["colbert_vecs"][0].shape, (2, 3))
            self.assertEqual(result["colbert_vecs"][1].shape, (3, 3))
            self.assertEqual(
                backend.session.options.entries["session.dynamic_block_base"],
                "4",
            )

    def test_embed_applies_policy_and_dense_normalisation(self):
        with tempfile.TemporaryDirectory() as temporary:
            backend = self._backend(Path(temporary))
            backend.batch_size_for_tokens = lambda tokens: 2 if tokens <= 3 else 1

            result = backend.embed(
                ["short", "short", "long"],
                return_sparse=False,
                return_colbert=False,
                normalize_dense=True,
            )

            norms = np.linalg.norm(result["dense_vecs"], axis=1)
            np.testing.assert_allclose(norms, np.ones(3), atol=1e-6)

    def test_rebuilds_session_for_thread_execution_and_spin_topology(self):
        with tempfile.TemporaryDirectory() as temporary:
            backend = self._backend(Path(temporary))
            original = backend.session

            rebuilt = backend.configure_session(
                threads=2,
                execution_mode="parallel",
                spin_policy="bounded",
                inter_op_threads=2,
            )

            self.assertTrue(rebuilt)
            self.assertIsNot(backend.session, original)
            self.assertEqual(backend.session.options.intra_op_num_threads, 2)
            self.assertEqual(backend.session.options.inter_op_num_threads, 2)
            self.assertEqual(backend.session.options.execution_mode, "parallel")
            self.assertEqual(
                backend.session.options.entries["session.intra_op.spin_duration_us"],
                "1000",
            )
            self.assertEqual(
                backend.session.options.entries["session.intra_op.spin_backoff_max"],
                "8",
            )
            self.assertFalse(
                backend.configure_session(
                    threads=2,
                    execution_mode="parallel",
                    spin_policy="bounded",
                    inter_op_threads=2,
                )
            )


if __name__ == "__main__":
    unittest.main()

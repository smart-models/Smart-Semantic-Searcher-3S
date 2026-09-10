"""Tests for the Qwen3 ONNX backend used by the advanced profile.

Fake ``onnxruntime`` sessions stand in for the real ones, so artifact
discovery and session configuration are checked without an ONNX export.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import advanced_qwen_onnx_backend as backend  # noqa: E402


class FakeSessionOptions:
    """Stand-in for onnxruntime SessionOptions, recording what is set."""

    def __init__(self):
        self.intra_op_num_threads = 0
        self.inter_op_num_threads = 0
        self.execution_mode = None
        self.graph_optimization_level = None
        self.entries = {}

    def add_session_config_entry(self, name, value):
        self.entries[name] = value


class FakeInferenceSession:
    """Stand-in ONNX session recording its construction and returning fixed output."""

    instances = []

    def __init__(self, path, *, sess_options, providers):
        self.path = path
        self.options = sess_options
        self.providers = providers
        self.last_inputs = None
        self.__class__.instances.append(self)

    def get_providers(self):
        return list(self.providers)

    def run(self, outputs, inputs):
        self.last_inputs = inputs
        rows = inputs["input_ids"].shape[0]
        return [np.ones((rows, 1024), dtype=np.float32)]


class FakeOrt:
    """Stand-in onnxruntime module exposing only what the backend uses."""

    class ExecutionMode:
        """The onnxruntime execution-mode constants the backend reads."""

        ORT_SEQUENTIAL = "sequential"

    class GraphOptimizationLevel:
        """The onnxruntime optimization-level constants the backend reads."""

        ORT_ENABLE_ALL = "all"

    SessionOptions = FakeSessionOptions
    InferenceSession = FakeInferenceSession


class QwenOnnxArtifactTests(unittest.TestCase):
    """Export, quantization, caching and session setup for the Qwen backend."""

    def setUp(self):
        FakeInferenceSession.instances.clear()

    def _snapshot(self, root: Path) -> Path:
        snapshot = root / "hub" / "models--Qwen--test" / "snapshots" / "rev-1"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(
            json.dumps({"hidden_size": 1024}), encoding="utf-8"
        )
        return snapshot

    def test_fp32_export_is_atomic_versioned_and_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = self._snapshot(root)
            calls = []

            def export(snapshot_path, output_path, tokenizer, torch_module):
                calls.append((snapshot_path, tokenizer, torch_module))
                output_path.write_bytes(b"valid-onnx")

            kwargs = {
                "snapshot_path": snapshot,
                "runtime": "onnx-fp32",
                "tokenizer": object(),
                "torch_module": object(),
                "cache_root": root / "exports",
                "ort_module": FakeOrt,
                "fp32_exporter": export,
                "validate": lambda _path, _ort: None,
            }
            first = backend.ensure_qwen_dense_onnx_artifact(**kwargs)
            with patch.object(backend, "ProfileLock", side_effect=OSError("read-only")):
                second = backend.ensure_qwen_dense_onnx_artifact(**kwargs)

            self.assertEqual(first, second)
            self.assertEqual(len(calls), 1)
            self.assertEqual(first.read_bytes(), b"valid-onnx")
            manifest = json.loads(
                (first.parent / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["identity"]["model_revision"], "rev-1")
            self.assertEqual(manifest["files"]["model.onnx"]["size_bytes"], 10)
            self.assertFalse(
                any(".tmp-" in path.name for path in first.parent.parent.iterdir())
            )

            first.write_bytes(b"corrupt-xx")
            regenerated = backend.ensure_qwen_dense_onnx_artifact(**kwargs)
            self.assertEqual(regenerated.read_bytes(), b"valid-onnx")
            self.assertEqual(len(calls), 2)

    def test_dynamic_int8_builds_from_valid_fp32_and_uses_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = self._snapshot(root)
            quantize_calls = []

            def export(_snapshot, output_path, _tokenizer, _torch):
                output_path.write_bytes(b"fp32")

            def quantize(fp32_path, output_path, *, reduce_range):
                quantize_calls.append((fp32_path, reduce_range))
                output_path.write_bytes(b"int8")

            kwargs = {
                "snapshot_path": snapshot,
                "runtime": "onnx-int8",
                "tokenizer": object(),
                "torch_module": object(),
                "cache_root": root / "exports",
                "reduce_range": True,
                "ort_module": FakeOrt,
                "fp32_exporter": export,
                "int8_quantizer": quantize,
                "validate": lambda _path, _ort: None,
            }
            artifact = backend.ensure_qwen_dense_onnx_artifact(**kwargs)
            with patch.object(backend, "ProfileLock", side_effect=OSError("read-only")):
                reused = backend.ensure_qwen_dense_onnx_artifact(**kwargs)

            self.assertEqual(artifact, reused)
            self.assertEqual(artifact.read_bytes(), b"int8")
            self.assertEqual(len(quantize_calls), 1)
            self.assertTrue(quantize_calls[0][1])
            self.assertEqual(quantize_calls[0][0].read_bytes(), b"fp32")

    def test_session_applies_threads_passive_spin_and_validates_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.onnx"
            model.write_bytes(b"model")
            preparation = {}

            def prepare(**kwargs):
                preparation.update(kwargs)
                return model

            session = backend.QwenDenseOnnxSession(
                snapshot_path=str(Path(directory) / "snapshots" / "rev"),
                runtime="onnx-int8",
                tokenizer=object(),
                torch_module=object(),
                threads=6,
                inter_op_threads=2,
                spin_policy="passive",
                instruction_sets=("avx2",),
                ort_module=FakeOrt,
                artifact_preparer=prepare,
            )
            result = session.run(
                {
                    "input_ids": [[1, 2], [3, 4]],
                    "attention_mask": [[1, 1], [1, 1]],
                }
            )

            self.assertTrue(preparation["reduce_range"])
            self.assertEqual(result.shape, (2, 1024))
            actual = FakeInferenceSession.instances[-1]
            self.assertEqual(actual.options.intra_op_num_threads, 6)
            self.assertEqual(actual.options.inter_op_num_threads, 2)
            self.assertEqual(
                actual.options.entries["session.intra_op.allow_spinning"], "0"
            )
            self.assertEqual(actual.last_inputs["input_ids"].dtype, np.int64)

    def test_vnni_does_not_enable_avx2_reduce_range_workaround(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.onnx"
            model.write_bytes(b"model")
            observed = {}

            def prepare(**kwargs):
                observed.update(kwargs)
                return model

            backend.QwenDenseOnnxSession(
                snapshot_path=str(Path(directory) / "snapshots" / "rev"),
                runtime="onnx-int8",
                tokenizer=object(),
                torch_module=object(),
                threads=1,
                instruction_sets=("avx2", "avx_vnni"),
                ort_module=FakeOrt,
                artifact_preparer=prepare,
            )

            self.assertFalse(observed["reduce_range"])


if __name__ == "__main__":
    unittest.main()

"""BASIC ONNX Runtime backend for the official BAAI/bge-m3 FP32 artifact.

``BgeM3OnnxBackend`` reconstructs dense, learned-sparse, and ColBERT outputs
while applying the calibrated CPU session topology.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


class BgeM3OnnxBackend:
    """Reconstruct BGE-M3 dense, sparse and ColBERT outputs from ONNX tokens."""

    runtime = "onnx-fp32"

    def __init__(
        self,
        *,
        snapshot_path: str,
        tokenizer: Any,
        torch_module: Any,
        threads: int,
        concurrency: int = 1,
        execution_mode: str = "sequential",
        spin_policy: str = "default",
        inter_op_threads: int = 1,
        max_length: int = 8192,
        batch_size_for_tokens: Optional[Callable[[int], int]] = None,
        ort_module: Any = None,
        providers: Optional[list[Any]] = None,
        expected_provider: str = "CPUExecutionProvider",
    ):
        """Load BGE ONNX and projection artifacts with a tuned ORT session.

        Args:
            snapshot_path: Local BGE-M3 snapshot containing ONNX and head files.
            tokenizer: Tokenizer adapter used for padding and length estimates.
            torch_module: PyTorch module used to load projection heads.
            threads: ORT intra-op worker count.
            concurrency: Number of independent CPU inference lanes.
            execution_mode: Sequential or parallel ORT graph scheduling.
            spin_policy: ORT worker spinning policy.
            inter_op_threads: ORT inter-op worker count.
            max_length: Maximum input token length.
            batch_size_for_tokens: Optional length-aware batch selector.
            ort_module: Injectable ONNX Runtime module.
            providers: Ordered ORT execution providers.
            expected_provider: Provider required at highest priority.

        Raises:
            FileNotFoundError: If a required model artifact is absent.
            RuntimeError: If ORT activates an unexpected provider.
        """
        snapshot = Path(snapshot_path)
        model_path = snapshot / "onnx" / "model.onnx"
        if not model_path.is_file():
            raise FileNotFoundError(f"BGE-M3 ONNX artifact not found: {model_path}")
        for name in ("sparse_linear.pt", "colbert_linear.pt"):
            if not (snapshot / name).is_file():
                raise FileNotFoundError(f"BGE-M3 head not found: {snapshot / name}")

        if ort_module is None:
            import onnxruntime as ort_module

        self.ort = ort_module
        self.tokenizer = tokenizer
        self.torch = torch_module
        self.max_length = max_length
        self.concurrency = max(1, int(concurrency))
        self.batch_size_for_tokens = batch_size_for_tokens or (lambda _tokens: 1)
        self.model_path = model_path
        self.providers = providers or ["CPUExecutionProvider"]
        self.expected_provider = expected_provider
        self._session_lock = threading.Lock()
        self.session = None
        self.session_topology: Optional[tuple[int, str, str, int]] = None
        self.configure_session(
            threads=threads,
            execution_mode=execution_mode,
            spin_policy=spin_policy,
            inter_op_threads=inter_op_threads,
        )

        sparse = torch_module.load(
            snapshot / "sparse_linear.pt",
            map_location="cpu",
            weights_only=True,
        )
        colbert = torch_module.load(
            snapshot / "colbert_linear.pt",
            map_location="cpu",
            weights_only=True,
        )
        self.sparse_weight = sparse["weight"].float().contiguous()
        self.sparse_bias = sparse["bias"].float().contiguous()
        self.colbert_weight = colbert["weight"].float().contiguous()
        self.colbert_bias = colbert["bias"].float().contiguous()
        self._lane_executor = (
            ThreadPoolExecutor(max_workers=self.concurrency)
            if self.concurrency > 1
            else None
        )

    @staticmethod
    def _normalise_execution_mode(value: str) -> str:
        normalised = str(value).strip().lower()
        return normalised if normalised in {"sequential", "parallel"} else "sequential"

    @staticmethod
    def _normalise_spin_policy(value: str) -> str:
        normalised = str(value).strip().lower()
        return (
            normalised
            if normalised in {"default", "bounded", "disabled"}
            else "default"
        )

    def _make_session(
        self,
        *,
        threads: int,
        execution_mode: str,
        spin_policy: str,
        inter_op_threads: int,
    ):
        options = self.ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = inter_op_threads
        options.execution_mode = (
            self.ort.ExecutionMode.ORT_PARALLEL
            if execution_mode == "parallel"
            else self.ort.ExecutionMode.ORT_SEQUENTIAL
        )
        options.graph_optimization_level = (
            self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        options.add_session_config_entry("session.dynamic_block_base", "4")
        if spin_policy == "bounded":
            options.add_session_config_entry(
                "session.intra_op.spin_duration_us", "1000"
            )
            options.add_session_config_entry("session.intra_op.spin_backoff_max", "8")
        elif spin_policy == "disabled":
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        session = self.ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=self.providers,
        )
        actual_providers = session.get_providers()
        if not actual_providers or actual_providers[0] != self.expected_provider:
            raise RuntimeError(f"unexpected ONNX Runtime providers: {actual_providers}")
        return session

    def configure_session(
        self,
        *,
        threads: int,
        execution_mode: str = "sequential",
        spin_policy: str = "default",
        inter_op_threads: int = 1,
    ) -> bool:
        """Rebuild the fixed ORT thread pool when a calibration candidate changes."""
        topology = (
            max(1, int(threads)),
            self._normalise_execution_mode(execution_mode),
            self._normalise_spin_policy(spin_policy),
            max(1, int(inter_op_threads)),
        )
        with self._session_lock:
            if topology == self.session_topology:
                return False
            session = self._make_session(
                threads=topology[0],
                execution_mode=topology[1],
                spin_policy=topology[2],
                inter_op_threads=topology[3],
            )
            self.session = session
            self.session_topology = topology
        return True

    def _process_sparse(self, weights: np.ndarray, input_ids: np.ndarray) -> dict:
        result = defaultdict(float)
        unused = {
            self.tokenizer.cls_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
        }
        for weight, token_id in zip(weights, input_ids):
            value = float(weight)
            token = int(token_id)
            if token not in unused and value > result[str(token)]:
                result[str(token)] = value
        return dict(result)

    def _run_batch(
        self,
        sentences: list[str],
        *,
        max_length: int,
        return_dense: bool,
        return_sparse: bool,
        return_colbert: bool,
    ) -> dict:
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
        attention_mask = np.asarray(encoded["attention_mask"], dtype=np.int64)
        outputs = self.session.run(
            ["token_embeddings", "sentence_embedding"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        token_embeddings = np.asarray(outputs[0], dtype=np.float32)
        result = {}
        if return_dense:
            result["dense_vecs"] = np.asarray(outputs[1], dtype=np.float32)

        token_tensor = self.torch.from_numpy(token_embeddings)
        if return_sparse:
            sparse = self.torch.nn.functional.linear(
                token_tensor, self.sparse_weight, self.sparse_bias
            )
            sparse = self.torch.relu(sparse).squeeze(-1).numpy()
            result["lexical_weights"] = [
                self._process_sparse(weights, ids)
                for weights, ids in zip(sparse, input_ids)
            ]

        if return_colbert:
            colbert = self.torch.nn.functional.linear(
                token_tensor[:, 1:], self.colbert_weight, self.colbert_bias
            )
            mask = self.torch.from_numpy(attention_mask[:, 1:]).unsqueeze(-1)
            colbert = colbert * mask
            colbert = self.torch.nn.functional.normalize(colbert, dim=-1)
            colbert_array = colbert.numpy()
            result["colbert_vecs"] = [
                vectors[: max(0, int(mask_row.sum()) - 1)]
                for vectors, mask_row in zip(colbert_array, attention_mask)
            ]
        return result

    @staticmethod
    def _merge(parts: list[dict]) -> dict:
        if len(parts) == 1:
            return parts[0]
        merged = {}
        for key in ("dense_vecs", "lexical_weights", "colbert_vecs"):
            values = [part[key] for part in parts if key in part]
            if not values:
                continue
            if key == "dense_vecs":
                merged[key] = np.concatenate(values, axis=0)
            else:
                merged[key] = [item for value in values for item in value]
        return merged

    def _run_group(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        max_length: Optional[int] = None,
        return_dense: bool,
        return_sparse: bool,
        return_colbert: bool,
    ) -> dict:
        effective_max_length = min(max_length or self.max_length, self.max_length)
        chunks = [
            sentences[start : start + batch_size]
            for start in range(0, len(sentences), batch_size)
        ]
        if self._lane_executor is None or len(chunks) <= 1:
            return self._merge(
                [
                    self._run_batch(
                        chunk,
                        max_length=effective_max_length,
                        return_dense=return_dense,
                        return_sparse=return_sparse,
                        return_colbert=return_colbert,
                    )
                    for chunk in chunks
                ]
            )
        futures = [
            self._lane_executor.submit(
                self._run_batch,
                chunk,
                max_length=effective_max_length,
                return_dense=return_dense,
                return_sparse=return_sparse,
                return_colbert=return_colbert,
            )
            for chunk in chunks
        ]
        return self._merge([future.result() for future in futures])

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int = 1,
        max_length: int = 8192,
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert_vecs: bool = False,
    ) -> dict:
        """Encode texts through the FlagEmbedding-compatible interface.

        Args:
            sentences: Text batch in caller order.
            batch_size: Maximum microbatch size.
            max_length: Per-input truncation limit.
            return_dense: Include dense sentence vectors.
            return_sparse: Include learned lexical weights.
            return_colbert_vecs: Include token-level ColBERT vectors.

        Returns:
            Requested BGE outputs using FlagEmbedding-compatible keys.
        """
        return self._run_group(
            sentences,
            batch_size=max(1, int(batch_size)),
            max_length=max_length,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert=return_colbert_vecs,
        )

    def embed(
        self,
        sentences: list[str],
        *,
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert: bool = True,
        normalize_dense: bool = False,
        deadline=None,
    ) -> dict:
        """Embed texts with length-aware grouping and optional normalization.

        Args:
            sentences: Text batch in caller order.
            return_dense: Include dense sentence vectors.
            return_sparse: Include learned lexical weights.
            return_colbert: Include token-level ColBERT vectors.
            normalize_dense: L2-normalize dense vectors when requested.
            deadline: Optional cooperative inference deadline.

        Returns:
            Requested outputs restored to input order.
        """
        if deadline is not None:
            deadline.check()
        lengths = self.tokenizer.token_lengths(sentences, self.max_length)
        groups = []
        for sentence, token_length in zip(sentences, lengths):
            selected_batch = self.batch_size_for_tokens(token_length)
            if groups and groups[-1][0] == selected_batch:
                groups[-1][1].append(sentence)
            else:
                groups.append((selected_batch, [sentence]))

        parts = []
        for selected_batch, group in groups:
            if deadline is not None:
                deadline.check()
            parts.append(
                self._run_group(
                    group,
                    batch_size=selected_batch,
                    return_dense=return_dense,
                    return_sparse=return_sparse,
                    return_colbert=return_colbert,
                )
            )
        result = self._merge(parts)
        if return_dense and normalize_dense and len(result.get("dense_vecs", [])):
            dense = np.asarray(result["dense_vecs"])
            norms = np.maximum(np.linalg.norm(dense, axis=1, keepdims=True), 1e-12)
            result["dense_vecs"] = dense / norms
        return result

    def close(self) -> None:
        """Stop the optional multi-lane executor without waiting for queued work."""
        if self._lane_executor is not None:
            self._lane_executor.shutdown(wait=False, cancel_futures=True)

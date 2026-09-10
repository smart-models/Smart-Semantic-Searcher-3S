"""ADVANCED ONNX Runtime backends for Qwen3 dense embeddings.

The export is deliberately local and cache-first: model weights are never
downloaded here.  FP32 and dynamic INT8 artifacts are generated atomically,
validated with ONNX Runtime, and tied to the exact model revision and exporter
stack through a manifest.  The ADVANCED retrieval quality suite remains the
only authority allowed to promote either runtime.

``ensure_qwen_dense_onnx_artifact()`` owns export integrity and
``QwenDenseOnnxSession`` owns CPU inference.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
from basic_autotune import ProfileLock

EXPORT_SCHEMA_VERSION = 1
EXPORT_OPSET = 18
SUPPORTED_RUNTIMES = frozenset({"onnx-fp32", "onnx-int8"})
MODEL_CACHE_KEY = "qwen3-embedding-0.6b"


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _safe_component(value: str) -> str:
    normalised = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return normalised[:128] or "unknown"


def qwen_dense_onnx_cache_dir(
    snapshot_path: Path,
    *,
    cache_root: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Path:
    """Resolve the revision-scoped cache for Qwen dense ONNX artifacts.

    Args:
        snapshot_path: Local Hugging Face model snapshot.
        cache_root: Explicit artifact-cache root.
        env: Environment mapping used when ``cache_root`` is omitted.

    Returns:
        A stable cache directory unique to the local model revision.
    """
    values = os.environ if env is None else env
    if cache_root is None:
        configured = values.get("SMART_EMBEDDER_ADVANCED_ONNX_CACHE")
        if configured:
            cache_root = Path(configured)
        else:
            hf_home = Path(values.get("HF_HOME", "/app/model_cache"))
            cache_root = hf_home / "autotune" / "onnx"
    revision = (
        snapshot_path.name
        if snapshot_path.parent.name == "snapshots"
        else _sha256(snapshot_path / "config.json")[:16]
    )
    return cache_root / MODEL_CACHE_KEY / _safe_component(revision)


def _export_identity(
    snapshot_path: Path, runtime: str, *, reduce_range: bool = False
) -> dict:
    config_path = snapshot_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Qwen config not found: {config_path}")
    backend_source = Path(__file__).resolve()
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "runtime": runtime,
        "model_revision": snapshot_path.name,
        "model_config_sha256": _sha256(config_path),
        "backend_source_sha256": _sha256(backend_source),
        "opset": EXPORT_OPSET,
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "onnx": _package_version("onnx"),
        "onnxscript": _package_version("onnxscript"),
        "onnxruntime": _package_version("onnxruntime"),
        "quantization": (
            {
                "method": "dynamic",
                "weight_type": "qint8",
                "per_channel": True,
                "reduce_range": bool(reduce_range),
                "operators": ["Gemm", "MatMul"],
            }
            if runtime == "onnx-int8"
            else None
        ),
    }


def _read_manifest(directory: Path) -> Optional[dict]:
    try:
        value = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _artifact_is_valid(directory: Path, identity: Mapping[str, Any]) -> bool:
    manifest = _read_manifest(directory)
    if not manifest or manifest.get("identity") != dict(identity):
        return False
    files = manifest.get("files")
    if not isinstance(files, Mapping) or "model.onnx" not in files:
        return False
    for name, metadata in files.items():
        if not isinstance(name, str) or Path(name).name != name:
            return False
        if not isinstance(metadata, Mapping):
            return False
        path = directory / name
        try:
            expected_size = int(metadata["size_bytes"])
        except (KeyError, TypeError, ValueError):
            return False
        if not path.is_file() or path.is_symlink():
            return False
        try:
            if path.stat().st_size != expected_size:
                return False
        except OSError:
            return False
        expected_sha256 = metadata.get("sha256")
        if not isinstance(expected_sha256, str):
            return False
        try:
            if _sha256(path) != expected_sha256:
                return False
        except OSError:
            return False
    return True


def _write_manifest(directory: Path, identity: Mapping[str, Any]) -> None:
    files = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name == "manifest.json":
            continue
        files[path.name] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    if "model.onnx" not in files:
        raise RuntimeError("Qwen ONNX export did not produce model.onnx")
    payload = {
        "identity": dict(identity),
        "files": files,
        "created_at_unix": int(time.time()),
    }
    temporary = directory / f".manifest.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, directory / "manifest.json")


def _default_fp32_exporter(
    snapshot_path: Path,
    output_path: Path,
    tokenizer: Any,
    torch_module: Any,
) -> None:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        str(snapshot_path),
        dtype=torch_module.float32,
        attn_implementation="eager",
        local_files_only=True,
    ).eval()

    class LastTokenEmbedding(torch_module.nn.Module):
        """Export wrapper reducing the model's output to a last-token embedding.

        The ONNX graph has to be self-contained, so the causal mask and the
        last-token pooling that live in Python around the model at inference
        time are folded into ``forward`` here, ahead of the export.
        """

        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, input_ids, attention_mask):
            sequence_length = input_ids.shape[1]
            positions = torch_module.arange(sequence_length, device=input_ids.device)
            causal = positions[:, None] >= positions[None, :]
            allowed = causal[None, None, :, :] & attention_mask[:, None, None, :].to(
                dtype=torch_module.bool
            )
            zeros = torch_module.zeros(
                allowed.shape,
                dtype=torch_module.float32,
                device=input_ids.device,
            )
            causal_mask = torch_module.where(
                allowed,
                zeros,
                torch_module.full_like(
                    zeros, torch_module.finfo(torch_module.float32).min
                ),
            )
            output = self.wrapped(
                input_ids=input_ids,
                attention_mask=causal_mask,
                use_cache=False,
                return_dict=False,
            )
            return output[0][:, -1, :]

    wrapper = LastTokenEmbedding(model).eval()
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = 1
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id
    input_ids = torch_module.tensor(
        [
            [int(pad_id), int(pad_id), int(eos_id), int(eos_id)],
            [int(eos_id), int(eos_id), int(eos_id), int(eos_id)],
        ],
        dtype=torch_module.long,
    )
    attention_mask = torch_module.tensor(
        [[0, 0, 1, 1], [1, 1, 1, 1]], dtype=torch_module.long
    )
    with torch_module.inference_mode():
        reference = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,
        )[0][:, -1, :]
        masked = wrapper(input_ids, attention_mask)
    maximum_error = float((reference - masked).abs().max().item())
    if maximum_error > 1e-5:
        raise RuntimeError(
            "export-safe Qwen causal mask changed the model output: "
            f"max_abs_error={maximum_error}"
        )
    batch = torch_module.export.Dim("batch", min=1)
    sequence = torch_module.export.Dim("sequence", min=1)
    dynamic_shapes = {
        "input_ids": {0: batch, 1: sequence},
        "attention_mask": {0: batch, 1: sequence},
    }
    try:
        torch_module.onnx.export(
            wrapper,
            (input_ids, attention_mask),
            str(output_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["sentence_embedding"],
            opset_version=EXPORT_OPSET,
            dynamo=True,
            external_data=True,
            dynamic_shapes=dynamic_shapes,
        )
    except Exception:
        logging.exception(
            "torch.export ONNX path failed; trying the compatibility exporter"
        )
        for path in output_path.parent.iterdir():
            if path.is_file():
                path.unlink()
        torch_module.onnx.export(
            wrapper,
            (input_ids, attention_mask),
            str(output_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["sentence_embedding"],
            opset_version=EXPORT_OPSET,
            dynamo=False,
            external_data=True,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "sentence_embedding": {0: "batch"},
            },
        )
    del wrapper
    del model


def _default_int8_quantizer(
    fp32_path: Path,
    output_path: Path,
    *,
    reduce_range: bool,
) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(fp32_path),
        str(output_path),
        per_channel=True,
        reduce_range=reduce_range,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Gemm", "MatMul"],
        use_external_data_format=True,
    )


def _validate_session(model_path: Path, ort_module: Any) -> None:
    options = ort_module.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort_module.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    for shape in ((1, 2), (2, 5)):
        output = session.run(
            ["sentence_embedding"],
            {
                "input_ids": np.ones(shape, dtype=np.int64),
                "attention_mask": np.ones(shape, dtype=np.int64),
            },
        )[0]
        if output.shape != (shape[0], 1024) or not np.isfinite(output).all():
            raise RuntimeError(
                f"invalid Qwen ONNX validation output shape={output.shape}"
            )


def ensure_qwen_dense_onnx_artifact(
    *,
    snapshot_path: Path,
    runtime: str,
    tokenizer: Any,
    torch_module: Any,
    cache_root: Optional[Path] = None,
    reduce_range: bool = False,
    ort_module: Any = None,
    fp32_exporter: Callable[..., None] = _default_fp32_exporter,
    int8_quantizer: Callable[..., None] = _default_int8_quantizer,
    validate: Callable[[Path, Any], None] = _validate_session,
) -> Path:
    """Return a valid cached artifact, generating it atomically if needed."""
    if runtime not in SUPPORTED_RUNTIMES:
        raise ValueError(f"unsupported Qwen ONNX runtime: {runtime}")
    if ort_module is None:
        import onnxruntime as ort_module

    revision_root = qwen_dense_onnx_cache_dir(snapshot_path, cache_root=cache_root)
    target = revision_root / runtime
    identity = _export_identity(
        snapshot_path,
        runtime,
        reduce_range=reduce_range if runtime == "onnx-int8" else False,
    )
    # Serving mounts prepared artifacts read-only; only exporters need a lock.
    if _artifact_is_valid(target, identity):
        return target / "model.onnx"
    lock_path = revision_root / ".export.lock"
    with ProfileLock(
        lock_path,
        timeout_seconds=1800.0,
        stale_seconds=7200.0,
        poll_seconds=0.25,
    ):
        runtimes = ["onnx-fp32"]
        if runtime == "onnx-int8":
            runtimes.append(runtime)
        for current_runtime in runtimes:
            identity = _export_identity(
                snapshot_path,
                current_runtime,
                reduce_range=(
                    reduce_range if current_runtime == "onnx-int8" else False
                ),
            )
            target = revision_root / current_runtime
            if _artifact_is_valid(target, identity):
                continue
            temporary = revision_root / (
                f".{current_runtime}.tmp-{os.getpid()}-{time.time_ns()}"
            )
            temporary.mkdir(parents=True, exist_ok=False)
            try:
                if current_runtime == "onnx-fp32":
                    fp32_exporter(
                        snapshot_path,
                        temporary / "model.onnx",
                        tokenizer,
                        torch_module,
                    )
                else:
                    fp32 = revision_root / "onnx-fp32" / "model.onnx"
                    int8_quantizer(
                        fp32,
                        temporary / "model.onnx",
                        reduce_range=reduce_range,
                    )
                validate(temporary / "model.onnx", ort_module)
                _write_manifest(temporary, identity)
                if target.exists():
                    shutil.rmtree(target)
                os.replace(temporary, target)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
    return revision_root / runtime / "model.onnx"


class QwenDenseOnnxSession:
    """Thread-configured ORT session for a cached Qwen dense artifact."""

    def __init__(
        self,
        *,
        snapshot_path: str,
        runtime: str,
        tokenizer: Any,
        torch_module: Any,
        threads: int,
        inter_op_threads: int = 1,
        spin_policy: str = "default",
        instruction_sets: tuple[str, ...] = (),
        cache_root: Optional[Path] = None,
        ort_module: Any = None,
        artifact_preparer: Callable[..., Path] = (ensure_qwen_dense_onnx_artifact),
    ):
        """Create a validated CPU session for a cached Qwen artifact.

        Args:
            snapshot_path: Local Qwen model snapshot.
            runtime: ONNX FP32 or dynamic INT8 runtime identifier.
            tokenizer: Tokenizer used when an artifact must be exported.
            torch_module: PyTorch module used by the local exporter.
            threads: ORT intra-op worker count.
            inter_op_threads: ORT inter-op worker count.
            spin_policy: Worker wait policy; ``passive`` disables spinning.
            instruction_sets: CPU capabilities used for safe INT8 export.
            cache_root: Optional artifact-cache root.
            ort_module: Injectable ONNX Runtime module.
            artifact_preparer: Injectable export/cache preparation function.

        Raises:
            ValueError: If ``runtime`` is unsupported.
            RuntimeError: If ORT does not activate its CPU provider.
        """
        if runtime not in SUPPORTED_RUNTIMES:
            raise ValueError(f"unsupported Qwen ONNX runtime: {runtime}")
        if ort_module is None:
            import onnxruntime as ort_module

        instruction_set = {value.lower() for value in instruction_sets}
        reduce_range = (
            runtime == "onnx-int8"
            and "avx2" in instruction_set
            and not ({"avx_vnni", "avx512_vnni"} & instruction_set)
        )
        model_path = artifact_preparer(
            snapshot_path=Path(snapshot_path),
            runtime=runtime,
            tokenizer=tokenizer,
            torch_module=torch_module,
            cache_root=cache_root,
            reduce_range=reduce_range,
            ort_module=ort_module,
        )
        options = ort_module.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = max(1, int(inter_op_threads))
        options.execution_mode = ort_module.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = (
            ort_module.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        if spin_policy == "passive":
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.runtime = runtime
        self.model_path = model_path
        self.session = ort_module.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        providers = self.session.get_providers()
        if not providers or providers[0] != "CPUExecutionProvider":
            raise RuntimeError(f"unexpected Qwen ORT providers: {providers}")

    def run(self, padded: Mapping[str, Any]) -> np.ndarray:
        """Run one padded batch and validate its dense embedding matrix.

        Args:
            padded: Integer ``input_ids`` and ``attention_mask`` tensors.

        Returns:
            A finite FP32 matrix with shape ``[batch, 1024]``.

        Raises:
            RuntimeError: If the backend returns an invalid shape or values.
        """
        inputs = {
            "input_ids": np.asarray(padded["input_ids"], dtype=np.int64),
            "attention_mask": np.asarray(padded["attention_mask"], dtype=np.int64),
        }
        output = np.asarray(
            self.session.run(["sentence_embedding"], inputs)[0],
            dtype=np.float32,
        )
        if output.ndim != 2 or output.shape[1] != 1024:
            raise RuntimeError(
                f"Qwen ORT returned shape {output.shape}, expected [batch, 1024]"
            )
        if not np.isfinite(output).all():
            raise RuntimeError("Qwen ORT returned NaN or Inf")
        return output

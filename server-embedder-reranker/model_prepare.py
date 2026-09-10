"""Provision pinned Smart Embedder models before the read-only server starts."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

from autotune_entrypoint import prepare
from model_revisions import MODEL_REVISIONS


BASIC_PROFILE = "basic"
ADVANCED_PROFILE = "advanced"
PROFILE_MODELS = {
    BASIC_PROFILE: (
        "BAAI/bge-m3",
        "BAAI/bge-reranker-v2-m3",
    ),
    ADVANCED_PROFILE: (
        "Qwen/Qwen3-Embedding-0.6B",
        "omai-research/milco-650m",
        "BAAI/bge-m3-unsupervised",
        "Qwen/Qwen3-Reranker-0.6B",
    ),
}


def models_for_profile(profile: str) -> tuple[str, ...]:
    """Return the complete pinned model set required by one profile."""
    normalized = profile.strip().lower()
    try:
        models = PROFILE_MODELS[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported EMBEDDER_PROFILE: {profile!r}") from exc
    missing = [model_id for model_id in models if model_id not in MODEL_REVISIONS]
    if missing:
        raise ValueError(f"models without immutable revisions: {', '.join(missing)}")
    return models


def prefetch_models(
    profile: str,
    *,
    downloader: Callable[..., str] | None = None,
    cache_dir: Path | None = None,
) -> dict[str, str]:
    """Ensure every pinned snapshot exists without importing remote code."""
    if downloader is None:
        from huggingface_hub import snapshot_download

        downloader = snapshot_download
    resolved_cache = cache_dir or Path(
        os.getenv(
            "HF_HUB_CACHE",
            str(Path(os.getenv("HF_HOME", "/app/model_cache")) / "hub"),
        )
    )
    snapshots: dict[str, str] = {}
    for model_id in models_for_profile(profile):
        revision = MODEL_REVISIONS[model_id]
        options = {
            "repo_id": model_id,
            "revision": revision,
            "cache_dir": str(resolved_cache),
        }
        try:
            snapshot = downloader(local_files_only=True, **options)
            logging.info("Pinned model already cached: %s@%s", model_id, revision)
        except Exception:  # local cache probes use backend-specific exceptions
            logging.info("Downloading pinned model: %s@%s", model_id, revision)
            snapshot = downloader(**options)
        snapshot_path = Path(snapshot)
        if not snapshot_path.is_dir():
            raise RuntimeError(
                f"downloaded snapshot is not a directory: {model_id}@{revision}"
            )
        if snapshot_path.name != revision:
            raise RuntimeError(
                f"snapshot revision mismatch for {model_id}: "
                f"expected {revision}, got {snapshot_path.name}"
            )
        snapshots[model_id] = str(snapshot_path)
    return snapshots


def main() -> int:
    """Provision models and persist the applicable hardware calibration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    profile = os.getenv("EMBEDDER_PROFILE", BASIC_PROFILE)
    prefetch_models(profile)
    result = prepare()
    if result.get("action") == "fallback":
        logging.warning("Hardware autotune preparation fallback: %s", result)
    else:
        logging.info("Hardware autotune preparation result: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

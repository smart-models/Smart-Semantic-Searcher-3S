"""Provision pinned Smart Embedder models before the read-only server starts."""

from __future__ import annotations

import fnmatch
import json
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
# MILCO builds its multilingual encoder from the sub-config cached in its own
# config.json and reads only this repository's tokenizer, so neither of its
# 2.3 GB weight files is ever loaded.
DOWNLOAD_IGNORE_PATTERNS = {
    "BAAI/bge-m3-unsupervised": ("*.safetensors", "*.bin"),
}
# Folder beside a repository's ``snapshots/`` recording finished downloads. The
# hub cache links each file into its snapshot as soon as that file lands, so a
# local-only lookup still returns the snapshot an interrupted download left
# behind. Each record is the inventory the Hub listed for that revision, written
# only once every listed file is present, so it proves the download finished.
COMPLETION_MARKERS = ".smart-embedder-complete"
# Older releases recorded only the skipped patterns, sometimes for a partial
# snapshot, so a record without this format is validated against the Hub again.
MARKER_FORMAT = 2


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


def completion_marker(cache_dir: Path, model_id: str, revision: str) -> Path:
    """Return the record of one pinned snapshot's finished download."""
    repository = "models--" + model_id.replace("/", "--")
    return cache_dir / repository / COMPLETION_MARKERS / revision


def _recorded_inventory(
    marker: Path, ignore_patterns: tuple[str, ...]
) -> list[str] | None:
    """Return the verified inventory a marker records, or None if it proves nothing."""
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        not isinstance(record, dict)
        or record.get("format") != MARKER_FORMAT
        or record.get("ignore_patterns") != list(ignore_patterns)
        or not isinstance(record.get("files"), list)
    ):
        return None
    return [str(name) for name in record["files"]]


def _missing_files(snapshot: Path, names: list[str]) -> list[str]:
    """Return the listed files a snapshot folder does not hold."""
    return [name for name in names if not (snapshot / name).exists()]


def prefetch_models(
    profile: str,
    *,
    downloader: Callable[..., str] | None = None,
    lister: Callable[..., list[str]] | None = None,
    cache_dir: Path | None = None,
) -> dict[str, str]:
    """Fully download every pinned snapshot without importing remote code.

    Raises:
        RuntimeError: When a snapshot cannot be proven complete, including when
            the Hub is unreachable and no earlier download was verified.
    """
    if downloader is None:
        from huggingface_hub import snapshot_download

        downloader = snapshot_download
    if lister is None:
        from huggingface_hub import list_repo_files

        lister = list_repo_files
    resolved_cache = cache_dir or Path(
        os.getenv(
            "HF_HUB_CACHE",
            str(Path(os.getenv("HF_HOME", "/app/model_cache")) / "hub"),
        )
    )
    snapshots: dict[str, str] = {}
    for model_id in models_for_profile(profile):
        revision = MODEL_REVISIONS[model_id]
        ignore_patterns = DOWNLOAD_IGNORE_PATTERNS.get(model_id, ())
        options = {
            "repo_id": model_id,
            "revision": revision,
            "cache_dir": str(resolved_cache),
        }
        if ignore_patterns:
            options["ignore_patterns"] = list(ignore_patterns)
        marker = completion_marker(resolved_cache, model_id, revision)
        snapshot = None
        inventory = _recorded_inventory(marker, ignore_patterns)
        if inventory is not None:
            try:
                snapshot = downloader(local_files_only=True, **options)
            except Exception:  # a pruned cache may have lost a recorded snapshot
                snapshot = None
            else:
                if _missing_files(Path(snapshot), inventory):
                    snapshot = None  # a recorded file was pruned since
                else:
                    logging.info(
                        "Pinned model already cached: %s@%s", model_id, revision
                    )
        if snapshot is None:
            # With the Hub unreachable, snapshot_download silently returns
            # whatever folder an interrupted download left behind; only the
            # listing taken here can tell a complete snapshot from that one.
            try:
                listed = lister(model_id, revision=revision)
            except Exception as exc:  # unreachable, rate limited, gated...
                raise RuntimeError(
                    f"cannot verify pinned model {model_id}@{revision}: listing "
                    f"its files on the Hub failed ({exc.__class__.__name__}); "
                    "retry with network access"
                ) from exc
            # The inventory leaves the skipped files out, so a wider download
            # set never reuses a snapshot that was completed without them.
            inventory = [
                name
                for name in listed
                if not any(fnmatch.fnmatch(name, p) for p in ignore_patterns)
            ]
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
        missing = _missing_files(snapshot_path, inventory)
        if missing:
            raise RuntimeError(
                f"incomplete snapshot for {model_id}@{revision}: "
                f"{len(missing)} file(s) missing, e.g. {missing[0]}"
            )
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "format": MARKER_FORMAT,
                    "ignore_patterns": list(ignore_patterns),
                    "files": sorted(inventory),
                }
            ),
            encoding="utf-8",
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

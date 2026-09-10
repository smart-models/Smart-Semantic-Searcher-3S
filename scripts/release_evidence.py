"""Hash a release evidence directory and reject an incomplete bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one evidence file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    root: Path,
    required: list[str],
    *,
    output_name: str = "manifest.json",
    commit: str = "unknown",
) -> dict:
    """Return a deterministic inventory plus completeness status for ``root``."""
    missing = sorted(item for item in required if not (root / item).is_file())
    files = []
    for path in sorted(
        candidate for candidate in root.rglob("*") if candidate.is_file()
    ):
        relative = path.relative_to(root).as_posix()
        if relative == output_name:
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema": 1,
        "status": "complete" if not missing else "incomplete",
        "commit": commit,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "missing": missing,
        "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    """Write ``manifest.json`` and fail when required evidence is absent."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--require", action="append", default=[])
    parser.add_argument("--output-name", default="manifest.json")
    parser.add_argument("--commit", default=os.environ.get("GITHUB_SHA", "unknown"))
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        print(
            f"[ERROR] evidence directory does not exist: {args.root}", file=sys.stderr
        )
        return 2
    manifest = build_manifest(
        args.root,
        args.require,
        output_name=args.output_name,
        commit=args.commit,
    )
    output = args.root / args.output_name
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if manifest["missing"]:
        print(
            f"[ERROR] missing release evidence: {', '.join(manifest['missing'])}",
            file=sys.stderr,
        )
        return 1
    print(f"Release evidence complete: {len(manifest['files'])} hashed file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

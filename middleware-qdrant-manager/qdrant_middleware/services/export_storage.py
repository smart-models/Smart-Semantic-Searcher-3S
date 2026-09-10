"""Safe naming, containment and download headers for collection exports."""

import os
import re
from pathlib import Path, PureWindowsPath
from urllib.parse import quote


_FORBIDDEN_STEM = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED_STEMS = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def export_filename(collection_name: str, stamp: str, nonce: str) -> str:
    """Return a bounded basename derived without trusting collection validation."""
    stem = _FORBIDDEN_STEM.sub("_", collection_name).strip(". ") or "collection"
    encoded = stem.encode("utf-8")[:120]
    stem = encoded.decode("utf-8", errors="ignore") or "collection"
    if stem.casefold() in _WINDOWS_RESERVED_STEMS:
        stem = f"{stem}_"
    return f"{stem}_export_{stamp}_{nonce[:12]}.json"


def attachment_disposition(filename: str) -> str:
    """Encode a Unicode attachment name without non-ASCII header bytes."""
    fallback = re.sub(r"[^A-Za-z0-9_.-]", "_", filename)
    encoded = quote(filename, safe="")
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{encoded}"


def export_target(root: Path, filename: str) -> Path:
    """Return a contained export target after rejecting non-basename input."""
    if (
        not filename
        or Path(filename).name != filename
        or PureWindowsPath(filename).name != filename
        or any(char in filename for char in ("/", "\\", "\r", "\n"))
    ):
        raise ValueError("export filename must be a basename")
    resolved_root = root.resolve()
    target = resolved_root / filename
    if target.resolve().parent != resolved_root or os.path.islink(target):
        raise ValueError("export target escapes the export root")
    return target

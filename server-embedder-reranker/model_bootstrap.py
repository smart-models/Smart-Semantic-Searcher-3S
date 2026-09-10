"""Prepare named-volume ownership, then run model preparation unprivileged."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_UID = 1000
DEFAULT_GID = 1000
WRITABLE_MOUNTS = (Path("/app/model_cache"), Path("/app/runtime_state"))


def _numeric_identity(variable: str, default: int) -> int:
    """Read a positive numeric UID or GID from the environment."""
    raw_value = os.getenv(variable, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{variable} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{variable} must be a positive integer")
    return value


def prepare_mount_ownership(paths: Iterable[Path], uid: int, gid: int) -> None:
    """Give the unprivileged preparation user ownership of mount roots only."""
    for path in paths:
        if not path.is_dir():
            raise RuntimeError(f"required writable mount is not a directory: {path}")
        os.chown(path, uid, gid)


def drop_privileges(uid: int, gid: int) -> None:
    """Irreversibly leave root before importing model or network libraries."""
    if os.geteuid() != 0:
        raise RuntimeError("model bootstrap must start as root")
    os.setgroups([])
    os.setgid(gid)
    os.setuid(uid)
    if os.geteuid() != uid or os.getegid() != gid:
        raise RuntimeError("model bootstrap failed to drop privileges")


def main() -> int:
    """Initialize volume roots and replace this process as UID/GID 1000."""
    uid = _numeric_identity("SMART_PREPARE_UID", DEFAULT_UID)
    gid = _numeric_identity("SMART_PREPARE_GID", DEFAULT_GID)
    prepare_mount_ownership(WRITABLE_MOUNTS, uid, gid)
    drop_privileges(uid, gid)
    preparer = Path(__file__).with_name("model_prepare.py")
    os.execv(sys.executable, [sys.executable, str(preparer)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

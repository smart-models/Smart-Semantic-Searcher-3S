"""Tests for the release evidence manifest builder."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

import release_evidence
from release_evidence import build_manifest, main


def test_manifest_hashes_files_and_reports_complete(tmp_path) -> None:
    """Inventory exact bytes while excluding the manifest itself."""
    (tmp_path / "check.log").write_text("pass\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text("old", encoding="utf-8")

    manifest = build_manifest(tmp_path, ["check.log"], commit="abc123")

    assert manifest["status"] == "complete"
    assert manifest["commit"] == "abc123"
    assert [entry["path"] for entry in manifest["files"]] == ["check.log"]
    content = (tmp_path / "check.log").read_bytes()
    assert manifest["files"][0]["sha256"] == hashlib.sha256(content).hexdigest()
    assert manifest["files"][0]["bytes"] == len(content)


def test_main_writes_incomplete_manifest_and_fails(tmp_path) -> None:
    """A missing required artifact is an explicit non-zero release gate."""
    result = main([str(tmp_path), "--require", "missing.log"])

    assert result == 1
    assert '"status": "incomplete"' in (tmp_path / "manifest.json").read_text()


def test_main_publishes_complete_manifest(tmp_path):
    """Successful evidence keeps the manifest schema and exact file digest."""
    (tmp_path / "check.log").write_bytes(b"pass\n")
    assert main([str(tmp_path), "--require", "check.log"]) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["files"] == [
        {
            "path": "check.log",
            "bytes": 5,
            "sha256": hashlib.sha256(b"pass\n").hexdigest(),
        }
    ]


def test_failed_manifest_replace_preserves_previous_file(tmp_path, monkeypatch):
    """An interrupted publication cannot truncate the previous manifest."""
    output = tmp_path / "manifest.json"
    output.write_bytes(b"previous manifest")

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(release_evidence.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        main([str(tmp_path)])
    assert output.read_bytes() == b"previous manifest"
    assert list(tmp_path.iterdir()) == [output]


def test_hashing_rejects_evidence_modified_during_read(tmp_path, monkeypatch):
    """The manifest must not mix a pre-read size with a later digest."""
    path = tmp_path / "changing.log"
    path.write_bytes(b"pass\n")
    states = iter(
        [
            SimpleNamespace(st_size=4, st_mtime_ns=1, st_ctime_ns=1),
            SimpleNamespace(st_size=5, st_mtime_ns=2, st_ctime_ns=2),
        ]
    )
    monkeypatch.setattr(release_evidence.os, "fstat", lambda descriptor: next(states))
    with pytest.raises(OSError, match="evidence changed while hashing"):
        build_manifest(tmp_path, ["changing.log"])

"""Tests for the release evidence manifest builder."""

from __future__ import annotations

from release_evidence import build_manifest, main


def test_manifest_hashes_files_and_reports_complete(tmp_path) -> None:
    """Inventory exact bytes while excluding the manifest itself."""
    (tmp_path / "check.log").write_text("pass\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text("old", encoding="utf-8")

    manifest = build_manifest(tmp_path, ["check.log"], commit="abc123")

    assert manifest["status"] == "complete"
    assert manifest["commit"] == "abc123"
    assert [entry["path"] for entry in manifest["files"]] == ["check.log"]
    assert len(manifest["files"][0]["sha256"]) == 64


def test_main_writes_incomplete_manifest_and_fails(tmp_path) -> None:
    """A missing required artifact is an explicit non-zero release gate."""
    result = main([str(tmp_path), "--require", "missing.log"])

    assert result == 1
    assert '"status": "incomplete"' in (tmp_path / "manifest.json").read_text()

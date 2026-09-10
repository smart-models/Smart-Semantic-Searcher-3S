"""Tests for the scheduled dependency-audit issue renderer."""

from __future__ import annotations

import pytest

from render_dependency_audit_issue import render_issue


def test_render_issue_sorts_findings_and_links_run() -> None:
    """Render stable rows without losing aliases or lock attribution."""
    report = {
        "scan_status": "vulnerable",
        "unique_cves": 2,
        "advisory_occurrences": 3,
        "affected_locks": 2,
        "findings": [
            {
                "package": "h2",
                "version": "4.4.0",
                "lock": "middleware.txt",
                "canonical_vulnerabilities": ["CVE-2"],
                "advisories": ["GHSA-2"],
            },
            {
                "package": "datasets",
                "version": "5.0.0",
                "lock": "embedder.txt",
                "canonical_vulnerabilities": ["CVE-1"],
                "advisories": ["PYSEC-1", "GHSA-1"],
            },
        ],
    }

    body = render_issue(report, "https://github.example/run/7")

    assert body.index("datasets==5.0.0") < body.index("h2==4.4.0")
    assert "`CVE-1`" in body
    assert "`PYSEC-1`, `GHSA-1`" in body
    assert "https://github.example/run/7" in body


def test_render_issue_rejects_non_vulnerable_report() -> None:
    """Do not create a vulnerability issue from an unavailable scan."""
    with pytest.raises(ValueError, match="not vulnerable"):
        render_issue({"scan_status": "unavailable"}, "https://example.test")

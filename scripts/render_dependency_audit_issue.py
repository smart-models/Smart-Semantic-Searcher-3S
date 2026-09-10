"""Render a stable GitHub issue body from a vulnerable dependency audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _code(value: object) -> str:
    """Return a value safe inside one Markdown code span."""
    return str(value).replace("`", "'")


def render_issue(report: dict, run_url: str) -> str:
    """Return the deduplicated vulnerability issue body for ``report``."""
    if report.get("scan_status") != "vulnerable":
        raise ValueError("the audit report is not vulnerable")
    findings = report.get("findings")
    if not isinstance(findings, list) or not findings:
        raise ValueError("the vulnerable audit report has no findings")

    rows = []
    for finding in sorted(
        findings,
        key=lambda item: (
            str(item.get("package", "")).lower(),
            str(item.get("lock", "")),
        ),
    ):
        canonical = ", ".join(
            f"`{_code(value)}`"
            for value in finding.get("canonical_vulnerabilities", [])
        )
        advisories = ", ".join(
            f"`{_code(value)}`" for value in finding.get("advisories", [])
        )
        rows.append(
            f"| `{_code(finding.get('package', ''))}=="
            f"{_code(finding.get('version', ''))}` | "
            f"`{_code(finding.get('lock', ''))}` | {canonical} | {advisories} |"
        )

    return (
        "\n".join(
            [
                "The scheduled OSV audit found known vulnerabilities in the "
                "committed locks.",
                "",
                f"- Unique vulnerabilities: **{int(report.get('unique_cves', 0))}**",
                "- Advisory occurrences: "
                f"**{int(report.get('advisory_occurrences', 0))}**",
                f"- Affected locks: **{int(report.get('affected_locks', 0))}**",
                f"- Workflow run: {run_url}",
                "",
                "| Package | Lock | Canonical vulnerability | OSV advisory |",
                "|---|---|---|---|",
                *rows,
                "",
                "This issue is updated in place on later vulnerable runs and closed "
                "automatically after a clean audit. OSV unavailability and verifier "
                "errors fail the workflow without being mislabeled as vulnerabilities.",
            ]
        )
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Read an audit report and write its GitHub issue body."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run-url", required=True)
    args = parser.parse_args(argv)
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        body = render_issue(report, args.run_url)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit the pinned dependency locks for known vulnerabilities (and emit SBOMs).

The pinned locks in ``requirements/locks/`` reproduce dependency versions, not
artifact bytes (they do not carry hashes). That is not the same as vulnerability
management: a lock freezes a version, it does not tell you the version has a
known CVE. This
verifier closes that gap by querying the OSV database (https://osv.dev) for each
locked ``name==version``.

It is deliberately stdlib-only and install-free. ``pip-audit -r`` cannot audit
these locks from a development machine, because it resolves the set by doing a
``pip install --dry-run`` that builds Linux-only wheels (uvloop) -- which is
impossible on the Windows host this repo is developed on. Reading the pins and
asking OSV about them directly is platform-independent and never installs
anything. The trade-off is that OSV's Python coverage is what it is; a finding
here is a real, published advisory, but an empty result is not a guarantee of
safety.

With ``--sbom <dir>`` it also writes a minimal CycloneDX document per lock,
turning the pinned manifests into machine-readable software bills of materials.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOCKS_DIR = REPOSITORY_ROOT / "requirements" / "locks"
OSV_QUERYBATCH_URL = "https://api.osv.dev/v1/querybatch"
OSV_VULN_URL = "https://api.osv.dev/v1/vulns"
REQUEST_TIMEOUT_SECONDS = 30
USER_AGENT = "smart-embedder-verify-dependencies"
EXIT_VULNERABLE = 1
EXIT_ERROR = 2
EXIT_UNAVAILABLE = 3


def parse_lock(path: Path) -> list[tuple[str, str]]:
    """Return the ``(name, version)`` pins in one lock file.

    Lock files are plain ``name==version`` lines; comment and blank lines are
    skipped, so the parser shares the file's own quoting rules rather than
    re-deriving them.
    """
    pins: list[tuple[str, str]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        name = name.strip()
        version = version.strip()
        if not separator or not name or not version or "==" in version:
            raise ValueError(f"{path}:{line_number} is not an exact name==version pin")
        pins.append((name, version))
    return pins


def _post_json(url: str, payload: dict) -> dict:
    """POST JSON to an OSV endpoint and return the decoded body."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read())


def _get_json(url: str) -> dict:
    """GET an OSV endpoint and return the decoded body."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read())


def query_vulnerabilities(
    pins: list[tuple[str, str]],
) -> dict[tuple[str, str], list[str]]:
    """Map each pinned ``(name, version)`` to its OSV vulnerability ids.

    One ``querybatch`` call covers the whole set (OSV accepts up to 1000
    queries per batch), so the whole matrix is a single round-trip.
    """
    if not pins:
        return {}
    queries = [
        {"package": {"name": name, "ecosystem": "PyPI"}, "version": version}
        for name, version in pins
    ]
    body = _post_json(OSV_QUERYBATCH_URL, {"queries": queries})
    if not isinstance(body, dict):
        raise RuntimeError("OSV returned a non-object response")
    results = body.get("results", [])
    if not isinstance(results, list) or len(results) != len(pins):
        raise RuntimeError("OSV returned a result count that does not match the query")
    findings: dict[tuple[str, str], list[str]] = {}
    for (name, version), result in zip(pins, results):
        if not isinstance(result, dict):
            raise RuntimeError("OSV returned a non-object query result")
        vulnerabilities = result.get("vulns", [])
        if not isinstance(vulnerabilities, list):
            raise RuntimeError("OSV returned a non-list vulnerability result")
        ids = []
        for vulnerability in vulnerabilities:
            if not isinstance(vulnerability, dict):
                raise RuntimeError("OSV returned a non-object vulnerability")
            vuln_id = vulnerability.get("id")
            if not isinstance(vuln_id, str) or not vuln_id:
                raise RuntimeError("OSV returned a vulnerability without an id")
            ids.append(vuln_id)
        if ids:
            findings[(name, version)] = ids
    return findings


def describe_vulnerability(vuln_id: str) -> str:
    """Return a one-line human summary for one advisory, best-effort.

    A detail lookup can fail or carry no summary without invalidating the
    finding itself -- the advisory id is the durable fact. Callers treat the
    summary as decoration, so this never raises.
    """
    try:
        detail = _get_json(f"{OSV_VULN_URL}/{vuln_id}")
    except (OSError, TypeError, ValueError, urllib.error.URLError):
        return ""
    if not isinstance(detail, dict):
        return ""
    raw_aliases = detail.get("aliases", [])
    aliases = (
        [a for a in raw_aliases if isinstance(a, str) and a.startswith("CVE")]
        if isinstance(raw_aliases, list)
        else []
    )
    summary = str(detail.get("summary") or "").strip()
    parts = [", ".join(aliases)] if aliases else []
    if summary:
        parts.append(summary)
    return " - ".join(parts)


def vulnerability_identity(vuln_id: str) -> tuple[str, str]:
    """Return a canonical CVE/advisory identity and one-line description."""
    try:
        detail = _get_json(f"{OSV_VULN_URL}/{vuln_id}")
    except (OSError, TypeError, ValueError, urllib.error.URLError):
        return vuln_id, ""
    if not isinstance(detail, dict):
        return vuln_id, ""
    raw_aliases = detail.get("aliases", [])
    cves = sorted(
        alias
        for alias in raw_aliases
        if isinstance(raw_aliases, list)
        and isinstance(alias, str)
        and alias.startswith("CVE")
    )
    canonical = cves[0] if cves else vuln_id
    summary = str(detail.get("summary") or "").strip()
    return canonical, summary


def _write_json_report(path: Path | None, report: dict) -> bool:
    """Write a requested JSON report, returning whether the write succeeded."""
    if path is None:
        return True
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"[ERROR] Could not write JSON report {path}: {exc}", file=sys.stderr)
        return False
    return True


def sbom_document(lock_name: str, pins: list[tuple[str, str]]) -> dict:
    """Return a minimal CycloneDX 1.5 document for one lock file.

    ``name`` and ``version`` plus a package URL are enough for a valid library
    component; the manifest's whole point is that the pin list is complete.
    """
    components = [
        {
            "type": "library",
            "bom-ref": f"pkg:pypi/{name.lower()}@{version}",
            "name": name,
            "version": version,
            "purl": f"pkg:pypi/{name.lower()}@{version}",
        }
        for name, version in sorted(pins, key=lambda item: item[0].lower())
    ]
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {"component": {"type": "application", "name": lock_name}},
        "components": components,
    }


def main(argv: list[str] | None = None) -> int:
    """Audit every lock in the locks directory and report vulnerabilities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--locks-dir",
        type=Path,
        default=LOCKS_DIR,
        help="directory holding the pinned lock files",
    )
    parser.add_argument(
        "--sbom",
        type=Path,
        default=None,
        help="also write a CycloneDX SBOM per lock into this directory",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="write scan status and deduplicated findings as JSON",
    )
    args = parser.parse_args(argv)

    locks_dir = args.locks_dir
    if not locks_dir.is_dir():
        print(f"[ERROR] locks directory does not exist: {locks_dir}", file=sys.stderr)
        return EXIT_ERROR
    locks = sorted(locks_dir.glob("*.txt"))
    if not locks:
        print(f"[ERROR] no lock files found in {locks_dir}", file=sys.stderr)
        return EXIT_ERROR

    # Query every lock's pins in one batch; the (lock, package) grouping is
    # recovered from the response order, which mirrors the query order.
    try:
        lock_pins = {lock: parse_lock(lock) for lock in locks}
        if args.sbom is not None:
            args.sbom.mkdir(parents=True, exist_ok=True)
            for lock in locks:
                document = sbom_document(lock.stem, lock_pins[lock])
                out_path = args.sbom / f"{lock.stem}.cdx.json"
                out_path.write_text(
                    json.dumps(document, indent=2) + "\n", encoding="utf-8"
                )
                print(
                    f"[SBOM] {out_path.name}: {len(document['components'])} components"
                )
    except (OSError, ValueError) as exc:
        _write_json_report(
            args.json_output,
            {"scan_status": "error", "error": str(exc)},
        )
        print(f"[ERROR] Could not prepare dependency audit: {exc}", file=sys.stderr)
        return EXIT_ERROR
    all_pins = [pin for lock in locks for pin in lock_pins[lock]]
    try:
        findings = query_vulnerabilities(all_pins)
    except (OSError, urllib.error.URLError) as exc:
        report_written = _write_json_report(
            args.json_output,
            {"scan_status": "unavailable", "error": str(exc)},
        )
        print(f"[UNAVAILABLE] OSV query failed: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE if report_written else EXIT_ERROR
    except (RuntimeError, TypeError, ValueError) as exc:
        _write_json_report(
            args.json_output,
            {"scan_status": "error", "error": str(exc)},
        )
        print(f"[ERROR] Invalid OSV response: {exc}", file=sys.stderr)
        return EXIT_ERROR

    # Reattribute the flat findings back to the locks that carry them.
    per_lock: dict[Path, list[tuple[str, str, list[str]]]] = {}
    for lock in locks:
        entries = [
            (name, version, findings[(name, version)])
            for name, version in lock_pins[lock]
            if (name, version) in findings
        ]
        if entries:
            per_lock[lock] = entries

    if not per_lock:
        report_written = _write_json_report(
            args.json_output,
            {
                "scan_status": "clean",
                "unique_cves": 0,
                "advisory_occurrences": 0,
                "affected_locks": 0,
                "locked_packages": len(all_pins),
            },
        )
        if not report_written:
            return EXIT_ERROR
        print(
            f"No known vulnerabilities in {len(all_pins)} locked packages "
            f"across {len(locks)} lock file(s)."
        )
        return 0

    advisory_ids = {
        vuln_id
        for entries in per_lock.values()
        for _name, _version, ids in entries
        for vuln_id in ids
    }
    identities = {vuln_id: vulnerability_identity(vuln_id) for vuln_id in advisory_ids}
    unique_vulnerabilities = {identity for identity, _summary in identities.values()}
    occurrences = 0
    report_findings: list[dict] = []
    for lock in sorted(per_lock):
        for name, version, ids in per_lock[lock]:
            occurrences += len(ids)
            print(f"[VULN] {lock.name}: {name}=={version}", file=sys.stderr)
            canonical_ids = sorted({identities[vuln_id][0] for vuln_id in ids})
            report_findings.append(
                {
                    "lock": lock.name,
                    "package": name,
                    "version": version,
                    "advisories": ids,
                    "canonical_vulnerabilities": canonical_ids,
                }
            )
            for vuln_id in ids:
                canonical, summary = identities[vuln_id]
                identity = f" -- {canonical}" if canonical != vuln_id else ""
                description = f" - {summary}" if summary else ""
                print(f"       {vuln_id}{identity}{description}", file=sys.stderr)
    report = {
        "scan_status": "vulnerable",
        "unique_cves": len(unique_vulnerabilities),
        "advisory_occurrences": occurrences,
        "affected_locks": len(per_lock),
        "locked_packages": len(all_pins),
        "findings": report_findings,
    }
    if not _write_json_report(args.json_output, report):
        return EXIT_ERROR
    print(
        f"[ERROR] {len(unique_vulnerabilities)} unique vulnerability(s), "
        f"{occurrences} advisory occurrence(s), {len(per_lock)} affected lock(s)",
        file=sys.stderr,
    )
    return EXIT_VULNERABLE


if __name__ == "__main__":
    raise SystemExit(main())

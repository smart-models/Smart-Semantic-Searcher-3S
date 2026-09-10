"""Backend-layer evidence for the console test book is declared, not assumed.

``frontend/tests/testbook/coverage-contract.json`` names the book cases this
suite owns. Every one of them must be carried by at least one collected test
marked ``@pytest.mark.book("<case>")``, every marker must name a real book case,
and every marked case must be listed in the contract -- so the contract, the
markers and the book cannot drift apart silently. The browser-side gate
(``frontend/tests/coverage-gate.test.mjs``) trusts this file for the
``middleware`` owner.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

FRONTEND_TESTS = Path(__file__).resolve().parents[1] / "frontend" / "tests"
TEST_BOOK = FRONTEND_TESTS / "testbook" / "test-book-e2e.md"
CONTRACT = FRONTEND_TESTS / "testbook" / "coverage-contract.json"

_HEADING = re.compile(r"^#{2,4} (\d+(?:\.\d+[a-z]?)*)\.?\s")
_ROW = re.compile(r"^\| ([A-Z][A-Za-z0-9]*(?:[-/][A-Za-z0-9]+)*)\s+\|")


def _expand_ids(value: str) -> list[str]:
    """Mirror ``expandIds`` in ``case-coverage.mjs`` (``RE4/5/7`` -> three ids)."""
    first, *rest = value.split("/")
    prefix = re.sub(r"\d+$", "", first)
    expanded = [first]
    for part in rest:
        if re.fullmatch(r"\d+", part):
            expanded.append(prefix + part)
        elif "-" in first and "-" not in part:
            expanded.append(first[: first.rfind("-") + 1] + part)
        else:
            expanded.append(part)
    return expanded


def book_case_keys(markdown: str) -> set[str]:
    """Return the qualified case keys of the book, as the Node ledger derives them."""
    section, fenced = "", False
    keys: set[str] = set()
    for line in markdown.splitlines():
        if line.startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        heading = _HEADING.match(line)
        if heading:
            section = heading.group(1)
        if not section or int(section.split(".")[0]) > 13:
            continue
        row = _ROW.match(line)
        if not row:
            continue
        raw = row.group(1)
        if not re.search(r"\d", raw) and not re.fullmatch(r"[A-Z]+-[A-Z]+", raw):
            continue
        for case_id in _expand_ids(raw):
            keys.add(
                case_id if case_id.startswith("TB-") else f"book:{section}/{case_id}"
            )
    return keys


def _marked_cases(session: pytest.Session) -> dict[str, list[str]]:
    marked: dict[str, list[str]] = {}
    for item in session.items:
        for marker in item.iter_markers("book"):
            for case_id in marker.args:
                marked.setdefault(case_id, []).append(item.nodeid)
    return marked


def test_book_parser_matches_the_ledger_shape():
    keys = book_case_keys(TEST_BOOK.read_text(encoding="utf-8"))
    assert len(keys) >= 500
    assert {"TB-HAR-001", "book:8.2/C1", "book:6.2/C1", "book:5/RE7"} <= keys
    assert len([key for key in keys if key.startswith("TB-")]) == 96


def _require_whole_suite(request) -> None:
    """Skip the suite-wide ownership check when only this file was collected."""
    targets = [Path(arg.split("::")[0]).name for arg in request.config.args]
    if targets and all(name == Path(__file__).name for name in targets):
        pytest.skip("ownership gate needs the whole middleware suite collected")


def test_every_owned_case_has_a_collected_marked_test(request):
    _require_whole_suite(request)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    owned = set(contract["owners"]["middleware"])
    marked = _marked_cases(request.session)
    missing = sorted(owned - marked.keys())
    assert not missing, f"contract owns cases with no @pytest.mark.book test: {missing}"


def test_every_marked_case_is_a_real_book_case_listed_in_the_contract(request):
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    owned = set(contract["owners"]["middleware"])
    keys = book_case_keys(TEST_BOOK.read_text(encoding="utf-8"))
    marked = _marked_cases(request.session)
    unknown = sorted(case for case in marked if case not in keys)
    assert not unknown, f"markers name cases the book does not define: {unknown}"
    unlisted = sorted(case for case in marked if case not in owned)
    assert not unlisted, (
        "marked cases must be listed under owners.middleware in "
        f"coverage-contract.json so the browser gate can credit them: {unlisted}"
    )

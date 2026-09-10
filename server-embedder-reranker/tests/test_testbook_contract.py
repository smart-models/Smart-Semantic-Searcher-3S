"""The cases the coverage contract assigns to this suite are really carried here.

Mirrors ``middleware-qdrant-manager/tests/test_testbook_contract.py`` for the
``embedder`` owner: every owned case must be declared by at least one
discovered test through ``@book_case``, and every declared case must be a real
book row listed in the contract.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from tests.book_cases import declared_cases

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_TESTS = REPO_ROOT / "middleware-qdrant-manager" / "frontend" / "tests"
TEST_BOOK = FRONTEND_TESTS / "testbook" / "test-book-e2e.md"
CONTRACT = FRONTEND_TESTS / "testbook" / "coverage-contract.json"

_HEADING = re.compile(r"^#{2,4} (\d+(?:\.\d+[a-z]?)*)\.?\s")
_ROW = re.compile(r"^\| ([A-Z][A-Za-z0-9]*(?:[-/][A-Za-z0-9]+)*)\s+\|")


def _tb_case_ids(markdown: str) -> set[str]:
    """Return the globally unique ``TB-*`` ids the book defines in sections 0-13."""
    section, fenced, found = "", False, set()
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
        if row and row.group(1).startswith("TB-"):
            found.add(row.group(1))
    return found


def _declared_by_suite() -> dict[str, list[str]]:
    declared: dict[str, list[str]] = {}
    # Same start directory as the documented `discover -s tests` run, so the
    # modules resolve to the ones already imported instead of a second copy.
    suite = unittest.defaultTestLoader.discover(
        str(Path(__file__).parent), pattern="test_*.py"
    )

    def walk(item):
        if isinstance(item, unittest.TestSuite):
            for child in item:
                walk(child)
            return
        for case_id in declared_cases(item):
            declared.setdefault(case_id, []).append(item.id())

    walk(suite)
    return declared


class TestBookContract(unittest.TestCase):
    """Owned cases, declared cases and the book must agree."""

    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        cls.owned = set(cls.contract["owners"]["embedder"])
        cls.book_ids = _tb_case_ids(TEST_BOOK.read_text(encoding="utf-8"))
        cls.declared = _declared_by_suite()

    def test_book_defines_the_ninety_six_tb_cases(self):
        self.assertEqual(len(self.book_ids), 96)

    def test_every_owned_case_is_declared_by_a_discovered_test(self):
        missing = sorted(self.owned - self.declared.keys())
        self.assertEqual(
            missing, [], f"contract owns cases no @book_case test declares: {missing}"
        )

    def test_every_declared_case_is_real_and_listed_in_the_contract(self):
        unknown = sorted(case for case in self.declared if case not in self.book_ids)
        self.assertEqual(
            unknown, [], f"declared cases the book does not define: {unknown}"
        )
        unlisted = sorted(case for case in self.declared if case not in self.owned)
        self.assertEqual(
            unlisted, [], f"declared cases missing from owners.embedder: {unlisted}"
        )

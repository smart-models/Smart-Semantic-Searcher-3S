"""Declare which console test-book cases a unittest proves.

The book (``middleware-qdrant-manager/frontend/tests/testbook/test-book-e2e.md``)
assigns a few Backend-layer rows to this service -- lifecycle cancellation,
shutdown drain and the inference admission budget. ``@book_case`` records the
claim on the test method itself so ``test_testbook_contract.py`` can discover
it without parsing source text, and so the coverage contract can credit the
``embedder`` owner only for cases that are really carried by a collected test.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TypeVar

_CASE_ID = re.compile(r"^TB-[A-Z]+-\d{3}$|^book:\d+(?:\.\d+[a-z]?)*/[A-Za-z0-9-]+$")
_ATTRIBUTE = "_book_cases"

T = TypeVar("T", bound=Callable[..., object])


def book_case(*case_ids: str) -> Callable[[T], T]:
    """Mark a test method as evidence for the given qualified book case ids."""
    for case_id in case_ids:
        if not _CASE_ID.match(case_id):
            raise ValueError(f"not a qualified test-book case id: {case_id!r}")

    def decorate(method: T) -> T:
        existing = tuple(getattr(method, _ATTRIBUTE, ()))
        setattr(method, _ATTRIBUTE, existing + case_ids)
        return method

    return decorate


def declared_cases(test) -> tuple[str, ...]:
    """Return the case ids a unittest ``TestCase`` instance declares, if any."""
    method = getattr(test, getattr(test, "_testMethodName", ""), None)
    return tuple(getattr(method, _ATTRIBUTE, ()))

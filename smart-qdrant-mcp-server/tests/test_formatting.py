"""Rendering and pagination envelopes are produced in exactly one place."""

import json

import pytest

from smart_qdrant_mcp.formatting import ResponseFormat, bullet, paginate, render


def _text(result) -> str:
    return "\n".join(block.text for block in result.content)


def test_json_format_returns_parsable_payload():
    out = render({"count": 3}, ResponseFormat.JSON, title="Count", rows=lambda p: [])
    assert json.loads(_text(out)) == {"count": 3}
    assert out.structured_content == {"count": 3}
    assert out.is_error is False


def test_markdown_format_starts_with_title():
    out = render(
        {"count": 3},
        ResponseFormat.MARKDOWN,
        title="Point count",
        rows=lambda p: [bullet("Count", p["count"])],
    )
    text = _text(out)
    assert text.splitlines()[0] == "# Point count"
    assert "- **Count**: 3" in text
    assert out.structured_content == {"count": 3}


def test_markdown_omits_empty_rows():
    out = render({}, ResponseFormat.MARKDOWN, title="Empty", rows=lambda p: [])
    assert _text(out).strip() == "# Empty"


def test_bullet_renders_none_as_dash():
    assert bullet("Vectors", None) == "- **Vectors**: —"


def test_paginate_reports_more_results_available():
    page = paginate([1, 2], total=10, offset=0, limit=2)
    assert page["total"] == 10
    assert page["count"] == 2
    assert page["has_more"] is True
    assert page["next_offset"] == 2


def test_paginate_reports_end_of_results():
    page = paginate([1], total=1, offset=0, limit=20)
    assert page["has_more"] is False
    assert page["next_offset"] is None


def test_paginate_uses_explicit_cursor_when_given():
    page = paginate([1, 2], total=None, offset="abc", limit=2, next_offset="def")
    assert page["offset"] == "abc"
    assert page["next_offset"] == "def"
    assert page["has_more"] is True


def test_paginate_without_total_infers_more_from_full_page():
    page = paginate([1, 2], total=None, offset=0, limit=2)
    assert page["total"] is None
    assert page["has_more"] is True


def test_paginate_without_total_infers_end_from_short_page():
    assert paginate([1], total=None, offset=0, limit=2)["has_more"] is False


def test_paginate_unknown_total_cursor_offset_without_next_offset_stops():
    """A non-integer offset gives the function nothing to advance itself with."""
    page = paginate(["a", "b"], total=None, offset="point-uuid-123", limit=2)
    assert page["has_more"] is False
    assert page["next_offset"] is None


def test_paginate_unknown_total_cursor_offset_with_explicit_next_offset():
    """A caller-supplied cursor must still be returned and marked as more."""
    page = paginate(
        ["a", "b"],
        total=None,
        offset="point-uuid-123",
        limit=2,
        next_offset="next-cursor",
    )
    assert page["next_offset"] == "next-cursor"
    assert page["has_more"] is True


@pytest.mark.parametrize("limit", [0, -1])
def test_paginate_non_positive_limit_reports_no_further_pages(limit):
    page = paginate([], total=None, offset=0, limit=limit)
    assert page["has_more"] is False
    assert page["next_offset"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(items=[1, 2], total=10, offset=0, limit=2),
        dict(items=[1], total=1, offset=0, limit=20),
        dict(items=["a", "b"], total=None, offset="cursor", limit=2, next_offset="n"),
        dict(items=[1, 2], total=None, offset=0, limit=2),
        dict(items=[1], total=None, offset=0, limit=2),
        dict(items=["a", "b"], total=None, offset="cursor", limit=2),
        dict(items=[], total=None, offset=0, limit=0),
        dict(items=[], total=None, offset=0, limit=-1),
    ],
)
def test_paginate_has_more_matches_next_offset_presence(kwargs):
    """has_more is true if and only if next_offset is not None, on every branch."""
    items = kwargs.pop("items")
    page = paginate(items, **kwargs)
    assert page["has_more"] == (page["next_offset"] is not None)


def test_bullet_renders_falsy_but_real_values_verbatim():
    """Zero, False and an empty list are real data, not absence of it."""
    assert bullet("Count", 0) == "- **Count**: 0"
    assert bullet("Enabled", False) == "- **Enabled**: False"
    assert bullet("Tags", []) == "- **Tags**: []"


def test_bullet_renders_only_none_and_empty_string_as_dash():
    assert bullet("Vectors", None) == "- **Vectors**: —"
    assert bullet("Name", "") == "- **Name**: —"

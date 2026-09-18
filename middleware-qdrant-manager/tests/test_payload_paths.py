"""Tests for payload-path authorization before Qdrant sees a delete request."""

import pytest

from qdrant_middleware.api.routers import points
from qdrant_middleware.services.payload_paths import payload_path_root
from qdrant_middleware.services.payload_paths import (
    canonical_payload_index_field,
    qdrant_filter_index_path,
    qdrant_payload_selector,
)


@pytest.mark.parametrize(
    "path",
    [
        "a.b",
        "items[].label",
        '"a.b"',
        '"items[].label"',
        '"order items"[].label',
        "orders[].items[0].sku",
    ],
)
def test_filter_index_paths_preserve_literal_and_traversal_identity(path):
    assert qdrant_filter_index_path(path) == path
    assert canonical_payload_index_field(path) == path


@pytest.mark.parametrize("path", ["", '"unterminated', "a..b", "a[wrong]", "a."])
def test_invalid_filter_index_path_fails_without_literal_fallback(path):
    with pytest.raises(ValueError):
        qdrant_filter_index_path(path)


def test_selectors_stay_literal_and_space_indexes_stay_canonical():
    assert qdrant_payload_selector(["a.b", "items[].label"]) == [
        '"a.b"',
        '"items[].label"',
    ]
    assert canonical_payload_index_field('"Header 1"') == "Header 1"
    assert qdrant_filter_index_path("Header 1") == '"Header 1"'


@pytest.mark.parametrize(
    ("path", "root"),
    [
        ("text", "text"),
        ('"text"', "text"),
        ("text[0]", "text"),
        ("text[].value", "text"),
        ('"source_id".nested', "source_id"),
        ("metadata.text", "metadata"),
        ('"metadata.text"', "metadata.text"),
    ],
)
def test_payload_path_root_decodes_the_first_addressed_field(path, root):
    """Protection compares Qdrant's decoded root rather than raw text."""
    assert payload_path_root(path) == root


@pytest.mark.parametrize("path", ["", '"text', ".text", "text[", "text..value"])
def test_payload_path_root_rejects_invalid_syntax(path):
    """Malformed paths fail before a payload delete can reach Qdrant."""
    with pytest.raises(ValueError):
        payload_path_root(path)


@pytest.mark.parametrize("selector", ["²", "١", "１２", "1²"])
def test_non_ascii_array_selectors_fail_before_qdrant(selector):
    path = f"metadata.items[{selector}].value"
    with pytest.raises(ValueError, match="invalid array selector"):
        qdrant_filter_index_path(path)
    with pytest.raises(ValueError, match="invalid array selector"):
        points.PayloadKeysDeleteRequest(keys=[path])


@pytest.mark.parametrize("selector", ["", "0", "12", "001"])
def test_ascii_array_selectors_remain_valid(selector):
    path = f"metadata.items[{selector}].value"
    assert qdrant_filter_index_path(path) == path
    assert points.PayloadKeysDeleteRequest(keys=[path]).keys == [path]


def test_unicode_digits_remain_valid_inside_literal_field_names():
    path = '"metadata[²]"'
    assert payload_path_root(path) == "metadata[²]"
    assert qdrant_filter_index_path(path) == '"metadata[\\u00b2]"'


@pytest.mark.parametrize("path", ["text", '"text"', "text[0]", "text[].value"])
def test_protected_text_paths_are_rejected_by_the_shared_route_guard(path):
    """Quoted and nested paths cannot bypass protected payload ownership."""
    with pytest.raises(ValueError, match="Cannot delete protected"):
        points._validate_payload_delete_keys([path])


def test_delete_key_request_keeps_an_unrelated_nested_path():
    """A nested non-protected payload field remains deletable."""
    request = points.PayloadKeysDeleteRequest(keys=["metadata.text"])
    assert request.keys == ["metadata.text"]

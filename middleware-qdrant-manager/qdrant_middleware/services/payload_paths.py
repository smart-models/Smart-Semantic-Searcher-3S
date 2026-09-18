"""Shared payload field normalization and Qdrant path quoting helpers."""

import json
import re


_BARE_SEGMENT = re.compile(r"[^.\[\]\"\s]+")


def payload_path_root(path: str) -> str:
    """Return the decoded first field addressed by a Qdrant payload path."""
    if not path:
        raise ValueError("payload path cannot be blank")

    index = 0
    decoder = json.JSONDecoder()
    if path.startswith('"'):
        try:
            root, index = decoder.raw_decode(path)
        except json.JSONDecodeError as exc:
            raise ValueError("payload path has an invalid quoted field") from exc
        if not isinstance(root, str) or not root:
            raise ValueError("payload path field cannot be blank")
    else:
        match = _BARE_SEGMENT.match(path)
        if match is None:
            raise ValueError("payload path has an invalid field")
        root = match.group()
        index = match.end()

    while index < len(path):
        if path[index] == "[":
            closing = path.find("]", index + 1)
            selector = path[index + 1 : closing] if closing >= 0 else None
            if closing < 0 or (
                selector != "" and not (selector.isascii() and selector.isdigit())
            ):
                raise ValueError("payload path has an invalid array selector")
            index = closing + 1
            continue
        if path[index] != ".":
            raise ValueError("payload path has invalid syntax")
        index += 1
        if index == len(path):
            raise ValueError("payload path cannot end with a dot")
        if path[index] == '"':
            try:
                segment, segment_end = decoder.raw_decode(path[index:])
            except json.JSONDecodeError as exc:
                raise ValueError("payload path has an invalid quoted field") from exc
            if not isinstance(segment, str) or not segment:
                raise ValueError("payload path field cannot be blank")
            index += segment_end
            continue
        match = _BARE_SEGMENT.match(path, index)
        if match is None:
            raise ValueError("payload path has an invalid field")
        index = match.end()
    return root


def canonical_payload_field(field_name: str) -> str:
    """Return a normalized payload field name without surrounding quotes."""
    return field_name.strip().strip('"')


def qdrant_payload_path(field_name: str) -> str:
    """Return a Qdrant path that addresses one literal top-level field."""
    stripped = canonical_payload_field(field_name)
    return stripped if _BARE_SEGMENT.fullmatch(stripped) else json.dumps(stripped)


def qdrant_payload_selector(fields: list[str]) -> list[str]:
    """Quote each field name in a payload selector for Qdrant."""
    return [qdrant_payload_path(field) for field in fields]


def qdrant_filter_index_path(field_name: str) -> str:
    """Preserve traversal while quoting explicit literals and simple space names."""
    path = field_name.strip()
    if path and not any(char in path for char in '.[]"'):
        return path if _BARE_SEGMENT.fullmatch(path) else json.dumps(path)
    payload_path_root(path)
    if path.startswith('"'):
        literal, end = json.JSONDecoder().raw_decode(path)
        if end == len(path):
            return literal if _BARE_SEGMENT.fullmatch(literal) else json.dumps(literal)
    return path


def canonical_payload_index_field(field_name: str) -> str:
    """Keep literal/path identities distinct without changing space-name display."""
    path = qdrant_filter_index_path(field_name)
    if path.startswith('"'):
        literal, end = json.JSONDecoder().raw_decode(path)
        if end == len(path) and not any(char in literal for char in '.[]"'):
            return literal
    return path

"""Validated conversion from request payload filters to Qdrant filter models."""

import math
import uuid
from numbers import Real
from typing import Any

from pydantic import ValidationError
from qdrant_client import models

from qdrant_middleware.services.payload_paths import (
    qdrant_filter_index_path,
)

LOGICAL_FILTER_KEYS = frozenset({"must", "should", "must_not", "min_should"})
FIELD_CONDITION_KEYS = frozenset(
    {
        "match",
        "range",
        "geo_radius",
        "geo_bounding_box",
        "geo_polygon",
        "values_count",
        "is_empty",
        "is_null",
    }
)
RANGE_KEYS = frozenset({"gt", "gte", "lt", "lte"})


class FilterBuildContext:
    """Mutable limits and allowlist state for parsing one filter payload."""

    def __init__(self, settings: Any = None) -> None:
        """Initialize parser limits from settings or defaults."""
        self.allowed_fields = _allowed_fields(settings)
        self.max_depth = getattr(settings, "query_filter_max_depth", 20)
        self.max_conditions = getattr(settings, "query_filter_max_conditions", 100)
        self.condition_count = 0

    def count_condition(self) -> None:
        """Record one parsed condition and reject requests over the limit."""
        self.condition_count += 1
        if self.condition_count > self.max_conditions:
            raise ValueError(
                f"filters exceed maximum condition count {self.max_conditions}"
            )

    def check_depth(self, depth: int) -> None:
        """Reject requests deeper than the configured nesting limit."""
        if depth > self.max_depth:
            raise ValueError(f"filters exceed maximum depth {self.max_depth}")


def build_qdrant_filter(
    filters: dict[str, Any] | None, *, settings: Any = None
) -> models.Filter | None:
    """Convert validated request filters into a Qdrant Filter."""
    if not filters:
        return None
    if not isinstance(filters, dict):
        raise ValueError("filters must be an object")

    context = FilterBuildContext(settings)
    if _is_advanced_filter(filters):
        return _parse_filter(filters, context, depth=1, prefix="")
    return _parse_legacy_filter(filters, context)


def _is_advanced_filter(filters: dict[str, Any]) -> bool:
    """Return whether ``filters`` uses the logical (must/should/...) syntax."""
    logical_keys = set(filters) & LOGICAL_FILTER_KEYS
    if not logical_keys:
        return False
    if set(filters) - LOGICAL_FILTER_KEYS:
        raise ValueError("filters cannot mix logical keys with legacy field filters")
    return True


def _parse_legacy_filter(
    filters: dict[str, Any], context: FilterBuildContext
) -> models.Filter:
    """Parse the legacy flat ``{field: value-or-list}`` filter syntax."""
    conditions: list[models.FieldCondition] = []
    for key, value in filters.items():
        context.count_condition()
        _validate_field_allowed(key, context.allowed_fields)
        qdrant_key = qdrant_filter_index_path(key)
        if isinstance(value, list):
            if not value:
                raise ValueError(f"legacy filter '{key}' list must not be empty")
            _validate_match_list(value, f"legacy filter '{key}'")
            match = models.MatchAny(any=value)
        else:
            match = models.MatchValue(value=value)
        conditions.append(models.FieldCondition(key=qdrant_key, match=match))

    return models.Filter(must=conditions)


def _parse_filter(
    filters: dict[str, Any],
    context: FilterBuildContext,
    *,
    depth: int,
    prefix: str,
) -> models.Filter:
    """Parse one logical filter object (must/should/must_not/min_should)."""
    context.check_depth(depth)
    kwargs: dict[str, Any] = {}
    for key in ("must", "should", "must_not"):
        if key not in filters:
            continue
        value = filters[key]
        if not isinstance(value, list):
            raise ValueError(f"filters.{key} must be a list")
        if not value:
            raise ValueError(f"filters.{key} must be a non-empty list")
        kwargs[key] = [
            _parse_condition(condition, context, depth=depth + 1, prefix=prefix)
            for condition in value
        ]
    if "min_should" in filters:
        kwargs["min_should"] = _parse_min_should(
            filters["min_should"], context, depth=depth + 1, prefix=prefix
        )
    return models.Filter(**kwargs)


def _parse_condition(
    obj: Any, context: FilterBuildContext, *, depth: int, prefix: str
) -> Any:
    """Parse one atomic, nested, or recursive-filter condition object."""
    context.check_depth(depth)
    context.count_condition()
    if not isinstance(obj, dict):
        raise ValueError("filter condition must be an object")
    logical_keys = set(obj) & LOGICAL_FILTER_KEYS
    if logical_keys:
        if set(obj) - LOGICAL_FILTER_KEYS:
            raise ValueError(
                "recursive filter condition cannot mix logical and atomic keys"
            )
        return _parse_filter(obj, context, depth=depth + 1, prefix=prefix)
    if len(obj) == 1 and ("is_empty" in obj or "is_null" in obj):
        # Qdrant's native {"is_empty": {"key": k}} shape, rewritten to the flat
        # field condition so it passes the same allowlist and key checks.
        [(operator, field)] = obj.items()
        if not isinstance(field, dict) or set(field) != {"key"}:
            raise ValueError(f"{operator} must be true or an object with only key")
        obj = {"key": field["key"], operator: True}
    condition_kinds = [
        name for name in ("key", "has_id", "has_vector", "nested") if name in obj
    ]
    if len(condition_kinds) != 1:
        raise ValueError("filter condition must contain exactly one condition kind")
    kind = condition_kinds[0]
    if kind == "key":
        return _parse_field_condition(obj, context, prefix=prefix)
    if set(obj) != {kind}:
        raise ValueError(f"unsupported condition keys: {sorted(set(obj) - {kind})}")
    if kind == "has_id":
        value = obj["has_id"]
        if not isinstance(value, list) or not value:
            raise ValueError("has_id must be a non-empty list")
        return models.HasIdCondition(has_id=_parse_point_ids(value))
    if kind == "has_vector":
        value = obj["has_vector"]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("has_vector must be a non-empty string")
        return models.HasVectorCondition(has_vector=value)
    return _parse_nested_condition(
        obj["nested"], context, depth=depth + 1, prefix=prefix
    )


def _parse_min_should(
    obj: Any, context: FilterBuildContext, *, depth: int, prefix: str
) -> models.MinShould:
    """Parse a ``min_should`` clause: N of the listed conditions must match."""
    context.check_depth(depth)
    context.count_condition()
    if not isinstance(obj, dict):
        raise ValueError("min_should must be an object")
    if set(obj) != {"conditions", "min_count"}:
        raise ValueError("min_should must contain conditions and min_count")
    conditions = obj["conditions"]
    min_count = obj["min_count"]
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("min_should.conditions must be a non-empty list")
    if isinstance(min_count, bool) or not isinstance(min_count, int) or min_count < 1:
        raise ValueError("min_should.min_count must be a positive integer")
    if min_count > len(conditions):
        raise ValueError("min_should.min_count cannot exceed condition count")
    return models.MinShould(
        conditions=[
            _parse_condition(condition, context, depth=depth + 1, prefix=prefix)
            for condition in conditions
        ],
        min_count=min_count,
    )


def _parse_nested_condition(
    obj: Any, context: FilterBuildContext, *, depth: int, prefix: str
) -> models.NestedCondition:
    """Parse a ``nested`` condition targeting an array-of-objects payload field."""
    context.check_depth(depth)
    if not isinstance(obj, dict):
        raise ValueError("nested must be an object")
    if set(obj) != {"key", "filter"}:
        raise ValueError("nested must contain key and filter")
    key = obj["key"]
    if not isinstance(key, str) or not key.strip():
        raise ValueError("nested.key must be a non-empty string")
    full_key = f"{prefix}{key}"
    _validate_nested_root_allowed(full_key, context.allowed_fields)
    nested_filter = obj["filter"]
    if not isinstance(nested_filter, dict):
        raise ValueError("nested.filter must be an object")
    if not _is_advanced_filter(nested_filter):
        raise ValueError("nested.filter must use advanced logical filter syntax")
    return models.NestedCondition(
        nested=models.Nested(
            key=qdrant_filter_index_path(key),
            filter=_parse_filter(
                nested_filter,
                context,
                depth=depth + 1,
                prefix=f"{full_key}[].",
            ),
        )
    )


def _parse_field_condition(
    obj: dict[str, Any], context: FilterBuildContext, *, prefix: str
) -> models.FieldCondition:
    """Parse a keyed field condition carrying exactly one supported operator."""
    key = obj["key"]
    if not isinstance(key, str) or not key.strip():
        raise ValueError("field condition key must be a non-empty string")
    _validate_field_allowed(f"{prefix}{key}", context.allowed_fields)

    allowed_keys = {"key"} | FIELD_CONDITION_KEYS
    bad_keys = set(obj) - allowed_keys
    if bad_keys:
        raise ValueError(f"unsupported field condition keys: {sorted(bad_keys)}")
    operators = [name for name in FIELD_CONDITION_KEYS if name in obj]
    if len(operators) != 1:
        raise ValueError("field condition must contain exactly one supported operator")

    operator = operators[0]
    return models.FieldCondition(
        key=qdrant_filter_index_path(key),
        **_field_condition_value(operator, obj[operator]),
    )


def _field_condition_value(operator: str, value: Any) -> dict[str, Any]:
    """Parse one supported Qdrant field-condition operator payload."""
    if operator == "match":
        return {"match": _parse_match(value)}
    if operator == "range":
        return {"range": _parse_range(value)}
    if operator == "values_count":
        return {"values_count": _parse_values_count(value)}
    if operator in {"is_empty", "is_null"}:
        if value is not True:
            raise ValueError(f"{operator} must be true")
        return {operator: True}
    if operator == "geo_radius":
        return {"geo_radius": _parse_geo_radius(value)}
    if operator == "geo_bounding_box":
        return {"geo_bounding_box": _parse_geo_bounding_box(value)}
    if operator == "geo_polygon":
        return {"geo_polygon": _parse_geo_polygon(value)}
    raise ValueError(f"unsupported field condition operator: {operator}")


def _parse_match(obj: Any) -> Any:
    """Parse a ``match`` operator into the corresponding Qdrant match model."""
    if not isinstance(obj, dict):
        raise ValueError("match must be an object")
    match_keys = set(obj)
    if len(match_keys) != 1:
        raise ValueError("match must contain exactly one operator")
    operator = next(iter(match_keys))
    value = obj[operator]
    if operator == "value":
        if isinstance(value, float) or not isinstance(value, (str, int, bool)):
            raise ValueError("match.value must be a keyword, integer, or boolean")
        return models.MatchValue(value=value)
    if operator == "any":
        _validate_match_list(value, "match.any")
        return models.MatchAny(any=value)
    if operator == "except":
        _validate_match_list(value, "match.except")
        return models.MatchExcept(**{"except": value})
    if operator == "text":
        _validate_match_text(value, "match.text")
        return models.MatchText(text=value)
    if operator == "text_any":
        _validate_match_text(value, "match.text_any")
        return models.MatchTextAny(text_any=value)
    if operator == "phrase":
        _validate_match_text(value, "match.phrase")
        return models.MatchPhrase(phrase=value)
    raise ValueError(f"unsupported match operator: {operator}")


def _parse_range(obj: Any) -> models.Range | models.DatetimeRange:
    """Parse a ``range`` operator, choosing numeric or datetime bounds."""
    if not isinstance(obj, dict):
        raise ValueError("range must be an object")
    bad = set(obj) - RANGE_KEYS
    if bad:
        raise ValueError(f"unsupported range keys: {sorted(bad)}")
    if not obj:
        raise ValueError("range must contain at least one bound")
    values = [value for value in obj.values() if value is not None]
    if not values:
        raise ValueError("range must contain at least one non-null bound")
    all_numeric = all(_is_numeric(value) for value in values)
    all_datetime = all(isinstance(value, str) for value in values)
    if all_numeric:
        return models.Range(**obj)
    if all_datetime:
        return models.DatetimeRange(**obj)
    raise ValueError("range bounds must be all numeric or all ISO datetime strings")


def _parse_values_count(obj: Any) -> models.ValuesCount:
    """Parse a ``values_count`` operator bounding array payload lengths."""
    if not isinstance(obj, dict):
        raise ValueError("values_count must be an object")
    bad = set(obj) - RANGE_KEYS
    if bad:
        raise ValueError(f"unsupported values_count keys: {sorted(bad)}")
    if not obj:
        raise ValueError("values_count must contain at least one bound")
    if any(value is None for value in obj.values()):
        raise ValueError("values_count bounds must be non-null")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in obj.values()
    ):
        raise ValueError("values_count bounds must be non-negative integers")
    return models.ValuesCount(**obj)


def _validate_match_list(value: Any, name: str) -> None:
    """Require a non-empty homogeneous keyword or integer match list."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    if any(
        isinstance(item, bool) or not isinstance(item, (str, int)) for item in value
    ):
        raise ValueError(f"{name} values must be all keywords or all integers")
    value_types = {str if isinstance(item, str) else int for item in value}
    if len(value_types) != 1:
        raise ValueError(f"{name} values must be all keywords or all integers")


def _validate_match_text(value: Any, name: str) -> None:
    """Require non-blank text for token and phrase matching."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _parse_point_ids(values: list[Any]) -> list[int | str]:
    """Normalize JSON-safe decimal uint64 strings and validate UUID IDs."""
    parsed: list[int | str] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("has_id values must be uint64 integers or UUIDs")
        if isinstance(value, int):
            number = value
        elif isinstance(value, str) and value.isdecimal():
            number = int(value)
        elif isinstance(value, str):
            try:
                parsed.append(str(uuid.UUID(value)))
            except ValueError as exc:
                raise ValueError(
                    "has_id values must be uint64 integers or UUIDs"
                ) from exc
            continue
        else:
            raise ValueError("has_id values must be uint64 integers or UUIDs")
        if number < 0 or number > 2**64 - 1:
            raise ValueError("has_id numeric values must fit unsigned 64-bit range")
        parsed.append(number)
    return parsed


def _parse_geo_point(obj: Any, name: str) -> models.GeoPoint:
    """Validate one unprojected WGS84-like lon/lat pair."""
    if not isinstance(obj, dict):
        raise ValueError(f"{name} must be an object")
    if set(obj) != {"lon", "lat"}:
        raise ValueError(f"{name} must contain exactly lon and lat")
    lon = obj["lon"]
    lat = obj["lat"]
    if not _is_numeric(lon) or not _is_numeric(lat):
        raise ValueError(f"{name} longitude and latitude must be numeric")
    if not -180 <= lon <= 180:
        raise ValueError(f"{name} longitude must be between -180 and 180")
    if not -90 <= lat <= 90:
        raise ValueError(f"{name} latitude must be between -90 and 90")
    return models.GeoPoint(lon=float(lon), lat=float(lat))


def _parse_geo_radius(obj: Any) -> models.GeoRadius:
    """Validate a geo radius condition before model construction."""
    if not isinstance(obj, dict):
        raise ValueError("geo_radius must be an object")
    if set(obj) != {"center", "radius"}:
        raise ValueError("geo_radius must contain exactly center and radius")
    radius = obj["radius"]
    if not _is_numeric(radius) or radius <= 0:
        raise ValueError("geo_radius radius must be a positive number")
    return models.GeoRadius(
        center=_parse_geo_point(obj["center"], "geo_radius.center"),
        radius=float(radius),
    )


def _parse_geo_bounding_box(obj: Any) -> models.GeoBoundingBox:
    """Validate top-left and bottom-right coordinates."""
    if not isinstance(obj, dict):
        raise ValueError("geo_bounding_box must be an object")
    if set(obj) != {"top_left", "bottom_right"}:
        raise ValueError(
            "geo_bounding_box must contain exactly top_left and bottom_right"
        )
    top_left = _parse_geo_point(obj["top_left"], "geo_bounding_box.top_left")
    bottom_right = _parse_geo_point(
        obj["bottom_right"], "geo_bounding_box.bottom_right"
    )
    if top_left.lat < bottom_right.lat:
        raise ValueError(
            "geo_bounding_box top-left latitude must be north of bottom-right"
        )
    return models.GeoBoundingBox(top_left=top_left, bottom_right=bottom_right)


def _parse_geo_line_string(obj: Any, name: str) -> models.GeoLineString:
    """Validate one closed polygon ring."""
    if not isinstance(obj, dict) or set(obj) != {"points"}:
        raise ValueError(f"{name} must contain exactly points")
    raw_points = obj["points"]
    if not isinstance(raw_points, list) or len(raw_points) < 4:
        raise ValueError(f"{name}.points must contain at least 4 points")
    points = [
        _parse_geo_point(point, f"{name}.points[{index}]")
        for index, point in enumerate(raw_points)
    ]
    first = points[0]
    last = points[-1]
    if first.lon != last.lon or first.lat != last.lat:
        raise ValueError(f"{name} must be closed (first point equals last point)")
    if len({(point.lon, point.lat) for point in points[:-1]}) < 3:
        raise ValueError(f"{name} needs at least 3 distinct points")
    return models.GeoLineString(points=points)


def _parse_geo_polygon(obj: Any) -> models.GeoPolygon:
    """Validate an exterior ring and optional interior holes."""
    if not isinstance(obj, dict):
        raise ValueError("geo_polygon must be an object")
    if "exterior" not in obj or set(obj) - {"exterior", "interiors"}:
        raise ValueError("geo_polygon must contain exterior and optional interiors")
    raw_interiors = obj.get("interiors", [])
    if not isinstance(raw_interiors, list):
        raise ValueError("geo_polygon.interiors must be a list")
    try:
        return models.GeoPolygon(
            exterior=_parse_geo_line_string(obj["exterior"], "geo_polygon.exterior"),
            interiors=[
                _parse_geo_line_string(ring, f"geo_polygon.interiors[{index}]")
                for index, ring in enumerate(raw_interiors)
            ]
            or None,
        )
    except ValidationError as exc:
        raise ValueError(f"invalid geo_polygon: {exc}") from exc


def _is_numeric(value: Any) -> bool:
    """Return True for finite real numbers, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _allowed_fields(settings: Any) -> frozenset[str]:
    """Return the configured filter-field allowlist (empty means allow all)."""
    if settings is None:
        return frozenset()
    return getattr(settings, "query_filter_field_allowlist", frozenset())


def _validate_field_allowed(field_name: str, allowed_fields: frozenset[str]) -> None:
    """Reject a filter field outside the configured allowlist (if any)."""
    if allowed_fields and field_name not in allowed_fields:
        raise ValueError(
            f"filters may only use configured fields {sorted(allowed_fields)}; "
            f"not allowed: {field_name}"
        )


def _validate_nested_root_allowed(
    field_name: str, allowed_fields: frozenset[str]
) -> None:
    """Allow a nested root when at least one configured child path is allowed."""
    if not allowed_fields:
        return
    child_prefix = f"{field_name}[]."
    if field_name in allowed_fields or any(
        allowed.startswith(child_prefix) for allowed in allowed_fields
    ):
        return
    raise ValueError(
        f"filters may only use configured fields {sorted(allowed_fields)}; "
        f"not allowed: {field_name}"
    )

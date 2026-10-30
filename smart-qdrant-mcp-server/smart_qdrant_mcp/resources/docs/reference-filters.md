# Filter reference

Every tool that accepts `filters` — `qdrant_search`, `qdrant_scroll_points`,
`qdrant_count_points`, `qdrant_facet_points`, `qdrant_set_payload`,
`qdrant_delete_payload_keys`, `qdrant_clear_payload`, and
`qdrant_delete_points` — parses that object through the same middleware
filter grammar. This document is that grammar, not Qdrant's in general: it
covers exactly what this middleware accepts, including the limits it
enforces before anything reaches Qdrant.

## Two syntaxes, never mixed

**Legacy (flat)**: `{"field": value}` or `{"field": [v1, v2, ...]}` — each
key is an implicit exact-match (or match-any, for a list) condition, and all
keys are ANDed together. **Advanced (logical)**: an object built from
`must`, `should`, `must_not`, and `min_should`. A single filter object
cannot mix a logical key with a legacy field key — pick one syntax per
object.

## Advanced condition shapes

Filter and payload-index field names preserve traversal: `a.b` addresses a nested
object and `items[].label` projects an array. Explicitly quote a literal key with
punctuation (JSON field value `"\"a.b\""`). A name containing spaces, such as
`Header 1`, is quoted automatically. Literal and traversal indexes are distinct;
an old accidentally quoted index must not be reused as a traversal index.

Each condition inside `must`/`should`/`must_not` is exactly one of: a field
condition (`{"key": ..., <operator>: ...}`), `{"has_id": [...]}`,
`{"has_vector": "dense"}`, `{"nested": {"key": ..., "filter": {...}}}` for
array-of-objects payload fields, or a further logical filter nested inside
(recursion). A field condition takes exactly one operator: `match` (with
`value`, `any`, `except`, `text`, `text_any`, or `phrase`), `range` (`gt`,
`gte`, `lt`, `lte` — all numeric or all ISO datetime strings, never mixed),
`values_count` (bounds on an array payload field's length), `is_empty`,
`is_null`, `geo_radius`, `geo_bounding_box`, or `geo_polygon`. `is_empty` and
`is_null` take `true` (`{"key": "notes", "is_empty": true}`); Qdrant's native
`{"is_empty": {"key": "notes"}}` is accepted as the same condition. `min_should`
takes `{"conditions": [...], "min_count": N}` — matches when at least `N` of
the listed conditions match.

## Depth and condition limits

The middleware enforces two configured limits before any filter reaches
Qdrant: `QUERY_FILTER_MAX_DEPTH` (default 20, 1–100) caps how deeply
conditions may nest inside one another, and `QUERY_FILTER_MAX_CONDITIONS`
(default 100, 1–1000) caps the total number of conditions counted across the
whole filter. Exceeding either is rejected as an invalid request before any
Qdrant call is made. If a `QUERY_FILTER_FIELD_ALLOWLIST` is configured, a
filter naming a field outside it is rejected the same way; an empty
allowlist (the default) permits any field.

## Three worked filters

Single keyword: `{"must": [{"key": "lang", "match": {"value": "it"}}]}`.

Date range plus keyword: `{"must": [{"key": "created_at", "range":
{"gte": "2024-01-01T00:00:00Z"}}, {"key": "lang", "match": {"value":
"it"}}]}`.

Exclusion: `{"must_not": [{"key": "status", "match": {"value":
"draft"}}]}`.

## Every filtered key needs an index

The middleware creates every collection — and re-aligns older ones at
startup — in Qdrant strict mode with unindexed filtering refused for reads
and writes. A filter on a key without a fitting index is never scanned: the
search, count, scroll, facet, or filter-addressed payload write or delete
fails with `Error: invalid request — '<key>' has no payload index that can
serve this filter (it needs type <types>)`. Two causes produce it: the key has no index (create
one with `qdrant_create_payload_index`), or it has one of another type, so
the filter's value type is wrong for it — an integer `match` against a
`keyword` index, or a `range` against one. `qdrant_list_payload_indexes`
tells the two apart. Only conditions that need no payload index — `has_id`,
`has_vector` — work on any collection. See the upstream
[indexing contract](https://qdrant.tech/documentation/manage-data/indexing/#block-queries-that-filter-on-unindexed-fields).

## Related tools

`qdrant_search`, `qdrant_scroll_points`, `qdrant_count_points`,
`qdrant_facet_points`, `qdrant_delete_points`.

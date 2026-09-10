"""Tool-to-endpoint coverage map.

This is data, not documentation: the coverage test reads it and compares it
against the middleware's registered routes, so an endpoint added upstream
without a tool here fails the suite.
"""

from __future__ import annotations

TOOL_ROUTES: dict[str, tuple[tuple[str, str], ...]] = {
    # query
    "qdrant_search": (("POST", "/query/search"),),
    # collections
    "qdrant_list_collections": (("GET", "/collections"),),
    "qdrant_get_collection": (("GET", "/collections/{collection_name}"),),
    "qdrant_verify_collection": (("POST", "/collections/verify"),),
    "qdrant_create_collection": (("POST", "/collections"),),
    "qdrant_update_schema_toggles": (
        ("PUT", "/collections/{collection_name}/schema-toggles"),
        ("PATCH", "/collections/{collection_name}/schema-toggles"),
    ),
    "qdrant_delete_collection": (("DELETE", "/collections/{collection_name}"),),
    # payload indexes
    "qdrant_list_payload_keys": (
        ("GET", "/collections/{collection_name}/payload-keys"),
    ),
    "qdrant_list_payload_indexes": (
        ("GET", "/collections/{collection_name}/payload-indexes"),
    ),
    "qdrant_create_payload_index": (
        ("POST", "/collections/{collection_name}/payload-indexes"),
    ),
    "qdrant_delete_payload_index": (
        ("DELETE", "/collections/{collection_name}/payload-indexes/{field_name}"),
    ),
    # reindex
    "qdrant_reindex_preflight": (
        ("POST", "/collections/{collection_name}/reindex/preflight"),
    ),
    "qdrant_start_reindex": (("POST", "/collections/{collection_name}/reindex"),),
    "qdrant_get_reindex_status": (("GET", "/collections/{collection_name}/reindex"),),
    "qdrant_control_reindex": (
        ("POST", "/collections/{collection_name}/reindex/pause"),
        ("POST", "/collections/{collection_name}/reindex/resume"),
        ("POST", "/collections/{collection_name}/reindex/cancel"),
    ),
    # ingest
    "qdrant_ingest_file": (("POST", "/points/ingest"),),
    "qdrant_ingest_chunks": (("POST", "/points/ingest-chunks"),),
    # export
    "qdrant_export_collection": (("POST", "/points/export"),),
    # point reads
    "qdrant_read_points": (
        ("GET", "/points/{point_id}"),
        ("POST", "/points/retrieve"),
    ),
    "qdrant_scroll_points": (("POST", "/points/scroll"),),
    "qdrant_count_points": (("POST", "/points/count"),),
    "qdrant_facet_points": (("POST", "/points/facet"),),
    # point writes
    "qdrant_upsert_points": (("POST", "/points"), ("POST", "/points/batch")),
    "qdrant_delete_points": (
        ("DELETE", "/points/{point_id}"),
        ("DELETE", "/points"),
        ("POST", "/points/delete-by-filter"),
    ),
    # payload writes
    "qdrant_set_payload": (
        ("PATCH", "/points/{point_id}/payload"),
        ("PUT", "/points/{point_id}/payload"),
        ("PATCH", "/points/payload"),
    ),
    "qdrant_delete_payload_keys": (
        ("POST", "/points/{point_id}/payload/delete-keys"),
        ("POST", "/points/payload/delete-keys"),
    ),
    "qdrant_clear_payload": (
        ("POST", "/points/{point_id}/payload/clear"),
        ("POST", "/points/payload/clear"),
    ),
    # monitoring
    "qdrant_check_health": (
        ("GET", "/health"),
        ("GET", "/readyz"),
        ("GET", "/livez"),
        ("GET", "/stats"),
    ),
}

# Prometheus text output is a resource (qhm://metrics), never a tool: an agent
# has no use for a scrape payload in its context window.
RESOURCE_ONLY_ROUTES: frozenset[tuple[str, str]] = frozenset({("GET", "/metrics")})

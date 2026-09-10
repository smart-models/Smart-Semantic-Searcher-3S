"""Pin and review every direct Qdrant call outside the repository module."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path


_SOURCE_ROOT = Path(__file__).parents[1] / "qdrant_middleware"
_QDRANT_METHODS = {
    "collection_exists",
    "count",
    "create_collection",
    "create_payload_index",
    "create_vector_name",
    "delete",
    "delete_collection",
    "delete_payload",
    "delete_payload_index",
    "delete_vector_name",
    "delete_vectors",
    "facet",
    "get_collection",
    "get_collections",
    "info",
    "overwrite_payload",
    "query_points",
    "retrieve",
    "scroll",
    "set_payload",
    "update_collection",
    "update_vectors",
    "upsert",
}
_MUTATING_METHODS = {
    "create_collection",
    "create_payload_index",
    "create_vector_name",
    "delete",
    "delete_collection",
    "delete_payload",
    "delete_payload_index",
    "delete_vector_name",
    "delete_vectors",
    "overwrite_payload",
    "set_payload",
    "update_collection",
    "update_vectors",
    "upsert",
}

# This is an allowlist of the current architecture, not a claim that all access
# belongs in one module. Counts catch extra calls inside an existing function.
_ALLOWED_COUNTS = Counter(
    {
        (
            "api/routers/collections.py",
            "delete_collection_endpoint",
            "delete_collection",
        ): 1,
        ("api/routers/points.py", "_delete_chunk", "delete"): 1,
        ("api/routers/points.py", "_existing_point_ids", "retrieve"): 1,
        ("api/routers/points.py", "_resolve_payload_filter", "count"): 1,
        ("api/routers/points.py", "_retrieve_point_or_404", "retrieve"): 1,
        ("api/routers/points.py", "_scroll_source_point_ids", "scroll"): 1,
        ("api/routers/points.py", "_upsert", "upsert"): 1,
        ("api/routers/points.py", "batch_upsert_points", "upsert"): 1,
        ("api/routers/points.py", "bulk_delete_points", "delete"): 1,
        ("api/routers/points.py", "clear_payload", "overwrite_payload"): 1,
        ("api/routers/points.py", "clear_payload_by_filter", "delete_payload"): 1,
        ("api/routers/points.py", "count_points", "count"): 1,
        ("api/routers/points.py", "delete_payload_keys", "delete_payload"): 1,
        (
            "api/routers/points.py",
            "delete_payload_keys_by_filter",
            "delete_payload",
        ): 1,
        ("api/routers/points.py", "delete_point", "delete"): 1,
        ("api/routers/points.py", "delete_points_by_filter", "count"): 1,
        ("api/routers/points.py", "delete_points_by_filter", "delete"): 1,
        ("api/routers/points.py", "export_collection", "count"): 1,
        ("api/routers/points.py", "facet_points", "facet"): 1,
        ("api/routers/points.py", "patch_payload", "set_payload"): 1,
        ("api/routers/points.py", "patch_payload_by_filter", "set_payload"): 1,
        ("api/routers/points.py", "put_payload", "overwrite_payload"): 1,
        ("api/routers/points.py", "retrieve_points", "retrieve"): 1,
        ("api/routers/points.py", "scroll_points", "count"): 1,
        ("api/routers/points.py", "scroll_points", "scroll"): 1,
        ("api/routers/points.py", "upsert_point", "upsert"): 1,
        ("services/collection_export.py", "build_export_document", "scroll"): 1,
        ("services/health_probe.py", "probe_once", "get_collections"): 1,
        ("services/query_executor.py", "_do", "query_points"): 1,
        (
            "services/vector_reindex.py",
            "_create_target_vectors",
            "create_vector_name",
        ): 2,
        (
            "services/vector_reindex.py",
            "_create_target_vectors",
            "update_collection",
        ): 1,
        (
            "services/vector_reindex.py",
            "_delete_vector_generation",
            "delete_vector_name",
        ): 1,
        ("services/vector_reindex.py", "_finalize", "count"): 3,
        ("services/vector_reindex.py", "_run_batches", "scroll"): 1,
        ("services/vector_reindex.py", "_run_batches", "update_vectors"): 1,
        ("services/vector_reindex.py", "preflight_reindex", "count"): 1,
        ("services/vector_reindex.py", "preflight_reindex", "info"): 1,
        ("services/vector_reindex.py", "preflight_reindex", "scroll"): 1,
    }
)

# Review artifact for raw mutations: guard/retry/cache are intentionally named.
# Schema-generation calls avoid blind retry where a lost success response is not
# safely idempotent; the durable reindex job provides cleanup/resume instead.
_MUTATION_REVIEWS = {
    ("api/routers/collections.py", "delete_collection_endpoint", "delete_collection"): (
        "owner=collection endpoint; guard=exact confirm + require_collection_mutable; "
        "retry=none on ambiguous delete response; "
        "cache=both collection caches invalidated"
    ),
    ("api/routers/points.py", "_delete_chunk", "delete"): (
        "owner=replace-source ingest; guard=guard_vector_operation; "
        "retry=retry_async; cache=none"
    ),
    ("api/routers/points.py", "_upsert", "upsert"): (
        "owner=ingest durable writer; guard=guard_vector_operation; "
        "retry=retry_async; cache=none"
    ),
    ("api/routers/points.py", "batch_upsert_points", "upsert"): (
        "owner=batch point CRUD; guard=guard_vector_operation; "
        "retry=_qdrant; cache=none"
    ),
    ("api/routers/points.py", "bulk_delete_points", "delete"): (
        "owner=bulk point CRUD; guard=guard_collection_mutation; "
        "retry=_qdrant; cache=none"
    ),
    ("api/routers/points.py", "clear_payload", "overwrite_payload"): (
        "owner=point payload CRUD; guard=guard_collection_mutation; "
        "retry=_qdrant; cache=none"
    ),
    ("api/routers/points.py", "clear_payload_by_filter", "delete_payload"): (
        "owner=filtered payload CRUD; guard=_resolve_payload_filter; "
        "retry=_qdrant; cache=none"
    ),
    ("api/routers/points.py", "delete_payload_keys", "delete_payload"): (
        "owner=point payload CRUD; guard=guard_collection_mutation; "
        "retry=_qdrant; cache=none"
    ),
    ("api/routers/points.py", "delete_payload_keys_by_filter", "delete_payload"): (
        "owner=filtered payload CRUD; guard=_resolve_payload_filter; "
        "retry=_qdrant; cache=none"
    ),
    ("api/routers/points.py", "delete_point", "delete"): (
        "owner=point CRUD; guard=guard_collection_mutation; retry=_qdrant; cache=none"
    ),
    ("api/routers/points.py", "delete_points_by_filter", "delete"): (
        "owner=filtered point CRUD; guard=guard_collection_mutation; "
        "retry=_qdrant; cache=none"
    ),
    ("api/routers/points.py", "patch_payload", "set_payload"): (
        "owner=point payload CRUD; guard=guard_collection_mutation; "
        "retry=_qdrant; cache=none"
    ),
    ("api/routers/points.py", "patch_payload_by_filter", "set_payload"): (
        "owner=filtered payload CRUD; guard=_resolve_payload_filter; "
        "retry=_qdrant; cache=none"
    ),
    ("api/routers/points.py", "put_payload", "overwrite_payload"): (
        "owner=point payload CRUD; guard=guard_collection_mutation; "
        "retry=_qdrant; cache=none"
    ),
    ("api/routers/points.py", "upsert_point", "upsert"): (
        "owner=point CRUD; guard=guard_vector_operation; retry=_qdrant; cache=none"
    ),
    ("services/vector_reindex.py", "_create_target_vectors", "create_vector_name"): (
        "owner=reindex schema phase; guard=persisted reindex lock; "
        "retry=compensating cleanup; cache=active metadata changes at finalize"
    ),
    ("services/vector_reindex.py", "_create_target_vectors", "update_collection"): (
        "owner=reindex schema tuning; guard=persisted reindex lock; "
        "retry=compensating cleanup; cache=active metadata changes at finalize"
    ),
    ("services/vector_reindex.py", "_delete_vector_generation", "delete_vector_name"): (
        "owner=reindex cleanup; guard=inactive generation from persisted job; "
        "retry=best effort per name; cache=no vector-generation cache"
    ),
    ("services/vector_reindex.py", "_run_batches", "update_vectors"): (
        "owner=reindex batch writer; guard=persisted reindex lock; "
        "retry=resume from durable batch checkpoint; cache=active metadata at finalize"
    ),
}


class _DirectCallVisitor(ast.NodeVisitor):
    """Collect Qdrant-looking calls made through a local ``client`` binding."""

    def __init__(self) -> None:
        self.functions: list[str] = []
        self.calls: Counter[tuple[str, str]] = Counter()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track a synchronous function while visiting its body."""
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Track an asynchronous function while visiting its body."""
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Visit one function with its name on the owner stack."""
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    def visit_Call(self, node: ast.Call) -> None:
        """Count known Qdrant methods on ``client`` and ``self.client``."""
        if isinstance(node.func, ast.Attribute) and node.func.attr in _QDRANT_METHODS:
            receiver = node.func.value
            is_client = isinstance(receiver, ast.Name) and receiver.id == "client"
            is_self_client = (
                isinstance(receiver, ast.Attribute)
                and receiver.attr == "client"
                and isinstance(receiver.value, ast.Name)
                and receiver.value.id == "self"
            )
            if (is_client or is_self_client) and self.functions:
                self.calls[(self.functions[-1], node.func.attr)] += 1
        self.generic_visit(node)


def _direct_calls() -> Counter[tuple[str, str, str]]:
    """Return raw Qdrant calls outside the collection repository directory."""
    calls: Counter[tuple[str, str, str]] = Counter()
    for path in sorted(_SOURCE_ROOT.rglob("*.py")):
        if "repositories" in path.parts:
            continue
        visitor = _DirectCallVisitor()
        visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
        relative = path.relative_to(_SOURCE_ROOT).as_posix()
        for (function, method), count in visitor.calls.items():
            calls[(relative, function, method)] += count
    return calls


def test_direct_qdrant_access_matches_reviewed_allowlist() -> None:
    """Reject unreviewed raw calls, removed calls and extra duplicate calls."""
    actual = _direct_calls()
    assert actual == _ALLOWED_COUNTS, (
        f"unreviewed={actual - _ALLOWED_COUNTS}; "
        f"stale_allowlist={_ALLOWED_COUNTS - actual}"
    )


def test_every_direct_mutation_has_a_review_artifact() -> None:
    """Require owner, guard, retry and cache review for each raw mutation."""
    mutation_keys = {key for key in _ALLOWED_COUNTS if key[2] in _MUTATING_METHODS}
    assert set(_MUTATION_REVIEWS) == mutation_keys
    for review in _MUTATION_REVIEWS.values():
        assert all(
            label in review for label in ("owner=", "guard=", "retry=", "cache=")
        )

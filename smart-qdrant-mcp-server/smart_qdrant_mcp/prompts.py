"""Prompts for the multi-tool sequences that are easy to get wrong.

Each prompt is a fixed, numbered walk through tools that must run in a
specific order — bootstrapping a collection, diagnosing bad search results,
or reindexing without losing data. They interpolate the caller's arguments
directly into the instructions rather than leaving them as a template, and
they point at the ``docs://`` document that explains the reasoning instead
of restating it. Every tool named below must exist in
``smart_qdrant_mcp.tool_map.TOOL_ROUTES`` and every ``docs://`` URI must be one of
the documents in ``smart_qdrant_mcp.resources.docs.DOC_RESOURCES`` —
``tests/test_prompts.py`` enforces both so a prompt can never point at
something that isn't real.
"""

from __future__ import annotations

from typing import Annotated

from mcp.server import MCPServer
from pydantic import Field


def register(mcp: MCPServer) -> None:
    """Register the workflow prompts."""

    @mcp.prompt(
        name="qdrant_bootstrap_collection",
        title="Bootstrap a Qdrant collection",
    )
    def bootstrap_collection(
        collection_name: Annotated[
            str, Field(description="Collection to create and load (e.g. 'docs')")
        ],
        source_path: Annotated[
            str,
            Field(
                description=(
                    "Path to the corpus file to ingest, relative to the "
                    "middleware's data directory (e.g. 'corpus/manual.json')"
                )
            ),
        ],
    ) -> str:
        """Create a collection and ingest a corpus file into it, in a safe order."""
        return (
            f"Set up the collection {collection_name!r} and load "
            f"{source_path!r} into it.\n\n"
            "Follow this order and stop if any step reports an error:\n"
            "1. qdrant_check_health — confirm Qdrant and the embedder are ready.\n"
            f"2. qdrant_list_collections — check whether {collection_name!r} "
            "already exists. If it does, do NOT recreate it without asking me "
            "first.\n"
            f"3. qdrant_create_collection — create {collection_name!r}. Read "
            "docs://concepts/qdrant-data-model first to choose the four "
            "required toggles deliberately rather than guessing.\n"
            "4. qdrant_create_payload_index — index the payload keys the "
            "corpus will be filtered by, at minimum source_id as a keyword "
            "index. Strict mode refuses any filter on an unindexed key.\n"
            f"5. qdrant_ingest_file — ingest {source_path!r} into "
            f"{collection_name!r}. The path is relative to the middleware's "
            "data directory, and the file must already be there: no tool on "
            "this server can put it there. If the ingest reports the file is "
            "missing, ask me to stage it, or send the chunks inline with "
            "qdrant_ingest_chunks if they are small enough. Read "
            "docs://workflows/ingest for the staging step.\n"
            "6. qdrant_count_points — confirm the number of points matches "
            "what the ingest reported.\n"
            "7. qdrant_search — run one sanity query and show me the top "
            "hits.\n"
        )

    @mcp.prompt(
        name="qdrant_diagnose_search_quality",
        title="Diagnose poor search quality",
    )
    def diagnose_search_quality(
        collection_name: Annotated[
            str,
            Field(
                description=(
                    "Collection whose search results to diagnose (e.g. 'docs')"
                )
            ),
        ],
        query: Annotated[
            str,
            Field(
                description=(
                    "Query text that returns poor or empty results "
                    "(e.g. 'how do I rotate the token')"
                )
            ),
        ],
    ) -> str:
        """Work through why a query returns poor or empty results."""
        return (
            f"Search quality for {query!r} in {collection_name!r} is poor. "
            "Diagnose it, do not guess.\n\n"
            "Read docs://workflows/search-recipes and "
            "docs://concepts/fusion-and-rerank first, then work through:\n"
            f"1. qdrant_count_points on {collection_name!r} with no filter — "
            "is there any data at all?\n"
            "2. qdrant_verify_collection — does the collection satisfy the "
            "active embedding contract? A mismatch makes qdrant_search "
            "refuse with 409; see docs://concepts/embedding-contract.\n"
            f"3. qdrant_search with the query as given, "
            "response_format='json' — record fusion, reranked, candidates "
            "and the score spread.\n"
            "4. Re-run with a higher candidate_limit, then with rerank=true "
            "and a small rerank_top_k. Report which change moved the "
            "needle.\n"
            "5. If a filter is involved: qdrant_list_payload_indexes — is "
            "the filtered key indexed? qdrant_facet_points — do the values "
            "you filter on actually exist?\n"
            "6. qdrant_scroll_points with the same filter — does the "
            "content you expect exist but rank badly, or is it absent "
            "entirely? These have different fixes.\n\n"
            "Report your conclusion as: cause, evidence, recommended change."
        )

    @mcp.prompt(
        name="qdrant_safe_reindex",
        title="Reindex a collection safely",
    )
    def safe_reindex(
        collection_name: Annotated[
            str, Field(description="Collection to reindex (e.g. 'docs')")
        ],
    ) -> str:
        """Reindex a collection's vectors without losing data or getting stuck."""
        return (
            f"Reindex the vectors of {collection_name!r} safely.\n\n"
            "Read docs://workflows/reindex first. Then:\n"
            f"1. qdrant_reindex_preflight on {collection_name!r} — report "
            "can_start, blockers, total_points, eligible_points and "
            "invalid_points. If can_start is false, stop and explain why.\n"
            "2. If invalid_points is above zero, show me the invalid ids "
            "and ask before proceeding — those points will not be "
            "reindexed.\n"
            f"3. qdrant_start_reindex with confirm set to "
            f"{collection_name!r} exactly. Nothing else is accepted.\n"
            "4. qdrant_get_reindex_status — poll until the job leaves the "
            "running state. Do not poll faster than once every few "
            "seconds.\n"
            "5. While the job runs, do not call any mutating tool on this "
            "collection: any such call is rejected with a 409 conflict "
            "while a job is active or paused. That rejection is intended, "
            "not a fault — it protects the in-progress checkpoint.\n"
            "6. When it completes: qdrant_verify_collection, then one "
            "qdrant_search to confirm results are sane.\n\n"
            "If the job must be interrupted, use qdrant_control_reindex "
            "with action='pause' — the checkpoint is kept and 'resume' "
            "continues. action='cancel' abandons the job's progress."
        )

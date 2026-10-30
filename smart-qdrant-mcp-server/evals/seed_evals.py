"""Seed the deterministic `mcp_eval` collection through the MCP server.

Run this once before running the evaluation harness. It is idempotent: point ids
are derived from source_id and chunk_id, so re-running overwrites rather than
duplicating.

This drives the **MCP server itself** over streamable HTTP, not the middleware
directly, so seeding exercises the same path the evaluation will. The
``mcp.client.Client`` dataclass (see ``docs/superpowers/notes/mcp-v2-api.md``,
Task 0) accepts a URL string directly, but that path has no way to attach an
``Authorization`` header. To carry the bearer token, this script instead
builds the ``streamable_http_client(...)`` transport itself with a
pre-configured ``httpx2.AsyncClient`` (the underlying HTTP client `mcp==2.2.0`
depends on) and hands that ``Transport`` to ``Client(...)`` — the third
constructor shape ``Client`` documents (``Server | MCPServer | Transport |
str``), verified directly against the installed ``mcp==2.2.0`` package.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx2
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client

CORPUS_PATH = Path(__file__).parent / "corpus" / "mcp_eval_corpus.json"
COLLECTION = "mcp_eval"
SOURCE_ID = "mcp-eval-corpus"
INDEXED_KEYS = [
    ("doc_type", "keyword"),
    ("author", "keyword"),
    ("component", "keyword"),
    ("published_on", "datetime"),
    ("version", "keyword"),
    ("language", "keyword"),
    ("severity", "keyword"),
    ("source_id", "keyword"),
]


def load_corpus() -> list[dict]:
    """Return the fixture chunks."""
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


async def call(client: Client, tool: str, arguments: dict) -> object:
    """Call one tool and raise if it reports an error.

    Every tool takes a single Pydantic argument named ``params``, so the wire
    arguments nest one level under that key. A refused call raises rather than
    printing, so a failed seed cannot pass for a successful one.
    """
    result = await client.call_tool(tool, {"params": arguments})
    if result.is_error:
        text = "".join(getattr(block, "text", "") for block in result.content)
        raise RuntimeError(f"{tool} failed: {text}")
    return result


async def seed(url: str, token: str, *, recreate: bool) -> None:
    """Create the collection, index its payload keys, and ingest the corpus."""
    async with httpx2.AsyncClient(
        headers={"Authorization": f"Bearer {token}"}
    ) as http_client:
        transport = streamable_http_client(url, http_client=http_client)
        async with Client(transport) as client:
            create = {
                "collection_name": COLLECTION,
                "dense_on_disk": False,
                "quantization": False,
                "sparse_idf": True,
                "sparse_on_disk": False,
            }
            if recreate:
                create |= {"recreate": True, "confirm": COLLECTION}
            await call(client, "qdrant_create_collection", create)
            await call(
                client,
                "qdrant_create_payload_index",
                {
                    "collection_name": COLLECTION,
                    "indexes": [
                        {"field_name": key, "field_schema": schema}
                        for key, schema in INDEXED_KEYS
                    ],
                },
            )
            result = await call(
                client,
                "qdrant_ingest_chunks",
                {
                    "chunks": load_corpus(),
                    "source_id": SOURCE_ID,
                    "collection_name": COLLECTION,
                    "replace_source": True,
                    # replace_source=True is gated by confirm, which must equal
                    # source_id exactly (smart_qdrant_mcp.guardrails.require_confirm,
                    # called with noun="source_id" from tools/ingest.py) — not
                    # the collection name and not a bare `true`.
                    "confirm": SOURCE_ID,
                },
            )
            print(result)
            count = await call(
                client,
                "qdrant_count_points",
                {"collection_name": COLLECTION, "response_format": "json"},
            )
            print(count)


def main() -> None:
    """Parse arguments and run the seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:6337/mcp")
    parser.add_argument("--token", required=True)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="drop and rebuild the collection before ingesting",
    )
    args = parser.parse_args()
    asyncio.run(seed(args.url, args.token, recreate=args.recreate))


if __name__ == "__main__":
    main()

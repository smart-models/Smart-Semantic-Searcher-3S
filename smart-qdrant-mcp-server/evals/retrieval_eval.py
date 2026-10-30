"""Score hybrid retrieval on the seeded `mcp_eval` collection against a golden set.

Run ``seed_evals.py`` first. Every query in ``retrieval_golden.json`` is searched
twice through the MCP server, with and without the cross-encoder reranker, and
scored against its labelled chunk ids (binary relevance). The aggregate scores
are compared with ``retrieval_baseline.json`` for the active embedder profile;
any metric falling more than ``TOLERANCE`` below it exits non-zero.
``--write-baseline`` records the current scores for that profile instead.

Like the seed, this needs the live stack with models loaded and the MCP
service token; it is not part of any offline pass.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

import httpx2
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client
from seed_evals import COLLECTION, call

HERE = Path(__file__).parent
GOLDEN_PATH = HERE / "retrieval_golden.json"
BASELINE_PATH = HERE / "retrieval_baseline.json"
K = 10
# Absorbs GPU floating-point jitter between runs, not real regressions.
TOLERANCE = 0.05
METRICS = ("hit@1", "recall@5", "recall@10", "mrr", "ndcg@10")
CONFIGS = {"fusion": False, "rerank": True}


def score(ranked: list[str], relevant: set[str]) -> dict[str, float]:
    """Return every metric for one ranked list of chunk ids."""
    hits = [chunk_id in relevant for chunk_id in ranked[:K]]
    first = next((rank for rank, hit in enumerate(hits, 1) if hit), None)
    dcg = sum(1 / math.log2(rank + 1) for rank, hit in enumerate(hits, 1) if hit)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), K) + 1))
    return {
        "hit@1": float(bool(hits) and hits[0]),
        "recall@5": sum(hits[:5]) / len(relevant),
        "recall@10": sum(hits) / len(relevant),
        "mrr": 1 / first if first else 0.0,
        "ndcg@10": dcg / ideal,
    }


def mean(rows: list[dict[str, float]]) -> dict[str, float]:
    """Average each metric over a list of per-query scores."""
    return {m: round(sum(row[m] for row in rows) / len(rows), 4) for m in METRICS}


def regressions(report: dict, baseline: dict) -> list[str]:
    """List aggregate metrics that fell more than TOLERANCE below the baseline."""
    failed = []
    for config, scores in baseline.items():
        for metric, base in scores["all"].items():
            current = report[config]["all"][metric]
            if current < base - TOLERANCE:
                failed.append(f"{config} {metric}: {base} -> {current}")
    return failed


async def evaluate(url: str, token: str) -> tuple[str, dict]:
    """Search every golden query under both configs; return profile and report."""
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    scored: dict[str, list[tuple[str, dict]]] = {name: [] for name in CONFIGS}
    async with httpx2.AsyncClient(
        headers={"Authorization": f"Bearer {token}"}
    ) as http_client:
        transport = streamable_http_client(url, http_client=http_client)
        async with Client(transport) as client:
            health = await call(client, "qdrant_check_health", {})
            profile = health.structured_content["embedder_profile"]
            for entry in golden:
                for name, rerank in CONFIGS.items():
                    result = await call(
                        client,
                        "qdrant_search",
                        {
                            "query": entry["query"],
                            "collection_name": COLLECTION,
                            "limit": K,
                            "rerank": rerank,
                            "with_payload": ["chunk_id"],
                        },
                    )
                    # A rerank that silently did not run once made both
                    # configs score identically; fail instead of measuring it.
                    if result.structured_content["reranked"] is not rerank:
                        raise RuntimeError(f"{entry['id']}: {name} not applied")
                    ranked = [
                        hit["payload"]["chunk_id"]
                        for hit in result.structured_content["items"]
                    ]
                    scores = score(ranked, set(entry["relevant"]))
                    scored[name].append((entry["category"], scores))
    categories = sorted({entry["category"] for entry in golden})
    report = {
        name: {
            "all": mean([s for _, s in rows]),
            **{cat: mean([s for c, s in rows if c == cat]) for cat in categories},
        }
        for name, rows in scored.items()
    }
    return profile, report


def main() -> None:
    """Parse arguments, score the golden set, and gate or record the baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:6337/mcp")
    parser.add_argument("--token", required=True)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="record these scores as the baseline for the active profile",
    )
    args = parser.parse_args()
    profile, report = asyncio.run(evaluate(args.url, args.token))
    print(json.dumps({"profile": profile, **report}, indent=2))

    baselines = (
        json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        if BASELINE_PATH.exists()
        else {}
    )
    if args.write_baseline:
        baselines[profile] = report
        BASELINE_PATH.write_text(json.dumps(baselines, indent=2) + "\n", "utf-8")
        print(f"baseline recorded for profile {profile!r}")
        return
    if profile not in baselines:
        print(f"no baseline for profile {profile!r}; nothing gated", file=sys.stderr)
        return
    failed = regressions(report, baselines[profile])
    if failed:
        sys.exit("retrieval regressed:\n" + "\n".join(failed))
    print(f"no aggregate metric fell more than {TOLERANCE} below the baseline")


if __name__ == "__main__":
    main()

"""qdrant_check_health aggregates the four monitoring endpoints.

The fixtures below are copies of what the running middleware really answers,
captured from a live stack. An earlier version of this file invented a nested
shape (`qdrant: {"reachable": true}`), which passed every mocked test and then
raised AttributeError against the real service, whose `/health` is flat and
whose counters are named total_requests / rejected_requests.
"""

import json

HEALTH = {
    "status": "healthy",
    "qdrant": "reachable",
    "collections": 1,
    "default_collection": "",
    "embedder": "reachable",
    "embedder_profile": "advanced",
    "embedder_models": {
        "dense": "Qwen/Qwen3-Embedding-0.6B",
        "sparse": "omai-research/milco-650m",
        "colbert": None,
        "reranker": "Qwen/Qwen3-Reranker-0.6B",
    },
}
STATS = {
    "uptime": "0:20:00",
    "uptime_seconds": 1200,
    "total_requests": 42,
    "rejected_requests": 3,
    "collection": "",
}


def arm(routes, *, readyz_status=200):
    routes.add("GET", "/health", body=HEALTH)
    routes.add("GET", "/readyz", status=readyz_status, body={"status": "ready"})
    routes.add("GET", "/livez", body={"status": "alive"})
    routes.add("GET", "/stats", body=STATS)


async def test_markdown_reports_a_single_verdict(call, routes):
    arm(routes)
    out = await call("qdrant_check_health")
    assert out.splitlines()[0].startswith("# ")
    assert "Ready" in out


async def test_markdown_renders_the_real_flat_health_shape(call, routes):
    arm(routes)
    out = await call("qdrant_check_health")
    assert "- **Qdrant**: reachable" in out
    assert "- **Embedder**: reachable" in out
    assert "- **Embedder profile**: advanced" in out
    assert "- **Uptime (s)**: 1200" in out
    assert "- **Requests served**: 42" in out
    assert "- **Requests rejected**: 3" in out


async def test_markdown_lists_the_active_models(call, routes):
    arm(routes)
    out = await call("qdrant_check_health")
    assert "dense=Qwen/Qwen3-Embedding-0.6B" in out
    # colbert is null on the advanced profile and must not render as "None".
    assert "colbert" not in out


async def test_json_format_carries_every_section(call, routes):
    arm(routes)
    payload = json.loads(await call("qdrant_check_health", response_format="json"))
    assert payload["ready"] is True
    assert payload["qdrant"] == "reachable"
    assert payload["embedder_profile"] == "advanced"
    assert payload["collections"] == 1
    assert payload["stats"]["total_requests"] == 42


async def test_not_ready_is_reported_without_raising(call, routes):
    arm(routes, readyz_status=503)
    payload = json.loads(await call("qdrant_check_health", response_format="json"))
    assert payload["ready"] is False


async def test_calls_all_four_endpoints(call, routes):
    arm(routes)
    await call("qdrant_check_health")
    called = {request.url.path for request in routes.calls}
    assert called == {"/health", "/readyz", "/livez", "/stats"}

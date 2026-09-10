"""Live resources proxy middleware state."""

import json

import pytest
from mcp import MCPError

EXPECTED_TITLES = {
    "qhm://collections": "Current Qdrant collections",
    "qhm://health": "Middleware dependency health",
    "qhm://metrics": "Middleware Prometheus metrics",
}


async def test_all_three_live_resources_are_listed(mcp_client):
    uris = {
        str(resource.uri) for resource in (await mcp_client.list_resources()).resources
    }
    assert {"qhm://collections", "qhm://health", "qhm://metrics"} <= uris


async def test_each_live_resource_exposes_a_title_for_ui_display(mcp_client):
    """Live resources advertise an explicit human-facing title.

    Names and descriptions already come from the function names and
    docstrings; ``title`` is the label a client UI shows instead.
    """
    resources = {
        str(resource.uri): resource
        for resource in (await mcp_client.list_resources()).resources
    }
    for uri, title in EXPECTED_TITLES.items():
        assert resources[uri].title == title, uri


async def test_collections_resource_returns_json(mcp_client, routes):
    routes.add("GET", "/collections", body={"collections": ["docs"]})
    result = await mcp_client.read_resource("qhm://collections")
    assert json.loads(result.contents[0].text) == {"collections": ["docs"]}


async def test_health_resource_returns_json(mcp_client, routes):
    routes.add("GET", "/health", body={"status": "ok"})
    result = await mcp_client.read_resource("qhm://health")
    assert json.loads(result.contents[0].text)["status"] == "ok"


async def test_metrics_resource_returns_prometheus_text(mcp_client, routes):
    routes.add("GET", "/metrics", body="qdrant_requests_total 5")
    result = await mcp_client.read_resource("qhm://metrics")
    assert "qdrant_requests_total 5" in result.contents[0].text


async def test_failure_raises_a_protocol_error_not_fabricated_content(
    mcp_client, routes
):
    """An upstream health failure surfaces as a protocol-level error.

    A resource read has no `is_error` flag the way a tool call does, so text
    content is the client's only signal. Returning the upstream failure as
    "Error: ..." text used to make it indistinguishable from a genuine
    (albeit oddly-worded) health document.
    """
    routes.add("GET", "/health", status=503, body={"detail": "not ready"})
    with pytest.raises(MCPError):
        await mcp_client.read_resource("qhm://health")

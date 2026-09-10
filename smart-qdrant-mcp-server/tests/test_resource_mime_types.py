"""Every resource declares the MIME type a client needs to render or parse it.

``mcp.resource(...)`` accepts ``mime_type`` and defaults to ``text/plain``
(``mcp/server/mcpserver/resources/types.py``) when it is omitted. Before this
test existed, every resource went unset and therefore all reported
``text/plain`` -- including the Markdown docs and the two JSON live
resources. This asserts every declared type explicitly, keyed by URI, so a
future resource that forgets ``mime_type`` fails loudly here instead of
silently advertising the wrong type.
"""

from __future__ import annotations

from smart_qdrant_mcp.resources.docs import published_documents

EXPECTED_MIME_TYPES: dict[str, str] = {
    # Every published document: Markdown, except the guide pages, which are
    # served as the HTML they were authored as.
    **{
        uri: ("text/html" if file_name.endswith(".html") else "text/markdown")
        for uri, (file_name, _, _) in published_documents().items()
    },
    # Two JSON snapshots proxied from the middleware.
    "qhm://collections": "application/json",
    "qhm://health": "application/json",
    # Prometheus exposition text -- the SDK's own default, left unset.
    "qhm://metrics": "text/plain",
}


async def test_every_resource_declares_its_expected_mime_type(mcp_client):
    assert len(EXPECTED_MIME_TYPES) == 23
    declared = {
        str(resource.uri): resource.mime_type
        for resource in (await mcp_client.list_resources()).resources
    }
    assert declared.keys() >= EXPECTED_MIME_TYPES.keys()
    for uri, expected in EXPECTED_MIME_TYPES.items():
        assert declared[uri] == expected, uri

"""Documentation resources are present, non-empty, and internally consistent."""

import re
from pathlib import Path

import pytest
from mcp import MCPError

from smart_qdrant_mcp.resources.docs import (
    APP_RESOURCES,
    DOC_RESOURCES,
    docs_directory,
    published_documents,
    resource_name,
)
from smart_qdrant_mcp.tool_map import TOOL_ROUTES

# Each copied application document, mapped to the repository file it was
# copied from. The copies live inside the package because the container image
# ships ``smart_qdrant_mcp/`` alone; this map is what keeps that duplication
# honest (see ``test_every_app_document_matches_its_repository_original``).
APP_DOCUMENT_ORIGINS = {
    "guide/getting-started.html": "guide/getting-started.html",
    "guide/cli.html": "guide/cli.html",
    "guide/environment.html": "guide/environment.html",
    "guide/architecture.html": "guide/architecture.html",
    "guide/hybrid-search-rerank.html": "guide/hybrid-search-rerank.html",
    "reference/middleware-api.md": "middleware-qdrant-manager/README.md",
    "reference/launcher-scripts.md": "scripts/README.md",
    "reference/smart-embedder.md": "server-embedder-reranker/README.md",
    "reference/mcp-server.md": "smart-qdrant-mcp-server/README.md",
    "reference/agent-skills.md": "smart-skill/README.md",
    "reference/smart-qdrant-skill.md": "smart-skill/smart-qdrant-mcp/SKILL.md",
}

EXPECTED_APP_URIS = {
    "docs://guide/getting-started",
    "docs://guide/cli",
    "docs://guide/environment",
    "docs://guide/architecture",
    "docs://guide/hybrid-search-rerank",
    "docs://reference/middleware-api",
    "docs://reference/launcher-scripts",
    "docs://reference/smart-embedder",
    "docs://reference/mcp-server",
    "docs://reference/agent-skills",
    "docs://reference/smart-qdrant-skill",
}

EXPECTED_URIS = {
    "docs://concepts/hybrid-search",
    "docs://concepts/fusion-and-rerank",
    "docs://concepts/qdrant-data-model",
    "docs://concepts/embedding-contract",
    "docs://reference/filters",
    "docs://workflows/ingest",
    "docs://workflows/search-recipes",
    "docs://workflows/reindex",
    "docs://troubleshooting",
}


def test_every_expected_document_is_declared():
    assert set(DOC_RESOURCES) == EXPECTED_URIS


@pytest.mark.parametrize("uri", sorted(EXPECTED_URIS))
def test_document_file_exists_and_has_content(uri):
    path = docs_directory() / DOC_RESOURCES[uri][0]
    assert path.is_file()
    assert len(path.read_text(encoding="utf-8").split()) >= 200


@pytest.mark.parametrize("uri", sorted(EXPECTED_URIS))
def test_document_names_related_tools(uri):
    text = (docs_directory() / DOC_RESOURCES[uri][0]).read_text(encoding="utf-8")
    assert "Related tools" in text
    assert any(tool in text for tool in TOOL_ROUTES)


@pytest.mark.parametrize("uri", sorted(EXPECTED_URIS))
def test_document_only_references_real_tool_names(uri):
    text = (docs_directory() / DOC_RESOURCES[uri][0]).read_text(encoding="utf-8")
    for token in text.replace("`", " ").replace(",", " ").split():
        if token.startswith("qdrant_"):
            assert token.rstrip(".:;)") in TOOL_ROUTES, token


def test_no_document_contains_a_placeholder():
    for file_name, _, _ in DOC_RESOURCES.values():
        text = (docs_directory() / file_name).read_text(encoding="utf-8")
        for marker in ("TODO", "TBD", "FIXME", "Lorem ipsum"):
            assert marker not in text, f"{file_name} contains {marker}"


async def test_documents_are_readable_over_mcp(mcp_client):
    result = await mcp_client.read_resource("docs://reference/filters")
    assert "filter" in result.contents[0].text.lower()


async def test_missing_document_file_raises_a_protocol_error_not_fabricated_content(
    mcp_client, monkeypatch, tmp_path
):
    """A missing document file raises instead of returning error text.

    It must behave like every other resource-read failure in this service
    (live.py's resources): raise, so the SDK turns it into a genuine
    protocol-level error, rather than returning "Error: ..." text a client
    could mistake for the document's real content -- a resource read has no
    `is_error` flag the way a tool call does, so returned text is the only
    thing a client sees either way.
    ``docs_directory`` is looked up by name at read time (not captured at
    registration time), so patching it here after the server is already
    built still redirects every registered resource to the empty directory.
    """
    monkeypatch.setattr(
        "smart_qdrant_mcp.resources.docs.docs_directory", lambda: tmp_path
    )
    with pytest.raises(MCPError):
        await mcp_client.read_resource("docs://troubleshooting")


def test_docs_directory_contains_no_unpublished_files():
    published = {file_name for file_name, _, _ in published_documents().values()}
    root = Path(docs_directory())
    on_disk = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.suffix in {".md", ".html"}
    }
    assert on_disk == published


def test_every_expected_app_document_is_declared():
    assert set(APP_RESOURCES) == EXPECTED_APP_URIS


def test_app_documents_are_disjoint_from_the_teaching_documents():
    """The two groups may never collide on a URI.

    They are merged into one registration set, and a duplicate key would
    silently drop one document rather than fail.
    """
    assert not set(APP_RESOURCES) & set(DOC_RESOURCES)
    assert len(published_documents()) == len(APP_RESOURCES) + len(DOC_RESOURCES)


@pytest.mark.parametrize("uri", sorted(EXPECTED_APP_URIS))
def test_app_document_file_exists_and_has_content(uri):
    path = docs_directory() / APP_RESOURCES[uri][0]
    assert path.is_file()
    assert len(path.read_text(encoding="utf-8").split()) >= 200


@pytest.mark.parametrize("file_name", sorted(APP_DOCUMENT_ORIGINS))
def test_every_app_document_matches_its_repository_original(file_name):
    """Each copy is byte-identical to the file it was copied from.

    Copying rather than linking is deliberate -- the deployed image contains
    only ``smart_qdrant_mcp/`` -- and the price of that decision is drift: an
    edit to ``guide/`` or to a component README leaves the served copy stale
    with nothing to say so. This is the thing that says so.
    """
    original = Path(__file__).parents[2] / APP_DOCUMENT_ORIGINS[file_name]
    if not original.is_file():
        pytest.skip(f"{original} is outside this checkout")
    copied = docs_directory() / file_name
    assert copied.read_bytes() == original.read_bytes(), (
        f"{file_name} has drifted from {APP_DOCUMENT_ORIGINS[file_name]}; re-copy it"
    )


@pytest.mark.parametrize(
    "uri", sorted(uri for uri in EXPECTED_APP_URIS if uri.startswith("docs://guide/"))
)
async def test_guide_page_is_served_without_its_inline_assets(mcp_client, uri):
    """A guide page reaches a client as prose, not as a stylesheet.

    Inline ``<style>`` and ``<script>`` are around 85% of each page's bytes
    and mean nothing to a client reading the document, so they are stripped
    at read time -- while the file on disk stays byte-identical to its
    original, which is what the drift guard above compares.
    """
    served = (await mcp_client.read_resource(uri)).contents[0].text
    stored = (docs_directory() / APP_RESOURCES[uri][0]).read_text(encoding="utf-8")
    assert "<style" in stored
    assert "<style" not in served
    assert "<script" not in served
    assert len(served) < len(stored)
    assert "<h1" in served or "<h2" in served


@pytest.mark.parametrize(
    "uri", sorted(uri for uri in EXPECTED_APP_URIS if uri.startswith("docs://guide/"))
)
async def test_guide_page_links_to_documents_this_server_serves(mcp_client, uri):
    """The guide's deep links must resolve for a client that has no checkout.

    Each page links to the component READMEs by repository path
    (``../scripts/README.md#recipes``). Those are rewritten at read time to
    the ``docs://`` URIs serving the same documents here, so no served page
    may still point at a checkout path, and every rewritten target must be a
    document this server actually publishes.
    """
    served = (await mcp_client.read_resource(uri)).contents[0].text
    assert not re.search(r'href="\.\./[^"]*\.(md|html)', served)
    targets = set(re.findall(r'href="(docs://[^"]*)"', served))
    assert targets, uri
    assert targets <= set(published_documents())


async def test_each_document_exposes_its_own_name_title_and_description(mcp_client):
    """Regression guard for the documents' identity in a resource listing.

    The SDK captures a resource's name at decoration time, so assigning
    ``__name__`` after the decorator ran had no effect: all nine documents
    used to be advertised under the one function name ``_read_document``,
    with the same generic docstring as every description, leaving a client
    listing resources unable to tell the documents apart. Each document must
    now surface the ``read_<file_stem>`` name derived from its file, plus
    the title and description declared for it in ``DOC_RESOURCES`` or
    ``APP_RESOURCES``.
    """
    resources = {
        str(resource.uri): resource
        for resource in (await mcp_client.list_resources()).resources
    }
    for uri, (file_name, title, description) in published_documents().items():
        expected_name = resource_name(file_name)
        resource = resources[uri]
        assert resource.name == expected_name, uri
        assert resource.title == title, uri
        assert resource.description == description, uri


async def test_all_resource_names_are_unique(mcp_client):
    """No two advertised resources share a name.

    The nine teaching documents, the eleven application documents and the
    three live resources share one listing, so all twenty-three names must be
    distinct.
    """
    resources = (await mcp_client.list_resources()).resources
    assert len(resources) == 23
    names = [resource.name for resource in resources]
    assert len(names) == len(set(names))


async def test_docs_resource_descriptions_are_non_empty_and_distinct(mcp_client):
    """Every ``docs://`` resource describes its own document.

    Each description must be non-empty and never identical to the description
    of another document.
    """
    descriptions = [
        resource.description
        for resource in (await mcp_client.list_resources()).resources
        if str(resource.uri).startswith("docs://")
    ]
    assert len(descriptions) == len(published_documents())
    assert all(description and description.strip() for description in descriptions)
    assert len(descriptions) == len(set(descriptions))


def test_ingest_document_explains_that_this_server_cannot_stage_files():
    """The staging gap must be answerable from the MCP server itself.

    An agent holding a corpus file has no tool here that can put it under
    the middleware's data/ directory -- that absence is deliberate, not an
    oversight, and this document is the only surface that can say so to a
    client with no access to the repository. It must also name the real
    ceiling that forces the file path in the first place: the 4 MiB
    streamable-HTTP transport cap, not the middleware's much larger body
    limit.
    """
    text = (docs_directory() / DOC_RESOURCES["docs://workflows/ingest"][0]).read_text(
        encoding="utf-8"
    )
    normalized = " ".join(text.split())
    assert "4 MiB" in normalized
    assert "read-only" in normalized
    assert "no tool on this server" in normalized.lower()

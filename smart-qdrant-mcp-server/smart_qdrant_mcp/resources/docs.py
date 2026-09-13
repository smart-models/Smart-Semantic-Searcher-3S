"""Teaching and reference documents, published as MCP resources.

These are static files that change only when this repository changes. In
principle a client could cache them aggressively, but `mcp==2.0.0` wires cache
hints per *method* (`resources/read`), not per resource URI (verified in Task 0,
`docs/superpowers/notes/mcp-v2-api.md`), and `resources/read` is shared with the
short-lived resources in `live.py`. No `cache_hints` entry is set here; see
`live.py` for the same trade-off.
"""

from __future__ import annotations

from pathlib import Path
import re

from mcp.server import MCPServer

# uri -> (file_name, title, description). The resource's ``name`` is not
# stored here: it is derived from the file name at registration time
# (``read_concepts_hybrid_search`` for ``concepts-hybrid-search.md``). Each
# description summarizes its own document's content, because it is what a
# client listing resources sees to tell the nine documents apart.
DOC_RESOURCES: dict[str, tuple[str, str, str]] = {
    "docs://concepts/hybrid-search": (
        "concepts-hybrid-search.md",
        "Hybrid search",
        "Explains why qdrant_search runs dense and sparse retrieval in "
        "parallel and fuses their rankings, what each half contributes, and "
        "why ColBERT is not used in this stack.",
    ),
    "docs://concepts/fusion-and-rerank": (
        "concepts-fusion-and-rerank.md",
        "Fusion and reranking",
        "Explains how dense and sparse rankings are fused into one via RRF "
        "(the default) or DBSF, what candidate_limit trades off, and what "
        "the cross-encoder reranker adds for its extra cost.",
    ),
    "docs://concepts/qdrant-data-model": (
        "concepts-qdrant-data-model.md",
        "Qdrant's data model",
        "Explains the Qdrant storage pieces these tools touch: collections, "
        "points, payloads, named dense and sparse vectors, payload indexes, "
        "HNSW, quantization with rescore, and the on-disk toggles.",
    ),
    "docs://concepts/embedding-contract": (
        "concepts-embedding-contract.md",
        "The embedding contract",
        "Explains the fixed basic and advanced embedding profiles, why a "
        "collection stamped by one profile rejects the other with HTTP 409, "
        "and the only two fixes: switch the profile back or reindex.",
    ),
    "docs://reference/filters": (
        "reference-filters.md",
        "Filter reference",
        "The middleware filter grammar accepted by every filters-taking "
        "tool: legacy flat syntax, advanced must/should/must_not logic, "
        "every operator, and the enforced depth and condition limits.",
    ),
    "docs://workflows/ingest": (
        "workflows-ingest.md",
        "Ingest workflow",
        "How qdrant_ingest_file and qdrant_ingest_chunks embed and write "
        "points, why deterministic UUID5 point ids make re-ingesting "
        "overwrite rather than duplicate, the two replace_source hazards, "
        "and why staging a large file under data/ is an operator step no "
        "tool here can perform.",
    ),
    "docs://workflows/search-recipes": (
        "workflows-search-recipes.md",
        "Search recipes",
        "Five concrete qdrant_search calls for common situations (plain, "
        "filtered, high-recall, paged, precision-first) plus a checklist "
        "for when a search comes back empty.",
    ),
    "docs://workflows/reindex": (
        "workflows-reindex.md",
        "Reindex workflow",
        "How the background reindex job regenerates every point's vectors "
        "under a new embedding contract: preflight first, pause, resume and "
        "cancel at batch boundaries, and the intentional 409 mutation lock.",
    ),
    "docs://troubleshooting": (
        "troubleshooting.md",
        "Troubleshooting",
        "Maps every error text an agent can see, whether reported by the "
        "middleware or refused by this server itself, to what it means and "
        "which tool to call next.",
    ),
}


# uri -> (file_name, title, description), the same shape as DOC_RESOURCES, for
# the application's own documentation: the six end-user guide pages and the
# six component references those pages link to for the technical detail.
#
# Every file is a copy kept inside this package, not a path reaching out of
# it. The container image copies ``smart_qdrant_mcp/`` and nothing else (see
# the Dockerfile's ``smart-qdrant-mcp`` stage), so ``../../guide/`` or
# ``../../../middleware-qdrant-manager/README.md`` would resolve to a file
# that exists in a checkout and is absent from every deployed server. The
# cost of copying is drift, and ``tests/test_resources_docs.py`` pays it back
# by comparing each copy byte-for-byte against its original.
APP_RESOURCES: dict[str, tuple[str, str, str]] = {
    "docs://guide/getting-started": (
        "guide/getting-started.html",
        "Guide: getting started",
        "The app's own end-user guide to a first run: writing .env, "
        "connecting a local or cloud Qdrant, starting the stack with "
        "up auto, and running a first search.",
    ),
    "docs://guide/cli": (
        "guide/cli.html",
        "Guide: starting the app",
        "The app's own end-user guide to the stack launcher: the PowerShell "
        "and Bash entry points, choosing CPU or GPU, reading logs, and "
        "applying settings changes.",
    ),
    "docs://guide/environment": (
        "guide/environment.html",
        "Guide: configuring .env",
        "The app's own end-user reference for the .env file: the settings "
        "that matter first, Qdrant connection and access keys, how a change "
        "is applied, and every parameter in the template.",
    ),
    "docs://guide/architecture": (
        "guide/architecture.html",
        "Guide: the local search architecture",
        "The app's own end-user tour of the whole local stack -- Smart "
        "Embedder, Qdrant, the middleware, this MCP server and the agent "
        "skills -- with the basic and advanced profiles on CPU or GPU.",
    ),
    "docs://guide/hybrid-search-rerank": (
        "guide/hybrid-search-rerank.html",
        "Guide: hybrid search and reranking",
        "The app's own plain-English walkthrough of hybrid search and "
        "reranking for end users, following one question from meaning and "
        "keywords through to a reranked set of results.",
    ),
    "docs://guide/embedding-models": (
        "guide/embedding-models.html",
        "Guide: the embedding models",
        "The app's own end-user guide to the two model profiles: which "
        "model fills the dense, sparse and reranking roles under basic "
        "(BGE) and advanced (Qwen + MILCO), their Hugging Face sources and "
        "licenses, why ColBERT is produced but never stored, and what the "
        "embedding contract means for a collection.",
    ),
    "docs://reference/middleware-api": (
        "reference/middleware-api.md",
        "Reference: the middleware REST API",
        "middleware-qdrant-manager's own README: the REST API this MCP "
        "server proxies -- its configuration, collection endpoints, "
        "ingestion and CRUD, and the query surface behind qdrant_search.",
    ),
    "docs://reference/launcher-scripts": (
        "reference/launcher-scripts.md",
        "Reference: the stack launchers",
        "scripts/'s own README: the stack.ps1 and stack.sh launchers -- "
        "prerequisites, parameters, automatic startup and GPU selection, "
        "startup calibration and console controls, and worked recipes.",
    ),
    "docs://reference/smart-embedder": (
        "reference/smart-embedder.md",
        "Reference: the embedding and reranking service",
        "server-embedder-reranker's own README: the self-hosted service "
        "producing the dense, sparse and rerank scores behind every search "
        "-- its models, profiles, configuration and endpoints.",
    ),
    "docs://reference/mcp-server": (
        "reference/mcp-server.md",
        "Reference: this MCP server",
        "smart-qdrant-mcp-server's own README: this server's design, its "
        "full tool surface, client configuration and authentication, and "
        "the guarantees each tool makes.",
    ),
    "docs://reference/agent-skills": (
        "reference/agent-skills.md",
        "Reference: the client-side agent skills",
        "smart-skill's own README: the agent skills shipped for this MCP "
        "server, what each one covers, and the recipes they encode.",
    ),
    "docs://reference/smart-qdrant-skill": (
        "reference/smart-qdrant-skill.md",
        "Reference: the smart-qdrant-mcp skill",
        "The SKILL.md an agent client loads to drive the qdrant_* tools: "
        "the operating instructions for searching, ingesting, inspecting "
        "and reindexing through this server.",
    ),
}

# Inline stylesheets and scripts in the guide pages. They are roughly 85% of
# each page's bytes and carry nothing a client reading the document needs, so
# they are stripped at read time and not at copy time: the files on disk stay
# byte-identical to the repository originals, which is what lets the drift
# guard in the tests be a straight comparison.
_ASSET_TAGS = re.compile(r"<(style|script)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)

# Repository-relative link target -> the URI serving that document here. The
# guide pages link to the component READMEs by checkout path
# (``../scripts/README.md#recipes``), which resolves to nothing for a client
# reading the page over MCP. Rewriting those hrefs at read time keeps the
# cross-references followable without touching the copies on disk. Anchors are
# dropped because a resource URI has no fragment to land on.
_LINKED_DOCS: dict[str, str] = {
    "middleware-qdrant-manager/README.md": "docs://reference/middleware-api",
    "scripts/README.md": "docs://reference/launcher-scripts",
    "server-embedder-reranker/README.md": "docs://reference/smart-embedder",
    "smart-qdrant-mcp-server/README.md": "docs://reference/mcp-server",
    "smart-skill/README.md": "docs://reference/agent-skills",
    "smart-skill/smart-qdrant-mcp/SKILL.md": "docs://reference/smart-qdrant-skill",
}

# Sibling guide pages also link to each other by bare filename
# (``href="cli.html#first-start"``), which has the same problem over MCP: the
# client has no checkout, so the relative path resolves to nothing. The same
# read-time rewrite applies, derived from ``APP_RESOURCES`` so it stays in sync
# with the registered guide pages rather than being hand-maintained. Anchors
# are dropped for the same reason as the README links above.
_GUIDE_LINKS: dict[str, str] = {
    Path(file_name).name: uri
    for uri, (file_name, _, _) in APP_RESOURCES.items()
    if file_name.startswith("guide/") and file_name.endswith(".html")
}

_LINK_HREF = re.compile(
    r'href="\.\./('
    + "|".join(re.escape(path) for path in _LINKED_DOCS)
    + r')(#[^"]*)?"'
)

_SIBLING_HREF = re.compile(
    r'href="(?!https?://|/|\.\./|#)('
    + "|".join(re.escape(name) for name in _GUIDE_LINKS)
    + r')(#[^"]*)?"'
)


def resource_name(file_name: str) -> str:
    """Derive a resource's advertised ``name`` from its file name.

    ``concepts-hybrid-search.md`` becomes ``read_concepts_hybrid_search`` and
    ``guide/getting-started.html`` becomes ``read_guide_getting_started``, so
    every document is advertised under a name of its own whichever
    subdirectory or extension it came from.
    """
    stem = Path(file_name).with_suffix("").as_posix()
    return "read_" + stem.replace("-", "_").replace("/", "_")


def _mime_type(file_name: str) -> str:
    """Return the MIME type a client needs to render this document."""
    return "text/html" if file_name.endswith(".html") else "text/markdown"


def _prepare(file_name: str, text: str) -> str:
    """Return the document text as a client should receive it.

    Markdown is served exactly as stored. An HTML guide page is served with
    its inline assets stripped and its links rewritten to the ``docs://`` URIs
    serving the same documents here: checkout-path links to the component
    READMEs and bare-filename links to sibling guide pages.
    """
    if not file_name.endswith(".html"):
        return text
    text = _ASSET_TAGS.sub("", text)
    text = _LINK_HREF.sub(lambda m: f'href="{_LINKED_DOCS[m.group(1)]}"', text)
    return _SIBLING_HREF.sub(lambda m: f'href="{_GUIDE_LINKS[m.group(1)]}"', text)


def docs_directory() -> Path:
    """Return the directory holding the published documents."""
    return Path(__file__).parent / "docs"


def published_documents() -> dict[str, tuple[str, str, str]]:
    """Return every published document, keyed by URI.

    The teaching documents and the application's own documentation are
    declared separately because their invariants differ -- only the teaching
    documents are written against the tool surface -- but they are registered,
    read and listed as one set.
    """
    return {**DOC_RESOURCES, **APP_RESOURCES}


def register(mcp: MCPServer) -> None:
    """Register every published document as a resource."""
    for uri, (file_name, title, description) in published_documents().items():
        _register_one(mcp, uri, file_name, title, description)


def _register_one(
    mcp: MCPServer, uri: str, file_name: str, title: str, description: str
) -> None:
    """Bind one document file to one resource URI.

    A closure factory is required here: a bare ``for`` loop over
    ``@mcp.resource(uri)`` decorators would define and rebind the same
    function name on every iteration, so every registered URI would end up
    serving whichever ``file_name`` the loop last held. Calling this
    function once per URI gives each registration its own closure over its
    own ``file_name``.

    The ``name`` must be passed to the decorator explicitly, derived from
    the file name by ``resource_name``: the SDK captures the resource's name
    at decoration time, so assigning ``__name__`` to the already-decorated
    function afterwards would change nothing on the wire and every document
    would be advertised under the same name.
    """

    @mcp.resource(
        uri,
        name=resource_name(file_name),
        title=title,
        description=description,
        mime_type=_mime_type(file_name),
    )
    async def _read_document() -> str:
        """Read one document file.

        A missing or unreadable file is left to raise rather than being
        caught and turned into "Error: ..." text, matching ``live.py``'s
        resources: a resource read has no ``is_error`` flag the way a tool
        call does, so text content is the client's only signal, and
        returning fabricated-looking "document" text for a read that failed
        is indistinguishable from the real document. Letting the exception
        propagate gets the SDK's own ``read_resource`` wrapper
        (``mcp/server/mcpserver/server.py``) to turn it into a genuine
        protocol-level error (``ResourceError`` -> ``MCPError``) instead.
        """
        path = docs_directory() / file_name
        return _prepare(file_name, path.read_text(encoding="utf-8"))

"""Workflow prompts guide multi-tool sequences."""

from __future__ import annotations

import re

from smart_qdrant_mcp.resources.docs import DOC_RESOURCES, published_documents
from smart_qdrant_mcp.server import (
    SERVER_CONFIRM_TOOL_NAMES,
    SERVER_FILESYSTEM_TOOL_NAMES,
    _SERVER_INSTRUCTIONS,
)
from smart_qdrant_mcp.tool_map import TOOL_ROUTES

EXPECTED = {
    "qdrant_bootstrap_collection",
    "qdrant_diagnose_search_quality",
    "qdrant_safe_reindex",
}

# Sample arguments for each prompt, used to render its full text for the
# reality checks below.
SAMPLE_ARGUMENTS = {
    "qdrant_bootstrap_collection": {
        "collection_name": "docs",
        "source_path": "corpus/manual.json",
    },
    "qdrant_diagnose_search_quality": {
        "collection_name": "docs",
        "query": "token rotation",
    },
    "qdrant_safe_reindex": {"collection_name": "docs"},
}

TOOL_NAME_RE = re.compile(r"qdrant_[a-z_]+")
DOC_URI_RE = re.compile(r"docs://[a-zA-Z0-9/_-]+")
PARAM_ASSIGNMENT_RE = re.compile(r"(\w+)=")


def _prompt_text(result) -> str:
    """Join every text block of a get_prompt result into one string."""
    return " ".join(
        message.content.text
        for message in result.messages
        if getattr(message.content, "text", None)
    )


def _schema_property_names(schema: object) -> set[str]:
    """Collect every ``properties`` key of a JSON schema, at any depth.

    A tool's real fields sit one level under ``params`` in its input schema
    (see the ``call`` fixture in conftest.py), and pydantic may park nested
    models under ``$defs`` behind a ``$ref`` — a shallow look at top-level
    properties would see ``params`` and nothing else.
    """
    names: set[str] = set()
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == "properties" and isinstance(value, dict):
                names.update(value)
            names.update(_schema_property_names(value))
    elif isinstance(schema, list):
        for item in schema:
            names.update(_schema_property_names(item))
    return names


async def test_all_prompts_are_listed(mcp_client):
    names = {prompt.name for prompt in (await mcp_client.list_prompts()).prompts}
    assert names == EXPECTED


async def test_bootstrap_prompt_takes_a_collection_argument(mcp_client):
    prompts = {p.name: p for p in (await mcp_client.list_prompts()).prompts}
    argument_names = {
        arg.name for arg in prompts["qdrant_bootstrap_collection"].arguments
    }
    assert {"collection_name", "source_path"} <= argument_names


async def test_every_prompt_has_a_title_and_argument_descriptions(mcp_client):
    """A prompt's title and argument descriptions are its user-facing label.

    Clients render ``title`` as the human label and each argument's
    ``description`` as the help text for filling it in; the SDK leaves both
    unset unless the decorator and the parameter annotations provide them.
    """
    prompts = (await mcp_client.list_prompts()).prompts
    assert prompts, "expected at least one registered prompt"
    for prompt in prompts:
        assert prompt.title, f"{prompt.name!r} has no human title"
        for argument in prompt.arguments or []:
            assert argument.description, (
                f"{prompt.name!r} argument {argument.name!r} has no description"
            )


async def test_bootstrap_prompt_names_the_tools_in_order(mcp_client):
    result = await mcp_client.get_prompt(
        "qdrant_bootstrap_collection",
        {"collection_name": "docs", "source_path": "corpus/manual.json"},
    )
    text = _prompt_text(result)
    assert text.index("qdrant_create_collection") < text.index("qdrant_ingest_file")
    assert "qdrant_create_payload_index" in text
    assert "docs" in text and "corpus/manual.json" in text


async def test_diagnose_prompt_references_the_quality_checklist(mcp_client):
    result = await mcp_client.get_prompt(
        "qdrant_diagnose_search_quality",
        {"collection_name": "docs", "query": "token rotation"},
    )
    text = _prompt_text(result)
    assert "docs://workflows/search-recipes" in text
    assert "qdrant_verify_collection" in text


async def test_safe_reindex_prompt_starts_with_preflight(mcp_client):
    result = await mcp_client.get_prompt(
        "qdrant_safe_reindex", {"collection_name": "docs"}
    )
    text = _prompt_text(result)
    assert text.index("qdrant_reindex_preflight") < text.index("qdrant_start_reindex")
    assert "confirm" in text


async def test_safe_reindex_prompt_explains_the_409_is_intended(mcp_client):
    result = await mcp_client.get_prompt(
        "qdrant_safe_reindex", {"collection_name": "docs"}
    )
    text = _prompt_text(result)
    assert "409" in text
    assert "intended" in text or "by design" in text


async def test_bootstrap_prompt_checks_before_recreating(mcp_client):
    result = await mcp_client.get_prompt(
        "qdrant_bootstrap_collection",
        {"collection_name": "docs", "source_path": "corpus/manual.json"},
    )
    text = _prompt_text(result)
    assert text.index("qdrant_list_collections") < text.index(
        "qdrant_create_collection"
    )
    assert "already exists" in text
    assert "ask" in text.lower()


async def test_every_named_tool_and_document_is_real(mcp_client):
    """A prompt must never name a tool or document that does not exist.

    This is the reality check the brief calls for: every ``qdrant_*`` token
    a prompt's rendered text contains must be a key in ``TOOL_ROUTES``, and
    every ``docs://`` URI it contains must be a key in ``DOC_RESOURCES``.
    """
    for name, arguments in SAMPLE_ARGUMENTS.items():
        result = await mcp_client.get_prompt(name, arguments)
        text = _prompt_text(result)
        tool_names = {match.rstrip("_") for match in TOOL_NAME_RE.findall(text)}
        # rstrip("_") only trims trailing underscores accidentally captured
        # by the regex at a word boundary; real tool names never end in "_".
        for tool_name in tool_names:
            assert tool_name in TOOL_ROUTES, (
                f"{name!r} names {tool_name!r}, which is not a registered tool"
            )
        doc_uris = set(DOC_URI_RE.findall(text))
        for doc_uri in doc_uris:
            assert doc_uri in DOC_RESOURCES, (
                f"{name!r} points at {doc_uri!r}, which is not a published document"
            )
        # Each prompt must name at least one real tool and, where it
        # references documentation, at least one real document — otherwise
        # this check would pass vacuously on an empty prompt.
        assert tool_names


async def test_server_instructions_only_name_real_tools_and_documents(mcp_client):
    """Always-injected server guidance is held to the prompt reality checks."""
    listed_tools = {tool.name for tool in (await mcp_client.list_tools()).tools}
    listed_prompts = {
        prompt.name for prompt in (await mcp_client.list_prompts()).prompts
    }
    mentioned_names = {
        match.rstrip("_") for match in TOOL_NAME_RE.findall(_SERVER_INSTRUCTIONS)
    }
    mentioned_documents = set(DOC_URI_RE.findall(_SERVER_INSTRUCTIONS))

    assert mentioned_names
    assert mentioned_names <= listed_tools | listed_prompts
    assert mentioned_names - listed_prompts <= set(TOOL_ROUTES)
    assert mentioned_documents
    assert mentioned_documents <= set(published_documents())


async def test_server_confirmation_claims_match_real_tool_schemas(mcp_client):
    """The injected confirm list is derived from the tools' public schemas."""
    tools = {tool.name: tool for tool in (await mcp_client.list_tools()).tools}
    schema_confirm_tools = {
        name
        for name, tool in tools.items()
        if "confirm" in _schema_property_names(tool.input_schema)
    }

    assert set(SERVER_CONFIRM_TOOL_NAMES) == schema_confirm_tools
    assert set(SERVER_CONFIRM_TOOL_NAMES) <= set(TOOL_ROUTES)
    for tool_name in SERVER_CONFIRM_TOOL_NAMES:
        assert tool_name in _SERVER_INSTRUCTIONS


def test_server_filesystem_claims_name_the_two_real_boundary_tools():
    """Ingest reads staged data; export is the only server-side file write."""
    assert set(SERVER_FILESYSTEM_TOOL_NAMES) == {
        "qdrant_export_collection",
        "qdrant_ingest_file",
    }
    assert set(SERVER_FILESYSTEM_TOOL_NAMES) <= set(TOOL_ROUTES)
    for tool_name in SERVER_FILESYSTEM_TOOL_NAMES:
        assert tool_name in _SERVER_INSTRUCTIONS
    assert "no tool here writes files" not in _SERVER_INSTRUCTIONS
    assert "cannot be changed without a full reindex" not in _SERVER_INSTRUCTIONS


def test_server_instructions_state_strict_mode_and_untrusted_payload_text():
    """Clients without the skill still learn the two rules that bite first."""
    assert "strict mode" in _SERVER_INSTRUCTIONS
    assert "data, not instructions" in _SERVER_INSTRUCTIONS


async def test_prompt_parameter_references_exist_in_named_tool_schemas(mcp_client):
    """Every ``param=value`` a prompt spells out must be a real tool parameter.

    The prompts write tool parameters inline — ``response_format='json'``,
    ``rerank=true``, ``action='pause'`` — so renaming a parameter on the tool
    side would silently turn a prompt's instructions into calls that fail
    validation. Render each prompt, collect the tools it names, and require
    every ``param=`` token to exist in at least one of those tools' input
    schemas.
    """
    tools = {tool.name: tool for tool in (await mcp_client.list_tools()).tools}
    for name, arguments in SAMPLE_ARGUMENTS.items():
        result = await mcp_client.get_prompt(name, arguments)
        text = _prompt_text(result)
        mentioned = {match.rstrip("_") for match in TOOL_NAME_RE.findall(text)}
        properties: set[str] = set()
        for tool_name in mentioned:
            assert tool_name in tools, (
                f"{name!r} names {tool_name!r}, which is not a listed tool"
            )
            properties.update(_schema_property_names(tools[tool_name].input_schema))
        for token in set(PARAM_ASSIGNMENT_RE.findall(text)):
            assert token in properties, (
                f"{name!r} instructs {token}=..., which no named tool accepts"
            )

    # "confirm" is spelled in prose, not as confirm=..., in
    # qdrant_safe_reindex (it must equal the collection name exactly) — it
    # gates a destructive call, so pin it directly as well.
    result = await mcp_client.get_prompt(
        "qdrant_safe_reindex", SAMPLE_ARGUMENTS["qdrant_safe_reindex"]
    )
    text = _prompt_text(result)
    assert "confirm" in text
    start_properties = _schema_property_names(
        tools["qdrant_start_reindex"].input_schema
    )
    assert "confirm" in start_properties


async def test_bootstrap_prompt_says_the_corpus_file_must_already_be_staged(mcp_client):
    """The bootstrap prompt anticipates a missing source file.

    Step 5 calls qdrant_ingest_file on a path the agent was handed, so the
    prompt is where a missing file has to be anticipated: nothing on this
    server can create it, and the fallback is either asking for it to be
    staged or sending the chunks inline.
    """
    result = await mcp_client.get_prompt(
        "qdrant_bootstrap_collection",
        {"collection_name": "docs", "source_path": "corpus/manual.json"},
    )
    text = _prompt_text(result)
    assert "must already be there" in " ".join(text.split())
    assert "qdrant_ingest_chunks" in text
    assert "docs://workflows/ingest" in text

"""Regression tests for the product rename and its compatibility boundary."""

from html.parser import HTMLParser
from pathlib import Path
import re

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PRODUCT_NAME = "Smart Semantic Searcher"
MARKDOWN_PRODUCT_NAME = "Smart Semantic Searcher (3S)"
COMPOSE_PROJECT_NAME = "smart-semantic-searcher"
PRIVATE_NETWORK_NAME = "smart-semantic-searcher-private"
LEGACY_DISPLAY_NAMES = (
    "Personal Smart Embedder",
    "Qdrant Hybrid Manager",
    "Hybrid Manager",
)
STABLE_COMPOSE_IDENTIFIERS = (
    "hybrid-manager/middleware:local",
    "hybrid-manager/smart-qdrant-mcp:local",
    "hybrid-manager/smart-embedder:cpu-local",
    "hybrid-manager/smart-embedder:gpu-local",
    "smart-embedder-hf-cache",
    "smart-embedder-state-cpu",
    "smart-embedder-state-gpu",
)


def _read(relative_path: str) -> str:
    """Read one repository file as UTF-8 text."""
    return (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")


class _HtmlTitles(HTMLParser):
    """Collect visible text inside HTML H1 elements, including nested markup."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.titles = []
        self.parts = None

    def handle_starttag(self, tag, attrs):
        if tag == "h1":
            self.parts = []

    def handle_data(self, data):
        if self.parts is not None:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == "h1" and self.parts is not None:
            self.titles.append(" ".join("".join(self.parts).split()))
            self.parts = None


def _readme_titles(markdown: str) -> list[str]:
    """Find the README's visible HTML or ATX level-one title."""
    parser = _HtmlTitles()
    parser.feed(markdown)
    atx = re.findall(r"^#\s+(.+?)\s*#*\s*$", markdown, re.MULTILINE)
    return parser.titles + atx


def test_root_markdown_title_uses_the_approved_product_name():
    assert _readme_titles(_read("README.md")) == [MARKDOWN_PRODUCT_NAME]


@pytest.mark.parametrize(
    "markup",
    [
        "# Smart Semantic Searcher (3S)\n\n## Introduction",
        '<h1 align="center">Smart Semantic Searcher (3S)</h1>',
        "<h1>Smart Semantic <em>Searcher</em> (3S)</h1>",
    ],
)
def test_branding_title_accepts_both_heading_formats(markup):
    assert _readme_titles(markup) == [MARKDOWN_PRODUCT_NAME]


@pytest.mark.parametrize(
    "markup",
    [
        "## Smart Semantic Searcher (3S)",
        "<h1>Personal Smart Embedder</h1>",
        '<img alt="Smart Semantic Searcher (3S)"><h1>Wrong title</h1>',
        "<h1>Smart Semantic Searcher (3S)</h1><h1>Another title</h1>",
    ],
)
def test_branding_title_does_not_accept_missing_or_wrong_titles(markup):
    assert _readme_titles(markup) != [MARKDOWN_PRODUCT_NAME]


def test_compose_uses_new_project_and_network_without_renaming_volumes():
    compose = _read("compose.yml")

    assert compose.splitlines()[0] == f"name: {COMPOSE_PROJECT_NAME}"
    assert f"    name: {PRIVATE_NETWORK_NAME}" in compose
    assert "personal-smart-embedder" not in compose
    for identifier in STABLE_COMPOSE_IDENTIFIERS:
        assert identifier in compose


def test_root_runtime_copy_uses_the_full_name_without_the_acronym():
    paths = (
        ".env.example",
        "Dockerfile",
        "stack.ps1",
        "stack.sh",
        "scripts/README.md",
    )

    for relative_path in paths:
        text = _read(relative_path)
        assert PRODUCT_NAME in text, relative_path
        assert "(3S)" not in text, relative_path
        for legacy_name in LEGACY_DISPLAY_NAMES:
            assert legacy_name not in text, relative_path


def test_field_guides_use_current_product_branding():
    pages = sorted((REPOSITORY_ROOT / "guide").glob("*.html"))
    assert pages, "No field guide HTML found"
    for page in pages:
        text = page.read_text(encoding="utf-8")
        for legacy_name in LEGACY_DISPLAY_NAMES:
            assert legacy_name not in text, page.name
        assert "PSE /" not in text, page.name

"""Regression tests for the product rename and its compatibility boundary."""

from pathlib import Path


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


def test_root_markdown_title_uses_the_approved_product_name():
    lines = _read("README.md").splitlines()
    heading = next(line for line in lines if line.startswith("# "))

    assert heading == f"# {MARKDOWN_PRODUCT_NAME}"


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

"""Static contracts for security-sensitive console source and documentation."""

import re
from pathlib import Path

FRONTEND_ROOT = Path(__file__).resolve().parents[1] / "frontend" / "dist"
TEST_BOOK = (
    Path(__file__).resolve().parents[1]
    / "frontend"
    / "tests"
    / "testbook"
    / "test-book-e2e.md"
)
REMOVED_SINGLE_TOKEN_NAME = re.compile(r"\bMIDDLEWARE_API_TOKEN\b")


def test_frontend_never_recommends_the_removed_single_token_setting():
    for relative_path in ("assets/api.js", "assets/app.js", "index.html"):
        text = (FRONTEND_ROOT / relative_path).read_text(encoding="utf-8")
        assert REMOVED_SINGLE_TOKEN_NAME.search(text) is None, relative_path


def test_frontend_explains_that_the_stored_key_is_sent_as_bearer_auth():
    api_source = (FRONTEND_ROOT / "assets" / "api.js").read_text(encoding="utf-8")
    index_source = (FRONTEND_ROOT / "index.html").read_text(encoding="utf-8")

    assert "MIDDLEWARE_API_TOKENS" in api_source
    assert 'h["Authorization"] = `Bearer ${token}`' in api_source
    assert "sent as a bearer" in index_source
    assert "token to this page origin" in index_source
    assert "never sent" not in index_source


def test_frontend_scopes_the_api_key_to_session_storage():
    api_source = (FRONTEND_ROOT / "assets" / "api.js").read_text(encoding="utf-8")
    app_source = (FRONTEND_ROOT / "assets" / "app.js").read_text(encoding="utf-8")

    assert 'sessionStorage.getItem("qhm_api_key")' in api_source
    assert "sessionStorage.getItem(APIKEY_STORAGE_KEY)" in app_source
    assert "sessionStorage.setItem(APIKEY_STORAGE_KEY" in app_source
    assert 'localStorage.getItem("qhm_api_key")' not in api_source
    assert "localStorage.getItem(APIKEY_STORAGE_KEY)" not in app_source

    test_book = TEST_BOOK.read_text(encoding="utf-8")
    h5_row = next(line for line in test_book.splitlines() if line.startswith("| H5 |"))
    assert "sessionStorage" in h5_row
    assert "localStorage" not in h5_row


def test_frontend_presents_the_approved_product_name():
    index_source = (FRONTEND_ROOT / "index.html").read_text(encoding="utf-8")
    package_source = (FRONTEND_ROOT.parent / "package.json").read_text(encoding="utf-8")
    app_source = (FRONTEND_ROOT / "assets" / "app.js").read_text(encoding="utf-8")
    asset_paths = (
        FRONTEND_ROOT / "assets" / "api.js",
        FRONTEND_ROOT / "assets" / "app.js",
        FRONTEND_ROOT / "assets" / "style.css",
        FRONTEND_ROOT.parent / "tests" / "helpers.js",
    )

    assert "<title>Smart Semantic Searcher</title>" in index_source
    assert "SMART SEMANTIC SEARCHER" in index_source
    assert "(3S)" not in index_source
    assert "Smart Semantic Searcher" in package_source
    assert '"name": "qdrant-hybrid-manager-frontend-tests"' in package_source
    assert "Qdrant Hybrid Manager" not in package_source
    assert "/assets/style.css?v=83" in index_source
    assert "/assets/app.js?v=123" in index_source
    assert 'import { api } from "/assets/api.js?v=63";' in app_source
    for asset_path in asset_paths:
        source = asset_path.read_text(encoding="utf-8")
        assert "Smart Semantic Searcher" in source
        assert "Qdrant Hybrid Manager" not in source
        assert "(3S)" not in source

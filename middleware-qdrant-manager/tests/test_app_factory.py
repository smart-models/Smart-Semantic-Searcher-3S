"""Tests for create_app wiring: routes, frontend exposure, and swagger toggles."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from qdrant_middleware.core.config import Settings
from qdrant_middleware.main import create_app


def _frontend_paths(app: FastAPI) -> list[str]:
    group = app.router._frontend_routes
    return [] if group is None else [route.path for route in group.routes]


def _route_paths(app: FastAPI) -> set[str]:
    return {route.path for route in app.routes if getattr(route, "path", None)}


def test_fastapi_frontend_registration_is_available():
    assert callable(getattr(FastAPI, "frontend", None))


def test_create_app_attaches_settings_and_runtime_state():
    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )

    app = create_app(settings)

    assert isinstance(app, FastAPI)
    assert app.state.settings is settings
    assert hasattr(app.state, "runtime")
    assert app.state.runtime.stats.total_requests == 0


def test_create_app_returns_independent_runtime_state():
    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )

    app_one = create_app(settings)
    app_two = create_app(settings)
    app_one.state.runtime.stats.total_requests = 99

    assert app_two.state.runtime.stats.total_requests == 0


def test_create_app_preserves_default_docs_and_frontend():
    app = create_app(
        Settings(
            QDRANT_API_HOST="127.0.0.1",
            MIDDLEWARE_API_TOKENS="",
            MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        )
    )

    assert app.title == "Smart Semantic Searcher"
    assert app.docs_url == "/docs"
    assert app.redoc_url == "/redoc"
    assert app.openapi_url == "/openapi.json"
    assert "/assets/config.js" in _route_paths(app)
    assert _frontend_paths(app) == ["/"]


def test_frontend_config_exposes_query_runtime_settings():
    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        QUERY_DEFAULT_LIMIT=7,
        QUERY_MAX_LIMIT=80,
        QUERY_DEFAULT_CANDIDATE_LIMIT=70,
        QUERY_MAX_CANDIDATE_LIMIT=400,
        QUERY_DEFAULT_FUSION="dbsf",
        QUERY_DEFAULT_RERANK=False,
        QUERY_DEFAULT_RERANK_TOP_K=30,
        QUERY_MAX_RERANK_TOP_K=300,
        QUERY_RERANK_MAX_CHARS=2500,
    )

    response = TestClient(create_app(settings)).get("/assets/config.js")
    payload = json.loads(
        response.text.removeprefix("window.QHM_CONFIG=").removesuffix(";")
    )

    assert payload == {
        "authRequired": False,
        "queryDefaultLimit": 7,
        "queryMaxLimit": 80,
        "queryDefaultCandidateLimit": 70,
        "queryMaxCandidateLimit": 400,
        "queryDefaultFusion": "dbsf",
        "queryDefaultRerank": False,
        "queryDefaultRerankTopK": 30,
        "queryMaxRerankTopK": 300,
        "queryRerankMaxChars": 2500,
        "denseVectorName": "dense",
        "sparseVectorName": "sparse",
        "queryFilterMaxDepth": 20,
        "queryFilterMaxConditions": 100,
    }
    assert response.headers["cache-control"] == "no-store"


def test_frontend_config_exposes_configured_default_rerank_window():
    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        QUERY_DEFAULT_LIMIT=5,
        QUERY_DEFAULT_CANDIDATE_LIMIT=5,
        QUERY_DEFAULT_RERANK_TOP_K=20,
    )

    response = TestClient(create_app(settings)).get("/assets/config.js")
    payload = json.loads(
        response.text.removeprefix("window.QHM_CONFIG=").removesuffix(";")
    )

    assert payload["queryDefaultRerankTopK"] == 20


def _console_settings():
    return Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )


def test_console_assets_carry_defense_in_depth_headers():
    response = TestClient(create_app(_console_settings())).get("/assets/config.js")

    assert response.headers["content-security-policy"].startswith("default-src 'self'")
    assert "img-src 'self' data:" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"


@pytest.mark.parametrize(
    "path", ["/console/history", "/console/history/", "/unknown/path"]
)
def test_spa_fallback_carries_the_same_headers_as_root(path):
    client = TestClient(create_app(_console_settings()))
    root = client.get("/")
    fallback = client.get(path)
    assert fallback.status_code == 200
    assert fallback.headers["content-type"].startswith("text/html")
    for header in (
        "content-security-policy",
        "referrer-policy",
        "x-frame-options",
        "x-content-type-options",
    ):
        assert fallback.headers[header] == root.headers[header]


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/docs/oauth2-redirect"])
def test_swagger_documents_keep_their_existing_header_policy(path):
    settings = _console_settings()
    settings.expose_middleware_swagger = True
    response = TestClient(create_app(settings)).get(path)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "content-security-policy" not in response.headers


def test_script_src_is_not_widened_by_the_style_allowance():
    # 'unsafe-inline' belongs to style-src alone. Scripts are where an XSS
    # actually executes, and every console script is loaded from this origin,
    # so default-src must keep covering them without an inline allowance.
    policy = (
        TestClient(create_app(_console_settings()))
        .get("/")
        .headers["content-security-policy"]
    )
    default_src = policy.split("default-src", 1)[1].split(";", 1)[0]

    assert "unsafe-inline" not in default_src
    assert "script-src" not in policy


def test_style_src_allows_the_inline_widths_the_console_computes():
    # frontend/dist/assets/app.js sizes its score bars with a computed
    # style="width:NN%" attribute, and style-src governs style attributes too
    # (style-src-attr falls back to it). Dropping 'unsafe-inline' here renders
    # every bar at zero width -- a broken console that no test but this one
    # would notice, since the Playwright suite needs a live stack.
    policy = (
        TestClient(create_app(_console_settings()))
        .get("/")
        .headers["content-security-policy"]
    )
    style_src = policy.split("style-src", 1)[1].split(";", 1)[0]

    assert "'unsafe-inline'" in style_src


def test_json_api_routes_do_not_get_console_headers():
    # CSP is for the HTML/JS console; the JSON API is not a rendered document,
    # so adding the headers there would only surprise strict clients. /livez is
    # a plain JSON route needing no lifespan state.
    response = TestClient(create_app(_console_settings())).get("/livez")

    assert "content-security-policy" not in response.headers
    assert "x-frame-options" not in response.headers


def test_console_headers_disabled_with_the_frontend():
    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        EXPOSE_FRONTEND=False,
    )
    response = TestClient(create_app(settings)).get("/livez")

    assert "content-security-policy" not in response.headers


def test_frontend_can_be_disabled_without_removing_api_routes():
    app = create_app(
        Settings(
            QDRANT_API_HOST="127.0.0.1",
            MIDDLEWARE_API_TOKENS="",
            MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
            EXPOSE_FRONTEND=False,
        )
    )
    paths = set(app.openapi()["paths"])

    assert "/assets/config.js" not in _route_paths(app)
    assert _frontend_paths(app) == []
    assert {"/collections", "/points", "/query/search", "/health", "/metrics"} <= paths


def test_middleware_swagger_can_be_disabled_without_removing_api_routes():
    app = create_app(
        Settings(
            QDRANT_API_HOST="127.0.0.1",
            MIDDLEWARE_API_TOKENS="",
            MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
            EXPOSE_MIDDLEWARE_SWAGGER=False,
        )
    )

    assert app.docs_url is None
    assert app.redoc_url is None
    assert app.openapi_url is None
    assert {"/collections", "/points", "/query/search", "/health", "/metrics"} <= set(
        app.openapi()["paths"]
    )
    assert _frontend_paths(app) == ["/"]

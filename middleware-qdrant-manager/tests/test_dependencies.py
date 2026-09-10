"""Tests for FastAPI dependencies: bearer auth and lazy embedder creation."""

import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from qdrant_middleware.api.dependencies import (
    get_embedder_client,
    get_qdrant_client,
    get_request_settings,
    match_bearer_token,
    require_bearer_auth,
    require_operational_readiness,
)
from qdrant_middleware.core.config import Settings


class FakeRequest:
    """Minimal stand-in for a Starlette request carrying only app state."""

    def __init__(self):
        self.app = SimpleNamespace(state=SimpleNamespace())
        self.url = SimpleNamespace(path="/collections")


def test_get_request_settings_reads_app_state():
    request = FakeRequest()
    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )
    request.app.state.settings = settings

    assert get_request_settings(request) is settings


def test_get_qdrant_client_reads_app_state():
    request = FakeRequest()
    client = object()
    request.app.state.qdrant_client = client

    assert get_qdrant_client(request) is client


@pytest.mark.asyncio
async def test_require_bearer_auth_rejects_bad_token():
    request = FakeRequest()
    settings = Settings(
        QDRANT_API_HOST="127.0.0.1", MIDDLEWARE_API_TOKENS="expected:test"
    )

    with pytest.raises(HTTPException) as exc:
        await require_bearer_auth(request, credentials=None, settings=settings)

    assert exc.value.status_code == 401


def test_operational_readiness_rejects_when_a_dependency_is_down():
    request = FakeRequest()
    request.app.state.health_probe = SimpleNamespace(
        reachable=True,
        embedder_reachable=False,
    )

    with pytest.raises(HTTPException) as exc:
        require_operational_readiness(request)

    assert exc.value.status_code == 503
    assert exc.value.detail["code"] == "dependencies_unavailable"
    assert exc.value.detail["unavailable"] == ["Smart Embedder"]


def test_operational_routes_return_503_when_embedder_is_down():
    from qdrant_middleware.main import create_app

    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )
    app = create_app(settings)
    app.state.health_probe = SimpleNamespace(
        reachable=True,
        embedder_reachable=False,
    )

    response = TestClient(app).get("/collections")

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "dependencies_unavailable"


def test_openapi_marks_protected_routes_with_http_bearer():
    from qdrant_middleware.main import create_app

    settings = Settings(
        QDRANT_API_HOST="127.0.0.1", MIDDLEWARE_API_TOKENS="expected:test"
    )
    schema = TestClient(create_app(settings)).get("/openapi.json").json()

    security_schemes = schema["components"]["securitySchemes"]
    assert "HTTPBearer" in security_schemes
    assert schema["paths"]["/collections"]["get"]["security"] == [{"HTTPBearer": []}]


def test_disabled_swagger_paths_are_absent_not_relocated():
    from qdrant_middleware.main import create_app

    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        EXPOSE_MIDDLEWARE_SWAGGER=False,
    )
    client = TestClient(create_app(settings))

    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/livez").status_code == 200


def test_get_embedder_client_initializes_once_under_concurrency(monkeypatch):
    from qdrant_middleware.main import create_app

    created = []

    class FakeEmbedder:
        """Stand-in embedder handle used to check dependency wiring."""

        def __init__(self, settings):
            time.sleep(0.02)
            self.settings = settings
            created.append(self)

    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )
    app = create_app(settings)
    request = SimpleNamespace(app=app)
    monkeypatch.setattr(
        "qdrant_middleware.api.dependencies.EmbedderClient",
        FakeEmbedder,
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: get_embedder_client(request), range(8)))

    assert len(created) == 1
    assert all(result is results[0] for result in results)


# --- Multi-token bearer auth: any configured token authenticates, and the
# label it maps to is what reaches the log and request state. ---


def test_match_bearer_token_returns_the_label_of_any_entry():
    tokens = {"t1": "alice", "t2": "bob"}
    assert match_bearer_token("t2", tokens) == "bob"


def test_match_bearer_token_returns_none_for_an_unknown_token():
    assert match_bearer_token("nope", {"t1": "alice"}) is None


def test_match_bearer_token_returns_none_for_an_empty_map():
    assert match_bearer_token("t1", {}) is None


def _auth_client(**overrides):
    """Build a TestClient whose protected routes stop right after the auth check.

    The app is marked as having an unreachable embedder, so a request that
    authenticates successfully is refused by ``require_operational_readiness``
    with 503 instead of reaching Qdrant. That makes 401 vs 503 a direct readout
    of the bearer check alone, with no live dependency involved.
    """
    from qdrant_middleware.main import create_app

    app = create_app(Settings(QDRANT_API_HOST="127.0.0.1", **overrides))
    app.state.health_probe = SimpleNamespace(
        reachable=True,
        embedder_reachable=False,
    )
    return TestClient(app)


def test_each_configured_token_is_accepted():
    client = _auth_client(MIDDLEWARE_API_TOKENS="t1:alice,t2:bob")
    for token in ("t1", "t2"):
        response = client.get(
            "/collections", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 503


def test_the_service_token_is_accepted_without_being_listed():
    client = _auth_client(MIDDLEWARE_API_TOKENS="t1:alice", SERVICE_TOKEN="mcp-secret")
    response = client.get(
        "/collections", headers={"Authorization": "Bearer mcp-secret"}
    )
    assert response.status_code == 503


def test_unknown_token_is_rejected():
    client = _auth_client(MIDDLEWARE_API_TOKENS="t1:alice")
    response = client.get("/collections", headers={"Authorization": "Bearer nope"})
    assert response.status_code == 401


def test_auth_disabled_serves_without_a_header():
    client = _auth_client(
        MIDDLEWARE_API_TOKENS="", MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True
    )
    assert client.get("/collections").status_code == 503


def test_the_matched_label_is_logged_and_the_token_is_not(caplog):
    client = _auth_client(MIDDLEWARE_API_TOKENS="s3cret-token:alice")
    with caplog.at_level("INFO", logger="qdrant_middleware.api.dependencies"):
        client.get("/collections", headers={"Authorization": "Bearer s3cret-token"})
    assert "alice" in caplog.text
    assert "s3cret-token" not in caplog.text


def test_a_non_ascii_token_is_a_401_not_a_500():
    """ASGI decodes headers latin-1, so a byte >= 0x80 reaches the matcher.

    Comparing such a string with secrets.compare_digest raises TypeError; the
    live stack answered HTTP 500 to `Authorization: Bearer café` before the
    matcher encoded both sides to UTF-8 first.
    """
    assert match_bearer_token("caf\u00e9", {"t1": "alice"}) is None
    client = _auth_client(MIDDLEWARE_API_TOKENS="t1:alice")
    # Sent as raw bytes: the test HTTP client refuses a non-ASCII header
    # *string*, while a real client puts the byte on the wire and ASGI hands
    # it to the application decoded latin-1.
    response = client.get(
        "/collections",
        headers={"Authorization": "Bearer caf\u00e9".encode("latin-1")},
    )
    assert response.status_code == 401

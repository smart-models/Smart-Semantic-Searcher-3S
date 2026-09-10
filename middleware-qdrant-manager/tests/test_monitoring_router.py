"""Tests for the monitoring endpoints (livez, readyz, health, stats, metrics)."""

from fastapi.testclient import TestClient

from qdrant_middleware.core.config import Settings
from qdrant_middleware.main import create_app


def test_livez_does_not_require_qdrant_client():
    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )
    app = create_app(settings)

    response = TestClient(app).get("/livez")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readyz_uses_cached_probe_state():
    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )
    app = create_app(settings)
    app.state.health_probe = type(
        "Probe",
        (),
        {"reachable": True, "embedder_reachable": True, "collections": 3},
    )()

    response = TestClient(app).get("/readyz")

    assert response.status_code == 200
    assert response.json()["qdrant"] == "reachable"


def _probe(**attrs):
    return type("Probe", (), attrs)()


def test_health_reports_embedder_reachable_with_models():
    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )
    app = create_app(settings)
    app.state.health_probe = _probe(
        reachable=True,
        collections=2,
        embedder_reachable=True,
        embedder_profile="bge-m3",
        embedder_models={
            "dense": "BAAI/bge-m3",
            "sparse": "BAAI/bge-m3",
            "colbert": None,
            "reranker": "BAAI/bge-reranker-v2-m3",
        },
    )

    body = TestClient(app).get("/health").json()

    assert body["embedder"] == "reachable"
    assert body["embedder_profile"] == "bge-m3"
    assert body["embedder_models"]["reranker"] == "BAAI/bge-reranker-v2-m3"


def test_health_reports_embedder_unreachable():
    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )
    app = create_app(settings)
    app.state.health_probe = _probe(
        reachable=True,
        collections=0,
        embedder_reachable=False,
        embedder_profile=None,
        embedder_models=None,
    )

    response = TestClient(app).get("/health")
    body = response.json()

    assert response.status_code == 503
    assert body["status"] == "unhealthy"
    assert body["embedder"] == "unreachable"
    assert "embedder_models" not in body


def test_readyz_requires_qdrant_and_embedder():
    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )
    app = create_app(settings)
    app.state.health_probe = _probe(
        reachable=True,
        collections=3,
        embedder_reachable=False,
    )

    response = TestClient(app).get("/readyz")

    assert response.status_code == 503
    assert response.json()["qdrant"] == "reachable"
    assert response.json()["embedder"] == "unreachable"


def test_health_tolerates_probe_without_embedder_attrs():
    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )
    app = create_app(settings)
    app.state.health_probe = _probe(reachable=False)

    response = TestClient(app).get("/health")

    assert response.status_code == 503
    assert response.json()["embedder"] == "unknown"


def test_shutdown_guard_counts_toward_rejected_requests():
    """A shutdown-guard 503 is counted in rejected_requests.

    Regression for M2 (docs/audit-monitoring-endpoints-2026-07.md): this 503
    used to bypass the counter entirely, unlike the auth/rate-limit/body-size
    rejection paths.
    """
    settings = Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
    )
    app = create_app(settings)
    app.state.runtime.is_shutting_down = True

    response = TestClient(app).get("/stats")

    assert response.status_code == 503
    assert response.json() == {"detail": "Server is shutting down"}
    assert app.state.runtime.stats.rejected_requests == 1

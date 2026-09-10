"""Tests for the embedder side of the background health probe."""

from types import SimpleNamespace
from typing import ClassVar

import pytest

from qdrant_middleware.services import health_probe as hp


class FakeQdrant:
    """Fake Qdrant client whose collection listing always succeeds."""

    async def get_collections(self):
        return SimpleNamespace(collections=[1, 2])


class FakeResponse:
    """Minimal stand-in for an httpx response with a fixed status and body."""

    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeHttpClient:
    """Async-context httpx.AsyncClient stand-in serving canned responses."""

    responses: ClassVar[dict[str, object]] = {}

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url):
        result = self.responses[url.rsplit("/", 1)[-1]]
        if isinstance(result, Exception):
            raise result
        return result


@pytest.mark.asyncio
async def test_probe_caches_embedder_models(monkeypatch):
    FakeHttpClient.responses = {
        "health": FakeResponse(200, {"status": "healthy"}),
        "models": FakeResponse(
            200,
            {
                "profile": "bge-m3",
                "version": "2.1.0",
                "embedding_contract": {
                    "schema_version": 1,
                    "profile": "basic",
                    "dense": {
                        "model": "BAAI/bge-m3",
                        "revision": "dense-revision",
                        "dimension": 1024,
                        "normalization": "qdrant_cosine",
                    },
                    "sparse": {
                        "model": "BAAI/bge-m3",
                        "revision": "sparse-revision",
                        "encoding": "indices_values",
                    },
                    "fingerprint": f"sha256:{'a' * 64}",
                },
                "embedder": {
                    "dense": "BAAI/bge-m3",
                    "sparse": "BAAI/bge-m3",
                    "colbert": None,
                },
                "reranker": "BAAI/bge-reranker-v2-m3",
            },
        ),
    }
    monkeypatch.setattr(hp.httpx, "AsyncClient", FakeHttpClient)
    probe = hp.HealthProbe(FakeQdrant(), 60, embedder_url="http://embedder:8000")

    await probe.probe_once()

    assert probe.reachable and probe.collections == 2
    assert probe.embedder_reachable
    assert probe.embedder_profile == "bge-m3"
    assert probe.embedder_models["dense"] == "BAAI/bge-m3"
    assert probe.embedder_models["reranker"] == "BAAI/bge-reranker-v2-m3"
    assert probe.embedder_contract["profile"] == "basic"
    assert probe.embedder_version == "2.1.0"


@pytest.mark.asyncio
async def test_probe_marks_embedder_unreachable_on_error(monkeypatch):
    FakeHttpClient.responses = {"health": ConnectionError("refused")}
    monkeypatch.setattr(hp.httpx, "AsyncClient", FakeHttpClient)
    probe = hp.HealthProbe(FakeQdrant(), 60, embedder_url="http://embedder:8000")

    await probe.probe_once()

    assert probe.reachable  # qdrant unaffected
    assert not probe.embedder_reachable
    assert probe.embedder_profile is None
    assert probe.embedder_models is None
    assert probe.embedder_contract is None
    assert probe.embedder_version is None


@pytest.mark.asyncio
async def test_probe_health_ok_but_models_failing_keeps_reachable(monkeypatch):
    FakeHttpClient.responses = {
        "health": FakeResponse(200, {"status": "healthy"}),
        "models": FakeResponse(500),
    }
    monkeypatch.setattr(hp.httpx, "AsyncClient", FakeHttpClient)
    probe = hp.HealthProbe(FakeQdrant(), 60, embedder_url="http://embedder:8000")

    await probe.probe_once()

    assert probe.embedder_reachable
    assert probe.embedder_models is None
    assert probe.embedder_contract is None
    assert probe.embedder_version is None


@pytest.mark.asyncio
async def test_probe_health_ok_but_models_erroring_keeps_reachable(monkeypatch):
    """/models raising (network reset, timeout, ...) must not undo a healthy /health.

    Regression for M1 (docs/audit-monitoring-endpoints-2026-07.md): only a
    non-200 HTTP response from /models used to leave embedder_reachable alone;
    an exception from that same call incorrectly reset it to False.
    """
    FakeHttpClient.responses = {
        "health": FakeResponse(200, {"status": "healthy"}),
        "models": ConnectionError("models endpoint reset"),
    }
    monkeypatch.setattr(hp.httpx, "AsyncClient", FakeHttpClient)
    probe = hp.HealthProbe(FakeQdrant(), 60, embedder_url="http://embedder:8000")

    await probe.probe_once()

    assert probe.embedder_reachable
    assert probe.embedder_profile is None
    assert probe.embedder_models is None
    assert probe.embedder_contract is None
    assert probe.embedder_version is None


@pytest.mark.asyncio
async def test_probe_uses_configured_timeout(monkeypatch):
    """probe_timeout is wired from Settings rather than a fixed constant (M3)."""
    seen = {}

    class RecordingHttpClient(FakeHttpClient):
        """HTTP client capturing the timeout it was constructed with."""

        def __init__(self, *args, **kwargs):
            seen["timeout"] = kwargs.get("timeout")
            super().__init__(*args, **kwargs)

    RecordingHttpClient.responses = {
        "health": FakeResponse(200, {"status": "healthy"}),
        "models": FakeResponse(200, {}),
    }
    monkeypatch.setattr(hp.httpx, "AsyncClient", RecordingHttpClient)
    probe = hp.HealthProbe(
        FakeQdrant(), 60, embedder_url="http://embedder:8000", probe_timeout=9.5
    )

    await probe.probe_once()

    assert seen["timeout"] == 9.5


@pytest.mark.asyncio
async def test_probe_authorizes_protected_models_endpoint(monkeypatch):
    """The operational /models probe must use the configured bearer token."""
    seen = {}

    class RecordingHttpClient(FakeHttpClient):
        """HTTP client capturing the headers it was constructed with."""

        def __init__(self, *args, **kwargs):
            seen["headers"] = kwargs.get("headers")
            super().__init__(*args, **kwargs)

    RecordingHttpClient.responses = {
        "health": FakeResponse(200, {"status": "healthy"}),
        "models": FakeResponse(200, {}),
    }
    monkeypatch.setattr(hp.httpx, "AsyncClient", RecordingHttpClient)
    probe = hp.HealthProbe(
        FakeQdrant(),
        60,
        embedder_url="http://embedder:8000",
        embedder_client_token="probe-token",
    )

    await probe.probe_once()

    assert seen["headers"] == {"Authorization": "Bearer probe-token"}


@pytest.mark.asyncio
async def test_probe_without_embedder_url_reports_unreachable():
    probe = hp.HealthProbe(FakeQdrant(), 60)

    await probe.probe_once()

    assert not probe.embedder_reachable
    assert probe.embedder_contract is None
    assert probe.embedder_version is None

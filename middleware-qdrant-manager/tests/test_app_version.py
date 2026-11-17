"""Tests that every surface reporting the middleware version reads one constant."""

import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from qdrant_middleware import __version__
from qdrant_middleware.core import lifespan as lifecycle
from qdrant_middleware.core.config import Settings
from qdrant_middleware.main import create_app


def test_package_version_is_a_release_number():
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)


def test_openapi_reports_the_package_version():
    app = create_app(
        Settings(
            QDRANT_API_HOST="127.0.0.1",
            MIDDLEWARE_API_TOKENS="",
            MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        )
    )

    assert app.version == __version__
    assert app.openapi()["info"]["version"] == __version__


async def test_server_info_metric_reports_the_package_version(monkeypatch):
    info = Mock()
    probe = SimpleNamespace(probe_once=AsyncMock(), start=Mock(), stop=AsyncMock())
    state = SimpleNamespace(
        runtime=SimpleNamespace(
            rate_limiter=SimpleNamespace(
                start_cleanup_task=Mock(), stop_cleanup_task=AsyncMock()
            )
        ),
        settings=SimpleNamespace(
            api_tokens={},
            collection_name="docs",
            health_probe_interval=1,
            smart_embedder_url="http://embedder",
            embedder_probe_timeout=1,
            smart_embedder_client_token_value="",
        ),
    )
    monkeypatch.setattr(lifecycle, "server_info", info)
    monkeypatch.setattr(
        lifecycle, "create_qdrant_client", lambda _: SimpleNamespace(close=AsyncMock())
    )
    monkeypatch.setattr(
        lifecycle, "EmbedderClient", lambda _: SimpleNamespace(close=AsyncMock())
    )
    monkeypatch.setattr(lifecycle, "HealthProbe", lambda *a, **k: probe)
    monkeypatch.setattr(lifecycle, "_align_existing_strict_mode", AsyncMock())

    async with lifecycle.lifespan(SimpleNamespace(state=state)):
        pass

    info.info.assert_called_once_with({"version": __version__, "collection": "docs"})

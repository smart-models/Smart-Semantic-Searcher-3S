"""Tests for the module-level app object and create_app independence."""

from fastapi import FastAPI

from qdrant_middleware.core.config import Settings


def test_package_entrypoint_exports_app_and_factory(monkeypatch):
    """The module-level app builds lazily; create_app stays independent."""
    monkeypatch.setenv("MIDDLEWARE_API_TOKENS", "entrypoint:test")
    from qdrant_middleware.main import app, create_app

    assert isinstance(app, FastAPI)
    assert isinstance(
        create_app(Settings(MIDDLEWARE_API_TOKENS="entrypoint:factory")),
        FastAPI,
    )

"""Shared pytest configuration for the middleware suite.

Keeps the suite hermetic: it must measure the code, never the machine it runs
on.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

import pytest

from qdrant_middleware.core import config as config_module
from qdrant_middleware.core.config import Settings
from qdrant_middleware.core.mutation_coordinator import MutationCoordinator
from qdrant_middleware.services.vector_reindex import VectorReindexManager


@pytest.fixture
def mutation_context():
    """Give direct helper calls one shared mutation context for the whole test."""
    coordinator = MutationCoordinator()
    return {
        "coordinator": coordinator,
        "manager": VectorReindexManager(coordinator),
    }


@pytest.fixture
def observe_mutation_attempt(monkeypatch):
    """Observe entry attempts without replacing the coordinator's real locking."""

    def observe(coordinator):
        attempted = asyncio.Event()
        original = coordinator.lease

        @asynccontextmanager
        async def lease(collection_name):
            attempted.set()
            async with original(collection_name):
                yield

        monkeypatch.setattr(coordinator, "lease", lease)
        return attempted

    return observe


@pytest.fixture(autouse=True, scope="session")
def isolate_from_the_repository_env_file():
    """Stop ``Settings()`` from reading the repository-root ``.env``.

    ``Settings`` points ``env_file`` at the real ``.env`` so the app picks up a
    deployment's configuration. Under pytest that turns a developer checkout --
    which has one -- into a different test subject from CI, which does not: a
    test asserting a documented default reads whatever the developer deployed
    instead, and passes or fails on that. Tests that need a file supply their
    own via an explicit ``_env_file=``, which still wins over this default;
    tests that need a variable set it through ``monkeypatch.setenv``, which
    reaches ``os.environ`` and is unaffected here.

    The same file is also what ``get_settings()`` scans for variable names the
    multi-token change removed, so that scan is pointed away from it too: a
    developer whose ``.env`` is not migrated yet must not fail the suite.
    """
    original_env_file = Settings.model_config.get("env_file")
    original_scan_file = config_module.QDRANT_ENV_FILE
    Settings.model_config["env_file"] = None
    config_module.QDRANT_ENV_FILE = None
    try:
        yield
    finally:
        Settings.model_config["env_file"] = original_env_file
        config_module.QDRANT_ENV_FILE = original_scan_file


@pytest.fixture(autouse=True, scope="session")
def provide_a_default_inbound_acl():
    """Give every bare ``Settings()`` in the suite a valid inbound ACL.

    An empty ``MIDDLEWARE_API_TOKENS`` is now a boot error unless the operator
    opts out explicitly, so a test that constructs ``Settings()`` to assert an
    unrelated default would otherwise have to carry a token argument it does
    not care about. The value lands in ``os.environ``, which loses to any
    explicit constructor argument, so the tests that *do* exercise the token
    contract still control it completely -- including the ones that pass an
    empty list on purpose.
    """
    previous = os.environ.get("MIDDLEWARE_API_TOKENS")
    os.environ["MIDDLEWARE_API_TOKENS"] = "conftest-token:tests"
    try:
        yield
    finally:
        if previous is None:
            del os.environ["MIDDLEWARE_API_TOKENS"]
        else:
            os.environ["MIDDLEWARE_API_TOKENS"] = previous

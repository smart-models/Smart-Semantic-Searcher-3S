"""Tests for transient Qdrant error classification (is_transient_qdrant)."""

import httpx

from qdrant_middleware.repositories.qdrant_collections import is_transient_qdrant


def test_transport_error_is_transient():
    assert is_transient_qdrant(httpx.ConnectError("boom")) is True


def test_value_error_is_not_transient():
    assert is_transient_qdrant(ValueError("bad input")) is False

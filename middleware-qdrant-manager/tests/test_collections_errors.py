"""Collection dependency failures expose context without upstream secrets."""

import json

import pytest
from fastapi.testclient import TestClient
from qdrant_client.http.exceptions import UnexpectedResponse

from qdrant_middleware.api.dependencies import get_qdrant_client
from qdrant_middleware.api.routers import collections
from qdrant_middleware.core.config import Settings
from qdrant_middleware.main import create_app


@pytest.mark.parametrize(
    "path, operation, message",
    [
        ("/collections", "list_qdrant_collections", "Qdrant collection list failed"),
        ("/collections/docs", "collection_report", "Qdrant collection lookup failed"),
    ],
)
@pytest.mark.parametrize("upstream_status", [None, 401, 500, 503])
def test_collection_dependency_failures_do_not_expose_raw_errors(
    monkeypatch, path, operation, message, upstream_status
):
    secret = "http://synthetic-user:synthetic-secret@private_host:6333"
    if upstream_status is None:
        failure = RuntimeError(secret)
    else:
        failure = UnexpectedResponse(
            status_code=upstream_status,
            reason_phrase="upstream failed",
            content=json.dumps({"status": {"error": secret}}).encode(),
            headers={},
        )

    async def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(collections, operation, fail)
    app = create_app(
        Settings(
            _env_file=None,
            MIDDLEWARE_API_TOKENS="",
            MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        )
    )
    app.dependency_overrides[get_qdrant_client] = lambda: object()
    response = TestClient(app).get(path)
    assert response.status_code == 502
    assert response.json() == {"detail": message}
    assert "synthetic-secret" not in response.text
    assert "private_host" not in response.text


@pytest.mark.parametrize("status", [400, 422])
def test_qdrant_validation_errors_keep_their_actionable_reason(status):
    failure = UnexpectedResponse(
        status_code=status,
        reason_phrase="invalid input",
        content=b'{"status":{"error":"Unknown vector name"}}',
        headers={},
    )
    mapped = collections._qdrant_failure(failure, "Collection lookup failed")
    assert mapped.status_code == 400
    assert mapped.detail == "Collection lookup failed: Unknown vector name"

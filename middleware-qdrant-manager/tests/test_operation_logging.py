"""Every non-GET request logs one ``operation`` line: who, what, outcome, how long.

The lines are read from the package's real stream handler, not ``caplog``:
``qdrant_middleware`` does not propagate, so ``caplog`` either misses these
records or, when pytest attaches to the logger itself, passes even if the
real handler were missing. Uvicorn's own access line is covered here too: it
stays, without its query string.
"""

import io
import logging
import re

from fastapi.testclient import TestClient

import qdrant_middleware.api.routers.query as query_router
from qdrant_middleware.api.dependencies import get_embedder_client, get_qdrant_client
from qdrant_middleware.core.config import Settings
from qdrant_middleware.main import create_app
from qdrant_middleware.services.query_executor import QueryHit
from test_points_ingest_source_id import FakeEmbedder, FakeQdrant

TOKEN = "s3cret-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
BAD_AUTH = {"Authorization": "Bearer wrong-token"}
SECRET_TEXT = "salary of the chief executive"
FILE_NAME = "payroll-2026.pdf"
QUERY_STRING = "?trace=query-string-value"
DURATION = r"duration=\d+\.\d{3}s"


def _client(fake=None, **overrides) -> TestClient:
    """Build the full app, with every middleware layer, over fake backends."""
    settings = {
        "QDRANT_COLLECTION": "docs",
        "MIDDLEWARE_API_TOKENS": f"{TOKEN}:alice",
        **overrides,
    }
    app = create_app(Settings(**settings))
    app.dependency_overrides[get_qdrant_client] = lambda: fake or object()
    app.dependency_overrides[get_embedder_client] = lambda: FakeEmbedder()
    return TestClient(app)


def _operation_lines(*requests) -> list[str]:
    """Send ``requests`` and return the ``operation`` lines the handler wrote."""
    handler = logging.getLogger("qdrant_middleware").handlers[0]
    buffer = io.StringIO()
    previous = handler.setStream(buffer)
    try:
        for send in requests:
            send()
    finally:
        handler.setStream(previous)
    output = buffer.getvalue()
    for never_logged in (SECRET_TEXT, FILE_NAME, TOKEN, "query-string-value"):
        assert never_logged not in output
    return [line for line in output.splitlines() if "operation " in line]


def _line(method: str, path: str, fields: str) -> str:
    """Return the full-line pattern for one operation."""
    return rf"INFO: +operation {method} {re.escape(path)}: {fields} {DURATION}"


def test_an_ingest_logs_its_collection_and_the_points_written():
    client = _client(FakeQdrant())
    body = {
        "chunks": [{"id": 1, "text": SECRET_TEXT}],
        "source_id": FILE_NAME,
        "collection_name": "docs",
    }
    lines = _operation_lines(
        lambda: client.post(
            f"/points/ingest-chunks{QUERY_STRING}", json=body, headers=AUTH
        ),
        lambda: client.get("/livez"),
    )
    assert len(lines) == 1
    assert re.fullmatch(
        _line(
            "POST",
            "/points/ingest-chunks",
            "caller=alice status=200 collection=docs points=1",
        ),
        lines[0],
    )


def test_a_search_logs_the_points_returned_and_never_the_query(monkeypatch):
    async def one_hit(*args, **kwargs):
        return [QueryHit(id="p1", score=0.9, payload={"text": SECRET_TEXT})]

    monkeypatch.setattr(query_router, "run_hybrid_query", one_hit)
    client = _client()
    body = {"query": SECRET_TEXT, "rerank": False}
    lines = _operation_lines(
        lambda: client.post(f"/query/search{QUERY_STRING}", json=body, headers=AUTH)
    )
    assert len(lines) == 1
    assert re.fullmatch(
        _line(
            "POST", "/query/search", "caller=alice status=200 collection=docs points=1"
        ),
        lines[0],
    )


def test_a_collection_operation_takes_the_collection_from_its_path():
    client = _client(
        MIDDLEWARE_API_TOKENS="", MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True
    )
    lines = _operation_lines(lambda: client.delete(f"/collections/docs{QUERY_STRING}"))
    assert len(lines) == 1
    assert re.fullmatch(
        _line(
            "DELETE",
            "/collections/docs",
            "caller=- status=400 collection=docs points=-",
        ),
        lines[0],
    )


def test_a_rejected_request_logs_no_caller_and_stays_on_one_line():
    client = _client()
    lines = _operation_lines(
        lambda: client.post("/points/scroll", json={}, headers=BAD_AUTH),
        lambda: client.delete("/collections/x%0AINFO:%20forged", headers=BAD_AUTH),
    )
    assert len(lines) == 2
    assert re.fullmatch(
        _line("POST", "/points/scroll", "caller=- status=401 collection=- points=-"),
        lines[0],
    )
    # request.url.path drops the newline (urlsplit strips it); the collection
    # path parameter keeps it and is escaped instead.
    assert re.fullmatch(
        _line(
            "DELETE",
            "/collections/xINFO: forged",
            r"caller=- status=401 collection=x\\nINFO: forged points=-",
        ),
        lines[1],
    )


def _access_line(path_with_query: str) -> str:
    """Log one access line exactly as uvicorn does; return what a handler got."""
    access = logging.getLogger("uvicorn.access")
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setLevel(logging.INFO)
    previous_level = access.level
    access.addHandler(handler)
    access.setLevel(logging.INFO)
    try:
        access.info(
            '%s - "%s %s HTTP/%s" %d',
            "127.0.0.1:50000",
            "POST",
            path_with_query,
            "1.1",
            401,
        )
    finally:
        access.removeHandler(handler)
        access.setLevel(previous_level)
    return buffer.getvalue()


def test_the_access_log_keeps_its_line_but_drops_the_query_string():
    _client()
    line = _access_line("/query/search?trace=qs-value")
    assert '"POST /query/search HTTP/1.1" 401' in line
    assert "qs-value" not in line

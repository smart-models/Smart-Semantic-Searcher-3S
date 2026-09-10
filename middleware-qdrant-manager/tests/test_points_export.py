"""Tests for the POST /points/export endpoint."""

import asyncio
import json
import threading
import httpx
import pytest
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

from qdrant_middleware.api.dependencies import get_qdrant_client
from qdrant_middleware.core.config import Settings
from qdrant_middleware.main import create_app


@pytest.mark.parametrize("destination", ["body", "file"])
def test_export_growth_past_cap_publishes_nothing(tmp_path, destination):
    qdrant = FakeQdrant(
        count=1,
        pages=[([_point("first")], "next"), ([_point("second")], None)],
    )
    app = _app(tmp_path, max_points=1)
    app.dependency_overrides[get_qdrant_client] = lambda: qdrant
    response = TestClient(app).post("/points/export", json={"destination": destination})
    assert response.status_code == 413
    assert "EXPORT_MAX_POINTS" in response.json()["detail"]
    assert qdrant.scroll_calls == 2
    assert list(tmp_path.iterdir()) == []


class FakeQdrant:
    """Fake Qdrant client serving a fixed page of points to the exporter."""

    def __init__(self, pages=None, count=1, scroll_error=None):
        self.pages = list(pages if pages is not None else [([_point()], None)])
        self.count_value = count
        self.scroll_error = scroll_error
        self.scroll_calls = 0

    async def scroll(self, **kwargs):
        self.scroll_calls += 1
        if self.scroll_error is not None:
            raise self.scroll_error
        return self.pages.pop(0)

    async def count(self, **kwargs):
        return SimpleNamespace(count=self.count_value)


def _point(point_id="uuid-a", payload=None):
    return SimpleNamespace(id=point_id, payload=payload or {"text": "alpha"})


def _app(tmp_path, **overrides):
    settings = Settings(
        QDRANT_COLLECTION="docs",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        EXPORT_MAX_POINTS=overrides.pop("max_points", 50_000),
        **overrides,
    )
    settings.export_data_root = tmp_path
    return create_app(settings)


@pytest.mark.parametrize("destination", ["body", "file"])
def test_export_byte_overflow_publishes_nothing_and_releases_capacity(
    tmp_path, destination
):
    app = _app(tmp_path, EXPORT_MAX_BYTES=20)
    qdrant = FakeQdrant()
    app.dependency_overrides[get_qdrant_client] = lambda: qdrant
    response = TestClient(app).post("/points/export", json={"destination": destination})
    assert response.status_code == 413
    assert "EXPORT_MAX_BYTES" in response.json()["detail"]
    assert app.state.runtime.active_exports == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("phase", ["count", "scroll"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_export_concurrency_refuses_before_count_and_releases(
    tmp_path, phase, cancel
):
    entered, release = asyncio.Event(), asyncio.Event()
    qdrant = FakeQdrant()
    original = getattr(qdrant, phase)

    async def blocked(**kwargs):
        entered.set()
        await release.wait()
        return await original(**kwargs)

    setattr(qdrant, phase, blocked)
    app = _app(tmp_path)
    app.dependency_overrides[get_qdrant_client] = lambda: qdrant
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        pending = asyncio.create_task(
            client.post("/points/export", json={"destination": "body"})
        )
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert app.state.runtime.active_exports == 1
            refused = await client.post("/points/export", json={"destination": "file"})
            assert refused.status_code == 503
            assert refused.headers["retry-after"] == "1"
            assert qdrant.scroll_calls == 0
            if cancel:
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                release.set()
                assert (await pending).status_code == 200
            assert app.state.runtime.active_exports == 0
            release.set()
            qdrant.pages = [([_point()], None)]
            assert (
                await client.post("/points/export", json={"destination": "body"})
            ).status_code == 200
            assert list(tmp_path.iterdir()) == []
        finally:
            release.set()
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


async def test_cancelled_export_keeps_capacity_until_serializer_finishes(
    tmp_path, monkeypatch
):
    entered, release = threading.Event(), threading.Event()
    original = json.JSONEncoder.iterencode

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    app = _app(tmp_path)
    app.dependency_overrides[get_qdrant_client] = lambda: FakeQdrant()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Patch after constructing the HTTP body to isolate export serialization.
        body = json.dumps({"destination": "file"})
        monkeypatch.setattr(json.JSONEncoder, "iterencode", blocked)
        pending = asyncio.create_task(
            client.post(
                "/points/export",
                content=body,
                headers={"content-type": "application/json"},
            )
        )
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            pending.cancel()
            await asyncio.sleep(0.01)
            assert app.state.runtime.active_exports == 1
            assert not pending.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, 2)
            assert app.state.runtime.active_exports == 0
            assert list(tmp_path.iterdir()) == []
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)


def test_export_to_file_writes_the_document_and_reports_the_path(tmp_path):
    qdrant = FakeQdrant()
    app = _app(tmp_path)
    app.dependency_overrides[get_qdrant_client] = lambda: qdrant

    response = TestClient(app).post("/points/export", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["collection_name"] == "docs"
    assert body["n_chunks"] == 1
    written = list(tmp_path.glob("docs_export_*.json"))
    assert len(written) == 1
    assert body["path"] == str(written[0])
    assert body["bytes"] == len(written[0].read_bytes())
    parsed = json.loads(written[0].read_text(encoding="utf-8"))
    assert parsed["chunks"] == [{"text": "alpha", "point_id": "uuid-a"}]
    assert parsed["metadata"]["mode"] == "collection-export"
    # No temp file survives an atomic write.
    assert [p.name for p in tmp_path.iterdir()] == [written[0].name]


def test_export_to_body_returns_the_same_bytes_as_an_attachment(tmp_path):
    qdrant = FakeQdrant()
    app = _app(tmp_path)
    app.dependency_overrides[get_qdrant_client] = lambda: qdrant

    response = TestClient(app).post("/points/export", json={"destination": "body"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    disposition = response.headers["content-disposition"]
    assert disposition.startswith('attachment; filename="docs_export_')
    assert "; filename*=UTF-8''docs_export_" in disposition
    assert json.loads(response.text)["metadata"]["n_chunks"] == 1
    # Nothing is written to disk for the browser path.
    assert list(tmp_path.iterdir()) == []


def test_export_of_an_empty_collection_writes_no_file(tmp_path):
    qdrant = FakeQdrant(pages=[([], None)], count=0)
    app = _app(tmp_path)
    app.dependency_overrides[get_qdrant_client] = lambda: qdrant

    response = TestClient(app).post("/points/export", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["n_chunks"] == 0
    assert body["path"] is None
    assert list(tmp_path.iterdir()) == []


def test_export_refuses_a_collection_over_the_cap_before_scrolling(tmp_path):
    qdrant = FakeQdrant(count=51)
    app = _app(tmp_path, max_points=50)
    app.dependency_overrides[get_qdrant_client] = lambda: qdrant

    response = TestClient(app).post("/points/export", json={})

    assert response.status_code == 413
    detail = response.json()["detail"]
    assert "51" in detail and "50" in detail
    assert qdrant.scroll_calls == 0
    assert list(tmp_path.iterdir()) == []


def test_export_maps_a_failing_scroll_to_502_and_writes_nothing(tmp_path):
    qdrant = FakeQdrant(scroll_error=RuntimeError("qdrant is down"))
    app = _app(tmp_path)
    app.dependency_overrides[get_qdrant_client] = lambda: qdrant

    response = TestClient(app).post("/points/export", json={})

    assert response.status_code == 502
    assert "qdrant is down" in response.json()["detail"]
    assert list(tmp_path.iterdir()) == []


def test_export_rejects_an_unknown_destination(tmp_path):
    qdrant = FakeQdrant()
    app = _app(tmp_path)
    app.dependency_overrides[get_qdrant_client] = lambda: qdrant

    response = TestClient(app).post("/points/export", json={"destination": "s3"})

    # The app maps every RequestValidationError to 400 (see main.py's
    # exception handler) rather than FastAPI's default 422 — consistent
    # with every other router's body-validation failures.
    assert response.status_code == 400


def test_export_maps_an_unwritable_export_root_to_502(tmp_path):
    # A missing mount or a full disk is a dependency failure like an
    # unreachable Qdrant: the documented contract for a failed export is 502
    # with nothing written, not a bare 500 from an uncaught OSError.
    qdrant = FakeQdrant()
    app = _app(tmp_path)
    app.dependency_overrides[get_qdrant_client] = lambda: qdrant

    with mock.patch("os.replace", side_effect=OSError("read-only file system")):
        response = TestClient(app).post("/points/export", json={})

    assert response.status_code == 502
    assert "read-only file system" in response.json()["detail"]
    # The temp file is cleaned up, so no half-written export is left behind.
    assert list(tmp_path.iterdir()) == []


def test_export_fsyncs_the_payload_before_publishing_it(tmp_path):
    # The file must be forced onto stable storage before the rename publishes
    # it, so a power cut cannot leave a zero-length file under the final name.
    qdrant = FakeQdrant()
    app = _app(tmp_path)
    app.dependency_overrides[get_qdrant_client] = lambda: qdrant
    calls = []

    real_fsync = __import__("os").fsync
    real_replace = __import__("os").replace

    def recording_fsync(fd):
        calls.append("fsync")
        return real_fsync(fd)

    def recording_replace(src, dst):
        calls.append("replace")
        return real_replace(src, dst)

    with (
        mock.patch("os.fsync", recording_fsync),
        mock.patch("os.replace", recording_replace),
    ):
        response = TestClient(app).post("/points/export", json={})

    assert response.status_code == 200
    # The payload fsync happens before the rename into place.
    assert calls[:2] == ["fsync", "replace"]

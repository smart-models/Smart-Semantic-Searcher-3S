"""Backend-layer acceptance for the console test book (`@pytest.mark.book`).

Each test proves the Backend oracle of one test-book row that the coverage
contract assigns to this suite: TB-API-001/002, TB-EXP-003/004/009/010,
TB-ING-005/006, TB-LIFE-002/004, TB-REI-006/007 and NF-X6b. Everything runs
against fakes and event barriers; nothing here needs Qdrant, Smart Embedder,
a browser or the network. Rows with UI clauses keep those clauses in the
browser suite -- this file never claims them.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from qdrant_client import models
from test_vector_reindex import _Embedder as _ReindexEmbedder
from test_vector_reindex import _Qdrant as _ReindexQdrant
from test_vector_reindex import _wait_for_worker

from qdrant_middleware.api.dependencies import (
    get_embedder_client,
    get_mutation_coordinator,
    get_qdrant_client,
    get_vector_reindex_manager,
)
from qdrant_middleware.api.routers import points
from qdrant_middleware.clients.embedder import EmbeddingResult
from qdrant_middleware.core.config import Settings
from qdrant_middleware.core.mutation_coordinator import MutationCoordinator
from qdrant_middleware.main import create_app
from qdrant_middleware.repositories.qdrant_collections import (
    EmbeddingContractViolation,
    get_embedding_metadata_state,
    require_collection_mutable,
)
from qdrant_middleware.services import collection_export, query_executor
from qdrant_middleware.services.collection_schema import make_point_id
from qdrant_middleware.services.vector_reindex import VectorReindexManager

BOOK_EXPORT = [
    {
        "text": f"export document {i}",
        "source_id": "e2e_book_export",
        "id": i,
        "marker": f"m-{i}",
        "unicode": "à漢🙂",
    }
    for i in range(1, 2002)
]


def _settings(**overrides) -> Settings:
    values = dict(
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        QDRANT_COLLECTION="docs",
        RETRY_MAX_ATTEMPTS=1,
        RETRY_BASE_DELAY=0,
        EXPOSE_FRONTEND=False,
    )
    values.update(overrides)
    return Settings(**values)


class _Store:
    """Record Qdrant effects while preserving original point payloads."""

    def __init__(self):
        self.points: dict = {}
        self.upserts: list = []
        self.deletes: list = []
        self.indexes: list = []
        self.upload_failure: int | None = None
        self.prune_failure = False

    async def get_collection(self, name):
        return SimpleNamespace(config=SimpleNamespace(), payload_schema={})

    async def retrieve(self, **kwargs):
        return [SimpleNamespace(id=i) for i in kwargs["ids"] if i in self.points]

    async def scroll(self, **kwargs):
        expected = kwargs["scroll_filter"].must[0].match.value
        return [
            SimpleNamespace(id=i)
            for i, p in self.points.items()
            if p.payload["source_id"] == expected
        ], None

    async def create_payload_index(self, **kwargs):
        self.indexes.append(kwargs)

    async def upsert(self, **kwargs):
        self.upserts.append(kwargs)
        if len(self.upserts) == self.upload_failure:
            raise RuntimeError("injected upload failure")
        for point in kwargs["points"]:
            self.points[point.id] = deepcopy(point)

    async def delete(self, **kwargs):
        self.deletes.append(kwargs)
        if self.prune_failure:
            raise RuntimeError("injected prune failure")
        for point_id in kwargs["points_selector"].points:
            self.points.pop(point_id, None)


class _Embedder:
    """Return deterministic vectors and capture the requested text order."""

    max_batch_size = 1

    def __init__(self, fail_call: int | None = None):
        self.calls: list[list[str]] = []
        self.fail_call = fail_call

    async def embed(self, texts):
        self.calls.append(list(texts))
        if len(self.calls) == self.fail_call:
            raise RuntimeError("injected embedding failure")
        return [
            EmbeddingResult(
                dense=[1.0, 0.0],
                sparse=models.SparseVector(indices=[1], values=[1.0]),
            )
            for _ in texts
        ]


# --------------------------------------------------------------------------
# TB-API-001 / TB-API-002 / NF-X6b
# --------------------------------------------------------------------------


@pytest.mark.book("TB-API-001")
@pytest.mark.parametrize(
    "mode,existing,expected",
    [
        ("insert_only", [1, 2], []),
        ("update_only", [], []),
        ("insert_only", [1], [2]),
        ("update_only", [1], [1]),
    ],
)
async def test_skipped_update_modes_invoke_no_embedding_or_upsert(
    mode, existing, expected
):
    store, embedder, coordinator = _Store(), _Embedder(), MutationCoordinator()
    store.points = dict.fromkeys(existing)
    result = await points.batch_upsert_points(
        points.BatchUpsertRequest(
            points=[{"point_id": str(i), "text": f"text-{i}"} for i in (1, 2)],
            update_mode=mode,
        ),
        store,
        _settings(),
        coordinator,
        VectorReindexManager(coordinator),
        embedder,
    )
    assert result.upserted == len(expected)
    assert result.skipped == 2 - len(expected)
    assert embedder.calls == ([[f"text-{i}" for i in expected]] if expected else [])
    assert [p.id for call in store.upserts for p in call["points"]] == expected


@pytest.mark.book("TB-API-002")
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("override", [False, True])
async def test_search_params_reach_their_qdrant_location(
    quantized, override, monkeypatch
):
    captured = []

    async def has_quantization(*args):
        return quantized

    async def query_points(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(points=[])

    monkeypatch.setattr(query_executor, "collection_has_quantization", has_quantization)
    conf = _settings(
        QUERY_HNSW_EF=128,
        QUERY_QUANTIZATION_RESCORE=True,
        QUERY_QUANTIZATION_OVERSAMPLING=2,
        MIDDLEWARE_REQUEST_TIMEOUT=17,
    )
    options = (
        dict(hnsw_ef=256, exact=True, rescore=False, oversampling=4, timeout=30)
        if override
        else {}
    )
    request = query_executor.QueryRequest(
        query="query", rerank=False, **options
    ).apply_defaults(conf)
    embedding = (await _Embedder().embed(["query"]))[0]
    await query_executor.run_hybrid_query(
        request, embedding, SimpleNamespace(query_points=query_points), conf
    )
    (call,) = captured
    assert call["timeout"] == (30 if override else 17)
    assert len(call["prefetch"]) == 2
    for prefetch in call["prefetch"]:
        assert prefetch.params.hnsw_ef == (256 if override else 128)
        assert prefetch.params.exact is (True if override else None)
        if quantized:
            # An explicit rescore=false must survive as False, never be dropped.
            assert prefetch.params.quantization.rescore is (not override)
            assert prefetch.params.quantization.oversampling == (4 if override else 2)
        else:
            assert prefetch.params.quantization is None


@pytest.mark.book("TB-API-002")
@pytest.mark.parametrize(
    "invalid",
    [{"hnsw_ef": 0}, {"oversampling": 0.5}, {"timeout": 0}, {"timeout": 3601}],
)
def test_invalid_search_params_are_refused_before_any_embedding_or_query(invalid):
    store, embedder = _Store(), _Embedder()
    app = create_app(_settings())
    app.dependency_overrides[get_qdrant_client] = lambda: store
    app.dependency_overrides[get_embedder_client] = lambda: embedder
    response = TestClient(app).post("/query/search", json={"query": "q", **invalid})
    assert response.status_code == 400
    assert embedder.calls == [] and store.upserts == []


@pytest.mark.book("book:13.6/NF-X6b")
async def test_offset_beyond_the_matching_set_yields_an_empty_page_without_error():
    class _Qdrant:
        async def query_points(self, **kwargs):
            return SimpleNamespace(
                points=[
                    SimpleNamespace(id=i, score=1.0 - i / 10, payload={"text": f"t{i}"})
                    for i in range(3)
                ]
            )

    conf = _settings()
    request = query_executor.QueryRequest(
        query="query", rerank=False, limit=5, candidate_limit=20, offset=10
    ).apply_defaults(conf)
    embedding = (await _Embedder().embed(["query"]))[0]
    hits = await query_executor.run_hybrid_query(request, embedding, _Qdrant(), conf)
    assert len(hits) == 3
    # The offset is applied exactly once, on the full pool: a window that starts
    # beyond the matching set is an empty page, not an error.
    response = query_executor._build_response(
        request, hits, reranked=False, candidates=len(hits)
    )
    assert response.results == []
    assert response.returned == 0
    assert response.offset == 10


# --------------------------------------------------------------------------
# TB-EXP-004 / TB-EXP-009 / TB-EXP-010 (backend clauses)
# --------------------------------------------------------------------------


class _ExportStore:
    """Serve deterministic pages and capture every offset and vector selector."""

    def __init__(self, size, count=None, page_size=None, text="à漢"):
        self.records = [
            SimpleNamespace(
                id=i + 1,
                payload={"id": i, "text": f"{text}-{i}", "source_id": "e2e-book"},
            )
            for i in range(size)
        ]
        self.count_value = size if count is None else count
        self.page_size = page_size
        self.calls: list[dict] = []

    async def count(self, **kwargs):
        return SimpleNamespace(count=self.count_value)

    async def scroll(self, **kwargs):
        self.calls.append(kwargs)
        start = kwargs.get("offset") or 0
        end = min(start + (self.page_size or kwargs["limit"]), len(self.records))
        return self.records[start:end], end if end < len(self.records) else None


def _export_app(tmp_path, store, **overrides):
    settings = _settings(**{"EXPORT_MAX_POINTS": 50_000, **overrides})
    settings.export_data_root = tmp_path
    app = create_app(settings)
    app.dependency_overrides[get_qdrant_client] = lambda: store
    return app


async def _export(app, destination):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.post("/points/export", json={"destination": destination})


def _published(tmp_path, response, destination):
    if destination == "body":
        return response.content
    return Path(response.json()["path"]).read_bytes()


@pytest.mark.book("TB-EXP-003")
async def test_unicode_collection_name_yields_an_ascii_rfc5987_attachment(tmp_path):
    """A Unicode collection name is delivered through ``filename*`` only.

    The header must stay ASCII (no raw non-ASCII bytes), the fallback filename
    must be sanitized, the encoded name must decode back to the collection name
    and the body must carry the Unicode text intact.
    """
    name = "e2e_book_à漢"
    app = _export_app(tmp_path, _ExportStore(1, text="à漢🙂"))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/points/export", json={"collection_name": name, "destination": "body"}
        )
    assert response.status_code == 200, response.text
    disposition = response.headers["content-disposition"]
    assert disposition.isascii()
    fallback = re.search(r'filename="([^"]+)"', disposition).group(1)
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", fallback)
    encoded = re.search(r"filename\*=UTF-8''(\S+)", disposition).group(1)
    decoded = unquote(encoded)
    assert decoded.startswith(f"{name}_export_")
    assert decoded.endswith(".json")
    document = json.loads(response.content.decode("utf-8"))
    assert document["metadata"]["n_chunks"] == 1
    assert document["chunks"][0]["point_id"] == "1"
    assert document["chunks"][0]["text"] == "à漢🙂-0"


@pytest.mark.book("TB-EXP-004")
async def test_body_and_file_deliver_the_identical_document_bytes(
    tmp_path, monkeypatch
):
    document = collection_export.ExportDocument(
        '{"chunks":[{"text":"à漢"}]}', 1, "fixed.json"
    )

    async def fixed_document(*args, **kwargs):
        return document

    monkeypatch.setattr(points, "build_export_document", fixed_document)
    body = await _export(_export_app(tmp_path, _ExportStore(1)), "body")
    saved = await _export(_export_app(tmp_path, _ExportStore(1)), "file")
    assert body.status_code == saved.status_code == 200
    assert body.content == Path(saved.json()["path"]).read_bytes()
    assert body.content == document.text.encode("utf-8")


@pytest.mark.book("TB-EXP-004")
async def test_two_exports_in_the_same_second_never_overwrite_each_other(
    tmp_path, monkeypatch
):
    real_builder = collection_export.build_export_document

    async def frozen_builder(*args, **kwargs):
        return await real_builder(
            *args, **kwargs, now=datetime(2026, 9, 5, tzinfo=timezone.utc)
        )

    monkeypatch.setattr(points, "build_export_document", frozen_builder)
    first = await _export(_export_app(tmp_path, _ExportStore(1, text="first")), "file")
    original = Path(first.json()["path"]).read_bytes()
    second = await _export(
        _export_app(tmp_path, _ExportStore(1, text="second")), "file"
    )
    assert first.json()["path"] != second.json()["path"]
    assert Path(first.json()["path"]).read_bytes() == original
    assert len(list(tmp_path.iterdir())) == 2


@pytest.mark.book("TB-EXP-004")
@pytest.mark.parametrize("failure", ["write", "fsync", "replace", "second-page"])
async def test_io_or_scroll_failure_publishes_nothing_and_keeps_earlier_exports(
    tmp_path, failure, monkeypatch
):
    previous = tmp_path / "previous.json"
    previous.write_bytes(b"completed export")
    store = _ExportStore(2001)
    if failure == "second-page":
        original_scroll = store.scroll

        async def failing_second_page(**kwargs):
            if kwargs.get("offset") == 1000:
                store.calls.append(kwargs)
                raise RuntimeError("injected scroll page failure")
            return await original_scroll(**kwargs)

        store.scroll = failing_second_page
    elif failure in {"fsync", "replace"}:

        def fail(*args, **kwargs):
            raise OSError(f"injected {failure} failure")

        monkeypatch.setattr(os, failure, fail)
    else:
        original_fdopen = os.fdopen

        class _FailedWriter:
            def __init__(self, *args, **kwargs):
                self.handle = original_fdopen(*args, **kwargs)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.handle.close()

            def write(self, data):
                raise OSError("injected write failure")

        monkeypatch.setattr(os, "fdopen", _FailedWriter)
    response = await _export(_export_app(tmp_path, store), "file")
    assert response.status_code == 502
    assert list(tmp_path.iterdir()) == [previous]
    assert previous.read_bytes() == b"completed export"
    if failure == "second-page":
        assert len(store.calls) == 2


@pytest.mark.book("TB-EXP-004")
@pytest.mark.parametrize("destination", ["body", "file"])
async def test_transient_second_page_exhausts_retries_and_publishes_nothing(
    tmp_path, destination
):
    store = _ExportStore(2001)
    original_scroll = store.scroll

    async def failed_second_page(**kwargs):
        if kwargs.get("offset") == 1000:
            store.calls.append(kwargs)
            raise httpx.ConnectError("injected lost dependency")
        return await original_scroll(**kwargs)

    store.scroll = failed_second_page
    app = _export_app(tmp_path, store, RETRY_MAX_ATTEMPTS=3)
    response = await _export(app, destination)
    assert response.status_code == 502
    assert [call["offset"] for call in store.calls] == [None, 1000, 1000, 1000]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.book("TB-EXP-008")
@pytest.mark.parametrize("destination", ["body", "file"])
async def test_export_at_the_point_cap_succeeds_and_one_over_is_refused_before_io(
    tmp_path, destination
):
    """cap=3: a three-point collection exports whole; four points are a 413.

    The over-cap refusal must come from the count preflight, so no scroll page
    is read and no file is created or left behind.
    """
    at_cap = _ExportStore(3)
    app = _export_app(tmp_path, at_cap, EXPORT_MAX_POINTS=3)
    response = await _export(app, destination)
    assert response.status_code == 200, response.text
    document = json.loads(_published(tmp_path, response, destination))
    assert [chunk["point_id"] for chunk in document["chunks"]] == ["1", "2", "3"]
    assert document["metadata"]["n_chunks"] == 3
    published = list(tmp_path.iterdir())
    over_cap = _ExportStore(4)
    refused = await _export(
        _export_app(tmp_path, over_cap, EXPORT_MAX_POINTS=3), destination
    )
    assert refused.status_code == 413
    assert "EXPORT_MAX_POINTS=3" in refused.json()["detail"]
    assert over_cap.calls == []
    assert list(tmp_path.iterdir()) == published


@pytest.mark.book("TB-EXP-009")
@pytest.mark.parametrize("destination", ["body", "file"])
async def test_point_cap_is_enforced_incrementally_after_a_low_preflight_count(
    tmp_path, destination
):
    # Preflight counts 2, then the pages total 4 against a cap of 3: the walk
    # must stop at the violating page and publish nothing.
    store = _ExportStore(4, count=2, page_size=2)
    app = _export_app(tmp_path, store, EXPORT_MAX_POINTS=3)
    response = await _export(app, destination)
    assert response.status_code == 413
    assert "EXPORT_MAX_POINTS" in response.json()["detail"]
    assert len(store.calls) == 2
    assert list(tmp_path.iterdir()) == []
    assert app.state.runtime.active_exports == 0


@pytest.mark.book("TB-EXP-009")
@pytest.mark.parametrize(
    "text", ["x" * 4200, "漢" * 1400], ids=["ascii", "multibyte-straddling"]
)
@pytest.mark.parametrize("destination", ["body", "file"])
async def test_byte_budget_counts_the_whole_utf8_document_and_aborts_early(
    tmp_path, text, destination
):
    store = _ExportStore(1, text=text)
    app = _export_app(tmp_path, store, EXPORT_MAX_BYTES=4096)
    response = await _export(app, destination)
    assert response.status_code == 413
    assert "EXPORT_MAX_BYTES" in response.json()["detail"]
    assert len(store.calls) == 1
    assert list(tmp_path.iterdir()) == []
    # The same point fits when the budget is measured against its real size.
    fitting = await _export(
        _export_app(tmp_path, _ExportStore(1, text=text), EXPORT_MAX_BYTES=1 << 20),
        "body",
    )
    assert fitting.status_code == 200
    assert 4096 < len(fitting.content)
    if text.startswith("漢"):
        assert len(fitting.content.decode("utf-8")) < 4096


@pytest.mark.book("TB-EXP-010")
async def test_serialization_never_blocks_an_unrelated_request(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original_add_points = collection_export._ExportBuffer.add_points

    def held_add_points(self, page):
        entered.set()
        assert release.wait(5), "serialization barrier was never released"
        return original_add_points(self, page)

    monkeypatch.setattr(collection_export._ExportBuffer, "add_points", held_add_points)
    store = _ExportStore(2001)
    app = _export_app(tmp_path, store)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        export = asyncio.create_task(
            client.post("/points/export", json={"destination": "body"})
        )
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            # Serialization is held on its worker thread; the loop must still
            # answer an unrelated liveness probe before it is released.
            heartbeat = await asyncio.wait_for(client.get("/livez"), 2)
            assert heartbeat.status_code == 200
        finally:
            release.set()
        response = await asyncio.wait_for(export, 10)
    assert response.status_code == 200
    assert json.loads(response.content)["metadata"]["n_chunks"] == 2001


# --------------------------------------------------------------------------
# TB-ING-005 / TB-ING-006
# --------------------------------------------------------------------------


@pytest.mark.book("TB-ING-005")
def test_ingest_requires_the_shared_coordinator_and_never_invents_one():
    with pytest.raises(TypeError, match="coordinator"):
        points._run_ingest(
            chunks=[],
            collection_name="docs",
            batch_size=1,
            replace_source=True,
            sid="e2e-book",
            metadata={},
            embedder=_Embedder(),
            client=_Store(),
            settings=_settings(),
            manager=VectorReindexManager(),
        )
    app = create_app(_settings())
    request = SimpleNamespace(app=app)
    assert get_vector_reindex_manager(request).coordinator is get_mutation_coordinator(
        request
    )
    with pytest.raises(AttributeError):
        get_mutation_coordinator(
            SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
        )


@pytest.mark.book("TB-ING-005")
async def test_concurrent_same_source_replacements_are_serialized_by_the_lease():
    coordinator = MutationCoordinator()
    store = _Store()
    first_entered, release_first, second_entered = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    original_upsert = store.upsert

    async def gated_upsert(**kwargs):
        text = kwargs["points"][0].payload["text"]
        if text.startswith("first"):
            first_entered.set()
            await release_first.wait()
        else:
            second_entered.set()
        return await original_upsert(**kwargs)

    store.upsert = gated_upsert

    def run(chunks):
        return asyncio.create_task(
            points._run_ingest(
                chunks=chunks,
                collection_name="docs",
                batch_size=1,
                replace_source=True,
                sid="e2e-book",
                metadata={},
                embedder=_Embedder(),
                client=store,
                settings=_settings(
                    INGEST_UPLOAD_CONCURRENCY=1, SMART_EMBEDDER_EMBED_CONCURRENCY=1
                ),
                coordinator=coordinator,
                manager=VectorReindexManager(coordinator),
            )
        )

    first = run([{"id": 1, "text": "first-1"}])
    await asyncio.wait_for(first_entered.wait(), 2)
    second = run([{"id": 2, "text": "second-2"}])
    await asyncio.sleep(0.05)
    assert not second_entered.is_set(), "second replacement entered the mutation body"
    release_first.set()
    await asyncio.wait_for(asyncio.gather(first, second), 5)
    assert second_entered.is_set()
    # The latest replacement is the only complete version of the source.
    assert {p.payload["text"] for p in store.points.values()} == {"second-2"}
    assert coordinator._entries == {}


@pytest.mark.book("TB-ING-006")
@pytest.mark.parametrize("phase", ["embedding", "upload", "prune"])
async def test_ingest_failure_phases_preserve_the_original_source_and_control(phase):
    store, coordinator = _Store(), MutationCoordinator()
    original_source = {
        make_point_id("e2e-book", 1): SimpleNamespace(
            payload={"source_id": "e2e-book", "text": "old"}
        )
    }
    control = make_point_id("control", 1)
    store.points = {
        **deepcopy(original_source),
        control: SimpleNamespace(payload={"source_id": "control", "text": "control"}),
    }
    originals = deepcopy(store.points)
    embedder = _Embedder(fail_call=2 if phase == "embedding" else None)
    store.upload_failure = 2 if phase == "upload" else None
    store.prune_failure = phase == "prune"
    invocation = points._run_ingest(
        chunks=[{"id": 2, "text": "second"}, {"id": 3, "text": "third"}],
        collection_name="docs",
        batch_size=1,
        replace_source=True,
        sid="e2e-book",
        metadata={},
        embedder=embedder,
        client=store,
        settings=_settings(
            INGEST_UPLOAD_CONCURRENCY=1, SMART_EMBEDDER_EMBED_CONCURRENCY=1
        ),
        coordinator=coordinator,
        manager=VectorReindexManager(coordinator),
    )
    if phase == "prune":
        with pytest.raises(HTTPException) as caught:
            await invocation
        assert caught.value.status_code == 502
        assert len(store.upserts) == 2
    else:
        response = await invocation
        assert not response.ok
        assert response.ingested == (0 if phase == "embedding" else 1)
        assert response.failed_at_chunk == 1
        assert store.deletes == []
        if phase == "embedding":
            assert store.upserts == store.indexes == []
            assert store.points == originals
    assert store.points[control] == originals[control]
    assert all(store.points[i] == original for i, original in original_source.items())
    assert coordinator._entries == {}


# --------------------------------------------------------------------------
# TB-LIFE-002 / TB-LIFE-004
# --------------------------------------------------------------------------


@pytest.mark.book("TB-LIFE-004")
@pytest.mark.parametrize("cancel_owner", [False, True])
@pytest.mark.parametrize("termination", ["cancel", "timeout"])
async def test_cancelled_or_timed_out_lease_users_never_enter_later(
    cancel_owner, termination
):
    coordinator = MutationCoordinator()
    owner_entered, release, waiter_entered = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )

    async def owner():
        async with coordinator.lease("same"):
            owner_entered.set()
            await release.wait()

    async def waiter():
        async with coordinator.lease("same"):
            waiter_entered.set()

    first = asyncio.create_task(owner())
    await owner_entered.wait()
    second = asyncio.create_task(waiter())

    async def both_registered():
        while coordinator._entries["same"].users != 2:
            await asyncio.sleep(0)

    await asyncio.wait_for(both_registered(), 2)
    async with coordinator.lease("different"):
        assert not waiter_entered.is_set()
    cancelled = first if cancel_owner else second
    if termination == "cancel":
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
    else:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(cancelled, 0.01)
    release.set()
    await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 2)
    assert waiter_entered.is_set() is cancel_owner
    async with coordinator.lease("same"):
        pass
    assert coordinator._entries == {}


@pytest.mark.book("TB-LIFE-002")
async def test_request_timeout_cancels_the_handler_before_it_writes():
    app = create_app(_settings(MIDDLEWARE_REQUEST_TIMEOUT=1))
    release, response_started = asyncio.Event(), asyncio.Event()
    writes, cancellations, cleanup = [], [], []
    coordinator = MutationCoordinator()

    @app.get("/acceptance-held-write")
    async def held_write():
        async with coordinator.lease("docs"):
            try:
                await release.wait()
                writes.append("new write after timeout")
                return {"ok": True}
            except asyncio.CancelledError:
                cancellations.append("cancelled")
                raise
            finally:
                cleanup.append("released")

    delivered = False
    messages = []

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await asyncio.Event().wait()

    async def send(message):
        messages.append(message)
        if message["type"] == "http.response.start":
            response_started.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/acceptance-held-write",
        "raw_path": b"/acceptance-held-write",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("127.0.0.1", 123),
        "server": ("test", 80),
    }
    task = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(response_started.wait(), 3)
        assert messages[0]["status"] == 504
        release.set()
        await asyncio.wait_for(task, 2)
        assert cleanup == ["released"] and coordinator._entries == {}
        assert writes == [], "handler issued a new write after the 504"
        assert cancellations == ["cancelled"]
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


# --------------------------------------------------------------------------
# TB-REI-006 / TB-REI-007
# --------------------------------------------------------------------------


@pytest.mark.book("TB-REI-006")
@pytest.mark.parametrize("failure", ["embedding", "upload", "interrupted"])
@pytest.mark.parametrize("terminal", ["complete", "cancel"])
async def test_reindex_recovers_from_the_committed_checkpoint(failure, terminal):
    class _RecordingStore(_ReindexQdrant):
        def __init__(self):
            super().__init__(deepcopy(BOOK_EXPORT))
            self.uploads = 0
            self.offsets: list = []

        async def scroll(self, **kwargs):
            self.offsets.append(kwargs.get("offset"))
            return await super().scroll(**kwargs)

        async def update_vectors(self, **kwargs):
            self.uploads += 1
            assert kwargs["wait"] is True
            if failure == "upload" and self.uploads == 2:
                raise RuntimeError("injected second vector upload failure")
            return await super().update_vectors(**kwargs)

    class _FailingEmbedder(_ReindexEmbedder):
        def __init__(self):
            super().__init__()
            self.second_entered, self.release = asyncio.Event(), asyncio.Event()

        async def embed(self, texts):
            if self.embed_calls == 2:
                if failure == "embedding":
                    raise RuntimeError("injected second worker embedding failure")
                if failure == "interrupted":
                    self.second_entered.set()
                    await self.release.wait()
            return await super().embed(texts)

    store, embedder, manager = _RecordingStore(), _FailingEmbedder(), None
    manager = VectorReindexManager()
    originals = deepcopy([p["payload"] for p in store.points])
    started = await manager.start(store, embedder, "docs", _settings())
    if failure == "interrupted":
        await asyncio.wait_for(embedder.second_entered.wait(), 2)
        await manager.stop()
        assert manager._tasks == {}
    else:
        await _wait_for_worker(manager)
    _, _, active, checkpoint = await get_embedding_metadata_state(
        store, "docs", _settings()
    )
    assert checkpoint.status == (
        "interrupted" if failure == "interrupted" else "failed"
    )
    assert checkpoint.processed_points == checkpoint.last_offset == 2
    assert active == checkpoint.source_vectors and active != started.target_vectors
    assert [p["payload"] for p in store.points] == originals
    with pytest.raises(EmbeddingContractViolation):
        await require_collection_mutable(store, "docs", _settings())
    # A new manager stands for a process restart: only the durable metadata in
    # the fake Qdrant carries progress across instances.
    restarted = VectorReindexManager()
    if terminal == "cancel":
        await restarted.set_control_status(store, "docs", _settings(), "cancelled")
    else:
        store.offsets.clear()
        await restarted.resume(store, _ReindexEmbedder(), "docs", _settings())
        await _wait_for_worker(restarted)
        assert store.offsets[0] == 2, "resume re-read data before the checkpoint"
    _, _, final_active, final = await get_embedding_metadata_state(
        store, "docs", _settings()
    )
    assert final.status == ("completed" if terminal == "complete" else "cancelled")
    assert final.processed_points == (2001 if terminal == "complete" else 2)
    assert final_active == (
        started.target_vectors if terminal == "complete" else started.source_vectors
    )
    assert [p["payload"] for p in store.points] == originals
    await require_collection_mutable(store, "docs", _settings())
    await restarted.stop()


@pytest.mark.book("TB-REI-007")
@pytest.mark.parametrize("fail_commit", [False, True])
async def test_cutover_only_after_durable_data_and_commit_failure_stays_coherent(
    fail_commit,
):
    class _HeldCommitStore(_ReindexQdrant):
        def __init__(self):
            super().__init__(deepcopy(BOOK_EXPORT))
            self.entered, self.release = asyncio.Event(), asyncio.Event()
            self.fail_once = fail_commit
            self.updates_durable = 0
            self.queries: list = []

        async def query_points(self, **kwargs):
            self.queries.append(kwargs)
            return SimpleNamespace(points=[])

        async def update_vectors(self, **kwargs):
            assert kwargs["wait"] is True
            await super().update_vectors(**kwargs)
            self.updates_durable += len(kwargs["points"])

        async def update_collection(self, **kwargs):
            metadata = kwargs.get("metadata") or {}
            job = metadata.get("hybrid_manager", {}).get("reindex_job") or {}
            if job.get("status") == "completed":
                self.entered.set()
                await self.release.wait()
                if self.fail_once:
                    self.fail_once = False
                    raise RuntimeError("injected failure before the active commit")
            return await super().update_collection(**kwargs)

    store, manager = _HeldCommitStore(), VectorReindexManager()

    async def inspect_query(active):
        request = query_executor.QueryRequest(
            query="query", collection_name="docs", rerank=False
        ).apply_defaults(_settings())
        embedding = (await _Embedder().embed(["query"]))[0]
        await query_executor.run_hybrid_query(
            request, embedding, store, _settings(), vector_names=active
        )
        assert [p.using for p in store.queries[-1]["prefetch"]] == [
            active.dense,
            active.sparse,
        ]

    originals = deepcopy([p["payload"] for p in store.points])
    started = await manager.start(store, _ReindexEmbedder(), "docs", _settings())
    try:
        await asyncio.wait_for(store.entered.wait(), 5)
        _, contract, active, job = await get_embedding_metadata_state(
            store, "docs", _settings()
        )
        assert active == started.source_vectors
        assert contract == started.source_contract
        await inspect_query(active)
        assert job.status == "finalizing"
        assert store.updates_durable == len(store.points)
        assert all(
            started.target_vectors.dense in p["vectors"]
            and started.target_vectors.sparse in p["vectors"]
            for p in store.points
        )
        store.release.set()
        await _wait_for_worker(manager)
        _, _, active, job = await get_embedding_metadata_state(
            store, "docs", _settings()
        )
        if fail_commit:
            assert job.status == "failed" and active == started.source_vectors
            await inspect_query(active)
            assert started.source_vectors.dense not in store.deleted_names
            restarted = VectorReindexManager()
            await restarted.resume(store, _ReindexEmbedder(), "docs", _settings())
            await _wait_for_worker(restarted)
            await restarted.stop()
            raw_job = store.config.metadata["hybrid_manager"]["reindex_job"]
            assert raw_job["processed_points"] <= raw_job["total_points"], (
                "resume corrupted the checkpoint: "
                f"processed={raw_job['processed_points']}, "
                f"total={raw_job['total_points']}, "
                f"offset={raw_job['last_offset']}, status={raw_job['status']}"
            )
        _, contract, active, completed = await get_embedding_metadata_state(
            store, "docs", _settings()
        )
        assert active == started.target_vectors
        assert contract == started.target_contract
        await inspect_query(active)
        assert completed.status == "completed"
        assert [p["payload"] for p in store.points] == originals
    finally:
        store.release.set()
        await manager.stop()

"""Safety and resumability tests for collection vector regeneration."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest
from qdrant_client.models import SparseVector

from qdrant_middleware.api.dependencies import get_embedder_client, get_qdrant_client
from qdrant_middleware.clients.embedder import EmbeddingResult
from qdrant_middleware.core.config import Settings
from qdrant_middleware.main import create_app
from qdrant_middleware.repositories.qdrant_collections import (
    EmbeddingContractViolation,
    get_embedding_metadata_state,
    patch_embedding_metadata,
    require_collection_mutable,
    require_embedding_fingerprint,
)
from qdrant_middleware.schemas.embedding_contract import (
    EmbedderContractInfo,
    EmbeddingContract,
)
from qdrant_middleware.services.vector_reindex import (
    VectorReindexManager,
    preflight_reindex,
)


def _settings() -> Settings:
    return Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        RETRY_MAX_ATTEMPTS=1,
        RETRY_BASE_DELAY=0,
    )


def _contract(digest: str) -> EmbeddingContract:
    return EmbeddingContract(
        schema_version=1,
        profile="basic",
        dense={
            "model": "dense-model",
            "revision": "dense-revision",
            "dimension": 2,
            "normalization": "qdrant_cosine",
        },
        sparse={
            "model": "sparse-model",
            "revision": "sparse-revision",
            "encoding": "indices_values",
        },
        fingerprint=f"sha256:{digest * 64}",
    )


class _Embedder:
    """Embedder stand-in serving a contract and fixed vectors to the job."""

    max_batch_size = 2

    def __init__(self, *, bad_worker_fingerprint: bool = False, block_worker=False):
        self.info = EmbedderContractInfo(
            contract=_contract("b"),
            server_version="test",
        )
        self.bad_worker_fingerprint = bad_worker_fingerprint
        self.block_worker = block_worker
        self.embed_calls = 0
        self.worker_entered = asyncio.Event()
        self.worker_release = asyncio.Event()

    async def get_embedding_contract(self):
        return self.info

    async def embed(self, texts):
        self.embed_calls += 1
        if self.block_worker and self.embed_calls == 2:
            self.worker_entered.set()
            await self.worker_release.wait()
        fingerprint = self.info.contract.fingerprint
        if self.bad_worker_fingerprint and self.embed_calls >= 2:
            fingerprint = _contract("c").fingerprint
        return [
            EmbeddingResult(
                dense=[float(index + 1), float(index + 2)],
                sparse=SparseVector(indices=[index + 1], values=[1.0]),
                contract_fingerprint=fingerprint,
            )
            for index, _text in enumerate(texts)
        ]


class _Qdrant:
    """Fake Qdrant client backing the job's reads, writes and schema edits."""

    def __init__(
        self,
        payloads,
        *,
        source_dense_on_disk: bool | None = None,
        source_sparse_on_disk: bool | None = None,
        source_sparse_idf: bool | None = None,
    ):
        source = _contract("a")
        dense_vector_cfg = (
            SimpleNamespace(on_disk=source_dense_on_disk)
            if source_dense_on_disk is not None
            else object()
        )
        if source_sparse_on_disk is not None or source_sparse_idf is not None:
            from qdrant_client.models import Modifier

            sparse_vector_cfg = SimpleNamespace(
                index=SimpleNamespace(on_disk=source_sparse_on_disk),
                modifier=Modifier.IDF if source_sparse_idf else Modifier.NONE,
            )
        else:
            sparse_vector_cfg = object()
        self.config = SimpleNamespace(
            metadata={
                "hybrid_manager": {
                    "metadata_schema": 2,
                    "embedding_contract": source.model_dump(mode="json"),
                    "active_vectors": {"dense": "dense", "sparse": "sparse"},
                }
            },
            params=SimpleNamespace(
                vectors={"dense": dense_vector_cfg},
                sparse_vectors={"sparse": sparse_vector_cfg},
            ),
        )
        self.points = [
            {
                "id": index + 1,
                "payload": deepcopy(payload),
                "vectors": {"dense": object(), "sparse": object()},
            }
            for index, payload in enumerate(payloads)
        ]
        self.created_names = []
        self.deleted_names = []
        self.last_vectors_config = None
        self.last_sparse_vectors_config = None
        self.total_count_delta = 0
        self.block_checkpoint = False
        self.metadata_updates = 0
        self.checkpoint_entered = asyncio.Event()
        self.checkpoint_release = asyncio.Event()

    async def info(self):
        return SimpleNamespace(version="1.19.0")

    async def get_collection(self, _collection_name):
        return SimpleNamespace(config=self.config)

    async def scroll(self, **kwargs):
        start = int(kwargs.get("offset") or 0)
        end = min(start + kwargs["limit"], len(self.points))
        records = [
            SimpleNamespace(id=point["id"], payload=deepcopy(point["payload"]))
            for point in self.points[start:end]
        ]
        return records, end if end < len(self.points) else None

    async def count(self, **kwargs):
        count_filter = kwargs.get("count_filter")
        if count_filter is None:
            return SimpleNamespace(count=len(self.points) + self.total_count_delta)
        vector_name = count_filter.must[0].has_vector
        return SimpleNamespace(
            count=sum(vector_name in point["vectors"] for point in self.points)
        )

    async def create_vector_name(self, **kwargs):
        self.created_names.append(kwargs["vector_name"])
        return True

    async def delete_vector_name(self, **kwargs):
        vector_name = kwargs["vector_name"]
        self.deleted_names.append(vector_name)
        for point in self.points:
            point["vectors"].pop(vector_name, None)
        return True

    async def update_collection(self, **kwargs):
        if kwargs.get("vectors_config") is not None:
            self.last_vectors_config = kwargs["vectors_config"]
        if kwargs.get("sparse_vectors_config") is not None:
            self.last_sparse_vectors_config = kwargs["sparse_vectors_config"]
        metadata = kwargs.get("metadata")
        if metadata is not None:
            self.metadata_updates += 1
            if self.block_checkpoint and self.metadata_updates == 2:
                self.checkpoint_entered.set()
                await self.checkpoint_release.wait()
            current = dict(self.config.metadata or {})
            current.update(deepcopy(metadata))
            self.config.metadata = current
        return True

    async def update_vectors(self, **kwargs):
        by_id = {point["id"]: point for point in self.points}
        for update in kwargs["points"]:
            by_id[update.id]["vectors"].update(update.vector)
        return True


async def _wait_for_worker(manager, collection_name="docs"):
    for _attempt in range(1000):
        if not manager.is_running(collection_name):
            return
        await asyncio.sleep(0)
    pytest.fail("background reindex worker did not finish")


@pytest.mark.parametrize("size", [3, 2001])
@pytest.mark.parametrize("status", ["failed", "paused", "interrupted"])
async def test_resume_final_checkpoint_does_not_rescan_or_reembed(size, status):
    """A failed cutover retries verification without replaying committed batches."""

    class FailedCutoverStore(_Qdrant):
        """Fail the metadata cutover once and count durable batch operations."""

        fail_commit = True
        scroll_calls = 0
        write_calls = 0

        async def scroll(self, **kwargs):
            self.scroll_calls += 1
            return await super().scroll(**kwargs)

        async def update_vectors(self, **kwargs):
            self.write_calls += 1
            assert kwargs["wait"] is True
            return await super().update_vectors(**kwargs)

        async def update_collection(self, **kwargs):
            job = (
                (kwargs.get("metadata") or {})
                .get("hybrid_manager", {})
                .get("reindex_job", {})
            )
            if job.get("status") == "completed" and self.fail_commit:
                self.fail_commit = False
                raise RuntimeError("cutover commit unavailable")
            return await super().update_collection(**kwargs)

    payloads = [
        {"text": f"point {i}", "source_id": "source", "id": i} for i in range(size)
    ]
    client = FailedCutoverStore(payloads)
    embedder = _Embedder()
    embedder.max_batch_size = 64
    manager = VectorReindexManager()
    started = await manager.start(client, embedder, "docs", _settings())
    await _wait_for_worker(manager)
    _, contract, active, failed = await get_embedding_metadata_state(
        client, "docs", _settings()
    )
    assert failed.status == "failed"
    assert failed.processed_points == failed.total_points == size
    assert failed.last_offset is None
    assert active == started.source_vectors
    assert contract == started.source_contract
    assert not client.deleted_names
    before = (client.scroll_calls, client.write_calls, embedder.embed_calls)
    client.config.metadata["hybrid_manager"]["reindex_job"]["status"] = status

    restarted = VectorReindexManager()
    await restarted.resume(client, embedder, "docs", _settings())
    await _wait_for_worker(restarted)
    _, contract, active, completed = await get_embedding_metadata_state(
        client, "docs", _settings()
    )
    assert completed.status == "completed"
    assert completed.processed_points == completed.total_points == size
    assert (client.scroll_calls, client.write_calls, embedder.embed_calls) == before
    assert active == started.target_vectors
    assert contract == started.target_contract
    assert [point["payload"] for point in client.points] == payloads


async def test_http_reindex_waits_for_admitted_write_and_blocks_later_writes(
    observe_mutation_attempt,
):
    """Exercise both sides of admission through the app's actual dependencies."""

    class PausedStore(_Qdrant):
        """Pause a non-protected payload update after mutation admission."""

        def __init__(self):
            super().__init__([{"text": "original", "tag": "old"}])
            self.retrieve_entered = asyncio.Event()
            self.retrieve_release = asyncio.Event()
            self.writes = 0

        async def retrieve(self, **kwargs):
            self.retrieve_entered.set()
            await self.retrieve_release.wait()
            return [SimpleNamespace(id=1, payload=self.points[0]["payload"])]

        async def set_payload(self, **kwargs):
            self.writes += 1
            self.points[0]["payload"].update(kwargs["payload"])

    store = PausedStore()
    embedder = _Embedder(block_worker=True)
    app = create_app(_settings())
    app.dependency_overrides[get_qdrant_client] = lambda: store
    app.dependency_overrides[get_embedder_client] = lambda: embedder
    tasks = []
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        try:
            tasks.append(
                asyncio.create_task(
                    client.patch(
                        "/points/1/payload",
                        json={"collection_name": "docs", "payload": {"tag": "new"}},
                    )
                )
            )
            await asyncio.wait_for(store.retrieve_entered.wait(), 2)
            attempted = observe_mutation_attempt(app.state.runtime.mutations)
            tasks.append(
                asyncio.create_task(
                    client.post(
                        "/collections/docs/reindex", json={"confirm_collection": "docs"}
                    )
                )
            )
            await asyncio.wait_for(attempted.wait(), 2)
            assert embedder.embed_calls == 0
            assert not tasks[1].done()
            store.retrieve_release.set()
            mutation, admission = await asyncio.wait_for(asyncio.gather(*tasks), 2)
            assert mutation.status_code == 200, mutation.text
            assert admission.status_code == 200, admission.text
            await asyncio.wait_for(embedder.worker_entered.wait(), 2)
            rejected = await client.patch(
                "/points/1/payload",
                json={"collection_name": "docs", "payload": {"tag": "too late"}},
            )
            assert rejected.status_code == 409, rejected.text
            assert store.writes == 1
            assert store.points[0]["payload"]["tag"] == "new"
        finally:
            store.retrieve_release.set()
            embedder.worker_release.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await _wait_for_worker(app.state.vector_reindex_manager)


@pytest.mark.asyncio
async def test_preflight_scans_all_payloads_and_reports_invalid_ids():
    client = _Qdrant([{"text": "ok"}, {}, {"text": "  "}])

    result = await preflight_reindex(
        client,
        _Embedder(),
        "docs",
        _settings(),
    )

    assert result.can_start is False
    assert result.total_points == 3
    assert result.eligible_points == 1
    assert result.invalid_points == 2
    assert result.invalid_point_ids == ["2", "3"]
    assert "payload.text" in result.blockers[-1]


@pytest.mark.parametrize(
    "confirmation", [" docs ", " docs", "docs ", "\tdocs\n", "Docs", "other"]
)
async def test_http_reindex_rejects_raw_confirmation_without_work(confirmation):
    store = _Qdrant([{"text": "one"}])
    embedder = _Embedder()
    app = create_app(_settings())
    app.dependency_overrides[get_qdrant_client] = lambda: store
    app.dependency_overrides[get_embedder_client] = lambda: embedder
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/collections/docs/reindex", json={"confirm_collection": confirmation}
        )
    await _wait_for_worker(app.state.vector_reindex_manager)
    assert response.status_code == 400
    assert embedder.embed_calls == 0
    assert store.metadata_updates == 0


@pytest.mark.asyncio
async def test_preflight_distinguishes_count_drift_from_invalid_payloads():
    client = _Qdrant([{"text": "one"}, {"text": "two"}])
    client.total_count_delta = 1

    result = await preflight_reindex(
        client,
        _Embedder(),
        "docs",
        _settings(),
    )

    assert result.can_start is False
    assert result.eligible_points == 2
    assert result.invalid_points == 0
    assert any("counted 3, scanned 2" in item for item in result.blockers)


@pytest.mark.asyncio
async def test_reindex_preserves_payload_and_cuts_over_only_after_exact_validation():
    client = _Qdrant(
        [
            {"text": "one", "kind": "a"},
            {"text": "two", "kind": "b"},
            {"text": "three", "kind": "c"},
        ]
    )
    original_payloads = [deepcopy(point["payload"]) for point in client.points]
    manager = VectorReindexManager()

    started = await manager.start(
        client,
        _Embedder(),
        "docs",
        _settings(),
    )
    await _wait_for_worker(manager)
    _, contract, active_vectors, job = await get_embedding_metadata_state(
        client,
        "docs",
        _settings(),
    )

    assert job is not None
    assert job.status == "completed"
    assert job.processed_points == 3
    assert contract == _contract("b")
    assert active_vectors == started.target_vectors
    assert [point["payload"] for point in client.points] == original_payloads
    for point in client.points:
        assert "dense" not in point["vectors"]
        assert "sparse" not in point["vectors"]
        assert started.target_vectors.dense in point["vectors"]
        assert started.target_vectors.sparse in point["vectors"]
    assert "dense" in client.deleted_names
    assert "sparse" in client.deleted_names


@pytest.mark.asyncio
async def test_reindex_target_preserves_source_vector_toggles_not_global_defaults():
    """A reindex must carry over the collection's own on_disk/idf toggles.

    Settings() defaults to dense_on_disk=True, sparse_on_disk=True,
    sparse_idf=True. This collection was actually configured the opposite
    way (RAM-resident, no IDF) via qdrant_create_collection or
    qdrant_update_schema_toggles. A reindex regenerates vectors under a new
    embedding contract -- it must not silently re-tier storage or flip the
    IDF modifier back to whatever the operator's current .env happens to say.
    """
    from qdrant_client.models import Modifier

    client = _Qdrant(
        [{"text": "one"}, {"text": "two"}],
        source_dense_on_disk=False,
        source_sparse_on_disk=False,
        source_sparse_idf=False,
    )
    manager = VectorReindexManager()

    started = await manager.start(client, _Embedder(), "docs", _settings())
    await _wait_for_worker(manager)

    dense_diff = client.last_vectors_config[started.target_vectors.dense]
    sparse_params = client.last_sparse_vectors_config[started.target_vectors.sparse]
    assert dense_diff.on_disk is False
    assert sparse_params.index.on_disk is False
    assert sparse_params.modifier == Modifier.NONE


@pytest.mark.asyncio
async def test_worker_failure_keeps_old_generation_active():
    client = _Qdrant([{"text": "one"}, {"text": "two"}])
    manager = VectorReindexManager()

    started = await manager.start(
        client,
        _Embedder(bad_worker_fingerprint=True),
        "docs",
        _settings(),
    )
    await _wait_for_worker(manager)
    _, contract, active_vectors, job = await get_embedding_metadata_state(
        client,
        "docs",
        _settings(),
    )

    assert job is not None
    assert job.status == "failed"
    assert "fingerprint changed" in job.error
    assert contract == _contract("a")
    assert active_vectors.dense == "dense"
    assert active_vectors.sparse == "sparse"
    assert started.target_vectors != active_vectors


@pytest.mark.asyncio
async def test_pause_commits_batch_boundary_and_resume_finishes():
    client = _Qdrant([{"text": "one"}, {"text": "two"}, {"text": "three"}])
    embedder = _Embedder(block_worker=True)
    manager = VectorReindexManager()

    await manager.start(client, embedder, "docs", _settings())
    await asyncio.wait_for(embedder.worker_entered.wait(), timeout=1)
    await manager.set_control_status(
        client,
        "docs",
        _settings(),
        "paused",
    )
    embedder.worker_release.set()
    await _wait_for_worker(manager)
    _, _, _, paused = await get_embedding_metadata_state(client, "docs", _settings())

    assert paused is not None
    assert paused.status == "paused"
    assert paused.processed_points == 2

    await manager.resume(client, embedder, "docs", _settings())
    await _wait_for_worker(manager)
    _, _, _, completed = await get_embedding_metadata_state(client, "docs", _settings())
    assert completed is not None
    assert completed.status == "completed"
    assert completed.processed_points == 3


@pytest.mark.asyncio
async def test_pause_cannot_be_overwritten_by_an_in_flight_checkpoint():
    client = _Qdrant([{"text": f"point {index}"} for index in range(5)])
    client.block_checkpoint = True
    manager = VectorReindexManager()

    await manager.start(client, _Embedder(), "docs", _settings())
    await asyncio.wait_for(client.checkpoint_entered.wait(), timeout=1)
    pause = asyncio.create_task(
        manager.set_control_status(client, "docs", _settings(), "paused")
    )
    await asyncio.sleep(0)
    assert not pause.done()

    client.checkpoint_release.set()
    await asyncio.wait_for(pause, timeout=1)
    await _wait_for_worker(manager)
    _, _, _, paused = await get_embedding_metadata_state(client, "docs", _settings())

    assert paused is not None
    assert paused.status == "paused"
    assert paused.processed_points < len(client.points)


@pytest.mark.parametrize("control", [None, "paused", "cancelled", "replaced"])
async def test_empty_final_scroll_respects_concurrent_control(control):
    """A terminal empty page must not overwrite an acknowledged control."""

    class EmptyFinalPageStore(_Qdrant):
        """Return a final cursor, then hold its empty page until released."""

        def __init__(self):
            super().__init__([{"text": "one"}, {"text": "two"}])
            self.block_empty = False
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def scroll(self, **kwargs):
            records, offset = await super().scroll(**kwargs)
            if records and offset is None:
                return records, len(self.points)
            if not records and self.block_empty:
                self.entered.set()
                await self.release.wait()
            return records, offset

    client = EmptyFinalPageStore()
    manager = VectorReindexManager()
    settings = _settings()
    started = await manager.start(client, _Embedder(), "docs", settings)
    client.block_empty = True
    try:
        await asyncio.wait_for(client.entered.wait(), 2)
        if control == "replaced":
            _, _, _, current = await get_embedding_metadata_state(
                client, "docs", settings
            )
            replacement = current.model_copy(update={"job_id": "replacement"})
            await patch_embedding_metadata(
                client, "docs", settings=settings, reindex_job=replacement
            )
        elif control is not None:
            await manager.set_control_status(client, "docs", settings, control)
    finally:
        client.release.set()
        await asyncio.wait_for(_wait_for_worker(manager), 2)
    _, contract, active, job = await get_embedding_metadata_state(
        client, "docs", settings
    )
    if control == "replaced":
        assert job == replacement
    else:
        assert job.status == (control or "completed")
    assert active == (started.source_vectors if control else started.target_vectors)
    assert contract == (started.source_contract if control else started.target_contract)
    if control:
        assert "dense" not in client.deleted_names
        assert "sparse" not in client.deleted_names
    if control == "cancelled":
        assert started.target_vectors.dense in client.deleted_names
        assert started.target_vectors.sparse in client.deleted_names


@pytest.mark.asyncio
async def test_exact_count_drift_fails_without_cutover():
    client = _Qdrant([{"text": "one"}])
    manager = VectorReindexManager()
    embedder = _Embedder(block_worker=True)

    started = await manager.start(client, embedder, "docs", _settings())
    await asyncio.wait_for(embedder.worker_entered.wait(), timeout=1)
    client.total_count_delta = 1
    embedder.worker_release.set()
    await _wait_for_worker(manager)
    _, contract, active_vectors, job = await get_embedding_metadata_state(
        client, "docs", _settings()
    )

    assert job is not None
    assert job.status == "failed"
    assert "point count changed" in job.error
    assert contract == _contract("a")
    assert active_vectors != started.target_vectors


@pytest.mark.asyncio
async def test_active_job_blocks_mutations_but_allows_query_compatibility_check():
    client = _Qdrant([{"text": "one"}, {"text": "two"}, {"text": "three"}])
    embedder = _Embedder(block_worker=True)
    manager = VectorReindexManager()

    started = await manager.start(client, embedder, "docs", _settings())
    await asyncio.wait_for(embedder.worker_entered.wait(), timeout=1)

    with pytest.raises(EmbeddingContractViolation) as exc:
        await require_collection_mutable(client, "docs", _settings())
    assert exc.value.status == "reindex_in_progress"

    vectors = await require_embedding_fingerprint(
        client,
        "docs",
        _contract("a").fingerprint,
        settings=_settings(),
        allow_during_reindex=True,
    )
    assert vectors.dense == "dense"

    await manager.set_control_status(
        client,
        "docs",
        _settings(),
        "cancelled",
    )
    embedder.worker_release.set()
    await _wait_for_worker(manager)
    _, _, _, cancelled = await get_embedding_metadata_state(client, "docs", _settings())
    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert started.target_vectors.dense in client.deleted_names
    assert started.target_vectors.sparse in client.deleted_names

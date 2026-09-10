"""Safe, resumable regeneration of a collection's dense and sparse vectors."""

import asyncio
import logging
import re
import uuid
from datetime import datetime, timezone

from qdrant_client import AsyncQdrantClient, models

from qdrant_middleware.clients.embedder import EmbedderClient
from qdrant_middleware.core.config import Settings
from qdrant_middleware.core.mutation_coordinator import MutationCoordinator
from qdrant_middleware.repositories.qdrant_collections import (
    get_embedding_metadata_state,
    patch_embedding_metadata,
)
from qdrant_middleware.schemas.embedding_contract import (
    ActiveVectorNames,
    EmbedderContractInfo,
    ReindexJobState,
)
from qdrant_middleware.schemas.reindex import ReindexPreflightResponse
from qdrant_middleware.services.collection_schema import (
    _get_named_config,
    _sparse_vector_params,
)
from qdrant_middleware.services.embedding_guard import response_embedding_fingerprint

log = logging.getLogger(__name__)

REINDEX_PAYLOAD_FIELD = "text"
REINDEX_SCROLL_BATCH = 256
REINDEX_INVALID_ID_LIMIT = 20
ACTIVE_JOB_STATUSES = {"running", "paused", "interrupted", "finalizing"}


def _utc_now() -> str:
    """Return a stable UTC timestamp for persisted job state."""
    return datetime.now(timezone.utc).isoformat()


def _version_tuple(value: str | None) -> tuple[int, int, int] | None:
    """Parse the numeric prefix of a Qdrant semantic version."""
    if not value:
        return None
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value)
    if match is None:
        return None
    return tuple(int(item or 0) for item in match.groups())


def _persistable_offset(value: object) -> int | str | None:
    """Convert Qdrant offsets to the JSON-safe job representation."""
    if value is None or isinstance(value, (int, str)):
        return value
    return str(value)


def _target_vector_names(
    contract: EmbedderContractInfo,
    job_id: str,
) -> ActiveVectorNames:
    """Build collision-resistant generation names that remain readable in UI."""
    digest = contract.contract.fingerprint.removeprefix("sha256:")[:8]
    suffix = job_id.replace("-", "")[:6]
    return ActiveVectorNames(
        dense=f"dense__{digest}_{suffix}",
        sparse=f"sparse__{digest}_{suffix}",
    )


class _SourceVectorToggles:
    """The source collection's own on_disk/IDF storage toggles.

    A reindex regenerates vectors under a new embedding contract; it must
    not also silently re-tier storage or flip the IDF modifier to whatever
    the operator's current .env happens to say. These three values -- and
    only these three -- are read from the collection's own live config
    instead of ``Settings``, mirroring the fields ``qdrant_create_collection``
    and ``qdrant_update_schema_toggles`` actually expose per collection.
    Every other reindex-time knob (HNSW params, datatype, sparse full-scan
    threshold) is not a per-collection toggle at all, so it legitimately
    still comes from ``Settings``.
    """

    def __init__(self, dense_on_disk: bool, sparse_on_disk: bool, sparse_idf: bool):
        self.dense_on_disk = dense_on_disk
        self.sparse_on_disk = sparse_on_disk
        self.sparse_idf = sparse_idf


def _source_vector_toggles(
    config: object,
    active_vectors: ActiveVectorNames,
    settings: Settings,
) -> _SourceVectorToggles:
    """Read the source collection's live on_disk/IDF toggles.

    Defaults to ``settings`` only when the live config carries no usable
    value (should not happen for a collection that already passed reindex
    preflight).
    """
    dense_cfg = _get_named_config(
        getattr(config.params, "vectors", None), active_vectors.dense
    )
    sparse_cfg = _get_named_config(
        getattr(config.params, "sparse_vectors", None), active_vectors.sparse
    )
    sparse_index = getattr(sparse_cfg, "index", None) if sparse_cfg else None
    dense_on_disk = getattr(dense_cfg, "on_disk", None) if dense_cfg else None
    sparse_on_disk = getattr(sparse_index, "on_disk", None) if sparse_index else None
    sparse_modifier = getattr(sparse_cfg, "modifier", None) if sparse_cfg else None
    return _SourceVectorToggles(
        dense_on_disk=(
            dense_on_disk if dense_on_disk is not None else settings.dense_on_disk
        ),
        sparse_on_disk=(
            sparse_on_disk if sparse_on_disk is not None else settings.sparse_on_disk
        ),
        sparse_idf=(
            sparse_modifier == models.Modifier.IDF
            if sparse_modifier is not None
            else settings.sparse_idf
        ),
    )


async def preflight_reindex(
    client: AsyncQdrantClient,
    embedder: EmbedderClient,
    collection_name: str,
    settings: Settings,
) -> ReindexPreflightResponse:
    """Validate every source payload and the active target contract."""
    (
        info,
        source_contract,
        source_vectors,
        current_job,
    ) = await get_embedding_metadata_state(client, collection_name, settings)
    target = await embedder.get_embedding_contract()
    blockers: list[str] = []

    qdrant_version: str | None = None
    try:
        server_info = await client.info()
        qdrant_version = getattr(server_info, "version", None)
    except Exception as exc:  # capability must be explicit, not guessed
        blockers.append(f"Unable to determine Qdrant version: {exc}")
    parsed_version = _version_tuple(qdrant_version)
    if qdrant_version is None:
        blockers.append("Qdrant did not report its server version.")
    elif parsed_version is not None and parsed_version < (1, 18, 0):
        blockers.append("Qdrant 1.18 or newer is required for named-vector migration.")
    elif qdrant_version is not None and parsed_version is None:
        blockers.append(f"Unrecognized Qdrant version: {qdrant_version}")

    if current_job is not None and current_job.status in ACTIVE_JOB_STATUSES:
        blockers.append(
            f"A reindex job is already {current_job.status} for this collection."
        )

    count_result = await client.count(collection_name=collection_name, exact=True)
    total_points = int(count_result.count)
    eligible = 0
    invalid_points = 0
    scanned_points = 0
    invalid_ids: list[str] = []
    sample_text: str | None = None
    offset = None
    while True:
        records, offset = await client.scroll(
            collection_name=collection_name,
            limit=REINDEX_SCROLL_BATCH,
            offset=offset,
            with_payload=[REINDEX_PAYLOAD_FIELD],
            with_vectors=False,
        )
        for record in records:
            scanned_points += 1
            text = (record.payload or {}).get(REINDEX_PAYLOAD_FIELD)
            if not isinstance(text, str) or not text.strip():
                invalid_points += 1
                if len(invalid_ids) < REINDEX_INVALID_ID_LIMIT:
                    invalid_ids.append(str(record.id))
                continue
            eligible += 1
            sample_text = sample_text or text
        if offset is None:
            break

    if total_points == 0:
        # Starting a job here would create, swap and drop a whole vector
        # generation for nothing, and lock the collection while doing it.
        blockers.append("Collection has no points to regenerate.")
    if scanned_points != total_points:
        blockers.append(
            "Collection point count changed during preflight: "
            f"counted {total_points}, scanned {scanned_points}; retry it."
        )
    if invalid_points:
        blockers.append(
            f"{invalid_points} points do not contain a non-empty payload.text string."
        )

    if sample_text is not None:
        sample_embeddings = await embedder.embed([sample_text])
        if len(sample_embeddings) != 1:
            blockers.append("Smart Embedder sample returned an invalid result count.")
        else:
            fingerprint = response_embedding_fingerprint(sample_embeddings)
            if fingerprint != target.contract.fingerprint:
                blockers.append(
                    "Smart Embedder sample fingerprint differs from its "
                    "/models contract."
                )
            if len(sample_embeddings[0].dense) != target.contract.dense.dimension:
                blockers.append(
                    "Smart Embedder sample dense dimension differs from its contract."
                )

    params = getattr(getattr(info, "config", None), "params", None)
    if params is None:
        blockers.append("Qdrant collection configuration is unavailable.")
    else:
        if (
            _get_named_config(
                getattr(params, "vectors", None),
                source_vectors.dense,
            )
            is None
        ):
            blockers.append(f"Active dense vector '{source_vectors.dense}' is missing.")
        if (
            _get_named_config(
                getattr(params, "sparse_vectors", None),
                source_vectors.sparse,
            )
            is None
        ):
            blockers.append(
                f"Active sparse vector '{source_vectors.sparse}' is missing."
            )

    return ReindexPreflightResponse(
        collection_name=collection_name,
        can_start=not blockers,
        qdrant_version=qdrant_version,
        source_contract=source_contract,
        target_contract=target.contract,
        source_vectors=source_vectors,
        total_points=total_points,
        eligible_points=eligible,
        invalid_points=invalid_points,
        invalid_point_ids=invalid_ids,
        blockers=blockers,
    )


class VectorReindexManager:
    """Own background reindex tasks while progress remains persisted in Qdrant."""

    def __init__(self, coordinator: MutationCoordinator | None = None) -> None:
        """Initialize per-collection task and lock registries."""
        self.coordinator = coordinator or MutationCoordinator()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_worker = asyncio.Semaphore(1)

    def _lock(self, collection_name: str) -> asyncio.Lock:
        return self._locks.setdefault(collection_name, asyncio.Lock())

    def is_running(self, collection_name: str) -> bool:
        """Return whether this process owns a live task for the collection."""
        task = self._tasks.get(collection_name)
        return task is not None and not task.done()

    async def start(
        self,
        client: AsyncQdrantClient,
        embedder: EmbedderClient,
        collection_name: str,
        settings: Settings,
    ) -> ReindexJobState:
        """Run preflight, create target schemas, persist the job and spawn work."""
        async with self.coordinator.lease(collection_name), self._lock(collection_name):
            if self.is_running(collection_name):
                raise ValueError("a reindex task is already running")
            preflight = await preflight_reindex(
                client, embedder, collection_name, settings
            )
            if not preflight.can_start:
                raise ValueError("; ".join(preflight.blockers))
            target_info = await embedder.get_embedding_contract()
            if (
                target_info.contract.fingerprint
                != preflight.target_contract.fingerprint
            ):
                raise ValueError("embedder contract changed during preflight")
            info, _, active_vectors, previous_job = await get_embedding_metadata_state(
                client, collection_name, settings
            )
            if (
                previous_job is not None
                and previous_job.status in {"failed", "cancelled"}
                and previous_job.target_vectors != active_vectors
            ):
                await self._delete_vector_generation(
                    client,
                    collection_name,
                    previous_job.target_vectors,
                )

            job_id = str(uuid.uuid4())
            target_vectors = _target_vector_names(target_info, job_id)
            source_toggles = _source_vector_toggles(
                info.config, active_vectors, settings
            )
            await self._create_target_vectors(
                client,
                collection_name,
                target_vectors,
                target_info,
                settings,
                source_toggles,
            )
            now = _utc_now()
            job = ReindexJobState(
                job_id=job_id,
                status="running",
                source_contract=preflight.source_contract,
                target_contract=target_info.contract,
                source_vectors=preflight.source_vectors,
                target_vectors=target_vectors,
                total_points=preflight.total_points,
                processed_points=0,
                started_at=now,
                updated_at=now,
            )
            try:
                await patch_embedding_metadata(
                    client,
                    collection_name,
                    settings=settings,
                    reindex_job=job,
                )
            except Exception:
                await self._delete_vector_generation(
                    client,
                    collection_name,
                    target_vectors,
                )
                raise
            self._spawn(client, embedder, collection_name, settings)
            return job

    async def resume(
        self,
        client: AsyncQdrantClient,
        embedder: EmbedderClient,
        collection_name: str,
        settings: Settings,
    ) -> ReindexJobState:
        """Resume a paused, interrupted or failed job from its committed offset."""
        async with self.coordinator.lease(collection_name), self._lock(collection_name):
            if self.is_running(collection_name):
                raise ValueError("a reindex task is already running")
            _, _, _, job = await get_embedding_metadata_state(
                client, collection_name, settings
            )
            if job is None or job.status not in {"paused", "interrupted", "failed"}:
                raise ValueError(
                    "no paused, interrupted or failed reindex job to resume"
                )
            target = await embedder.get_embedding_contract()
            if target.contract.fingerprint != job.target_contract.fingerprint:
                raise ValueError("active embedder no longer matches the job target")
            resumed = job.model_copy(
                update={"status": "running", "updated_at": _utc_now(), "error": None}
            )
            await patch_embedding_metadata(
                client,
                collection_name,
                settings=settings,
                reindex_job=resumed,
            )
            self._spawn(client, embedder, collection_name, settings)
            return resumed

    async def set_control_status(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        settings: Settings,
        status: str,
    ) -> ReindexJobState:
        """Persist a cooperative pause or cancellation request."""
        if status not in {"paused", "cancelled"}:
            raise ValueError("unsupported reindex control status")
        async with self.coordinator.lease(collection_name), self._lock(collection_name):
            _, _, _, job = await get_embedding_metadata_state(
                client, collection_name, settings
            )
            allowed = (
                {"running"}
                if status == "paused"
                else {"running", "paused", "interrupted", "failed"}
            )
            if job is None or job.status not in allowed:
                raise ValueError("no controllable reindex job for this action")
            updated = job.model_copy(
                update={"status": status, "updated_at": _utc_now()}
            )
            await patch_embedding_metadata(
                client,
                collection_name,
                settings=settings,
                reindex_job=updated,
            )
            if status == "cancelled" and not self.is_running(collection_name):
                await self._delete_vector_generation(
                    client,
                    collection_name,
                    updated.target_vectors,
                )
            return updated

    async def stop(self) -> None:
        """Cancel and await process-owned tasks before clients are closed."""
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    def _spawn(
        self,
        client: AsyncQdrantClient,
        embedder: EmbedderClient,
        collection_name: str,
        settings: Settings,
    ) -> None:
        task = asyncio.create_task(
            self._run(client, embedder, collection_name, settings),
            name=f"vector-reindex:{collection_name}",
        )
        self._tasks[collection_name] = task

        def _forget(done: asyncio.Task[None]) -> None:
            if self._tasks.get(collection_name) is done:
                self._tasks.pop(collection_name, None)

        task.add_done_callback(_forget)

    async def _create_target_vectors(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        names: ActiveVectorNames,
        target: EmbedderContractInfo,
        settings: Settings,
        source_toggles: _SourceVectorToggles,
    ) -> None:
        created_names: list[str] = []
        try:
            await client.create_vector_name(
                collection_name=collection_name,
                vector_name=names.dense,
                vector_name_config=models.DenseVectorNameConfig(
                    dense=models.DenseVectorConfig(
                        size=target.contract.dense.dimension,
                        distance=models.Distance.COSINE,
                        datatype=models.Datatype.FLOAT16
                        if settings.dense_float16
                        else None,
                    )
                ),
                wait=True,
            )
            created_names.append(names.dense)
            await client.create_vector_name(
                collection_name=collection_name,
                vector_name=names.sparse,
                vector_name_config=models.SparseVectorNameConfig(
                    sparse=models.SparseVectorConfig(
                        modifier=models.Modifier.IDF
                        if settings.sparse_idf
                        else models.Modifier.NONE,
                        datatype=models.Datatype.FLOAT16
                        if settings.sparse_float16
                        else None,
                    )
                ),
                wait=True,
            )
            created_names.append(names.sparse)
            await client.update_collection(
                collection_name=collection_name,
                vectors_config={
                    names.dense: models.VectorParamsDiff(
                        on_disk=source_toggles.dense_on_disk,
                        hnsw_config=models.HnswConfigDiff(
                            m=settings.hnsw_m,
                            ef_construct=settings.hnsw_ef_construct,
                            full_scan_threshold=settings.hnsw_full_scan_threshold,
                            inline_storage=settings.hnsw_inline_storage,
                        ),
                    )
                },
                sparse_vectors_config={
                    names.sparse: _sparse_vector_params(
                        source_toggles.sparse_idf,
                        settings,
                        on_disk=source_toggles.sparse_on_disk,
                    )
                },
            )
        except Exception:
            await self._delete_vector_generation(
                client,
                collection_name,
                names,
                only=created_names,
            )
            raise

    async def _delete_vector_generation(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        names: ActiveVectorNames,
        *,
        only: list[str] | None = None,
    ) -> None:
        """Best-effort deletion of one inactive named-vector generation."""
        vector_names = only if only is not None else [names.dense, names.sparse]
        for vector_name in reversed(vector_names):
            try:
                await client.delete_vector_name(
                    collection_name=collection_name,
                    vector_name=vector_name,
                    wait=True,
                )
            except Exception:  # noqa: PERF203 - deletion is best-effort per name
                log.exception(
                    "Failed to delete inactive vector schema '%s'",
                    vector_name,
                )

    async def _run(
        self,
        client: AsyncQdrantClient,
        embedder: EmbedderClient,
        collection_name: str,
        settings: Settings,
    ) -> None:
        try:
            async with self._global_worker:
                await self._run_batches(client, embedder, collection_name, settings)
        except asyncio.CancelledError:
            await self._mark_interrupted(client, collection_name, settings)
            raise
        except Exception as exc:  # state must survive every background failure
            log.exception("Vector reindex failed for '%s'", collection_name)
            await self._mark_failed(client, collection_name, settings, str(exc))

    async def _run_batches(
        self,
        client: AsyncQdrantClient,
        embedder: EmbedderClient,
        collection_name: str,
        settings: Settings,
    ) -> None:
        batch_size = min(getattr(embedder, "max_batch_size", 64), 64)
        while True:
            _, _, _, job = await get_embedding_metadata_state(
                client, collection_name, settings
            )
            if job is None:
                raise RuntimeError("persisted reindex job disappeared")
            if job.status == "cancelled":
                await self._delete_vector_generation(
                    client,
                    collection_name,
                    job.target_vectors,
                )
                return
            if job.status == "paused":
                return
            if job.status not in {"running", "finalizing"}:
                raise RuntimeError(f"unexpected reindex status: {job.status}")

            if job.last_offset is None and job.processed_points == job.total_points:
                # The last batch is durable even if its metadata cutover failed.
                # Serialize the retry with control requests, like normal cutover.
                async with self._lock(collection_name):
                    _, _, _, current = await get_embedding_metadata_state(
                        client, collection_name, settings
                    )
                    if current == job:
                        await self._finalize(client, collection_name, settings, job)
                        return
                continue

            records, next_offset = await client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=job.last_offset,
                with_payload=[job.payload_field],
                with_vectors=False,
            )
            if not records:
                await self._finalize(client, collection_name, settings, job)
                return

            texts = []
            for record in records:
                text = (record.payload or {}).get(job.payload_field)
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError(
                        f"point {record.id} lost its non-empty payload.text"
                    )
                texts.append(text)
            embeddings = await embedder.embed(texts)
            if len(embeddings) != len(records):
                raise RuntimeError("Smart Embedder embedding count mismatch")
            fingerprint = response_embedding_fingerprint(embeddings)
            if fingerprint != job.target_contract.fingerprint:
                raise RuntimeError("Smart Embedder fingerprint changed during reindex")
            if any(
                len(embedding.dense) != job.target_contract.dense.dimension
                for embedding in embeddings
            ):
                raise RuntimeError(
                    "Smart Embedder dense dimension changed during reindex"
                )

            await client.update_vectors(
                collection_name=collection_name,
                points=[
                    models.PointVectors(
                        id=record.id,
                        vector={
                            job.target_vectors.dense: embedding.dense,
                            job.target_vectors.sparse: embedding.sparse,
                        },
                    )
                    for record, embedding in zip(records, embeddings)
                ],
                wait=True,
            )

            # Serialize the control read and checkpoint write with pause/cancel.
            # Otherwise the worker can overwrite a control request with a stale
            # running checkpoint.
            async with self._lock(collection_name):
                _, _, _, control = await get_embedding_metadata_state(
                    client, collection_name, settings
                )
                if control is None or control.job_id != job.job_id:
                    raise RuntimeError("reindex job changed while a batch was running")
                updated = job.model_copy(
                    update={
                        "status": control.status
                        if control.status in {"paused", "cancelled"}
                        else "running",
                        "processed_points": job.processed_points + len(records),
                        "last_offset": _persistable_offset(next_offset),
                        "updated_at": _utc_now(),
                    }
                )
                await patch_embedding_metadata(
                    client,
                    collection_name,
                    settings=settings,
                    reindex_job=updated,
                )
                if next_offset is None and updated.status == "running":
                    await self._finalize(client, collection_name, settings, updated)
                    return
            if updated.status == "cancelled":
                await self._delete_vector_generation(
                    client,
                    collection_name,
                    updated.target_vectors,
                )
                return
            if updated.status == "paused":
                return

    async def _finalize(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        settings: Settings,
        job: ReindexJobState,
    ) -> None:
        finalizing = job.model_copy(
            update={"status": "finalizing", "updated_at": _utc_now()}
        )
        await patch_embedding_metadata(
            client,
            collection_name,
            settings=settings,
            reindex_job=finalizing,
        )
        total = int(
            (await client.count(collection_name=collection_name, exact=True)).count
        )
        dense_count = int(
            (
                await client.count(
                    collection_name=collection_name,
                    count_filter=models.Filter(
                        must=[
                            models.HasVectorCondition(
                                has_vector=job.target_vectors.dense
                            )
                        ]
                    ),
                    exact=True,
                )
            ).count
        )
        sparse_count = int(
            (
                await client.count(
                    collection_name=collection_name,
                    count_filter=models.Filter(
                        must=[
                            models.HasVectorCondition(
                                has_vector=job.target_vectors.sparse
                            )
                        ]
                    ),
                    exact=True,
                )
            ).count
        )
        if total != job.total_points:
            raise RuntimeError(
                "point count changed during reindex: expected "
                f"{job.total_points}, got {total}"
            )
        if dense_count != total or sparse_count != total:
            raise RuntimeError(
                "target vector counts are incomplete: "
                f"points={total}, dense={dense_count}, sparse={sparse_count}"
            )

        completed = finalizing.model_copy(
            update={
                "status": "completed",
                "processed_points": total,
                "updated_at": _utc_now(),
                "error": None,
            }
        )
        await patch_embedding_metadata(
            client,
            collection_name,
            settings=settings,
            contract=job.target_contract,
            active_vectors=job.target_vectors,
            reindex_job=completed,
        )
        await self._delete_vector_generation(
            client,
            collection_name,
            job.source_vectors,
        )

    async def _mark_failed(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        settings: Settings,
        error: str,
    ) -> None:
        try:
            _, _, _, job = await get_embedding_metadata_state(
                client, collection_name, settings
            )
            if job is None or job.status in {"cancelled", "completed"}:
                return
            failed = job.model_copy(
                update={
                    "status": "failed",
                    "updated_at": _utc_now(),
                    "error": error[:2000],
                }
            )
            await patch_embedding_metadata(
                client,
                collection_name,
                settings=settings,
                reindex_job=failed,
            )
        except Exception:
            log.exception("Failed to persist reindex failure state")

    async def _mark_interrupted(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        settings: Settings,
    ) -> None:
        try:
            _, _, _, job = await get_embedding_metadata_state(
                client, collection_name, settings
            )
            if job is None or job.status not in {"running", "finalizing"}:
                return
            interrupted = job.model_copy(
                update={"status": "interrupted", "updated_at": _utc_now()}
            )
            await patch_embedding_metadata(
                client,
                collection_name,
                settings=settings,
                reindex_job=interrupted,
            )
        except Exception:
            log.exception("Failed to persist interrupted reindex state")

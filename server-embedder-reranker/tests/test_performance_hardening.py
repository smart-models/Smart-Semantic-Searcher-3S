"""Tests for the server's concurrency and batching hardening.

Drives the advanced profile's hybrid scheduling with fake tensors and
tokenizers to pin the batching, padding and thread-safety behaviour that
keeps the server stable under load.
"""

import asyncio
import concurrent.futures
import contextlib
import threading
import time
import unittest
from unittest.mock import patch

from tests.test_reranker_wrapper import install_support_fakes, load_server


class TestFakeCompatibility(unittest.TestCase):
    """The shared fakes still match the interfaces the server relies on."""

    def test_fake_numpy_concatenate_supports_axis_zero(self):
        with patch.dict("sys.modules", install_support_fakes()):
            import numpy as np

            self.assertEqual(np.concatenate([[1], [2, 3]], axis=0), [1, 2, 3])

    def test_server_imports_with_fake_orjson_response(self):
        module = load_server()
        self.assertTrue(hasattr(module, "FastAPI"))


class TestAdvancedHybridScheduling(unittest.TestCase):
    """Executor, queue policy and CPU affinity for advanced hybrid runs."""

    def test_advanced_inference_lanes_configure_shared_executor(self):
        module = load_server(
            {
                "EMBEDDER_PROFILE": "advanced",
                "ADVANCED_SCHEDULE_MODE": "parallel",
                "ADVANCED_INFERENCE_LANES": "2",
            }
        )
        try:
            self.assertEqual(module.ADVANCED_INFERENCE_LANES, 2)
            self.assertEqual(module.GPU_EXECUTOR._max_workers, 2)
            self.assertEqual(module.ADVANCED_EFFECTIVE_CONFIG["inference_lanes"], 2)
        finally:
            module.GPU_EXECUTOR.shutdown(wait=False, cancel_futures=True)

    def test_advanced_queue_policy_configures_http_batcher(self):
        module = load_server(
            {
                "EMBEDDER_PROFILE": "advanced",
                "ADVANCED_QUEUE_BATCH": "7",
                "ADVANCED_FLUSH_TIMEOUT_MS": "25",
            }
        )
        try:
            self.assertEqual(module.MAX_REQUESTS_IN_BATCH, 7)
            self.assertEqual(module.REQUEST_FLUSH_TIMEOUT, 0.025)
            self.assertEqual(module.processor.max_batch_size, 7)
            self.assertEqual(module.processor.accumulation_timeout, 0.025)
            self.assertEqual(module.ADVANCED_EFFECTIVE_CONFIG["queue_batch"], 7)
            self.assertEqual(module.ADVANCED_EFFECTIVE_CONFIG["flush_timeout_ms"], 25)
        finally:
            module.GPU_EXECUTOR.shutdown(wait=False, cancel_futures=True)

    def test_unsupported_qwen_runtime_is_never_silently_treated_as_torch(self):
        module = load_server()

        with (
            patch.object(module, "ADVANCED_DENSE_RUNTIME", "openvino"),
            self.assertRaisesRegex(RuntimeError, "unsupported"),
        ):
            module.QwenDenseEmbeddingBackend("Qwen/test", device="cpu")

    def test_parallel_schedule_runs_dense_and_sparse_concurrently(self):
        module = load_server()
        rendezvous = threading.Barrier(2)

        class DenseBackend:
            """Dense backend blocking on a barrier, proving it runs concurrently."""

            def embed_dense(self, sentences, normalize_dense=False, deadline=None):
                rendezvous.wait(timeout=1.0)
                return {"dense_vecs": [[1.0] * 1024 for _ in sentences]}

        class SparseBackend:
            """Sparse backend blocking on a barrier, proving it runs concurrently."""

            model = type("Model", (), {"config": type("Config", (), {})()})()

            def embed_sparse(self, sentences, deadline=None):
                rendezvous.wait(timeout=1.0)
                return {"lexical_weights": [{"1": 1.0} for _ in sentences]}

        service = module.EmbeddingService.__new__(module.EmbeddingService)
        service.profile = module.EMBEDDER_PROFILE_ADVANCED
        service.bge_backend = None
        service.qwen_dense_backend = DenseBackend()
        service.milco_sparse_backend = SparseBackend()
        service._hybrid_streams = (None, None)
        service._hybrid_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        try:
            with patch.object(module, "EMBED_LENGTH_AWARE_BATCHING", False):
                result = service.embed(
                    ["one", "two"],
                    return_dense=True,
                    return_sparse=True,
                    return_colbert=False,
                )
        finally:
            service.close()

        self.assertEqual(len(result["dense_vecs"]), 2)
        self.assertEqual(len(result["lexical_weights"]), 2)

    def test_pinned_policy_requests_nonblocking_copy(self):
        module = load_server()

        class Tensor:
            """Tensor stand-in recording how it was pinned and copied."""

            def __init__(self):
                self.pinned = False
                self.copy = None

            def pin_memory(self):
                self.pinned = True
                return self

            def to(self, target, **kwargs):
                self.copy = (target, kwargs)
                return self

        tensor = Tensor()
        with patch.object(module, "has_cuda", True):
            moved = module._move_advanced_tensor_to_device(
                tensor, "cuda", pinned_memory=True
            )

        self.assertIs(moved, tensor)
        self.assertTrue(tensor.pinned)
        self.assertEqual(tensor.copy, ("cuda", {"non_blocking": True}))

    def test_double_buffer_prepares_next_batch_before_current_is_consumed(self):
        module = load_server()
        second_prepared = threading.Event()

        def prepare(value):
            if value == 2:
                second_prepared.set()
            return value

        buffered = module._iter_double_buffered([1, 2], prepare, enabled=True)
        try:
            self.assertEqual(next(buffered), 1)
            self.assertTrue(second_prepared.wait(timeout=1.0))
            self.assertEqual(next(buffered), 2)
            with self.assertRaises(StopIteration):
                next(buffered)
        finally:
            buffered.close()

    def test_cpu_affinity_prefers_physical_cores_and_one_socket(self):
        module = load_server()
        cpuinfo = """\
processor : 0
physical id : 0
core id : 0

processor : 1
physical id : 0
core id : 0

processor : 2
physical id : 0
core id : 1

processor : 3
physical id : 1
core id : 0
"""

        compact = module._advanced_cpu_affinity_target(
            "compact", {0, 1, 2, 3}, 2, cpuinfo
        )
        socket_local = module._advanced_cpu_affinity_target(
            "socket-local", {0, 1, 2, 3}, 2, cpuinfo
        )

        self.assertEqual(compact, {0, 2})
        self.assertEqual(socket_local, {0, 2})


class TestRerankerAdmission(unittest.IsolatedAsyncioTestCase):
    """The per-instance reranker lock remains effective with multiple lanes."""

    async def asyncSetUp(self):
        self.module = load_server(
            {
                "EMBEDDER_PROFILE": "advanced",
                "ADVANCED_INFERENCE_LANES": "2",
            }
        )
        self.started = threading.Event()
        self.release = threading.Event()
        started, release = self.started, self.release

        class ExclusiveReranker:
            """Fail if concurrent preparation reaches one reranker instance."""

            model_name = "test-reranker"

            def __init__(self):
                self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
                self.preparing = 0
                self.calls = 0
                self.fail = False

            def prepare_pairs_for_scoring(self, query, passages):
                self.preparing += 1
                if self.preparing != 1:
                    raise RuntimeError("reranker preparation overlapped")
                self.calls += 1
                started.set()
                try:
                    if self.calls == 1:
                        release.wait(timeout=2)
                    return [[query, passage] for passage in passages], []
                finally:
                    self.preparing -= 1

            def score(self, pairs, normalize, deadline=None):
                del normalize, deadline
                if self.fail:
                    raise RuntimeError("synthetic worker failure")
                return [1.0 for _ in pairs]

        self.reranker = ExclusiveReranker()
        self.module.reranker = self.reranker
        self.module.rerank_slots = asyncio.Semaphore(32)
        self.module.reranker_instance_lock = asyncio.Lock()
        self.module.record_gpu_memory_metrics = lambda: None
        self.request = self.module.RerankRequest(query="query", passages=["passage"])
        self.tasks = []

    async def asyncTearDown(self):
        self.release.set()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await asyncio.to_thread(
            self.reranker.executor.shutdown, wait=True, cancel_futures=True
        )
        self.module.GPU_EXECUTOR.shutdown(wait=True, cancel_futures=True)
        await asyncio.sleep(0)

    def _start_request(self):
        task = asyncio.create_task(self.module.rerank(self.request))
        self.tasks.append(task)
        return task

    async def test_two_lane_reranks_do_not_prepare_the_same_instance_together(self):
        """An admitted request may wait more than 0.5 s for its instance."""
        first = self._start_request()
        self.assertTrue(await asyncio.to_thread(self.started.wait, 1))
        second = self._start_request()
        await asyncio.sleep(0.65)
        self.assertFalse(second.done())
        self.assertEqual(self.reranker.calls, 1)
        self.release.set()
        first_response, second_response = await asyncio.gather(first, second)

        self.assertEqual(len(first_response.results), 1)
        self.assertEqual(len(second_response.results), 1)
        self.assertEqual(self.reranker.calls, 2)

    async def test_instance_wait_uses_the_processing_deadline(self):
        self.module.RERANK_GPU_TIMEOUT = 0.15
        async with self.module.reranker_instance_lock:
            with self.assertRaises(self.module.HTTPException) as caught:
                await self.module.rerank(self.request)
            self.assertEqual(caught.exception.status_code, 504)
            self.assertEqual(self.reranker.calls, 0)
        self.release.set()
        response = await self.module.rerank(self.request)
        self.assertEqual(len(response.results), 1)

    async def test_cancelled_waiter_does_not_release_another_requests_lock(self):
        async with self.module.reranker_instance_lock:
            waiting = self._start_request()
            await asyncio.sleep(0.05)
            waiting.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiting
            self.assertTrue(self.module.reranker_instance_lock.locked())
            self.assertEqual(self.reranker.calls, 0)
        self.release.set()
        response = await self.module.rerank(self.request)
        self.assertEqual(len(response.results), 1)

    async def test_cancelled_caller_keeps_the_lock_until_its_worker_finishes(self):
        first = self._start_request()
        self.assertTrue(await asyncio.to_thread(self.started.wait, 1))
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        second = self._start_request()
        await asyncio.sleep(0.05)
        self.assertTrue(self.module.reranker_instance_lock.locked())
        self.assertEqual(self.reranker.calls, 1)
        self.release.set()
        response = await second
        self.assertEqual(len(response.results), 1)

    async def test_abandoned_worker_failure_is_consumed_and_releases_the_lock(self):
        loop = asyncio.get_running_loop()
        errors = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: errors.append(context))
        try:
            self.reranker.fail = True
            first = self._start_request()
            self.assertTrue(await asyncio.to_thread(self.started.wait, 1))
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            self.release.set()
            await asyncio.wait_for(self.module.reranker_instance_lock.acquire(), 2)
            self.module.reranker_instance_lock.release()
            await asyncio.sleep(0.01)
            self.assertEqual(errors, [])
        finally:
            loop.set_exception_handler(previous_handler)


# --- Fakes for micro-batching tests ---


class FakeBatchTensor:
    """Batch tensor stand-in that survives every reshaping call unchanged."""

    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), max((len(row) for row in rows), default=0))

    def to(self, device):
        return self

    def __getitem__(self, key):
        return self

    def bool(self):
        return self

    def unsqueeze(self, dim):
        return self

    def expand(self, size):
        return self

    def float(self):
        return self


class FakeDenseOutput:
    """Dense output stand-in convertible to plain nested lists."""

    def __init__(self, batch_size, dim=1024):
        self.shape = (batch_size, dim)
        self.normalized = False

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return [[0.0] * self.shape[1] for _ in range(self.shape[0])]


class FakeLastHiddenState:
    """Hidden-state stand-in yielding a FakeDenseOutput on any slice."""

    def __init__(self, batch_size, dim=1024):
        self._batch_size = batch_size
        self._dim = dim

    def __getitem__(self, key):
        return FakeDenseOutput(self._batch_size, self._dim)


class TrackingPadTokenizer:
    """Tokenizer stand-in recording the padding arguments it receives."""

    pad_token = "<eos>"
    eos_token = "<eos>"

    def __init__(self):
        self.pad_calls = []

    def __call__(self, sentences, **kwargs):
        max_length = kwargs["max_length"]
        input_ids = []
        attention_mask = []
        for sentence in sentences:
            count = int(sentence.split(":", 1)[1])
            row = list(range(min(count, max_length)))
            input_ids.append(row)
            attention_mask.append([1] * len(row))
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def pad(self, inputs, **kwargs):
        self.pad_calls.append([list(row) for row in inputs["input_ids"]])
        width = max(len(row) for row in inputs["input_ids"])
        padded = []
        masks = []
        for row in inputs["input_ids"]:
            prefix = [0] * (width - len(row))
            padded.append(prefix + row)
            masks.append([0] * len(prefix) + [1] * len(row))
        return {
            "input_ids": FakeBatchTensor(padded),
            "attention_mask": FakeBatchTensor(masks),
        }


class TrackingDenseModel:
    """Dense model stand-in recording every batch it was called with."""

    def __init__(self):
        self.calls = []

    def __call__(self, **inputs):
        batch_size = inputs["input_ids"].shape[0]
        self.calls.append(inputs["input_ids"].shape)
        return type(
            "Output",
            (),
            {"last_hidden_state": FakeLastHiddenState(batch_size)},
        )()


class CapturingExecutor:
    """Executor stand-in capturing submitted work instead of running it."""

    def __init__(self):
        self.submitted = []

    def submit(self, fn, *args):
        self.submitted.append((fn, args))
        future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future


class CapturingMetric:
    """Prometheus metric stand-in capturing labels and observed values."""

    def __init__(self):
        self.observations = []

    def labels(self, *args, **kwargs):
        return self

    def observe(self, value):
        self.observations.append(value)


class FakeCpuList:
    """Torch-tensor stand-in wrapping a plain list of CPU indices."""

    def __init__(self, data):
        self._data = data

    def cpu(self):
        return self

    def tolist(self):
        return list(self._data)


class FakeSparseValues:
    """Sparse-value tensor stand-in convertible to a plain list."""

    def __init__(self, n):
        self._n = n

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return [1.0] * self._n


class FakeSparseIndices:
    """Sparse-index tensor stand-in yielding a row on any slice."""

    def __init__(self, n):
        self._n = n

    def __getitem__(self, key):
        return FakeCpuList(list(range(self._n)))


class FakeSparseReps:
    """Sparse representation stand-in exposing coalesced indices and values."""

    def __init__(self, n_sentences):
        self._n = n_sentences

    def coalesce(self):
        return self

    def indices(self):
        return FakeSparseIndices(self._n)

    def values(self):
        return FakeSparseValues(self._n)

    @property
    def shape(self):
        return (self._n, self._n)


class TrackingMilcoModel:
    """MILCO model stand-in recording every batch it encoded."""

    def __init__(self):
        self.calls = []
        self.source_views = []

    def encode_text(self, sentences, source_view=False):
        self.calls.append(list(sentences))
        self.source_views.append(source_view)
        return FakeSparseReps(len(sentences))


class TestAdvancedEmbeddingMicroBatching(unittest.TestCase):
    """Micro-batching splits work by item count and token budget."""

    def test_dense_and_sparse_use_their_own_pinned_memory_policy(self):
        module = load_server()
        observed = []

        def capture_move(value, target, *, pinned_memory):
            observed.append((target, pinned_memory))
            return value

        dense = module.QwenDenseEmbeddingBackend.__new__(
            module.QwenDenseEmbeddingBackend
        )
        dense.tokenizer = TrackingPadTokenizer()
        dense.model = TrackingDenseModel()
        dense.device = "cuda"
        dense.max_length = 128

        sparse = module.MilcoSparseBackend.__new__(module.MilcoSparseBackend)
        sparse.tokenizer = TrackingPadTokenizer()
        sparse.model = object()
        sparse.device = "cuda"
        sparse.max_length = 128

        with (
            patch.object(module, "ADVANCED_DENSE_PINNED_MEMORY", True),
            patch.object(module, "ADVANCED_SPARSE_PINNED_MEMORY", False),
            patch.object(module, "_move_advanced_tensor_to_device", capture_move),
        ):
            dense.embed_dense(["tok:8"])

        self.assertTrue(observed)
        self.assertTrue(all(pinned for _target, pinned in observed))
        observed.clear()

        with (
            patch.object(module, "ADVANCED_DENSE_PINNED_MEMORY", True),
            patch.object(module, "ADVANCED_SPARSE_PINNED_MEMORY", False),
            patch.object(module, "_move_advanced_tensor_to_device", capture_move),
            patch.object(
                module,
                "memory_bounded_milco_forward",
                return_value=FakeSparseReps(1),
            ),
            patch.object(
                module.torch,
                "autocast",
                create=True,
                side_effect=lambda **_kwargs: contextlib.nullcontext(),
            ),
        ):
            sparse.embed_sparse(["tok:8"])

        self.assertTrue(observed)
        self.assertTrue(all(not pinned for _target, pinned in observed))

    def test_qwen_dense_chunks_by_item_and_token_budget(self):
        module = load_server()
        tokenizer = TrackingPadTokenizer()
        model = TrackingDenseModel()
        backend = module.QwenDenseEmbeddingBackend.__new__(
            module.QwenDenseEmbeddingBackend
        )
        backend.tokenizer = tokenizer
        backend.model = model
        backend.device = "cuda"
        backend.max_length = 128

        with (
            patch.object(module, "QWEN_EMBED_BATCH_SIZE", 2),
            patch.object(module, "QWEN_EMBED_MAX_BATCH_TOKENS", 160),
        ):
            vectors = backend.embed_dense(
                ["tok:100", "tok:80", "tok:70", "tok:10"], normalize_dense=False
            )

        self.assertEqual(len(vectors["dense_vecs"]), 4)
        self.assertEqual([len(call) for call in tokenizer.pad_calls], [1, 2, 1])
        self.assertEqual(model.calls, [(1, 100), (2, 80), (1, 10)])

    def test_qwen_dense_clamps_single_item_to_token_budget(self):
        module = load_server()
        tokenizer = TrackingPadTokenizer()
        model = TrackingDenseModel()
        backend = module.QwenDenseEmbeddingBackend.__new__(
            module.QwenDenseEmbeddingBackend
        )
        backend.tokenizer = tokenizer
        backend.model = model
        backend.device = "cuda"
        backend.max_length = 128

        with (
            patch.object(module, "QWEN_EMBED_BATCH_SIZE", 8),
            patch.object(module, "QWEN_EMBED_MAX_BATCH_TOKENS", 64),
        ):
            vectors = backend.embed_dense(["tok:100"], normalize_dense=False)

        self.assertEqual(len(vectors["dense_vecs"]), 1)
        self.assertEqual(model.calls, [(1, 64)])

    def test_milco_sparse_chunks_by_batch_size(self):
        module = load_server()
        backend = module.MilcoSparseBackend.__new__(module.MilcoSparseBackend)
        backend.model = TrackingMilcoModel()

        with patch.object(module, "MILCO_BATCH_SIZE", 2):
            result = backend.embed_sparse(["a", "b", "c", "d", "e"])

        self.assertEqual(len(result["lexical_weights"]), 5)
        self.assertEqual(backend.model.calls, [["a", "b"], ["c", "d"], ["e"]])
        self.assertEqual(backend.model.source_views, [True, True, True])


class FakeEmbeddingService:
    """Embedding service stand-in returning fixed output for metric tests."""

    def embed(self, sentences, **kwargs):
        return {"dense_vecs": [[1.0] for _ in sentences]}


class TestPhaseMetrics(unittest.TestCase):
    """Per-phase timings and per-item models reach the response metrics."""

    def test_embed_on_worker_returns_phase_timing(self):
        module = load_server()
        processor = module.RequestProcessor.__new__(module.RequestProcessor)
        processor.model_wrapper = FakeEmbeddingService()

        result, worker_started_at, worker_finished_at = processor._embed_on_worker(
            ["a", "b"], True, False, False, False
        )

        self.assertEqual(result, {"dense_vecs": [[1.0], [1.0]]})
        self.assertGreaterEqual(worker_finished_at, worker_started_at)

    def test_embedding_response_build_metric_includes_per_item_models(self):
        module = load_server()
        observed_metric = CapturingMetric()
        clock = {"now": 100.0}
        original_single_response = module.SingleEmbeddingResponse

        async def fake_process_request(request, *, lease=None):
            return {"dense_vecs": [[1.0], [2.0]]}

        def fake_monotonic():
            return clock["now"]

        def slow_single_embedding_response(**kwargs):
            clock["now"] += 0.25
            return original_single_response(**kwargs)

        module.processor.process_request = fake_process_request
        module.api_response_build_duration = observed_metric
        request = module.EmbedRequest(
            sentences=["alpha", "beta"],
            return_dense=True,
            return_sparse=False,
            return_colbert=False,
            normalize_dense=False,
            sparse_as_indices=False,
        )

        with (
            patch.object(module.time, "monotonic", fake_monotonic),
            patch.object(
                module,
                "SingleEmbeddingResponse",
                slow_single_embedding_response,
            ),
        ):
            response = asyncio.run(module.get_embeddings(request))

        self.assertEqual(len(response.data), 2)
        self.assertEqual(observed_metric.observations, [0.5])


class TestCooperativeDeadlines(unittest.TestCase):
    """Deadlines are checked between micro-batches, not only per request."""

    def test_embedding_batch_deadline_uses_worker_timeout(self):
        module = load_server()
        processor = module.RequestProcessor.__new__(module.RequestProcessor)
        processor.stats = None
        processor.executor = CapturingExecutor()
        processor.model_wrapper = FakeEmbeddingService()
        processor.response_futures = {}
        processor._reservations = {}

        loop = asyncio.new_event_loop()
        self.addCleanup(loop.close)
        request = module.EmbedRequest(
            sentences=["alpha"],
            return_dense=True,
            return_sparse=False,
            return_colbert=False,
            normalize_dense=False,
            sparse_as_indices=False,
        )
        processor.response_futures["req-1"] = loop.create_future()
        captured_timeouts = []

        class FakeDeadline:
            """Deadline stand-in that never expires."""

            def check(self):
                return None

        def fake_from_timeout(timeout_seconds, safety_margin_seconds=0.05):
            captured_timeouts.append(timeout_seconds)
            return FakeDeadline()

        async def run_batch():
            with (
                patch.object(module, "REQUEST_TIMEOUT", 90),
                patch.object(module, "GPU_PROCESS_TIMEOUT", 15),
                patch.object(
                    module.InferenceDeadline,
                    "from_timeout",
                    fake_from_timeout,
                ),
            ):
                await processor._run_model_on_batch(
                    ["alpha"],
                    ["req-1"],
                    [("req-1", request, time.monotonic())],
                    (True, False, False, False),
                )

        loop.run_until_complete(run_batch())

        self.assertEqual(captured_timeouts, [15])
        response = processor.response_futures["req-1"].result()
        self.assertEqual(len(response["dense_vecs"]), 1)

    def test_qwen_dense_checks_deadline_between_micro_batches(self):
        module = load_server()
        tokenizer = TrackingPadTokenizer()
        model = TrackingDenseModel()
        backend = module.QwenDenseEmbeddingBackend.__new__(
            module.QwenDenseEmbeddingBackend
        )
        backend.tokenizer = tokenizer
        backend.model = model
        backend.device = "cuda"
        backend.max_length = 128

        call_count = {"n": 0}

        class ExpiredDeadline:
            """Deadline stand-in expiring after the first check."""

            def check(self):
                call_count["n"] += 1
                if call_count["n"] > 1:
                    raise module.InferenceTimeoutError("expired")

        with (
            patch.object(module, "QWEN_EMBED_BATCH_SIZE", 1),
            patch.object(module, "QWEN_EMBED_MAX_BATCH_TOKENS", 999999),
            self.assertRaises(module.InferenceTimeoutError),
        ):
            backend.embed_dense(
                ["tok:10", "tok:10", "tok:10"],
                normalize_dense=False,
                deadline=ExpiredDeadline(),
            )

        self.assertEqual(len(model.calls), 1)

    def test_milco_sparse_checks_deadline_between_micro_batches(self):
        module = load_server()
        backend = module.MilcoSparseBackend.__new__(module.MilcoSparseBackend)
        backend.model = TrackingMilcoModel()

        call_count = {"n": 0}

        class ExpiredDeadline:
            """Deadline stand-in expiring after the first check."""

            def check(self):
                call_count["n"] += 1
                if call_count["n"] > 1:
                    raise module.InferenceTimeoutError("expired")

        with (
            patch.object(module, "MILCO_BATCH_SIZE", 1),
            self.assertRaises(module.InferenceTimeoutError),
        ):
            backend.embed_sparse(["a", "b", "c"], deadline=ExpiredDeadline())

        self.assertEqual(len(backend.model.calls), 1)

    def test_bge_rerank_uses_bge_batch_size_not_qwen_batch_size(self):
        module = load_server()
        wrapper = module.RerankerWrapper(module.BGE_RERANKER_MODEL)
        wrapper.model.calls = []

        with (
            patch.object(module, "BGE_RERANK_BATCH_SIZE", 2),
            patch.object(module, "QWEN_RERANK_BATCH_SIZE", 99),
        ):
            scores = wrapper.score(
                [["q", "p1"], ["q", "p2"], ["q", "p3"], ["q", "p4"], ["q", "p5"]],
                normalize=True,
            )

        self.assertEqual(len(scores), 5)
        chunk_sizes = [len(call[0]) for call in wrapper.model.calls]
        self.assertEqual(chunk_sizes, [2, 2, 1])

    def test_embedding_service_passes_deadline_to_backends(self):
        module = load_server()
        service = module.EmbeddingService.__new__(module.EmbeddingService)
        service.profile = module.EMBEDDER_PROFILE_ADVANCED
        service.bge_backend = None

        class DeadlineTrackingQwen:
            """Qwen dense backend stand-in recording the deadline it was passed."""

            def __init__(self):
                self.received_deadline = "not_set"

            def embed_dense(self, sentences, normalize_dense=False, deadline=None):
                self.received_deadline = deadline
                return {"dense_vecs": [[1.0] for _ in sentences]}

        class DeadlineTrackingMilco:
            """MILCO sparse backend stand-in recording the deadline it was passed."""

            def __init__(self):
                self.received_deadline = "not_set"

            def embed_sparse(self, sentences, deadline=None):
                self.received_deadline = deadline
                return {"lexical_weights": [{"7": 0.9} for _ in sentences]}

        service.qwen_dense_backend = DeadlineTrackingQwen()
        service.milco_sparse_backend = DeadlineTrackingMilco()
        service.bge_model_name = None
        service.dense_model_name = module.QWEN_DENSE_EMBEDDING_MODEL
        service.sparse_model_name = module.MILCO_SPARSE_MODEL
        service.colbert_model_name = None

        test_deadline = module.InferenceDeadline.from_timeout(60)
        service.embed(
            ["a"], return_dense=True, return_sparse=True, deadline=test_deadline
        )

        self.assertIs(service.qwen_dense_backend.received_deadline, test_deadline)
        self.assertIs(service.milco_sparse_backend.received_deadline, test_deadline)


class TestIncludeText(unittest.TestCase):
    """include_text decides whether the source text is echoed back."""

    def test_include_text_false_sets_text_to_none(self):
        import asyncio

        module = load_server()

        async def fake_process_request(request, *, lease=None):
            return {"dense_vecs": [[1.0]]}

        module.processor.process_request = fake_process_request
        request = module.EmbedRequest(
            sentences=["alpha"],
            return_dense=True,
            return_sparse=False,
            return_colbert=False,
            normalize_dense=False,
            sparse_as_indices=False,
            include_text=False,
        )

        response = asyncio.run(module.get_embeddings(request))

        self.assertIsInstance(response, module.EmbeddingsListResponse)
        self.assertIsNone(response.data[0].text)

    def test_include_text_default_keeps_text(self):
        import asyncio

        module = load_server()

        async def fake_process_request(request, *, lease=None):
            return {"dense_vecs": [[1.0]]}

        module.processor.process_request = fake_process_request
        request = module.EmbedRequest(
            sentences=["alpha"],
            return_dense=True,
            return_sparse=False,
            return_colbert=False,
            normalize_dense=False,
            sparse_as_indices=False,
        )

        response = asyncio.run(module.get_embeddings(request))

        self.assertEqual(response.data[0].text, "alpha")

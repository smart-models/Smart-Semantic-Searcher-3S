"""Request admission, cancellation and shutdown with real worker barriers."""

import asyncio
import threading
import time
import unittest

from tests.book_cases import book_case
from tests.test_reranker_wrapper import load_server


class Gauge:
    """Track the production request and queue accounting without Prometheus."""

    def __init__(self):
        self.value = 0

    def inc(self):
        self.value += 1

    def dec(self):
        self.value -= 1

    def set(self, value):
        self.value = value


class NativeBackend:
    """Keep a real thread active until the test releases its model call."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = []

    def embed(self, sentences, **kwargs):
        self.calls.append(list(sentences))
        self.entered.set()
        if not self.release.wait(5):
            raise RuntimeError("test did not release native worker")
        return {"dense_vecs": [[float(len(text)), 1.0] for text in sentences]}


class TestRequestProcessorLifecycle(unittest.IsolatedAsyncioTestCase):
    """Every caller settles while native resources keep their physical lifetime."""

    async def asyncSetUp(self):
        self.server = load_server()
        self.server.active_requests = Gauge()
        self.server.queue_size = Gauge()
        self.backend = NativeBackend()
        self.processor = self.server.RequestProcessor(
            self.backend, 2, 0.01, max_queue_size=1
        )
        self.tasks = []

    async def asyncTearDown(self):
        self.backend.release.set()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        loop = getattr(self.processor, "processing_loop_task", None)
        if loop:
            loop.cancel()
            await asyncio.gather(loop, return_exceptions=True)
        for future in self.processor.response_futures.values():
            future.cancel()
        await asyncio.to_thread(self.server.GPU_EXECUTOR.shutdown, wait=True)

    def submit(self, text):
        request = self.server.EmbedRequest(sentences=[text], return_colbert=False)
        task = asyncio.create_task(self.processor.process_request(request))
        self.tasks.append(task)
        return task

    async def until(self, predicate):
        deadline = time.monotonic() + 2
        while not predicate():
            if time.monotonic() >= deadline:
                self.fail("processor did not reach the expected state")
            await asyncio.sleep(0)

    @book_case("TB-LIFE-001")
    async def test_cancellation_during_admission_cleans_future_and_counter(self):
        self.processor.queue.put_nowait(("occupied", None, time.monotonic()))
        caller = self.submit("waiting")
        await self.until(lambda: bool(self.processor.response_futures))
        future = next(iter(self.processor.response_futures.values()))
        caller.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await caller
        self.assertTrue(future.done())
        self.assertEqual(self.processor.response_futures, {})
        self.assertEqual(self.server.active_requests.value, 0)

    @book_case("TB-LIFE-001")
    async def test_cancelled_batch_member_does_not_shift_survivor_vectors(self):
        self.processor.queue = asyncio.Queue(maxsize=4)
        first, survivor = self.submit("abandoned"), self.submit("survives")
        await self.until(lambda: self.processor.queue.qsize() == 2)
        await self.processor.start_processing_loop()
        self.assertTrue(await asyncio.to_thread(self.backend.entered.wait, 2))
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.backend.release.set()
        self.assertEqual((await survivor)["dense_vecs"], [[8.0, 1.0]])
        self.assertEqual(self.processor.response_futures, {})
        self.assertEqual(self.server.active_requests.value, 0)

    @book_case("TB-LIFE-003")
    async def test_shutdown_drains_accepted_work(self):
        self.processor.max_batch_size = 1
        await self.processor.start_processing_loop()
        first = self.submit("native")
        self.assertTrue(await asyncio.to_thread(self.backend.entered.wait, 2))
        second = self.submit("queued")
        await self.until(lambda: self.processor.queue.qsize() == 1)
        shutdown = asyncio.create_task(self.processor.graceful_shutdown(timeout=0.2))
        self.tasks.append(shutdown)
        await self.until(lambda: self.processor.is_shutting_down)
        self.backend.release.set()
        await shutdown
        self.assertTrue(first.done() and second.done())
        self.assertEqual((await second)["dense_vecs"], [[6.0, 1.0]])
        self.assertEqual(self.processor.response_futures, {})
        self.assertTrue(self.processor.queue.empty())

    @book_case("TB-LIFE-003")
    async def test_shutdown_deadline_settles_native_queued_and_admission_callers(self):
        self.processor.max_batch_size = 1
        await self.processor.start_processing_loop()
        self.submit("native")
        self.assertTrue(await asyncio.to_thread(self.backend.entered.wait, 2))
        self.submit("queued")
        self.submit("admission")
        await self.until(lambda: len(self.processor.response_futures) == 3)
        await self.processor.graceful_shutdown(timeout=0.03)
        self.assertTrue(all(task.done() for task in self.tasks))
        self.assertEqual(self.processor.response_futures, {})
        self.assertTrue(self.processor.queue.empty())
        self.assertEqual(self.server.active_requests.value, 0)
        self.assertEqual(self.server.queue_size.value, 0)
        for result in await asyncio.gather(*self.tasks, return_exceptions=True):
            self.assertEqual(result.status_code, 503)

    @book_case("TB-LIFE-003")
    async def test_shutdown_rejects_new_direct_caller_without_allocating(self):
        await self.processor.graceful_shutdown(timeout=0)
        with self.assertRaises(self.server.HTTPException) as caught:
            await asyncio.wait_for(self.submit("new"), 1)
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(self.processor.response_futures, {})

    @book_case("TB-LIFE-001")
    async def test_cancelled_queued_work_is_not_submitted(self):
        caller = self.submit("abandoned")
        await self.until(lambda: self.processor.queue.qsize() == 1)
        caller.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await caller
        self.backend.release.set()
        await self.processor.start_processing_loop()
        await self.until(lambda: self.processor.queue.empty())
        self.assertEqual(self.backend.calls, [])

    @book_case("TB-LIFE-003")
    async def test_lifespan_keeps_models_open_until_native_executor_finishes(self):
        closed = []
        self.server.processor = self.processor
        self.server.embedding_service.close = lambda: closed.append(True)
        drain = self.processor.graceful_shutdown

        async def short_drain(timeout=30):
            await drain(timeout=0.03)

        self.processor.graceful_shutdown = short_drain
        context = self.server.lifespan(None)
        await context.__aenter__()
        caller = self.submit("native")
        self.assertTrue(await asyncio.to_thread(self.backend.entered.wait, 2))
        shutdown = asyncio.create_task(context.__aexit__(None, None, None))
        self.tasks.append(shutdown)
        await self.until(caller.done)
        self.assertFalse(shutdown.done())
        self.assertEqual(closed, [])
        self.backend.release.set()
        await asyncio.wait_for(shutdown, 2)
        self.assertEqual(closed, [True])

    @book_case("TB-LIFE-003")
    async def test_cancelled_shutdown_waiter_keeps_native_rerank_resources_alive(self):
        closed = []
        self.server.processor = self.processor
        self.server.embedding_service.close = lambda: closed.append(True)
        context = self.server.lifespan(None)
        await context.__aenter__()
        # Reranking uses the same executor but is outside response_futures.
        native = self.server.GPU_EXECUTOR.submit(self.backend.embed, ["rerank"])
        self.assertTrue(await asyncio.to_thread(self.backend.entered.wait, 2))
        shutdown = asyncio.create_task(context.__aexit__(None, None, None))
        self.tasks.append(shutdown)
        await self.until(lambda: self.processor.processing_loop_task.done())
        self.assertFalse(shutdown.done())
        self.assertEqual(closed, [])
        shutdown.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await shutdown
        self.assertFalse(native.done())
        self.assertEqual(closed, [])
        self.backend.release.set()
        await asyncio.wrap_future(native)
        self.assertEqual(closed, [])

    def api_request(self, text="native"):
        return self.server.EmbedRequest(
            sentences=[text],
            return_dense=True,
            return_sparse=False,
            return_colbert=False,
            include_text=False,
        )

    def api_submit(self, request):
        self.server.processor = self.processor
        task = asyncio.create_task(self.server.get_embeddings(request))
        self.tasks.append(task)
        return task

    @book_case("TB-PER-001")
    async def test_oversized_api_request_is_refused_before_warning_or_model_work(self):
        self.server.inference_budget = self.server.RequestBudget(1, 1000000)
        with self.assertRaises(self.server.HTTPException) as caught:
            await self.api_submit(self.api_request())
        self.assertEqual(caught.exception.status_code, 413)
        self.assertEqual(caught.exception.detail["code"], "request_budget_exceeded")
        self.assertEqual(self.processor.response_futures, {})
        self.assertEqual(self.backend.calls, [])
        self.assertEqual(self.server.inference_budget.used.tokens, 0)

    @book_case("TB-PER-001")
    async def test_warning_work_runs_on_worker_and_cancelled_caller_keeps_reservation(
        self,
    ):
        request = self.api_request()
        cost = self.server._embedding_request_cost(request)
        self.server.inference_budget = self.server.RequestBudget(cost.tokens, 1000000)
        loop_thread = threading.get_ident()
        warning_threads = []

        def warnings(_request):
            warning_threads.append(threading.get_ident())
            return []

        self.server.collect_embedding_truncation_warnings = warnings
        await self.processor.start_processing_loop()
        caller = self.api_submit(request)
        self.assertTrue(await asyncio.to_thread(self.backend.entered.wait, 2))
        caller.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await caller
        self.assertEqual(self.server.inference_budget.used.tokens, cost.tokens)
        with self.assertRaises(self.server.HTTPException) as caught:
            await self.api_submit(request)
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(caught.exception.headers, {"Retry-After": "1"})
        self.assertEqual(len(self.backend.calls), 1)
        self.backend.release.set()
        await self.until(lambda: self.server.inference_budget.used.tokens == 0)
        response = await self.api_submit(request)
        self.assertEqual(len(response.data), 1)
        self.assertTrue(
            warning_threads and all(value != loop_thread for value in warning_threads)
        )
        self.assertEqual(self.server.inference_budget.used.tokens, 0)

    @book_case("TB-PER-001")
    async def test_cancelled_queued_api_request_releases_its_budget(self):
        caller = self.api_submit(self.api_request())
        await self.until(lambda: self.processor.queue.qsize() == 1)
        self.assertGreater(self.server.inference_budget.used.tokens, 0)
        caller.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await caller
        self.assertEqual(self.server.inference_budget.used.tokens, 0)
        self.assertEqual(self.processor._reservations, {})
        self.assertEqual(self.backend.calls, [])

    @book_case("TB-LIFE-001", "TB-PER-001")
    async def test_rerank_and_embedding_share_capacity_after_native_cancellation(self):
        rerank_request = self.server.RerankRequest(
            query="query", passages=["passage"], normalize=True
        )
        cost = self.server.estimate_rerank(
            "query", ["passage"], pair_limit=self.server.BGE_RERANK_MAX_LENGTH
        )
        self.server.inference_budget = self.server.RequestBudget(cost.tokens, 1000000)

        def score(pairs, normalize, deadline=None):
            self.backend.entered.set()
            if not self.backend.release.wait(5):
                raise RuntimeError("native rerank barrier was not released")
            return [0.5] * len(pairs)

        self.server.reranker.score = score
        caller = asyncio.create_task(self.server.rerank(rerank_request))
        self.tasks.append(caller)
        self.assertTrue(await asyncio.to_thread(self.backend.entered.wait, 2))
        caller.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await caller
        self.assertEqual(self.server.inference_budget.used.tokens, cost.tokens)
        with self.assertRaises(self.server.HTTPException) as caught:
            await self.api_submit(self.api_request())
        self.assertEqual(caught.exception.status_code, 503)
        self.backend.release.set()
        await self.until(lambda: self.server.inference_budget.used.tokens == 0)
        await self.processor.start_processing_loop()
        self.assertEqual(len((await self.api_submit(self.api_request())).data), 1)

    @book_case("TB-LIFE-003", "TB-PER-001")
    async def test_shutdown_settles_callers_but_native_work_retains_its_budget(self):
        self.processor.max_batch_size = 1
        await self.processor.start_processing_loop()
        first = self.api_submit(self.api_request("native"))
        self.assertTrue(await asyncio.to_thread(self.backend.entered.wait, 2))
        second = self.api_submit(self.api_request("queued"))
        await self.until(lambda: self.processor.queue.qsize() == 1)
        await self.processor.graceful_shutdown(timeout=0.03)
        results = await asyncio.gather(first, second, return_exceptions=True)
        self.assertTrue(all(result.status_code == 503 for result in results))
        self.assertEqual(
            self.server.inference_budget.used.tokens,
            self.server._embedding_request_cost(self.api_request("native")).tokens,
        )
        self.backend.release.set()
        await self.until(lambda: self.server.inference_budget.used.tokens == 0)

    @book_case("TB-LIFE-001")
    async def test_queued_and_submitted_cancellations_never_settle_cancelled_futures(
        self,
    ):
        # One request is cancelled while native, one while still queued; the
        # survivor of the native batch keeps its own vector and no cancelled
        # future is ever completed by the processor (TB-LIFE-001).
        self.processor.queue = asyncio.Queue(maxsize=4)
        self.processor.max_batch_size = 2
        native_cancelled = self.submit("abandoned")
        survivor = self.submit("survives")
        await self.until(lambda: self.processor.queue.qsize() == 2)
        await self.processor.start_processing_loop()
        self.assertTrue(await asyncio.to_thread(self.backend.entered.wait, 2))
        queued_cancelled = self.submit("queued-then-cancelled")
        await self.until(lambda: len(self.processor.response_futures) == 3)
        settled_after_cancel = []

        def guard(future):
            set_result, set_exception = future.set_result, future.set_exception

            def settle(original, *args):
                if future.cancelled():
                    settled_after_cancel.append(future)
                return original(*args)

            future.set_result = lambda *a: settle(set_result, *a)
            future.set_exception = lambda *a: settle(set_exception, *a)

        for future in list(self.processor.response_futures.values()):
            guard(future)
        native_cancelled.cancel()
        queued_cancelled.cancel()
        for task in (native_cancelled, queued_cancelled):
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.backend.release.set()
        self.assertEqual((await survivor)["dense_vecs"], [[8.0, 1.0]])
        await self.until(lambda: self.processor.queue.empty())
        self.assertEqual(settled_after_cancel, [])
        self.assertEqual(self.processor.response_futures, {})
        self.assertEqual(self.server.active_requests.value, 0)
        self.assertEqual(self.server.queue_size.value, 0)
        submitted = [text for call in self.backend.calls for text in call]
        self.assertNotIn("queued-then-cancelled", submitted)

    @book_case("TB-LIFE-003")
    async def test_shutdown_leaves_no_orphan_task_and_refuses_http_work(self):
        before = {task for task in asyncio.all_tasks() if not task.done()}
        self.processor.max_batch_size = 1
        await self.processor.start_processing_loop()
        native = self.submit("native")
        self.assertTrue(await asyncio.to_thread(self.backend.entered.wait, 2))
        queued = self.submit("queued")
        await self.until(lambda: self.processor.queue.qsize() == 1)
        shutdown = asyncio.create_task(self.processor.graceful_shutdown(timeout=0.2))
        await self.until(lambda: self.processor.is_shutting_down)

        async def call_next(_request):
            self.fail("shutdown middleware forwarded a request during drain")

        class Response:
            def __init__(self, status_code, content):
                self.status_code, self.content = status_code, content

        self.server.processor = self.processor
        self.server.JSONResponse = Response
        response = await self.server.shutdown_middleware(object(), call_next)
        self.assertEqual(response.status_code, 503)
        self.backend.release.set()
        await shutdown
        await asyncio.gather(native, queued, return_exceptions=True)
        await asyncio.to_thread(self.server.GPU_EXECUTOR.shutdown, wait=True)
        self.assertTrue(self.processor.queue.empty())
        self.assertEqual(self.processor.response_futures, {})
        leaked = {
            task
            for task in asyncio.all_tasks()
            if not task.done()
            and task not in before
            and task is not asyncio.current_task()
        }
        self.assertEqual(leaked, set())
        # The executor accepted the shutdown and every worker thread has exited,
        # so no native work item can still be pending anywhere.
        executor = self.server.GPU_EXECUTOR
        self.assertTrue(executor._shutdown)
        self.assertTrue(all(not thread.is_alive() for thread in executor._threads))
        # Accepted work drained: the native and the queued request were both run.
        self.assertEqual(len(self.backend.calls), 2)

    @book_case("TB-PER-001")
    async def test_endpoint_budget_admits_two_halves_refuses_one_unit_then_recovers(
        self,
    ):
        unit = self.server._embedding_request_cost(self.api_request("a"))
        # A 4096-unit budget expressed in whole requests: two requests fill it.
        budget = self.server.RequestBudget(unit.tokens * 2, 1000000)
        self.server.inference_budget = budget
        self.processor.max_batch_size = 1
        await self.processor.start_processing_loop()
        first = self.api_submit(self.api_request("a"))
        self.assertTrue(await asyncio.to_thread(self.backend.entered.wait, 2))
        second = self.api_submit(self.api_request("b"))
        await self.until(lambda: budget.used.tokens == unit.tokens * 2)
        with self.assertRaises(self.server.HTTPException) as caught:
            await self.api_submit(self.api_request("c"))
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(caught.exception.headers, {"Retry-After": "1"})
        self.assertEqual(len(self.backend.calls), 1)
        self.backend.release.set()
        await first
        await second
        await self.until(lambda: budget.used.tokens == 0)
        third = await self.api_submit(self.api_request("c"))
        self.assertEqual(len(third.data), 1)
        self.assertEqual(budget.used.tokens, 0)
        self.assertEqual(budget.used.output_bytes, 0)

    @book_case("TB-PER-001")
    async def test_colbert_output_reservation_is_distinct_and_can_be_oversized(self):
        without = self.server._embedding_request_cost(
            self.server.EmbedRequest(sentences=["x"], return_colbert=False)
        )
        with_colbert = self.server._embedding_request_cost(
            self.server.EmbedRequest(sentences=["x"], return_colbert=True)
        )
        self.assertGreater(with_colbert.output_bytes, without.output_bytes)
        self.assertEqual(with_colbert.tokens, without.tokens)
        self.server.inference_budget = self.server.RequestBudget(
            with_colbert.tokens, without.output_bytes
        )
        with self.assertRaises(self.server.HTTPException) as caught:
            await self.api_submit(
                self.server.EmbedRequest(sentences=["x"], return_colbert=True)
            )
        self.assertEqual(caught.exception.status_code, 413)
        self.assertEqual(self.backend.calls, [])
        self.backend.release.set()
        await self.processor.start_processing_loop()
        response = await self.api_submit(self.api_request("x"))
        self.assertEqual(len(response.data), 1)
        self.assertEqual(self.server.inference_budget.used.output_bytes, 0)

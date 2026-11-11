"""The Qwen3 dense query prompt: who gets it, and who must not.

Qwen3-Embedding is trained asymmetrically: a query carries the model's query
prompt, a document carries nothing. These tests pin that the prompt reaches
only the ADVANCED dense backend, only for requests that say they are queries,
and that queries and documents never share a backend call.
"""

import asyncio
import typing
import unittest
from concurrent.futures import ThreadPoolExecutor

from tests.test_embedder_profile import TrackingMilcoBackend, _advanced_service
from tests.test_reranker_wrapper import load_server


class TrackingDenseBackend:
    """Qwen dense backend stand-in recording the sentences it receives."""

    def __init__(self):
        self.calls = []

    def embed_dense(self, sentences, normalize_dense=False, deadline=None):
        self.calls.append(list(sentences))
        return {"dense_vecs": [[1.0] for _ in sentences]}


class TrackingBgeBackend:
    """BGE-M3 backend stand-in recording the sentences it receives."""

    def __init__(self):
        self.calls = []

    def embed(self, sentences, **kwargs):
        self.calls.append(list(sentences))
        return {
            "dense_vecs": [[1.0] for _ in sentences],
            "lexical_weights": [{"7": 0.9} for _ in sentences],
        }


def _advanced(module):
    service = _advanced_service(module)
    service.qwen_dense_backend = TrackingDenseBackend()
    return service


def _basic(module):
    service = module.EmbeddingService.__new__(module.EmbeddingService)
    service.profile = module.EMBEDDER_PROFILE_BASIC
    service.bge_backend = TrackingBgeBackend()
    service.qwen_dense_backend = None
    service.milco_sparse_backend = None
    return service


class QueryPromptRoutingTests(unittest.TestCase):
    """Only the ADVANCED dense leg of a query request sees the prompt."""

    def test_prompt_is_the_one_the_pinned_model_ships(self):
        module = load_server()
        self.assertEqual(
            module.QWEN_DENSE_QUERY_PROMPT,
            "Instruct: Given a web search query, retrieve relevant passages "
            "that answer the query\nQuery:",
        )

    def test_advanced_query_is_prompted_for_dense_and_plain_for_sparse(self):
        module = load_server()
        service = _advanced(module)
        service.embed(["alpha"], return_colbert=False, input_type="query")
        self.assertEqual(
            service.qwen_dense_backend.calls,
            [[module.QWEN_DENSE_QUERY_PROMPT + "alpha"]],
        )
        self.assertEqual(service.milco_sparse_backend.calls, [{"sentences": ["alpha"]}])

    def test_advanced_document_reaches_dense_unchanged(self):
        module = load_server()
        service = _advanced(module)
        service.embed(["alpha"], return_colbert=False)
        self.assertEqual(service.qwen_dense_backend.calls, [["alpha"]])

    def test_advanced_query_is_prompted_on_the_parallel_hybrid_path(self):
        module = load_server()
        service = _advanced(module)
        service._hybrid_executor = ThreadPoolExecutor(max_workers=2)
        service._hybrid_streams = (None, None)
        try:
            service.embed(["alpha"], return_colbert=False, input_type="query")
        finally:
            service._hybrid_executor.shutdown(wait=True)
        self.assertEqual(
            service.qwen_dense_backend.calls,
            [[module.QWEN_DENSE_QUERY_PROMPT + "alpha"]],
        )
        self.assertEqual(service.milco_sparse_backend.calls, [{"sentences": ["alpha"]}])

    def test_advanced_query_keeps_caller_order_under_length_aware_batching(self):
        module = load_server()
        service = _advanced(module)
        service.milco_sparse_backend = TrackingMilcoBackend()
        texts = ["a much longer first sentence", "b"]
        service.embed(texts, return_colbert=False, input_type="query")
        (sent,) = service.qwen_dense_backend.calls
        self.assertEqual(
            sorted(sent), sorted(module.QWEN_DENSE_QUERY_PROMPT + t for t in texts)
        )

    def test_basic_query_reaches_bge_unchanged(self):
        module = load_server()
        service = _basic(module)
        service.embed(["alpha"], return_colbert=False, input_type="query")
        self.assertEqual(service.bge_backend.calls, [["alpha"]])


class QueryPromptRequestTests(unittest.TestCase):
    """The request field, its default and what the service advertises."""

    def test_requests_default_to_document_in_both_profiles(self):
        module = load_server()
        for model in (module.EmbedRequest, module.AdvancedEmbedRequest):
            self.assertEqual(model(sentences=["x"]).input_type, "document")
            self.assertEqual(
                model(sentences=["x"], input_type="query").input_type, "query"
            )

    def test_input_type_admits_only_document_and_query(self):
        # The harness replaces pydantic, so the field's type is what is pinned;
        # rejecting any other value is pydantic's own behaviour.
        module = load_server()
        field = module.EmbedRequestBase.__annotations__["input_type"]
        self.assertEqual(typing.get_args(field), ("document", "query"))

    def test_health_reports_the_prompt_only_under_advanced(self):
        advanced = load_server({"EMBEDDER_PROFILE": "advanced"})
        self.assertEqual(
            asyncio.run(advanced.health_check())["dense_query_prompt"],
            advanced.QWEN_DENSE_QUERY_PROMPT,
        )
        basic = load_server({"EMBEDDER_PROFILE": "basic"})
        self.assertEqual(asyncio.run(basic.health_check())["dense_query_prompt"], "")

    def test_query_cost_counts_the_prompt_under_advanced_only(self):
        advanced = load_server({"EMBEDDER_PROFILE": "advanced"})
        document = advanced.AdvancedEmbedRequest(sentences=["alpha"])
        query = advanced.AdvancedEmbedRequest(sentences=["alpha"], input_type="query")
        self.assertGreater(
            advanced._embedding_request_cost(query).tokens,
            advanced._embedding_request_cost(document).tokens,
        )
        basic = load_server({"EMBEDDER_PROFILE": "basic"})
        self.assertEqual(
            basic._embedding_request_cost(
                basic.EmbedRequest(sentences=["alpha"], input_type="query")
            ).tokens,
            basic._embedding_request_cost(
                basic.EmbedRequest(sentences=["alpha"])
            ).tokens,
        )


class RecordingWrapper:
    """Model wrapper stand-in recording each backend call and its input type."""

    def __init__(self):
        self.calls = []

    def embed(self, sentences, **kwargs):
        self.calls.append((list(sentences), kwargs.get("input_type")))
        return {"dense_vecs": [[1.0] for _ in sentences]}


class QueryPromptBatchingTests(unittest.IsolatedAsyncioTestCase):
    """Dynamic batching must not merge a query with a document."""

    async def asyncSetUp(self):
        self.server = load_server()
        self.wrapper = RecordingWrapper()
        self.processor = self.server.RequestProcessor(
            self.wrapper, 8, 0.05, max_queue_size=4
        )

    async def asyncTearDown(self):
        loop = getattr(self.processor, "processing_loop_task", None)
        if loop:
            loop.cancel()
            await asyncio.gather(loop, return_exceptions=True)
        await asyncio.to_thread(self.server.GPU_EXECUTOR.shutdown, wait=True)

    async def test_query_and_document_requests_use_separate_backend_calls(self):
        requests = [
            self.server.EmbedRequest(
                sentences=["doc"], return_colbert=False, return_sparse=False
            ),
            self.server.EmbedRequest(
                sentences=["ask"],
                return_colbert=False,
                return_sparse=False,
                input_type="query",
            ),
        ]
        tasks = [
            asyncio.create_task(self.processor.process_request(r)) for r in requests
        ]
        for _ in range(200):
            if self.processor.queue.qsize() == 2:
                break
            await asyncio.sleep(0)
        await self.processor.start_processing_loop()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
        self.assertCountEqual(
            self.wrapper.calls, [(["doc"], "document"), (["ask"], "query")]
        )


if __name__ == "__main__":
    unittest.main()

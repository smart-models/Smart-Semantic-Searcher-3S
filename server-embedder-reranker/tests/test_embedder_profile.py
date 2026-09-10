"""Tests for profile-dependent behaviour of the Smart Embedder app.

Loads the server module under both profiles with fakes in place and checks
profile resolution, the embedding contract, request limits, metrics
sampling, rate limiting, MILCO mapping and service wiring.
"""

import os
import unittest
from unittest.mock import patch

from tests.test_reranker_wrapper import load_server


class ProfileResolutionTests(unittest.TestCase):
    """Which profile an unset or explicit value resolves to."""

    def test_unset_profile_resolves_to_basic(self):
        module = load_server()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EMBEDDER_PROFILE", None)
            p = module._resolve_profile()
        self.assertEqual(p["profile"], "basic")
        self.assertEqual(p["dense"], module.BGE_EMBEDDING_MODEL)
        self.assertEqual(p["sparse"], module.BGE_EMBEDDING_MODEL)
        self.assertEqual(p["colbert"], module.BGE_EMBEDDING_MODEL)
        self.assertEqual(p["reranker"], module.BGE_RERANKER_MODEL)

    def test_unsupported_profile_fails_closed(self):
        module = load_server()
        with patch.dict(os.environ, {"EMBEDDER_PROFILE": "turbo"}):
            with self.assertRaisesRegex(ValueError, "expected one of: advanced, basic"):
                module._resolve_profile()

    def test_advanced_profile_maps_all_types(self):
        module = load_server()
        with patch.dict(os.environ, {"EMBEDDER_PROFILE": "advanced"}):
            p = module._resolve_profile()
        self.assertEqual(p["profile"], "advanced")
        self.assertEqual(p["dense"], module.QWEN_DENSE_EMBEDDING_MODEL)
        self.assertEqual(p["sparse"], module.MILCO_SPARSE_MODEL)
        self.assertIsNone(p["colbert"])
        self.assertEqual(p["reranker"], module.QWEN_RERANKER_MODEL)


class EmbeddingContractTests(unittest.TestCase):
    """The contract fingerprint is deterministic and profile-sensitive."""

    def test_contract_fingerprint_is_deterministic_and_profile_sensitive(self):
        module = load_server()
        with patch.object(module, "_model_revision", return_value="revision-1"):
            first = module.build_embedding_contract(
                "basic", module.BGE_EMBEDDING_MODEL, module.BGE_EMBEDDING_MODEL
            )
            second = module.build_embedding_contract(
                "basic", module.BGE_EMBEDDING_MODEL, module.BGE_EMBEDDING_MODEL
            )
            advanced = module.build_embedding_contract(
                "advanced",
                module.QWEN_DENSE_EMBEDDING_MODEL,
                module.MILCO_SPARSE_MODEL,
            )

        self.assertEqual(first, second)
        self.assertNotEqual(first["fingerprint"], advanced["fingerprint"])
        self.assertEqual(first["dense"]["dimension"], 1024)
        self.assertEqual(first["dense"]["revision"], "revision-1")
        self.assertTrue(first["fingerprint"].startswith("sha256:"))


class RequestSizeLimitTests(unittest.TestCase):
    """Oversized embedding and rerank requests are rejected up front."""

    def test_embeddings_rejects_too_many_sentences(self):
        import asyncio

        module = load_server()

        async def fake_pr(request, *, lease=None):
            return {"dense_vecs": [[1.0]] * len(request.sentences)}

        module.processor.process_request = fake_pr
        request = module.EmbedRequest(sentences=["x"] * 5000)
        with self.assertRaises(module.HTTPException) as ctx:
            asyncio.run(module.get_embeddings(request))
        self.assertEqual(ctx.exception.status_code, 413)

    def test_embeddings_rejects_oversized_item(self):
        import asyncio

        module = load_server()

        async def fake_pr(request, *, lease=None):
            return {"dense_vecs": [[1.0]] * len(request.sentences)}

        module.processor.process_request = fake_pr
        request = module.EmbedRequest(sentences=["x" * 2_000_000])
        with self.assertRaises(module.HTTPException) as ctx:
            asyncio.run(module.get_embeddings(request))
        self.assertEqual(ctx.exception.status_code, 413)

    def test_rerank_rejects_too_many_passages(self):
        import asyncio

        module = load_server()
        request = module.RerankRequest(query="q", passages=["p"] * 5000)
        with self.assertRaises(module.HTTPException) as ctx:
            asyncio.run(module.rerank(request))
        self.assertEqual(ctx.exception.status_code, 413)


class GpuMetricsSamplingTests(unittest.TestCase):
    """GPU metric sampling is throttled rather than run per request."""

    def test_gpu_metrics_sampling_is_throttled(self):
        from unittest.mock import patch

        module = load_server()
        calls = {"n": 0}

        def counting_alloc():
            calls["n"] += 1
            return 100

        module.has_cuda = True
        module.torch.cuda.memory_allocated = counting_alloc

        clock = {"t": 5000.0}
        with patch("time.monotonic", lambda: clock["t"]):
            module.record_gpu_memory_metrics()  # first sample
            module.record_gpu_memory_metrics()  # within window -> throttled
            self.assertEqual(calls["n"], 1)
            clock["t"] = 5002.0  # past the sampling interval
            module.record_gpu_memory_metrics()  # samples again
            self.assertEqual(calls["n"], 2)


class RateLimiterClockTests(unittest.TestCase):
    """Token refill follows the monotonic clock, not wall time."""

    def test_token_refill_follows_monotonic_clock(self):
        import asyncio
        from unittest.mock import patch

        module = load_server()
        clock = {"t": 1000.0}
        with patch("time.monotonic", lambda: clock["t"]):
            limiter = module.RateLimiter(requests_per_minute=60, burst_size=1)
            # First request consumes the only token.
            self.assertTrue(asyncio.run(limiter.check_rate_limit("ip")))
            # No monotonic time has passed: bucket is empty, request denied.
            self.assertFalse(asyncio.run(limiter.check_rate_limit("ip")))
            # Advance the monotonic clock 1.5s: 60 rpm refills 1 token (capped
            # at burst), so the next request is allowed again.
            clock["t"] = 1001.5
            self.assertTrue(asyncio.run(limiter.check_rate_limit("ip")))


class MilcoMappingTests(unittest.TestCase):
    """MILCO rows map to Qdrant-compatible sparse lexical weights."""

    def test_rows_to_lexical_weights_groups_by_row_and_keeps_positive(self):
        module = load_server()
        rows = [0, 0, 1]
        cols = [10, 20, 30]
        vals = [0.5, -0.1, 0.9]  # negative weight must be dropped
        out = module._milco_rows_to_lexical_weights(rows, cols, vals, n=2, top_k=0)
        self.assertEqual(out, [{"10": 0.5}, {"30": 0.9}])

    def test_rows_to_lexical_weights_top_k_keeps_highest(self):
        module = load_server()
        rows = [0, 0, 0]
        cols = [1, 2, 3]
        vals = [0.2, 0.9, 0.5]
        out = module._milco_rows_to_lexical_weights(rows, cols, vals, n=1, top_k=2)
        self.assertEqual(out, [{"2": 0.9, "3": 0.5}])

    def test_rows_to_lexical_weights_handles_interleaved_coo_order(self):
        module = load_server()
        # COO entries are not grouped by row; the mapping must still bucket each
        # entry into its row regardless of input ordering.
        rows = [1, 0, 1, 0]
        cols = [30, 10, 31, 11]
        vals = [0.3, 0.1, 0.7, 0.2]
        out = module._milco_rows_to_lexical_weights(rows, cols, vals, n=2, top_k=0)
        self.assertEqual(out, [{"10": 0.1, "11": 0.2}, {"30": 0.3, "31": 0.7}])

    def test_keys_are_strings_and_values_floats_for_qdrant(self):
        module = load_server()
        out = module._milco_rows_to_lexical_weights([0], [7], [1.0], n=1, top_k=0)
        (key,), (val,) = out[0].keys(), out[0].values()
        self.assertIsInstance(key, str)
        self.assertIsInstance(val, float)

    def test_lexecho_source_vocabulary_indices_remain_qdrant_compatible(self):
        module = load_server()
        english_vocab_size = 30522
        source_index = english_vocab_size + 4242

        out = module._milco_rows_to_lexical_weights(
            [0, 0],
            [17, source_index],
            [0.8, 1.1],
            n=1,
            top_k=0,
        )

        self.assertEqual(out, [{"17": 0.8, str(source_index): 1.1}])

    def test_milco_source_view_defaults_on_and_can_be_disabled(self):
        default_module = load_server()
        disabled_module = load_server({"MILCO_SOURCE_VIEW": "false"})

        self.assertTrue(default_module.MILCO_SOURCE_VIEW)
        self.assertFalse(disabled_module.MILCO_SOURCE_VIEW)


class TrackingMilcoBackend:
    """MILCO backend stand-in recording the calls it receives."""

    def __init__(self):
        self.calls = []

    def embed_sparse(self, sentences, deadline=None):
        self.calls.append({"sentences": sentences})
        return {"lexical_weights": [{"7": 0.9} for _ in sentences]}


def _advanced_service(module):
    service = module.EmbeddingService.__new__(module.EmbeddingService)
    service.profile = module.EMBEDDER_PROFILE_ADVANCED
    service.bge_backend = None
    service.qwen_dense_backend = None
    service.milco_sparse_backend = TrackingMilcoBackend()
    service.bge_model_name = None
    service.dense_model_name = module.QWEN_DENSE_EMBEDDING_MODEL
    service.sparse_model_name = module.MILCO_SPARSE_MODEL
    service.colbert_model_name = None
    return service


class ServiceRoutingTests(unittest.TestCase):
    """Each profile loads and routes to its own backends only."""

    def test_advanced_sparse_request_uses_milco_not_bge(self):
        module = load_server()
        service = _advanced_service(module)
        result = service.embed(
            ["alpha"],
            return_dense=False,
            return_sparse=True,
            return_colbert=True,
            normalize_dense=False,
        )
        self.assertEqual(result["lexical_weights"], [{"7": 0.9}])
        self.assertNotIn("colbert_vecs", result)  # no BGE => no colbert
        self.assertEqual(service.milco_sparse_backend.calls, [{"sentences": ["alpha"]}])

    def test_advanced_profile_loads_qwen_and_milco_not_bge(self):
        module = load_server({"EMBEDDER_PROFILE": "advanced"})
        service = module.embedding_service
        self.assertIsNone(service.bge_backend)
        self.assertIsNotNone(service.qwen_dense_backend)
        self.assertIsNotNone(service.milco_sparse_backend)
        self.assertEqual(service.profile, "advanced")
        self.assertIsNone(service.colbert_model_name)

    def test_basic_profile_loads_bge_only(self):
        module = load_server({"EMBEDDER_PROFILE": "basic"})
        service = module.embedding_service
        self.assertIsNotNone(service.bge_backend)
        self.assertIsNone(service.qwen_dense_backend)
        self.assertIsNone(service.milco_sparse_backend)
        self.assertEqual(service.sparse_model_name, module.BGE_EMBEDDING_MODEL)


class WiringTests(unittest.TestCase):
    """Profile-dependent request models, endpoints and public routes."""

    def test_advanced_endpoint_request_model_omits_colbert_option(self):
        module = load_server({"EMBEDDER_PROFILE": "advanced"})
        request_model = module.get_embeddings.__annotations__["request"]
        self.assertIs(request_model, module.AdvancedEmbedRequest)
        self.assertNotIn("return_colbert", request_model.__annotations__)
        self.assertEqual(request_model.model_config["extra"], "forbid")

    def test_basic_endpoint_request_model_keeps_colbert_option(self):
        module = load_server({"EMBEDDER_PROFILE": "basic"})
        request_model = module.get_embeddings.__annotations__["request"]
        self.assertIs(request_model, module.EmbedRequest)
        self.assertIn("return_colbert", request_model.__annotations__)

    def test_active_tokenizers_skip_bge_in_advanced(self):
        module = load_server({"EMBEDDER_PROFILE": "advanced"})
        req = module.EmbedRequest(
            sentences=["x"],
            return_dense=False,
            return_sparse=True,
            return_colbert=True,
            normalize_dense=False,
            sparse_as_indices=False,
        )
        # Must not raise AttributeError on a None BGE backend, and must not
        # include any BGE tokenizer entry.
        tokenizers = module._active_embedding_tokenizers(req)
        names = [name for _, name, _ in tokenizers]
        self.assertNotIn(module.BGE_EMBEDDING_MODEL, names)

    def test_batch_tuning_dense_source_is_profile_resolved(self):
        module = load_server({"EMBEDDER_PROFILE": "advanced"})
        self.assertEqual(
            module.ACTIVE_PROFILE["dense"], module.QWEN_DENSE_EMBEDDING_MODEL
        )

    def test_colbert_model_name_accepts_none(self):
        module = load_server({"EMBEDDER_PROFILE": "advanced"})
        resp = module.EmbeddingsListResponse(
            data=[],
            model_name=module.QWEN_DENSE_EMBEDDING_MODEL,
            dense_model_name=module.QWEN_DENSE_EMBEDDING_MODEL,
            sparse_model_name=module.MILCO_SPARSE_MODEL,
            colbert_model_name=None,
            processing_time_ms=1.0,
        )
        self.assertIsNone(resp.colbert_model_name)

    def test_colbert_model_name_annotation_is_optional(self):
        import typing

        module = load_server()
        ann = module.EmbeddingsListResponse.__annotations__["colbert_model_name"]
        # Catches a regression to a bare `str` annotation even though the fake
        # FakeBaseModel does not enforce types at runtime.
        self.assertIn(type(None), typing.get_args(ann))

    def test_server_info_includes_profile_and_reranker(self):
        module = load_server({"EMBEDDER_PROFILE": "advanced"})
        info = module.server_info.last_info
        self.assertEqual(info["profile"], "advanced")
        self.assertEqual(info["reranker_model"], module.QWEN_RERANKER_MODEL)
        self.assertEqual(info["sparse_model_name"], module.MILCO_SPARSE_MODEL)

    def test_models_endpoint_reports_basic_profile_models(self):
        import asyncio

        module = load_server({"EMBEDDER_PROFILE": "basic"})

        response = asyncio.run(module.get_models())

        self.assertEqual(response.profile, "basic")
        self.assertEqual(response.embedder.dense, module.BGE_EMBEDDING_MODEL)
        self.assertEqual(response.embedder.sparse, module.BGE_EMBEDDING_MODEL)
        self.assertEqual(response.embedder.colbert, module.BGE_EMBEDDING_MODEL)
        self.assertEqual(response.reranker, module.BGE_RERANKER_MODEL)
        self.assertEqual(response.version, module.APP_VERSION)
        self.assertEqual(response.embedding_contract["profile"], "basic")
        self.assertEqual(
            response.embedding_contract["dense"]["dimension"],
            module.BGE_DENSE_VECTOR_SIZE,
        )

    def test_models_endpoint_reports_advanced_profile_models(self):
        import asyncio

        module = load_server({"EMBEDDER_PROFILE": "advanced"})

        response = asyncio.run(module.get_models())

        self.assertEqual(response.profile, "advanced")
        self.assertEqual(response.embedder.dense, module.QWEN_DENSE_EMBEDDING_MODEL)
        self.assertEqual(response.embedder.sparse, module.MILCO_SPARSE_MODEL)
        self.assertIsNone(response.embedder.colbert)
        self.assertEqual(response.reranker, module.QWEN_RERANKER_MODEL)

    def test_only_health_is_public_when_auth_is_configured(self):
        module = load_server({"API_TOKENS": "secret:test"})

        self.assertTrue(module.is_public_auth_path("/health"))
        self.assertFalse(module.is_public_auth_path("/models"))
        self.assertFalse(module.is_public_auth_path("/stats"))
        self.assertFalse(module.is_public_auth_path("/metrics"))

    def test_advanced_endpoint_omits_colbert_and_reports_null(self):
        import asyncio

        module = load_server({"EMBEDDER_PROFILE": "advanced"})

        async def fake_process_request(request, *, lease=None):
            return {"dense_vecs": [[1.0]], "lexical_weights": [{"7": 0.9}]}

        module.processor.process_request = fake_process_request
        request = module.EmbedRequest(
            sentences=["alpha"],
            return_dense=True,
            return_sparse=True,
            return_colbert=True,
            normalize_dense=False,
            sparse_as_indices=False,
        )
        response = asyncio.run(module.get_embeddings(request))
        self.assertIsNone(response.colbert_model_name)
        self.assertEqual(response.sparse_model_name, module.MILCO_SPARSE_MODEL)
        self.assertEqual(
            response.embedding_contract_fingerprint,
            module.build_embedding_contract(
                "advanced",
                module.QWEN_DENSE_EMBEDDING_MODEL,
                module.MILCO_SPARSE_MODEL,
            )["fingerprint"],
        )
        # colbert is never set on the item (no BGE backend in advanced).
        self.assertIsNone(getattr(response.data[0].embeddings, "colbert", None))

    def test_advanced_profile_inherits_include_text(self):
        module = load_server({"EMBEDDER_PROFILE": "advanced"})
        request = module.AdvancedEmbedRequest(
            sentences=["a"],
            return_dense=True,
            return_sparse=True,
            include_text=False,
        )
        self.assertFalse(request.include_text)

    def test_advanced_endpoint_include_text_false_sets_null(self):
        import asyncio

        module = load_server({"EMBEDDER_PROFILE": "advanced"})

        async def fake_process_request(request, *, lease=None):
            return {"dense_vecs": [[1.0]], "lexical_weights": [{"7": 0.9}]}

        module.processor.process_request = fake_process_request
        request = module.AdvancedEmbedRequest(
            sentences=["alpha"],
            return_dense=True,
            return_sparse=True,
            include_text=False,
        )
        response = asyncio.run(module.get_embeddings(request))
        self.assertIsNone(response.data[0].text)


if __name__ == "__main__":
    unittest.main()

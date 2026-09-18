"""Tests for the standalone ``benchmark.py`` load generator.

Covers failure reporting, input-token accounting, token resolution and the
Prometheus diagnostics it collects, all against a faked HTTP transport.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmark
import httpx


class TestBenchmarkFailureDetails(unittest.TestCase):
    """Failures are grouped by status and message in the summary."""

    def test_render_failure_details_groups_statuses_and_messages(self):
        result = benchmark.ScenarioResult(
            name="embed_dense",
            samples=[
                benchmark.Sample(
                    latency_ms=10.0, ok=False, status=504, error="timeout"
                ),
                benchmark.Sample(
                    latency_ms=11.0, ok=False, status=503, error="overload"
                ),
                benchmark.Sample(
                    latency_ms=12.0, ok=False, status=503, error="overload"
                ),
            ],
        )

        rendered = benchmark.render_failure_details([result])

        self.assertIn("Failure details", rendered)
        self.assertIn("embed_dense: 504=1, 503=2", rendered)
        self.assertIn("- overload: 2", rendered)
        self.assertIn("- timeout: 1", rendered)


class FakeTokenizer:
    """Tokenizer stand-in returning a fixed token count."""

    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        max_length = kwargs["max_length"]
        return {
            "input_ids": [
                list(range(min(len(text.split()) + 2, max_length))) for text in texts
            ]
        }


class WordTokenizer:
    """Whitespace tokenizer stand-in, so counts do not depend on a vocabulary."""

    def __call__(self, texts, **kwargs):
        max_length = kwargs["max_length"]
        return {
            "input_ids": [
                list(range(min(len(text.split()), max_length))) for text in texts
            ]
        }

    def encode(self, text, add_special_tokens=False, verbose=False):
        _ = add_special_tokens, verbose
        return list(range(len(text.split())))


class TestBenchmarkInputTokens(unittest.IsolatedAsyncioTestCase):
    """Input-token accounting across every benchmark scenario."""

    def test_counts_special_tokens_after_truncation(self):
        tokenizer = FakeTokenizer()
        counter = benchmark.EmbedTokenCounter(
            tokenizer=tokenizer,
            model_name="test/model",
            max_length=4,
        )

        total = counter.count(["one two", "one two three four five"])

        self.assertEqual(total, 8)
        _, kwargs = tokenizer.calls[0]
        self.assertEqual(kwargs["max_length"], 4)
        self.assertTrue(kwargs["truncation"])
        self.assertTrue(kwargs["add_special_tokens"])
        self.assertFalse(kwargs["padding"])

    def test_builds_exact_controlled_token_workload(self):
        counter = benchmark.EmbedTokenCounter(
            tokenizer=FakeTokenizer(),
            model_name="test/model",
            max_length=16,
        )

        texts = counter.texts_at_token_length(size=4, target_tokens=8)
        scenario = benchmark.build_scenarios(
            batch_size=4,
            timeout=1.0,
            include_colbert=False,
            token_counter=counter,
            target_tokens=8,
        )["embed_dense"]

        self.assertEqual(len(texts), 4)
        self.assertTrue(all(counter.count([text]) == 8 for text in texts))
        self.assertEqual(scenario[2], 32)

    def test_builds_mixed_exact_token_workload(self):
        counter = benchmark.EmbedTokenCounter(
            tokenizer=FakeTokenizer(),
            model_name="test/model",
            max_length=16,
        )

        texts = counter.texts_at_token_pattern(size=5, target_tokens=[4, 8])
        scenario = benchmark.build_scenarios(
            batch_size=5,
            timeout=1.0,
            include_colbert=False,
            token_counter=counter,
            mixed_target_tokens=[4, 8],
        )["embed_dense"]

        self.assertEqual([counter.count([text]) for text in texts], [4, 8, 4, 8, 4])
        self.assertEqual(scenario[2], 28)

    async def test_run_scenario_tracks_only_successful_input_tokens(self):
        samples = iter(
            [
                benchmark.Sample(latency_ms=1.0, ok=True, status=200),
                benchmark.Sample(latency_ms=1.0, ok=False, status=503),
                benchmark.Sample(latency_ms=1.0, ok=True, status=200),
            ]
        )

        async def request(_client, _base_url):
            return next(samples)

        async with httpx.AsyncClient() as client:
            result = await benchmark.run_scenario(
                client=client,
                base_url="http://test",
                name="embed_dense",
                fn=request,
                units_per_req=4,
                input_tokens_per_req=30,
                requests=3,
                concurrency=1,
                warmup=0,
            )

        self.assertEqual(result.total_units, 8)
        self.assertEqual(result.total_input_tokens, 60)
        self.assertIn("Tok/s", benchmark.summarize([result], concurrency=1))

    async def test_hybrid_scenario_disables_colbert(self):
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            self.assertTrue(payload["return_dense"])
            self.assertTrue(payload["return_sparse"])
            self.assertFalse(payload["return_colbert"])
            return httpx.Response(200, json={"processing_time_ms": 1.0})

        scenario, _, _ = benchmark.build_scenarios(
            batch_size=1,
            timeout=1.0,
            include_colbert=True,
        )["embed_hybrid"]
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            sample = await scenario(client, "http://test")

        self.assertTrue(sample.ok)

    async def test_rerank_controlled_workload_counts_useful_query_and_passage_tokens(
        self,
    ):
        counter = benchmark.EmbedTokenCounter(
            tokenizer=FakeTokenizer(),
            model_name="test/reranker",
            max_length=32,
        )

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            query_tokens = counter.count([payload["query"]])
            pair_tokens = [
                query_tokens + counter.count([passage])
                for passage in payload["passages"]
            ]
            self.assertEqual(pair_tokens, [12, 12])
            return httpx.Response(200, json={"processing_time_ms": 1.0})

        scenario, units, useful_tokens = benchmark.build_scenarios(
            batch_size=2,
            timeout=1.0,
            include_colbert=False,
            token_counter=counter,
            target_tokens=12,
            rerank_token_counter=counter,
        )["rerank"]
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            sample = await scenario(client, "http://test")

        self.assertTrue(sample.ok)
        self.assertEqual(units, 2)
        self.assertEqual(useful_tokens, 24)

    async def test_qwen_rerank_workload_counts_exact_full_prompt_sequence(self):
        base = benchmark.EmbedTokenCounter(
            tokenizer=WordTokenizer(),
            model_name="test/qwen-reranker",
            max_length=64,
        )
        counter = benchmark.RerankTokenCounter(
            base=base,
            instruction="retrieve relevant passages",
            prefix_tokens=5,
            suffix_tokens=3,
        )

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            self.assertEqual(
                [
                    counter.pair_count(payload["query"], passage)
                    for passage in payload["passages"]
                ],
                [32, 32],
            )
            return httpx.Response(200, json={"processing_time_ms": 1.0})

        scenario, units, useful_tokens = benchmark.build_scenarios(
            batch_size=2,
            timeout=1.0,
            include_colbert=False,
            token_counter=base,
            target_tokens=32,
            rerank_token_counter=counter,
        )["rerank"]
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            sample = await scenario(client, "http://test")

        self.assertTrue(sample.ok)
        self.assertEqual(units, 2)
        self.assertEqual(useful_tokens, 64)

    async def test_mixed_scenario_counts_both_http_requests(self):
        async def embed(_client, _base_url):
            return benchmark.Sample(latency_ms=4.0, ok=True, status=200)

        async def rerank(_client, _base_url):
            return benchmark.Sample(latency_ms=6.0, ok=True, status=200)

        mixed = benchmark.make_mixed_request(embed, rerank)
        async with httpx.AsyncClient() as client:
            result = await benchmark.run_scenario(
                client=client,
                base_url="http://test",
                name="mixed",
                fn=mixed,
                units_per_req=4,
                input_tokens_per_req=20,
                requests=2,
                concurrency=1,
                warmup=0,
            )

        self.assertEqual(result.requests, 4)
        self.assertEqual(result.success, 4)
        self.assertEqual(result.total_units, 8)
        self.assertEqual(result.total_input_tokens, 40)
        self.assertIn("4 measured requests", benchmark.render_verdict([result]))

    def test_resolves_profile_specific_max_length(self):
        self.assertEqual(
            benchmark.active_embedding_max_length(
                {"profile": "basic", "bge_embed_max_length": 8192}
            ),
            8192,
        )
        self.assertEqual(
            benchmark.active_embedding_max_length(
                {"profile": "advanced", "qwen_embed_max_length": 32768}
            ),
            32768,
        )
        self.assertEqual(
            benchmark.active_reranker_max_length(
                {
                    "reranker_model": "Qwen/Qwen3-Reranker-0.6B",
                    "qwen_rerank_max_length": 8192,
                }
            ),
            8192,
        )


class TestBenchmarkTokenResolution(unittest.TestCase):
    """Where the API token is read from, and which source wins."""

    def test_reads_token_from_root_env_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "# shared config\n"
                "SMART_EMBEDDER_API_TOKENS='root-token:bench' # comment\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                token = benchmark.default_api_token(env_path)

        self.assertEqual(token, "root-token")

    def test_process_environment_takes_precedence_over_root_env(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "SMART_EMBEDDER_API_TOKENS=root-token:bench\n", encoding="utf-8"
            )

            with patch.dict(
                os.environ, {benchmark.API_TOKEN_ENV: "process-token:bench"}, clear=True
            ):
                token = benchmark.default_api_token(env_path)

        self.assertEqual(token, "process-token")


class TestBenchmarkPrometheusDiagnostics(unittest.TestCase):
    """Padding and throughput derived from the scraped metrics."""

    def test_derives_padding_and_worker_token_throughput(self):
        pre = """
embedding_input_tokens_total{backend="bge"} 100
embedding_padded_tokens_total{backend="bge"} 125
embedding_tokenized_batches_total{backend="bge"} 2
api_model_forward_duration_seconds_count{endpoint="/embeddings",operation="embed"} 1
api_model_forward_duration_seconds_sum{endpoint="/embeddings",operation="embed"} 1
"""
        post = """
embedding_input_tokens_total{backend="bge"} 500
embedding_padded_tokens_total{backend="bge"} 625
embedding_tokenized_batches_total{backend="bge"} 6
api_model_forward_duration_seconds_count{endpoint="/embeddings",operation="embed"} 5
api_model_forward_duration_seconds_sum{endpoint="/embeddings",operation="embed"} 3
"""

        rendered = benchmark.render_metric_delta(pre, post)

        self.assertIn(
            "embedding_padding_efficiency ratio=0.8000 useful=400 padded=500",
            rendered,
        )
        self.assertIn(
            "embedding_worker_throughput useful_tok/s=200 padded_tok/s=250 "
            "worker_seconds=2.000000",
            rendered,
        )
        self.assertIn(
            "embedding_tokenized_batch_averages useful=100.0 padded=125.0 batches=4",
            rendered,
        )

    def test_reports_advanced_backend_padding_and_forward_throughput(self):
        pre = """
advanced_backend_input_tokens_total{backend="milco_sparse"} 100
advanced_backend_padded_tokens_total{backend="milco_sparse"} 128
advanced_backend_phase_duration_seconds_sum{backend="milco_sparse",phase="forward"} 1
"""
        post = """
advanced_backend_input_tokens_total{backend="milco_sparse"} 500
advanced_backend_padded_tokens_total{backend="milco_sparse"} 640
advanced_backend_phase_duration_seconds_sum{backend="milco_sparse",phase="forward"} 3
"""

        rendered = benchmark.render_metric_delta(pre, post)

        self.assertIn(
            "advanced_backend backend=milco_sparse padding_efficiency=0.7812 "
            "useful=400 padded=512 forward_useful_tok/s=200",
            "\n".join(rendered),
        )


class TestBenchmarkJsonReport(unittest.TestCase):
    """The JSON report is written atomically and stays comparable."""

    def test_writes_comparable_report_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "reports" / "baseline.json"
            benchmark.write_json_report(
                path,
                {"schema_version": 1, "label": "baseline", "runs": []},
            )

            payload = json.loads(path.read_text(encoding="utf-8"))
            temporary_files = list(path.parent.glob(".*.tmp"))

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["label"], "baseline")
        self.assertEqual(temporary_files, [])

    def test_failed_replace_preserves_report_and_removes_temporary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "baseline.json"
            path.write_bytes(b"previous report")
            with patch.object(benchmark.os, "replace", side_effect=OSError("disk")):
                with self.assertRaisesRegex(OSError, "disk"):
                    benchmark.write_json_report(path, {"runs": []})
            self.assertEqual(path.read_bytes(), b"previous report")
            self.assertEqual(list(path.parent.iterdir()), [path])


class TestBenchmarkAuthentication(unittest.IsolatedAsyncioTestCase):
    """A missing or rejected token is reported before the run starts."""

    async def test_reports_missing_token_before_benchmark(self):
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.method, "HEAD")
            return httpx.Response(
                401,
                json={"detail": "Invalid or missing bearer token."},
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            error = await benchmark.authentication_error(client, "http://test", "")

        self.assertIn("requires a bearer token", error)
        self.assertIn("root .env", error)

    async def test_accepts_authorized_method_not_allowed_response(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(405))
        async with httpx.AsyncClient(transport=transport) as client:
            error = await benchmark.authentication_error(
                client, "http://test", "valid-token"
            )

        self.assertEqual(error, "")


if __name__ == "__main__":
    unittest.main()

"""Tests for benchmark.py: helpers, scenario factories, and setup/teardown flow."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
from pathlib import Path

import httpx
import pytest

BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "benchmark.py"
SPEC = importlib.util.spec_from_file_location("qdrant_benchmark", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark

# benchmark.py calls load_dotenv() at import time so a human running it picks up
# the root .env. Importing it here would push that whole file -- deployment URLs
# and real API keys included -- into os.environ for the rest of the pytest
# process, where every later Settings() would read it as if it were the default.
# The import is what these tests need; the environment it leaves behind is not.
_ENVIRONMENT_BEFORE_IMPORT = dict(os.environ)
try:
    SPEC.loader.exec_module(benchmark)
finally:
    os.environ.clear()
    os.environ.update(_ENVIRONMENT_BEFORE_IMPORT)


def run(coro):
    return asyncio.run(coro)


class RecordingClient:
    """An httpx-shaped stand-in recording every request the benchmark makes."""

    def __init__(
        self,
        status_code: int = 200,
        payload: dict | None = None,
        text: str = "",
    ):
        self.status_code = status_code
        self.payload = payload
        self.text = text
        self.calls = []

    def _response(self, method: str, url: str) -> httpx.Response:
        request = httpx.Request(method, url)
        if self.payload is not None:
            return httpx.Response(self.status_code, json=self.payload, request=request)
        return httpx.Response(self.status_code, text=self.text, request=request)

    async def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self._response("GET", url)

    async def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self._response("POST", url)

    async def patch(self, url, **kwargs):
        self.calls.append(("PATCH", url, kwargs))
        return self._response("PATCH", url)

    async def put(self, url, **kwargs):
        self.calls.append(("PUT", url, kwargs))
        return self._response("PUT", url)

    async def delete(self, url, **kwargs):
        self.calls.append(("DELETE", url, kwargs))
        return self._response("DELETE", url)

    async def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self._response(method, url)


def test_percentile_uses_sorted_values_and_clamps_bounds():
    assert benchmark.percentile([], 95) == 0.0
    assert benchmark.percentile([30, 10, 20], -10) == 10
    assert benchmark.percentile([30, 10, 20], 0) == 10
    assert benchmark.percentile([30, 10, 20], 50) == 20
    assert benchmark.percentile([30, 10, 20], 100) == 30
    assert benchmark.percentile([30, 10, 20], 200) == 30


def test_formatting_status_and_table_helpers_are_stable():
    ok = benchmark.Sample(latency_ms=12.5, ok=True, status=200)
    failed_http = benchmark.Sample(
        latency_ms=2.0,
        ok=False,
        status=502,
        error="bad gateway",
    )
    failed_exc = benchmark.Sample(
        latency_ms=1.0,
        ok=False,
        status=0,
        error="TimeoutException",
    )
    result = benchmark.ScenarioResult(
        "mixed",
        [ok, failed_http, failed_exc],
        wall_time_s=1.0,
    )

    assert benchmark.fmt_ms(999.94) == "999.9"
    assert benchmark.fmt_ms(1200) == "1.20s"
    assert benchmark.fmt_int(1234567) == "1 234 567"
    assert benchmark.section("Title") == "Title\n-----"
    assert benchmark.sample_status_label(failed_http) == "502"
    assert benchmark.sample_status_label(failed_exc) == "EXC:TimeoutException"
    assert benchmark.status_counts(result) == {
        "200": 1,
        "502": 1,
        "EXC:TimeoutException": 1,
    }
    assert benchmark.format_status_counts(result) == (
        "200=1, 502=1, EXC:TimeoutException=1"
    )

    table = benchmark.render_table([["a", "bb"], ["ccc", "d"]], ["H1", "H2"])
    assert "| H1  | H2 |" in table
    assert "| ccc | d  |" in table


def test_summary_verdict_and_failure_details_report_benchmark_outcome():
    result = benchmark.ScenarioResult(
        "query_rrf_rerank",
        [
            benchmark.Sample(10.0, True, 200),
            benchmark.Sample(20.0, False, 429, "rate limited"),
        ],
        wall_time_s=0.5,
    )

    summary = benchmark.summarize([result], concurrency=2)
    assert "query_rrf_rerank" in summary
    assert "POST /query/search" in summary
    assert "FAIL" in summary
    assert "429=1" in summary

    verdict = benchmark.render_verdict([result])
    assert verdict.startswith("Verdict: FAIL")
    assert "1/2 requests failed" in verdict
    assert "query_rrf_rerank" in verdict

    details = benchmark.render_failure_details([result])
    assert "Failure details" in details
    assert "query_rrf_rerank: 429=1" in details
    assert "rate limited: 1" in details


def test_prometheus_parsing_and_metric_delta_ignore_unrelated_lines():
    pre = "\n".join(
        [
            "# HELP qdrant_manager_requests_total Total",
            'qdrant_manager_requests_total{path="/query"} 2',
            "unrelated_metric 99",
            "qdrant_manager_vectorstore_up 0",
        ]
    )
    post = "\n".join(
        [
            'qdrant_manager_requests_total{path="/query"} 5',
            "qdrant_manager_vectorstore_up 1",
            'qdrant_collection_operations_total{op="create"} 1',
        ]
    )

    assert benchmark.parse_prometheus_samples(pre) == {
        'qdrant_manager_requests_total{path="/query"}': 2.0,
        "qdrant_manager_vectorstore_up": 0.0,
    }
    assert benchmark.render_metric_delta(pre, post) == [
        'qdrant_collection_operations_total{op="create"} delta=1 current=1',
        'qdrant_manager_requests_total{path="/query"} delta=3 current=5',
        "qdrant_manager_vectorstore_up current=1",
    ]


def test_validate_args_reports_each_invalid_numeric_bound():
    args = argparse.Namespace(
        concurrency=0,
        requests=0,
        warmup=-1,
        timeout=0,
        sleep_between=-0.1,
    )

    assert benchmark.validate_args(args) == [
        "--concurrency must be >= 1",
        "--requests must be >= 1",
        "--warmup must be >= 0",
        "--timeout must be > 0",
        "--sleep-between must be >= 0",
    ]


def test_resolve_scenarios_expands_groups_all_and_deduplicates():
    resolved = benchmark.resolve_scenarios(["monitoring", "livez", "query"])

    assert resolved[:5] == benchmark.SCENARIO_GROUPS["monitoring"]
    assert resolved.count("livez") == 1
    assert resolved[-len(benchmark.QUERY_SCENARIOS) :] == benchmark.QUERY_SCENARIOS
    assert benchmark.resolve_scenarios(["all"]) == benchmark.ALL_SCENARIOS


def test_resolve_scenarios_rejects_unknown_token_with_context():
    with pytest.raises(ValueError) as exc:
        benchmark.resolve_scenarios(["missing"])

    message = str(exc.value)
    assert "Unknown scenario/group: 'missing'" in message
    assert "Groups:" in message
    assert "Scenarios:" in message


def test_build_scenarios_contains_expected_groups_and_payloads():
    scenarios = benchmark.build_scenarios("bench-collection", timeout=7.0)

    assert set(benchmark.ALL_SCENARIOS) <= set(scenarios)

    create_client = RecordingClient()
    create_sample = run(
        scenarios["collections_create"](create_client, "http://middleware")
    )

    assert create_sample.ok is True
    assert create_client.calls == [
        (
            "POST",
            "http://middleware/collections",
            {
                "json": {
                    "collection_name": "bench-collection",
                    "recreate": False,
                    "dense_on_disk": True,
                    "quantization": True,
                    "sparse_idf": True,
                    "sparse_on_disk": True,
                },
                "timeout": 7.0,
            },
        )
    ]

    client = RecordingClient(status_code=422, payload={"detail": "validation failed"})
    sample = run(scenarios["query_err_candidate_lt_limit"](client, "http://middleware"))

    assert sample.ok is True
    assert client.calls == [
        (
            "POST",
            "http://middleware/query/search",
            {
                "json": {
                    "query": benchmark.QUERY_RRF,
                    "collection_name": "bench-collection",
                    "limit": 10,
                    "candidate_limit": 5,
                },
                "timeout": 7.0,
            },
        )
    ]


def test_representative_scenarios_use_expected_methods_params_and_timeouts():
    scenarios = benchmark.build_scenarios("bench-collection", timeout=3.0)

    get_client = RecordingClient()
    get_sample = run(scenarios["get_point"](get_client, "http://middleware"))
    assert get_sample.ok is True
    assert get_client.calls == [
        (
            "GET",
            f"http://middleware/points/{benchmark.BENCH_POINT_ID}",
            {"params": {"collection_name": "bench-collection"}, "timeout": 3.0},
        )
    ]

    ingest_client = RecordingClient()
    ingest_sample = run(scenarios["ingest_file"](ingest_client, "http://middleware"))
    assert ingest_sample.ok is True
    assert ingest_client.calls == [
        (
            "POST",
            "http://middleware/points/ingest",
            {
                "json": {
                    "file_path": benchmark.BENCH_FILE_REPEAT,
                    "collection_name": "bench-collection",
                    "source_id": benchmark.BENCH_SOURCE_FILE,
                    "replace_source": True,
                },
                "timeout": 12.0,
            },
        )
    ]


def test_http_helpers_capture_unexpected_json_detail_text_and_exceptions():
    json_client = RecordingClient(
        status_code=503,
        payload={"detail": "service unavailable"},
    )
    json_sample = run(
        benchmark.make_get("/health", timeout=3.0)(json_client, "http://middleware")
    )
    assert json_sample.ok is False
    assert json_sample.status == 503
    assert json_sample.error == "service unavailable"

    text_client = RecordingClient(status_code=500, text="plain failure")
    text_sample = run(
        benchmark.make_post("/points", {}, timeout=3.0)(
            text_client,
            "http://middleware",
        )
    )
    assert text_sample.ok is False
    assert text_sample.status == 500
    assert text_sample.error == "plain failure"

    class RaisingClient(RecordingClient):
        """A RecordingClient whose delete raises, to exercise cleanup failures."""

        async def delete(self, url, **kwargs):
            raise TimeoutError("too slow")

    exc_sample = run(
        benchmark.make_delete("/points/1", timeout=3.0)(
            RaisingClient(),
            "http://middleware",
        )
    )
    assert exc_sample.ok is False
    assert exc_sample.status == 0
    assert exc_sample.error == "TimeoutError"


def test_delete_body_uses_delete_request_with_json_payload():
    client = RecordingClient()
    sample = run(
        benchmark.make_delete_body("/points", {"ids": ["1"]}, timeout=4.0)(
            client,
            "http://middleware",
        )
    )

    assert sample.ok is True
    assert client.calls == [
        ("DELETE", "http://middleware/points", {"json": {"ids": ["1"]}, "timeout": 4.0})
    ]


def test_run_setup_and_teardown_call_destructive_steps_in_expected_places():
    class SetupClient(RecordingClient):
        """Distinguish missing collections from successful creation."""

        def _response(self, method, url):
            return httpx.Response(
                404 if method == "GET" else 200,
                json={
                    "ok": True,
                    "created": True,
                    "collection_name": "bench-collection",
                },
                request=httpx.Request(method, url),
            )

    client = SetupClient()

    setup = run(
        benchmark.run_setup(
            client,
            "http://middleware",
            "bench-collection",
            timeout=2.0,
        )
    )
    assert [result.name for result in setup] == [
        "create_collection",
        "create_payload_indexes",
        "ingest_stellantis_doc",
        "seed_upsert_point",
        "seed_batch_upsert",
        "seed_inline_chunks",
        "seed_file_chunks",
    ]
    assert client.calls[1] == (
        "POST",
        "http://middleware/collections",
        {
            "json": {
                "collection_name": "bench-collection",
                "recreate": False,
                "dense_on_disk": True,
                "quantization": True,
                "sparse_idf": True,
                "sparse_on_disk": True,
            },
            "timeout": 2.0,
        },
    )

    client.calls.clear()
    teardown = run(
        benchmark.run_teardown(
            client,
            "http://middleware",
            "bench-collection",
            timeout=2.0,
        )
    )
    assert [result.name for result in teardown] == [
        "delete_point",
        "bulk_delete_points",
        "delete_payload_index",
        "delete_collection",
    ]
    assert client.calls[-1] == (
        "DELETE",
        "http://middleware/collections/bench-collection",
        {"params": {"confirm_delete": "bench-collection"}, "timeout": 2.0},
    )


@pytest.mark.parametrize("status,payload", [(200, {}), (503, {}), (200, None)])
def test_existing_or_unavailable_collection_is_never_seeded(status, payload):
    client = RecordingClient(status, payload)
    ownership = benchmark.CollectionOwnership()
    result = run(
        benchmark.run_setup(client, "http://middleware", "bench", 2, ownership)
    )
    assert not result[0].samples[0].ok
    assert not ownership.created
    assert [method for method, _, _ in client.calls] == ["GET"]


@pytest.mark.parametrize("payload", [{"created": False}, {}, [], {"created": 1}])
async def test_unconfirmed_creation_is_never_seeded(payload):
    calls = []

    def respond(request):
        calls.append(request.method)
        return httpx.Response(404 if request.method == "GET" else 200, json=payload)

    ownership = benchmark.CollectionOwnership()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await benchmark.run_setup(
            client, "http://middleware", "bench", 2, ownership
        )
    assert not result[0].samples[0].ok
    assert not ownership.created
    assert calls == ["GET", "POST"]


def _main_args():
    """Use a small offline benchmark without reading environment credentials."""
    return argparse.Namespace(
        url="http://middleware",
        token="",
        scenarios="health",
        requests=1,
        concurrency=1,
        warmup=0,
        timeout=2,
        sleep_between=0,
        no_post_check=True,
    )


def _mock_main_http(monkeypatch, handler):
    """Run the real main HTTP flow through a strict in-memory transport."""
    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        benchmark.httpx,
        "AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(handler), **kwargs),
    )


@pytest.mark.parametrize(
    "outcome",
    [
        "existing",
        "unavailable",
        "lookup-timeout",
        "create-timeout",
        "create-http-error",
        "invalid-json",
        "not-created",
        "numeric-created",
        "missing-fields",
        "wrong-collection",
        "not-ok",
    ],
)
async def test_main_never_seeds_or_deletes_without_confirmed_ownership(
    monkeypatch,
    outcome,
):
    calls = []
    collection = benchmark.BENCH_COLLECTION
    acknowledgement = {"ok": True, "created": True, "collection_name": collection}

    def respond(request):
        calls.append((request.method, request.url.path))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.method == "GET":
            if outcome == "lookup-timeout":
                raise httpx.ReadTimeout("lookup timed out", request=request)
            status = {"existing": 200, "unavailable": 503}.get(outcome, 404)
            return httpx.Response(status, json={})
        if request.method == "POST" and request.url.path == "/collections":
            if outcome == "create-timeout":
                raise httpx.ReadTimeout("acknowledgement lost", request=request)
            if outcome == "create-http-error":
                return httpx.Response(500, json=acknowledgement)
            if outcome == "invalid-json":
                return httpx.Response(200, text="not JSON")
            payloads = {
                "not-created": {**acknowledgement, "created": False},
                "numeric-created": {**acknowledgement, "created": 1},
                "missing-fields": {"created": True},
                "wrong-collection": {**acknowledgement, "collection_name": "other"},
                "not-ok": {**acknowledgement, "ok": False},
            }
            return httpx.Response(200, json=payloads[outcome])
        return httpx.Response(200, json={"ok": True})

    _mock_main_http(monkeypatch, respond)
    assert await benchmark.main_async(_main_args()) == 1
    expected = [("GET", "/health"), ("GET", f"/collections/{collection}")]
    if outcome not in {"existing", "unavailable", "lookup-timeout"}:
        expected.append(("POST", "/collections"))
    assert calls == expected


@pytest.mark.parametrize(
    "failure",
    [
        "setup-http",
        "setup-exception",
        "setup-cancelled",
        "benchmark-exception",
        "benchmark-cancelled",
    ],
)
async def test_main_cleans_up_acknowledged_creation_after_failure(monkeypatch, failure):
    calls = []
    collection = benchmark.BENCH_COLLECTION

    def respond(request):
        calls.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.method == "GET":
            return httpx.Response(404, json={})
        if request.url.path == "/collections":
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "created": True,
                    "collection_name": collection,
                },
            )
        if failure == "setup-http" and request.method == "POST":
            return httpx.Response(500, json={"detail": "setup failed"})
        return httpx.Response(200, json={"ok": True})

    original_run_once = benchmark._run_once

    async def run_once(client, base_url, name, fn):
        if name == "create_payload_indexes":
            if failure == "setup-exception":
                raise RuntimeError("injected setup failure")
            if failure == "setup-cancelled":
                raise asyncio.CancelledError()
        return await original_run_once(client, base_url, name, fn)

    async def run_selected(**kwargs):
        if failure == "benchmark-exception":
            raise RuntimeError("injected benchmark failure")
        if failure == "benchmark-cancelled":
            raise asyncio.CancelledError()
        pytest.fail("benchmark must be skipped after failed setup")

    _mock_main_http(monkeypatch, respond)
    monkeypatch.setattr(benchmark, "_run_once", run_once)
    monkeypatch.setattr(benchmark, "run_selected", run_selected)
    if failure == "setup-http":
        assert await benchmark.main_async(_main_args()) == 1
    else:
        expected = (
            asyncio.CancelledError if failure.endswith("cancelled") else RuntimeError
        )
        with pytest.raises(expected):
            await benchmark.main_async(_main_args())
    deletions = [request for request in calls if request.method == "DELETE"]
    assert [request.url.path for request in deletions] == [
        f"/points/{benchmark.BENCH_POINT_ID}",
        "/points",
        f"/collections/{collection}/payload-indexes/{benchmark.FACET_FIELD}",
        f"/collections/{collection}",
    ]
    assert deletions[-1].url.params["confirm_delete"] == collection


def test_run_scenario_honors_warmup_request_count_and_concurrency():
    state = {"active": 0, "max_active": 0, "calls": 0}

    async def scenario(client, base_url):
        assert base_url == "http://middleware"
        state["calls"] += 1
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        await asyncio.sleep(0.01)
        state["active"] -= 1
        return benchmark.Sample(1.0, True, 200)

    result = run(
        benchmark.run_scenario(
            client=object(),
            base_url="http://middleware",
            name="load",
            fn=scenario,
            requests=5,
            concurrency=2,
            warmup=3,
        )
    )

    assert result.name == "load"
    assert len(result.samples) == 5
    assert result.success == 5
    assert state["calls"] == 8
    assert state["max_active"] <= 2


def test_run_selected_preserves_order_and_uses_sleep_between(monkeypatch):
    calls = []
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    async def one(client, base_url):
        calls.append(("one", base_url))
        return benchmark.Sample(1.0, True, 200)

    async def two(client, base_url):
        calls.append(("two", base_url))
        return benchmark.Sample(1.0, True, 200)

    monkeypatch.setattr(benchmark.asyncio, "sleep", fake_sleep)
    results = run(
        benchmark.run_selected(
            client=object(),
            base_url="http://middleware",
            names=["one", "two"],
            scenario_map={"one": one, "two": two},
            requests=1,
            concurrency=1,
            warmup=0,
            sleep_between=0.25,
        )
    )

    assert [result.name for result in results] == ["one", "two"]
    assert calls == [("one", "http://middleware"), ("two", "http://middleware")]
    assert sleeps == [0.25]

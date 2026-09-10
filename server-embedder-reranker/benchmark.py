"""Smart Embedder server benchmark.

Measures latency (avg/p50/p95/p99) and throughput (req/s, sent/s, input token/s)
across configurable scenarios. Exact token counting uses the active model's
Transformers tokenizer, which is loaded before the timed benchmark window.

``build_scenarios()`` defines workloads, ``run_scenario()`` measures them, and
``main_async()`` coordinates preflight, metrics, and optional JSON reporting.

Usage:
    python benchmark.py --url http://127.0.0.1:8010 --concurrency 8 --requests 200
    python benchmark.py --scenarios all --concurrency 4 --requests 100
    python benchmark.py --scenarios embed_dense,embed_sparse,rerank
    python benchmark.py --token <token>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List

import httpx

API_TOKEN_ENV = "SMART_EMBEDDER_API_TOKENS"
ROOT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

DEFAULT_SENTENCES = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks to solve complex problems.",
    "Natural language processing enables machines to understand text.",
    "Vector databases power semantic search at scale.",
    "Retrieval-augmented generation grounds LLMs with external context.",
    "Transformers revolutionized sequence modeling.",
    "Embeddings map text to high-dimensional vector spaces.",
    "Cross-encoders score query-document pairs jointly.",
]

DEFAULT_PASSAGES = [
    "Machine learning is a subset of artificial intelligence.",
    "The weather forecast predicts rain tomorrow.",
    "Deep learning uses neural networks for complex tasks.",
    "Cooking pasta requires boiling water and salt.",
    "Vector databases enable efficient similarity search.",
    "The Eiffel Tower is located in Paris, France.",
    "Cross-encoders rerank retrieved candidates by relevance.",
    "Photosynthesis converts sunlight into chemical energy.",
]

CONTROLLED_TEXT_SEEDS = [
    (
        "Modern retrieval systems transform multilingual documents into dense and "
        "sparse vectors while preserving semantic relationships across languages, "
        "domains, and document lengths. Efficient batching reduces padding overhead, "
        "keeps accelerators busy, and improves throughput without changing the "
        "embedding model or its numerical behavior."
    ),
    (
        "Un motore di ricerca semantica prepara i documenti, calcola rappresentazioni "
        "vettoriali e combina segnali lessicali con similarità dense. Una pipeline "
        "efficiente misura separatamente tokenizzazione, inferenza, trasferimenti di "
        "memoria e serializzazione, mantenendo latenza e affidabilità sotto controllo."
    ),
    (
        "Reliable performance engineering uses deterministic inputs, repeatable warmup "
        "cycles, exact token counts, and explicit error reporting. Measurements cover "
        "short queries, medium passages, and long documents so that optimization "
        "decisions reflect both interactive latency and sustained service throughput."
    ),
    (
        "Vector databases support semantic search by indexing numerical "
        "representations of text. Production systems also need backpressure, bounded "
        "queues, health "
        "checks, authentication, and observable resource usage because raw model speed "
        "alone does not guarantee stable end-to-end application performance."
    ),
]


@dataclass
class Sample:
    """Observed outcome of one logical benchmark transaction."""

    latency_ms: float
    ok: bool
    status: int = 0
    error: str = ""
    server_ms: float = 0.0
    warnings: int = 0
    http_requests: int = 1
    successful_http_requests: int | None = None
    statuses: Dict[str, int] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    """Aggregate samples and workload totals for one named scenario."""

    name: str
    samples: List[Sample] = field(default_factory=list)
    wall_time_s: float = 0.0
    total_units: int = 0  # sentences embedded or pairs reranked
    total_input_tokens: int | None = None

    @property
    def ok_latencies(self) -> List[float]:
        """Return client latencies for successful logical transactions."""
        return [s.latency_ms for s in self.samples if s.ok]

    @property
    def ok_server_latencies(self) -> List[float]:
        """Return positive server timings from successful transactions."""
        return [s.server_ms for s in self.samples if s.ok and s.server_ms > 0]

    @property
    def warnings(self) -> int:
        """Return warnings emitted by successful HTTP responses."""
        return sum(s.warnings for s in self.samples if s.ok)

    @property
    def requests(self) -> int:
        """Return the number of physical HTTP requests represented."""
        return sum(sample.http_requests for sample in self.samples)

    @property
    def success(self) -> int:
        """Return the number of successful physical HTTP requests."""
        return sum(
            sample.successful_http_requests
            if sample.successful_http_requests is not None
            else int(sample.ok)
            for sample in self.samples
        )

    @property
    def failed(self) -> int:
        """Return the number of failed physical HTTP requests."""
        return self.requests - self.success

    @property
    def ok_percent(self) -> float:
        """Return the successful-request percentage."""
        return (self.success / self.requests * 100.0) if self.requests else 0.0


def percentile(values: List[float], p: float) -> float:
    """Return the nearest indexed percentile, or zero for no samples."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = max(
        0, min(len(sorted_vals) - 1, int(round(p / 100.0 * (len(sorted_vals) - 1))))
    )
    return sorted_vals[k]


def fmt_ms(v: float) -> str:
    """Format milliseconds, switching to seconds at one second."""
    return f"{v:.1f}" if v < 1000 else f"{v / 1000:.2f}s"


def fmt_int(v: int | float) -> str:
    """Format a number with spaces as thousands separators."""
    return f"{v:,}".replace(",", " ")


def section(title: str) -> str:
    """Render a plain-text section heading."""
    return f"{title}\n{'-' * len(title)}"


def sample_status_label(sample: Sample) -> str:
    """Return an HTTP status or stable client-exception label."""
    if sample.status:
        return str(sample.status)
    return f"EXC:{sample.error or 'client_error'}"


def status_counts(result: ScenarioResult) -> Dict[str, int]:
    """Aggregate deterministic status counts across a scenario."""
    counts: Counter[str] = Counter()
    for sample in result.samples:
        if sample.statuses:
            counts.update(sample.statuses)
        else:
            counts[sample_status_label(sample)] += 1

    def sort_key(item: tuple[str, int]) -> tuple[int, str]:
        key = item[0]
        if key.isdigit():
            return (0, f"{int(key):04d}")
        return (1, key)

    return dict(sorted(counts.items(), key=sort_key))


def format_status_counts(result: ScenarioResult) -> str:
    """Render scenario status counts as a compact comma-separated string."""
    counts = status_counts(result)
    return ", ".join(f"{status}={count}" for status, count in counts.items()) or "-"


def render_table(rows: List[List[str]], headers: List[str]) -> str:
    """Render rows as a width-aligned ASCII table."""
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(str(c)) for c in col) for col in cols]
    sep = "+".join("-" * (w + 2) for w in widths)
    sep = f"+{sep}+"

    def row(cells: List[str]) -> str:
        padded = [f" {c:<{w}} " for c, w in zip(cells, widths)]
        return f"|{'|'.join(padded)}|"

    out = [sep, row(headers), sep]
    for r in rows:
        out.append(row([str(c) for c in r]))
    out.append(sep)
    return "\n".join(out)


# --- Scenarios --------------------------------------------------------------

ScenarioFn = Callable[[httpx.AsyncClient, str], Awaitable[Sample]]


@dataclass(frozen=True)
class EmbedTokenCounter:
    """Count post-truncation input tokens exactly like the embedding backend."""

    tokenizer: Any
    model_name: str
    max_length: int

    def count(self, texts: List[str]) -> int:
        """Count exact post-truncation tokens across embedding inputs."""
        if not texts:
            return 0
        encoded = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        return sum(len(input_ids) for input_ids in encoded["input_ids"])

    def texts_at_token_length(self, size: int, target_tokens: int) -> List[str]:
        """Build deterministic texts that each encode to exactly target_tokens."""
        if target_tokens < 1 or target_tokens > self.max_length:
            raise ValueError(
                f"target token length must be between 1 and {self.max_length}"
            )

        fillers = (
            "benchmark ",
            "performance ",
            "data ",
            "test ",
            "vector ",
            "semantic ",
            "retrieval ",
            "embedding ",
        )
        texts = []
        for index in range(size):
            seed_words = CONTROLLED_TEXT_SEEDS[
                index % len(CONTROLLED_TEXT_SEEDS)
            ].split()
            copies = (target_tokens * 2 // len(seed_words)) + 2
            word_stream = seed_words * copies

            low = 0
            high = len(word_stream)
            base_text = ""
            while low <= high:
                word_count = (low + high) // 2
                text = " ".join(word_stream[:word_count]) + " "
                actual_tokens = self.count([text])
                if actual_tokens <= target_tokens:
                    base_text = text
                    low = word_count + 1
                else:
                    high = word_count - 1

            exact_text = base_text if self.count([base_text]) == target_tokens else ""
            for filler_offset in range(len(fillers)):
                if exact_text:
                    break
                filler = fillers[(index + filler_offset) % len(fillers)]
                low = 0
                high = target_tokens
                while low <= high:
                    repeats = (low + high) // 2
                    text = base_text + filler * repeats
                    actual_tokens = self.count([text])
                    if actual_tokens == target_tokens:
                        exact_text = text
                        break
                    if actual_tokens < target_tokens:
                        low = repeats + 1
                    else:
                        high = repeats - 1

            if not exact_text:
                raise ValueError(
                    f"could not synthesize an exact {target_tokens}-token input "
                    f"with tokenizer '{self.model_name}'"
                )
            texts.append(exact_text)

        return texts

    def texts_at_token_pattern(self, size: int, target_tokens: List[int]) -> List[str]:
        """Build deterministic texts cycling through exact token lengths."""
        if not target_tokens:
            raise ValueError("mixed token workloads require at least one length")
        generated = {
            target: self.texts_at_token_length(size, target)
            for target in dict.fromkeys(target_tokens)
        }
        return [
            generated[target_tokens[index % len(target_tokens)]][index]
            for index in range(size)
        ]


@dataclass(frozen=True)
class RerankTokenCounter:
    """Reproduce Qwen's full formatted query-document sequence length."""

    base: EmbedTokenCounter
    instruction: str
    prefix_tokens: int
    suffix_tokens: int

    @property
    def max_length(self) -> int:
        """Return the active reranker sequence limit."""
        return self.base.max_length

    def pair_count(self, query: str, passage: str) -> int:
        """Count the complete formatted Qwen query-document sequence."""
        formatted = (
            f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {passage}"
        )
        tokenizer = self.base.tokenizer
        try:
            token_ids = tokenizer.encode(
                formatted, add_special_tokens=False, verbose=False
            )
        except TypeError:
            token_ids = tokenizer.encode(formatted, add_special_tokens=False)
        return self.prefix_tokens + len(token_ids) + self.suffix_tokens

    def passages_at_total_length(
        self, size: int, target_tokens: int, query: str
    ) -> List[str]:
        """Build passages whose complete Qwen rerank sequence is exact."""
        if target_tokens < 1 or target_tokens > self.max_length:
            raise ValueError(
                f"rerank target must be between 1 and {self.max_length} tokens"
            )
        empty_total = self.pair_count(query, "")
        initial_content = target_tokens - empty_total
        if initial_content < 1:
            raise ValueError(
                f"rerank target {target_tokens} cannot fit the fixed prompt and query "
                f"overhead ({empty_total} tokens before passage content)"
            )

        # Token boundaries around '<Document>:' can shift the additive estimate
        # by a handful of tokens. Probe symmetrically around it while preserving
        # exact raw passage lengths; this stays outside the measured window.
        candidates = [initial_content]
        for delta in range(1, 65):
            candidates.extend((initial_content - delta, initial_content + delta))
        for content_tokens in candidates:
            if content_tokens < 1 or content_tokens > self.base.max_length:
                continue
            passage = self.base.texts_at_token_length(1, content_tokens)[0]
            if self.pair_count(query, passage) == target_tokens:
                return [passage for _ in range(size)]

        raise ValueError(
            f"could not synthesize an exact {target_tokens}-token Qwen rerank pair"
        )

    def passages_at_total_pattern(
        self, size: int, target_tokens: List[int], query: str
    ) -> List[str]:
        """Cycle exact total Qwen sequence lengths across passages.

        Args:
            size: Number of passages to generate.
            target_tokens: Non-empty sequence of complete pair lengths.
            query: Query included in every formatted pair.

        Returns:
            Passages whose complete pair lengths repeat the supplied pattern.

        Raises:
            ValueError: If ``target_tokens`` is empty.
        """
        if not target_tokens:
            raise ValueError("mixed rerank workloads require at least one length")
        generated = {
            target: self.passages_at_total_length(1, target, query)[0]
            for target in dict.fromkeys(target_tokens)
        }
        return [
            generated[target_tokens[index % len(target_tokens)]]
            for index in range(size)
        ]


def load_embed_token_counter(model_name: str, max_length: int) -> EmbedTokenCounter:
    """Load the active embedding tokenizer outside the measured time window."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "exact token counting requires transformers; install the embedder "
            "requirements or use --no-token-count"
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"could not load tokenizer '{model_name}': {type(exc).__name__}: {exc}"
        ) from exc

    return EmbedTokenCounter(
        tokenizer=tokenizer,
        model_name=model_name,
        max_length=max_length,
    )


def active_embedding_max_length(info: Dict) -> int:
    """Resolve the active profile's advertised embedding truncation limit."""
    key = (
        "qwen_embed_max_length"
        if str(info.get("profile", "")).lower() == "advanced"
        else "bge_embed_max_length"
    )
    try:
        return int(info.get(key) or info.get("max_input_length") or 0)
    except (TypeError, ValueError):
        return 0


def active_reranker_max_length(info: Dict) -> int:
    """Resolve the active reranker's advertised total sequence limit."""
    reranker_model = str(info.get("reranker_model", "")).lower()
    key = (
        "qwen_rerank_max_length"
        if "qwen" in reranker_model
        else "bge_rerank_max_length"
    )
    try:
        return int(info.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def response_summary(resp: httpx.Response) -> tuple[float, int, str]:
    """Extract server timing, warning count, and bounded error detail.

    Args:
        resp: HTTP response returned by a benchmark request.

    Returns:
        Server milliseconds, warning count, and a short detail string.
    """
    try:
        data = resp.json()
    except ValueError:
        return 0.0, 0, resp.text[:200]

    if not isinstance(data, dict):
        return 0.0, 0, resp.text[:200]

    server_ms = float(data.get("processing_time_ms") or 0.0)
    warnings = data.get("warnings") or []
    warning_count = len(warnings) if isinstance(warnings, list) else 0
    detail = data.get("detail", "")
    return server_ms, warning_count, str(detail)[:200]


def read_dotenv_value(path: Path, key: str) -> str:
    """Read one value from a Compose-style env file without exposing it."""
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return ""

    result = ""
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()

        name, separator, raw_value = line.partition("=")
        if not separator or name.strip() != key:
            continue

        value = raw_value.strip()
        if value.startswith(('"', "'")):
            quote = value[0]
            closing_quote = value.find(quote, 1)
            result = value[1:closing_quote] if closing_quote >= 0 else value[1:]
            continue

        for index, char in enumerate(value):
            if char == "#" and (index == 0 or value[index - 1].isspace()):
                value = value[:index]
                break
        result = value.rstrip()

    return result


def first_token(raw: str) -> str:
    """Return the token half of the first ``token:label`` entry in ``raw``."""
    for entry in (part.strip() for part in raw.split(",")):
        if not entry:
            continue
        token, separator, _ = entry.partition(":")
        if separator and token.strip():
            return token.strip()
    return ""


def default_api_token(env_path: Path = ROOT_ENV_PATH) -> str:
    """Resolve auth using process environment before the repository root .env.

    ``SMART_EMBEDDER_API_TOKENS`` holds a ``token:label`` list; the benchmark
    presents the first entry's token. Use ``--token`` to pick a different one.
    Taking the first entry is acceptable here because the benchmark is run by
    the operator who owns ``.env``, unlike a service-to-service credential,
    which has its own variable.
    """
    if API_TOKEN_ENV in os.environ:
        return first_token(os.environ[API_TOKEN_ENV])
    return first_token(read_dotenv_value(env_path, API_TOKEN_ENV))


async def authentication_error(
    client: httpx.AsyncClient, base_url: str, token: str
) -> str:
    """Probe auth middleware without running model inference."""
    try:
        response = await client.head(f"{base_url}/embeddings/", timeout=10.0)
    except httpx.HTTPError as exc:
        return f"authentication probe failed ({type(exc).__name__})"

    if response.status_code not in {401, 403}:
        return ""

    _, _, detail = response_summary(response)
    detail_suffix = f" Server response: {detail}" if detail else ""
    if token:
        return (
            f"the supplied bearer token was rejected ({response.status_code})."
            f"{detail_suffix}"
        )
    return (
        f"the server requires a bearer token ({response.status_code}); use --token, "
        f"export {API_TOKEN_ENV} (a token:label list; the first entry's token is "
        f"used), or configure it in the repository root .env."
        f"{detail_suffix}"
    )


def make_post_request(path: str, payload: Dict, timeout: float) -> ScenarioFn:
    """Build a timed POST scenario that converts failures into samples.

    Args:
        path: API path relative to the benchmark base URL.
        payload: JSON request body reused for each sample.
        timeout: Per-request timeout in seconds.

    Returns:
        An asynchronous scenario callable.
    """

    async def fn(client: httpx.AsyncClient, base_url: str) -> Sample:
        start = time.perf_counter()
        try:
            resp = await client.post(f"{base_url}{path}", json=payload, timeout=timeout)
            latency = (time.perf_counter() - start) * 1000
            server_ms, warnings, detail = response_summary(resp)
            return Sample(
                latency_ms=latency,
                ok=resp.status_code == 200,
                status=resp.status_code,
                error="" if resp.status_code == 200 else detail,
                server_ms=server_ms,
                warnings=warnings,
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return Sample(
                latency_ms=latency,
                ok=False,
                status=0,
                error=type(e).__name__,
            )

    return fn


def make_mixed_request(
    embed_request: ScenarioFn,
    rerank_request: ScenarioFn,
) -> ScenarioFn:
    """Run one embedding and one rerank HTTP request as a mixed transaction."""

    async def fn(client: httpx.AsyncClient, base_url: str) -> Sample:
        start = time.perf_counter()
        embed_sample, rerank_sample = await asyncio.gather(
            embed_request(client, base_url),
            rerank_request(client, base_url),
        )
        latency = (time.perf_counter() - start) * 1000
        samples = (embed_sample, rerank_sample)
        successes = sum(int(sample.ok) for sample in samples)
        statuses: Counter[str] = Counter(
            sample_status_label(sample) for sample in samples
        )
        errors = [sample.error for sample in samples if sample.error]
        first_failure = next((sample for sample in samples if not sample.ok), None)
        return Sample(
            latency_ms=latency,
            ok=successes == len(samples),
            status=first_failure.status if first_failure is not None else 200,
            error="; ".join(errors),
            server_ms=sum(sample.server_ms for sample in samples),
            warnings=sum(sample.warnings for sample in samples),
            http_requests=len(samples),
            successful_http_requests=successes,
            statuses=dict(statuses),
        )

    return fn


def repeat_to_size(values: List[str], size: int) -> List[str]:
    """Repeat a non-empty sequence and truncate it to ``size`` items."""
    copies = (size // len(values)) + 1
    return (values * copies)[:size]


def build_scenarios(
    batch_size: int,
    timeout: float,
    include_colbert: bool,
    token_counter: EmbedTokenCounter | None = None,
    target_tokens: int = 0,
    mixed_target_tokens: List[int] | None = None,
    rerank_token_counter: EmbedTokenCounter | RerankTokenCounter | None = None,
) -> Dict[str, tuple[ScenarioFn, int, int | None]]:
    """Construct all API scenarios for one workload shape.

    Args:
        batch_size: Sentences or passages sent per logical request.
        timeout: Per-request timeout in seconds.
        include_colbert: Whether the active profile supports ColBERT output.
        token_counter: Optional exact embedding token counter.
        target_tokens: Uniform exact embedding length, or zero for defaults.
        mixed_target_tokens: Optional repeating embedding-length pattern.
        rerank_token_counter: Counter used to synthesize reranker pairs.

    Returns:
        Scenario names mapped to callables, units, and exact input-token totals.

    Raises:
        ValueError: If token controls conflict or lack a required counter.
    """
    if target_tokens and mixed_target_tokens:
        raise ValueError("uniform and mixed token workloads are mutually exclusive")
    if mixed_target_tokens:
        if token_counter is None:
            raise ValueError("mixed token workloads require exact token counting")
        sents = token_counter.texts_at_token_pattern(batch_size, mixed_target_tokens)
    elif target_tokens:
        if token_counter is None:
            raise ValueError("target token workloads require exact token counting")
        sents = token_counter.texts_at_token_length(batch_size, target_tokens)
    else:
        sents = repeat_to_size(DEFAULT_SENTENCES, batch_size)
    embed_tokens = token_counter.count(sents) if token_counter is not None else None

    rerank_query = "What is machine learning?"
    rerank_tokens = None
    if isinstance(rerank_token_counter, RerankTokenCounter):
        if mixed_target_tokens:
            passages = rerank_token_counter.passages_at_total_pattern(
                batch_size,
                mixed_target_tokens,
                rerank_query,
            )
        elif target_tokens:
            passages = rerank_token_counter.passages_at_total_length(
                batch_size,
                target_tokens,
                rerank_query,
            )
        else:
            passages = repeat_to_size(DEFAULT_PASSAGES, batch_size)
        rerank_tokens = sum(
            rerank_token_counter.pair_count(rerank_query, passage)
            for passage in passages
        )
    elif rerank_token_counter is not None:
        query_tokens = rerank_token_counter.count([rerank_query])
        if mixed_target_tokens:
            passage_targets = [
                max(1, target - query_tokens) for target in mixed_target_tokens
            ]
            passages = rerank_token_counter.texts_at_token_pattern(
                batch_size,
                passage_targets,
            )
        elif target_tokens:
            passages = rerank_token_counter.texts_at_token_length(
                batch_size,
                max(1, target_tokens - query_tokens),
            )
        else:
            passages = repeat_to_size(DEFAULT_PASSAGES, batch_size)
        rerank_tokens = rerank_token_counter.count(
            [rerank_query] * batch_size + passages
        )
    else:
        passages = repeat_to_size(DEFAULT_PASSAGES, batch_size)

    embed_dense = {
        "sentences": sents,
        "return_dense": True,
        "return_sparse": False,
        "normalize_dense": True,
    }
    embed_sparse = {
        "sentences": sents,
        "return_dense": False,
        "return_sparse": True,
    }
    embed_sparse_indices = {
        **embed_sparse,
        "sparse_as_indices": True,
    }
    embed_hybrid = {
        "sentences": sents,
        "return_dense": True,
        "return_sparse": True,
        "return_colbert": False,
        "normalize_dense": True,
        "sparse_as_indices": True,
    }
    embed_full = {**embed_hybrid}
    if include_colbert:
        embed_dense["return_colbert"] = False
        embed_sparse["return_colbert"] = False
        embed_sparse_indices["return_colbert"] = False
        embed_full["return_colbert"] = True

    rerank_request = make_post_request(
        "/rerank",
        {
            "query": rerank_query,
            "passages": passages,
            "normalize": True,
        },
        timeout,
    )
    hybrid_request = make_post_request("/embeddings/", embed_hybrid, timeout)

    scenarios = {
        "embed_dense": (
            make_post_request("/embeddings/", embed_dense, timeout),
            batch_size,
            embed_tokens,
        ),
        "embed_sparse": (
            make_post_request("/embeddings/", embed_sparse, timeout),
            batch_size,
            embed_tokens,
        ),
        "embed_sparse_indices": (
            make_post_request("/embeddings/", embed_sparse_indices, timeout),
            batch_size,
            embed_tokens,
        ),
        "embed_hybrid": (
            hybrid_request,
            batch_size,
            embed_tokens,
        ),
        "embed_full": (
            make_post_request("/embeddings/", embed_full, timeout),
            batch_size,
            embed_tokens,
        ),
        "rerank": (
            rerank_request,
            batch_size,
            rerank_tokens,
        ),
        "mixed": (
            make_mixed_request(hybrid_request, rerank_request),
            batch_size * 2,
            (
                embed_tokens + rerank_tokens
                if embed_tokens is not None and rerank_tokens is not None
                else None
            ),
        ),
    }

    if include_colbert:
        scenarios["embed_colbert"] = (
            make_post_request(
                "/embeddings/",
                {
                    "sentences": sents,
                    "return_dense": False,
                    "return_sparse": False,
                    "return_colbert": True,
                },
                timeout,
            ),
            batch_size,
            embed_tokens,
        )

    return scenarios


# --- Runner -----------------------------------------------------------------


async def run_scenario(
    client: httpx.AsyncClient,
    base_url: str,
    name: str,
    fn: ScenarioFn,
    units_per_req: int,
    input_tokens_per_req: int | None,
    requests: int,
    concurrency: int,
    warmup: int,
) -> ScenarioResult:
    """Execute warmups and bounded concurrent measurements for one scenario.

    Args:
        client: Shared asynchronous HTTP client.
        base_url: Target Smart Embedder base URL.
        name: Scenario label used in reports.
        fn: Timed logical request callable.
        units_per_req: Successful sentences or pairs per logical request.
        input_tokens_per_req: Exact useful tokens, when available.
        requests: Number of measured logical requests.
        concurrency: Maximum simultaneous logical requests.
        warmup: Untimed requests issued before measurement.

    Returns:
        Samples and successful-work totals for the scenario.
    """
    # Warmups initialize model and transport paths outside the measured window.
    if warmup > 0:
        warm = [fn(client, base_url) for _ in range(warmup)]
        await asyncio.gather(*warm, return_exceptions=True)

    sem = asyncio.Semaphore(concurrency)

    async def bounded() -> Sample:
        async with sem:
            return await fn(client, base_url)

    start = time.perf_counter()
    samples = await asyncio.gather(*[bounded() for _ in range(requests)])
    wall = time.perf_counter() - start

    successful_requests = sum(1 for sample in samples if sample.ok)
    return ScenarioResult(
        name=name,
        samples=samples,
        wall_time_s=wall,
        total_units=units_per_req * successful_requests,
        total_input_tokens=(
            input_tokens_per_req * successful_requests
            if input_tokens_per_req is not None
            else None
        ),
    )


def summarize(results: List[ScenarioResult], concurrency: int) -> str:
    """Render comparable latency, throughput, and status results."""
    headers = [
        "Scenario",
        "Result",
        "Reqs",
        "OK",
        "OK%",
        "Fail",
        "Conc",
        "Wall(s)",
        "Req/s",
        "Units/s",
        "Tok/item",
        "Tok/s",
        "Avg(ms)",
        "SrvAvg",
        "P50",
        "P95",
        "P99",
        "Min",
        "Max",
        "Warn",
        "Statuses",
    ]
    rows: List[List[str]] = []
    for r in results:
        lat = r.ok_latencies
        server_lat = r.ok_server_latencies
        server_avg = statistics.fmean(server_lat) if server_lat else 0.0
        rps = r.success / r.wall_time_s if r.wall_time_s > 0 else 0
        ups = r.total_units / r.wall_time_s if r.wall_time_s > 0 else 0
        tokens_per_s = (
            r.total_input_tokens / r.wall_time_s
            if r.total_input_tokens is not None and r.wall_time_s > 0
            else None
        )
        tokens_per_unit = (
            r.total_input_tokens / r.total_units
            if r.total_input_tokens is not None and r.total_units > 0
            else None
        )
        if lat:
            avg = statistics.fmean(lat)
            p50 = percentile(lat, 50)
            p95 = percentile(lat, 95)
            p99 = percentile(lat, 99)
            mn = min(lat)
            mx = max(lat)
        else:
            avg = p50 = p95 = p99 = mn = mx = 0.0

        rows.append(
            [
                r.name,
                "PASS" if r.failed == 0 else "FAIL",
                fmt_int(r.requests),
                fmt_int(r.success),
                f"{r.ok_percent:.1f}%",
                fmt_int(r.failed),
                str(concurrency),
                f"{r.wall_time_s:.2f}",
                f"{rps:.1f}",
                f"{ups:.0f}",
                f"{tokens_per_unit:.1f}" if tokens_per_unit is not None else "-",
                f"{tokens_per_s:.0f}" if tokens_per_s is not None else "-",
                fmt_ms(avg),
                fmt_ms(server_avg),
                fmt_ms(p50),
                fmt_ms(p95),
                fmt_ms(p99),
                fmt_ms(mn),
                fmt_ms(mx),
                fmt_int(r.warnings),
                format_status_counts(r),
            ]
        )
    return render_table(rows, headers)


def scenario_result_record(result: ScenarioResult) -> Dict[str, Any]:
    """Serialize stable raw and derived values for before/after comparison."""
    latencies = result.ok_latencies
    server_latencies = result.ok_server_latencies
    return {
        "name": result.name,
        "result": "pass" if result.failed == 0 else "fail",
        "requests": result.requests,
        "success": result.success,
        "failed": result.failed,
        "warnings": result.warnings,
        "wall_time_s": result.wall_time_s,
        "total_units": result.total_units,
        "total_input_tokens": result.total_input_tokens,
        "requests_per_s": (
            result.success / result.wall_time_s if result.wall_time_s > 0 else 0.0
        ),
        "units_per_s": (
            result.total_units / result.wall_time_s if result.wall_time_s > 0 else 0.0
        ),
        "input_tokens_per_s": (
            result.total_input_tokens / result.wall_time_s
            if result.total_input_tokens is not None and result.wall_time_s > 0
            else None
        ),
        "latency_ms": {
            "average": statistics.fmean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "minimum": min(latencies) if latencies else 0.0,
            "maximum": max(latencies) if latencies else 0.0,
        },
        "server_average_ms": (
            statistics.fmean(server_latencies) if server_latencies else 0.0
        ),
        "statuses": status_counts(result),
        "samples": [
            {
                "latency_ms": sample.latency_ms,
                "server_ms": sample.server_ms,
                "ok": sample.ok,
                "status": sample.status,
                "error": sample.error,
                "warnings": sample.warnings,
                "http_requests": sample.http_requests,
                "successful_http_requests": sample.successful_http_requests,
                "statuses": sample.statuses,
            }
            for sample in result.samples
        ],
    }


def metric_delta_record(pre_metrics: str, post_metrics: str) -> Dict[str, float]:
    """Build machine-readable Prometheus deltas for a JSON report.

    Gauge-like memory series retain their current value; counters and
    histogram components are reported as post-minus-pre deltas.
    """
    pre = parse_prometheus_samples(pre_metrics)
    post = parse_prometheus_samples(post_metrics)
    return {
        key: (
            value
            if key.startswith(
                (
                    "advanced_backend_gpu_memory_bytes",
                    "advanced_backend_process_rss_peak_bytes",
                )
            )
            else value - pre.get(key, 0.0)
        )
        for key, value in sorted(post.items())
        if value - pre.get(key, 0.0) != 0.0
        or key.startswith(
            (
                "advanced_backend_gpu_memory_bytes",
                "advanced_backend_process_rss_peak_bytes",
            )
        )
    }


def write_json_report(path: Path, payload: Dict[str, Any]) -> None:
    """Persist a benchmark report atomically to avoid partial calibration data."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def render_verdict(results: List[ScenarioResult]) -> str:
    """Render a pass/fail verdict across measured HTTP requests."""
    total = sum(r.requests for r in results)
    failed = sum(r.failed for r in results)
    if failed == 0:
        return (
            "Verdict: PASS - "
            f"{fmt_int(total)} measured requests completed successfully."
        )

    failed_names = ", ".join(r.name for r in results if r.failed)
    return (
        f"Verdict: FAIL - {fmt_int(failed)}/{fmt_int(total)} measured requests failed "
        f"across: {failed_names}."
    )


def render_failure_details(results: List[ScenarioResult]) -> str:
    """Return per-scenario status and error-message breakdowns."""
    lines = []
    for result in results:
        if not result.failed:
            continue

        status_totals: Dict[str, int] = {}
        messages: Dict[str, int] = {}
        for sample in result.samples:
            if sample.ok:
                continue
            sample_statuses = sample.statuses or {sample_status_label(sample): 1}
            for status_label, count in sample_statuses.items():
                status_totals[status_label] = status_totals.get(status_label, 0) + count
            if sample.error:
                messages[sample.error] = messages.get(sample.error, 0) + 1

        detail = ", ".join(
            f"{status}={count}" for status, count in status_totals.items()
        )
        lines.append(f"  {result.name}: {detail}")
        for message, count in sorted(messages.items(), key=lambda item: item[0]):
            lines.append(f"    - {message}: {count}")

    if not lines:
        return ""
    return section("Failure details") + "\n" + "\n".join(lines)


def render_server_info(info: Dict) -> str:
    """Render relevant health metadata as a two-column table."""
    gpu = info.get("gpu", {}) if isinstance(info.get("gpu"), dict) else {}
    rows = [
        ["Status", str(info.get("status", "unknown"))],
        ["Version", str(info.get("version", "unknown"))],
        ["Profile", str(info.get("profile", "unknown"))],
        ["Dense model", str(info.get("dense_embedding_model", "unknown"))],
        ["Sparse model", str(info.get("sparse_model_name", "unknown"))],
        ["ColBERT model", str(info.get("colbert_model_name") or "disabled")],
        ["Reranker model", str(info.get("reranker_model", "unknown"))],
        ["GPU", str(gpu.get("device_info", "unknown"))],
        ["GPU available", str(gpu.get("available", "unknown"))],
        ["GPU memory", f"{gpu.get('memory_gb', 'unknown')} GB"],
        ["Device count", str(gpu.get("device_count", "unknown"))],
        ["Configured devices", ", ".join(map(str, gpu.get("configured_devices", [])))],
        ["BGE embed max", str(info.get("bge_embed_max_length", "unknown"))],
        ["Qwen embed max", str(info.get("qwen_embed_max_length", "unknown"))],
        ["BGE rerank max", str(info.get("bge_rerank_max_length", "unknown"))],
        ["Qwen rerank max", str(info.get("qwen_rerank_max_length", "unknown"))],
        [
            "Rerank content budget",
            str(info.get("reranker_content_budget", "unknown")),
        ],
        [
            "Rerank fixed overhead",
            str(info.get("reranker_fixed_overhead_tokens", "unknown")),
        ],
        ["MILCO source view", str(info.get("milco_source_view", "unknown"))],
        ["MILCO top-k", str(info.get("milco_top_k", "unknown"))],
        [
            "MILCO projection chunk",
            str(info.get("milco_projection_chunk_tokens", "unknown")),
        ],
        [
            "Length-aware batching",
            str(info.get("length_aware_batching", "unknown")),
        ],
        [
            "BGE embed batch",
            str(info.get("bge_embed_batch_size", info.get("batch_size", "unknown"))),
        ],
        ["Max requests/batch", str(info.get("max_requests_in_batch", "unknown"))],
    ]
    return render_table(rows, ["Field", "Value"])


def parse_prometheus_samples(metrics_text: str) -> Dict[str, float]:
    """Parse the benchmark's allowlisted Prometheus numeric series."""
    prefixes = (
        "embedding_requests_total",
        "embedding_requests_rejected_total",
        "embedding_input_tokens_total",
        "embedding_padded_tokens_total",
        "embedding_tokenized_batches_total",
        "advanced_backend_requests_total",
        "advanced_backend_input_tokens_total",
        "advanced_backend_padded_tokens_total",
        "advanced_backend_tokenized_batches_total",
        "advanced_backend_phase_duration_seconds_sum",
        "advanced_backend_phase_duration_seconds_count",
        "advanced_backend_gpu_memory_bytes",
        "advanced_backend_process_rss_peak_bytes",
        "embedding_queue_size",
        "embedding_active_requests",
        "api_model_forward_duration_seconds_sum",
        "api_model_forward_duration_seconds_count",
        "rerank_requests_total",
        "rerank_requests_rejected_total",
        "rerank_active_requests",
    )
    samples = {}
    for line in metrics_text.splitlines():
        if line.startswith("#") or not line.startswith(prefixes):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        try:
            samples[parts[0]] = float(parts[1])
        except ValueError:
            continue
    return samples


def prometheus_metric_delta(
    pre: Dict[str, float],
    post: Dict[str, float],
    metric_name: str,
    required_labels: tuple[str, ...] = (),
) -> float:
    """Sum a metric delta across series matching optional exact label fragments."""
    return sum(
        value - pre.get(key, 0.0)
        for key, value in post.items()
        if (key == metric_name or key.startswith(f"{metric_name}{{"))
        and all(label in key for label in required_labels)
    )


def render_metric_delta(pre_metrics: str, post_metrics: str) -> List[str]:
    """Render diagnostic metric changes and derived token throughput."""
    pre = parse_prometheus_samples(pre_metrics)
    post = parse_prometheus_samples(post_metrics)
    lines = []
    for key in sorted(post):
        delta = post[key] - pre.get(key, 0.0)
        if key.endswith(("_queue_size", "_active_requests")) or key.startswith(
            (
                "advanced_backend_gpu_memory_bytes",
                "advanced_backend_process_rss_peak_bytes",
            )
        ):
            lines.append(f"{key} current={post[key]:.0f}")
        elif (
            key.startswith(
                (
                    "api_model_forward_duration_seconds_sum",
                    "advanced_backend_phase_duration_seconds_sum",
                )
            )
        ) and delta:
            lines.append(f"{key} delta={delta:.6f}s current={post[key]:.6f}s")
        elif delta:
            lines.append(f"{key} delta={delta:.0f} current={post[key]:.0f}")

    useful_tokens = prometheus_metric_delta(pre, post, "embedding_input_tokens_total")
    padded_tokens = prometheus_metric_delta(pre, post, "embedding_padded_tokens_total")
    tokenized_batches = prometheus_metric_delta(
        pre, post, "embedding_tokenized_batches_total"
    )
    worker_seconds = prometheus_metric_delta(
        pre,
        post,
        "api_model_forward_duration_seconds_sum",
        ('endpoint="/embeddings"', 'operation="embed"'),
    )

    if useful_tokens > 0 and padded_tokens > 0:
        lines.append(
            "embedding_padding_efficiency "
            f"ratio={useful_tokens / padded_tokens:.4f} "
            f"useful={useful_tokens:.0f} padded={padded_tokens:.0f}"
        )
    if useful_tokens > 0 and worker_seconds > 0:
        lines.append(
            "embedding_worker_throughput "
            f"useful_tok/s={useful_tokens / worker_seconds:.0f} "
            f"padded_tok/s={padded_tokens / worker_seconds:.0f} "
            f"worker_seconds={worker_seconds:.6f}"
        )
    if tokenized_batches > 0:
        lines.append(
            "embedding_tokenized_batch_averages "
            f"useful={useful_tokens / tokenized_batches:.1f} "
            f"padded={padded_tokens / tokenized_batches:.1f} "
            f"batches={tokenized_batches:.0f}"
        )

    for backend in ("qwen_dense", "milco_sparse", "qwen_reranker"):
        label = (f'backend="{backend}"',)
        useful = prometheus_metric_delta(
            pre, post, "advanced_backend_input_tokens_total", label
        )
        padded = prometheus_metric_delta(
            pre, post, "advanced_backend_padded_tokens_total", label
        )
        forward_seconds = prometheus_metric_delta(
            pre,
            post,
            "advanced_backend_phase_duration_seconds_sum",
            (f'backend="{backend}"', 'phase="forward"'),
        )
        if useful > 0 and padded > 0:
            fields = [
                f"padding_efficiency={useful / padded:.4f}",
                f"useful={useful:.0f}",
                f"padded={padded:.0f}",
            ]
            if forward_seconds > 0:
                fields.extend(
                    (
                        f"forward_useful_tok/s={useful / forward_seconds:.0f}",
                        f"forward_seconds={forward_seconds:.6f}",
                    )
                )
            lines.append(f"advanced_backend backend={backend} " + " ".join(fields))
    return lines


def validate_args(args: argparse.Namespace) -> List[str]:
    """Return all command-line validation errors without raising."""
    errors = []

    if args.concurrency < 1:
        errors.append("--concurrency must be >= 1")
    if args.requests < 1:
        errors.append("--requests must be >= 1")
    if args.batch_size < 1:
        errors.append("--batch-size must be >= 1")
    if args.warmup < 0:
        errors.append("--warmup must be >= 0")
    if args.sleep_between < 0:
        errors.append("--sleep-between must be >= 0")
    if args.timeout <= 0:
        errors.append("--timeout must be > 0")
    if args.max_batch_size < 0:
        errors.append("--max-batch-size must be >= 0")
    if args.max_batch_size and args.batch_size > args.max_batch_size:
        errors.append(
            f"--batch-size exceeds --max-batch-size ({args.max_batch_size}); "
            "raise --max-batch-size or set it to 0 if the server allows larger payloads"
        )
    if args.matrix_sleep < 0:
        errors.append("--matrix-sleep must be >= 0")
    if args.tokenizer_max_length < 0:
        errors.append("--tokenizer-max-length must be >= 0")
    if args.reranker_tokenizer_max_length < 0:
        errors.append("--reranker-tokenizer-max-length must be >= 0")
    if args.input_tokens < 0:
        errors.append("--input-tokens must be >= 0")
    if args.mixed_input_tokens and (args.input_tokens or args.matrix_input_tokens):
        errors.append(
            "--mixed-input-tokens cannot be combined with --input-tokens or "
            "--matrix-input-tokens"
        )

    return errors


def parse_positive_int_list(value: str) -> List[int]:
    """Parse a comma-separated list containing only positive integers.

    Raises:
        argparse.ArgumentTypeError: If the list is empty or contains a
            non-positive value.
    """
    values = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed < 1:
            raise argparse.ArgumentTypeError("values must be positive integers")
        values.append(parsed)
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


async def run_selected_scenarios(
    client: httpx.AsyncClient,
    base_url: str,
    selected_names: List[str],
    scenario_map: Dict[str, tuple],
    requests: int,
    concurrency: int,
    warmup: int,
    sleep_between: float,
) -> List[ScenarioResult]:
    """Run selected scenarios sequentially with optional cooldowns."""
    results = []
    for idx, name in enumerate(selected_names):
        if idx > 0 and sleep_between > 0:
            print(f"Cooling down {sleep_between}s (rate-limit refill)...")
            await asyncio.sleep(sleep_between)
        fn, units, input_tokens = scenario_map[name]
        print(f"Running: {name} ...")
        res = await run_scenario(
            client=client,
            base_url=base_url,
            name=name,
            fn=fn,
            units_per_req=units,
            input_tokens_per_req=input_tokens,
            requests=requests,
            concurrency=concurrency,
            warmup=warmup,
        )
        results.append(res)
        print(
            f"  {name}: {'PASS' if res.failed == 0 else 'FAIL'} "
            f"({res.success}/{res.requests} HTTP OK, "
            f"{res.wall_time_s:.2f}s, statuses: {format_status_counts(res)})"
        )
    return results


async def main_async(args: argparse.Namespace) -> int:
    """Coordinate benchmark preflight, scenarios, metrics, and reporting.

    Args:
        args: Parsed benchmark options.

    Returns:
        Zero when all scenarios pass, one on runtime failure, or two for
        invalid arguments.
    """
    arg_errors = validate_args(args)
    if arg_errors:
        print("Invalid arguments:")
        for error in arg_errors:
            print(f"  - {error}")
        return 2

    requested_tokens = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    known_scenarios = {
        "all",
        "embed_dense",
        "embed_sparse",
        "embed_sparse_indices",
        "embed_hybrid",
        "embed_full",
        "embed_colbert",
        "rerank",
        "mixed",
    }
    unknown = [s for s in requested_tokens if s not in known_scenarios]
    if not requested_tokens:
        print(f"No scenarios selected. Available: {sorted(known_scenarios)}")
        return 2
    if unknown:
        print(f"Unknown scenarios: {unknown}. Available: {sorted(known_scenarios)}")
        return 2

    batch_size_values = args.matrix_batch_size or [args.batch_size]
    concurrency_values = args.matrix_concurrency or [args.concurrency]
    input_token_values = args.matrix_input_tokens or [args.input_tokens]
    mixed_input_tokens = args.mixed_input_tokens
    matrix_workloads = [
        (target_tokens, batch_size)
        for target_tokens in input_token_values
        for batch_size in batch_size_values
    ]
    max_concurrency = max(concurrency_values)

    limits = httpx.Limits(
        max_connections=max_concurrency * 2,
        max_keepalive_connections=max_concurrency * 2,
    )
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    async with httpx.AsyncClient(headers=headers, limits=limits) as client:
        # Health check (once)
        try:
            r = await client.get(f"{args.url}/health", timeout=10.0)
            if r.status_code != 200:
                print(f"Health check failed: {r.status_code}")
                return 1
            info = r.json()
            print(section("Pre-flight health"))
            print(render_server_info(info))
            print()
        except Exception as e:
            print(f"Cannot reach {args.url}: {e}")
            return 1

        auth_error = await authentication_error(client, args.url, args.token)
        if auth_error:
            print(f"Authentication check failed: {auth_error}")
            return 1

        token_counter = None
        tokenizer_model = "disabled"
        tokenizer_max_length = 0
        if not args.no_token_count:
            tokenizer_model = args.tokenizer_model or str(
                info.get("dense_embedding_model") or info.get("model") or ""
            )
            tokenizer_max_length = (
                args.tokenizer_max_length or active_embedding_max_length(info)
            )
            if not tokenizer_model or tokenizer_max_length < 1:
                print(
                    "Exact token counting could not resolve the active tokenizer "
                    "model or max length. Use --tokenizer-model and "
                    "--tokenizer-max-length, or --no-token-count."
                )
                return 2
            try:
                token_counter = load_embed_token_counter(
                    tokenizer_model, tokenizer_max_length
                )
            except RuntimeError as exc:
                print(f"Exact token counting unavailable: {exc}")
                return 2

        if (any(input_token_values) or mixed_input_tokens) and token_counter is None:
            print("Controlled token workloads cannot be used with --no-token-count.")
            return 2
        if token_counter is not None:
            try:
                for target_tokens in input_token_values:
                    if target_tokens:
                        token_counter.texts_at_token_length(1, target_tokens)
                if mixed_input_tokens:
                    token_counter.texts_at_token_pattern(1, mixed_input_tokens)
            except ValueError as exc:
                print(f"Invalid controlled token workload: {exc}")
                return 2

        rerank_token_counter = None
        reranker_tokenizer_model = "disabled"
        reranker_tokenizer_max_length = 0
        needs_rerank = "all" in requested_tokens or any(
            name in {"rerank", "mixed"} for name in requested_tokens
        )
        if not args.no_token_count and needs_rerank:
            reranker_tokenizer_model = args.reranker_tokenizer_model or str(
                info.get("reranker_model") or ""
            )
            reranker_tokenizer_max_length = (
                args.reranker_tokenizer_max_length or active_reranker_max_length(info)
            )
            if not reranker_tokenizer_model or reranker_tokenizer_max_length < 1:
                print(
                    "Exact rerank token counting could not resolve the active "
                    "tokenizer model or max length. Use --reranker-tokenizer-model "
                    "and --reranker-tokenizer-max-length, or --no-token-count."
                )
                return 2
            try:
                rerank_base_counter = load_embed_token_counter(
                    reranker_tokenizer_model,
                    reranker_tokenizer_max_length,
                )
                if "qwen" in reranker_tokenizer_model.lower():
                    instruction = str(info.get("reranker_instruction") or "")
                    if not instruction:
                        raise ValueError(
                            "the Qwen server does not advertise reranker_instruction; "
                            "upgrade the server before exact rerank benchmarking"
                        )
                    rerank_token_counter = RerankTokenCounter(
                        base=rerank_base_counter,
                        instruction=instruction,
                        prefix_tokens=int(info.get("reranker_prefix_tokens") or 0),
                        suffix_tokens=int(info.get("reranker_suffix_tokens") or 0),
                    )
                else:
                    rerank_token_counter = rerank_base_counter
                targets = mixed_input_tokens or [
                    target for target in input_token_values if target
                ]
                for target in targets:
                    if isinstance(rerank_token_counter, RerankTokenCounter):
                        rerank_token_counter.passages_at_total_length(
                            1,
                            target,
                            "What is machine learning?",
                        )
                    else:
                        query_tokens = rerank_token_counter.count(
                            ["What is machine learning?"]
                        )
                        if target <= query_tokens:
                            raise ValueError(
                                f"rerank target {target} must exceed the fixed "
                                f"query's {query_tokens} useful tokens"
                            )
                        rerank_token_counter.texts_at_token_length(
                            1,
                            target - query_tokens,
                        )
            except (RuntimeError, ValueError) as exc:
                print(f"Invalid controlled rerank workload: {exc}")
                return 2

        include_colbert = bool(info.get("colbert_model_name"))

        if "all" in requested_tokens:
            requested = list(
                build_scenarios(
                    args.batch_size,
                    args.timeout,
                    include_colbert,
                    token_counter,
                    rerank_token_counter=rerank_token_counter,
                )
            )
        else:
            requested = requested_tokens

        print(section("Benchmark plan"))
        print(
            render_table(
                [
                    ["Target", args.url],
                    ["Requested", args.scenarios],
                    ["Selected", ", ".join(requested)],
                    [
                        "Input token targets",
                        (
                            "mixed: " + ", ".join(map(str, mixed_input_tokens))
                            if mixed_input_tokens
                            else ", ".join(
                                str(value) if value else "default"
                                for value in input_token_values
                            )
                        ),
                    ],
                    [
                        "Matrix batch sizes",
                        ", ".join(str(v) for v in batch_size_values),
                    ],
                    [
                        "Matrix concurrencies",
                        ", ".join(str(v) for v in concurrency_values),
                    ],
                    ["Requests/scenario", str(args.requests)],
                    ["Warmup", f"{args.warmup} requests per scenario"],
                    ["Timeout", f"{args.timeout:.1f}s per request"],
                    ["Token counting", "exact" if token_counter else "disabled"],
                    ["Tokenizer model", tokenizer_model],
                    [
                        "Tokenizer max length",
                        str(tokenizer_max_length) if token_counter else "-",
                    ],
                    ["Reranker tokenizer", reranker_tokenizer_model],
                    [
                        "Reranker tokenizer max",
                        (
                            str(reranker_tokenizer_max_length)
                            if rerank_token_counter
                            else "-"
                        ),
                    ],
                    ["Auth", "Bearer token provided" if args.token else "disabled"],
                ],
                ["Setting", "Value"],
            )
        )
        print()

        all_results: List[ScenarioResult] = []
        matrix_records: List[Dict[str, Any]] = []
        any_fail = False

        for target_tokens, bs in matrix_workloads:
            for conc in concurrency_values:
                scenarios = build_scenarios(
                    bs,
                    args.timeout,
                    include_colbert,
                    token_counter,
                    target_tokens=target_tokens,
                    mixed_target_tokens=mixed_input_tokens,
                    rerank_token_counter=rerank_token_counter,
                )
                unavailable = [s for s in requested if s not in scenarios]
                if unavailable:
                    print(
                        f"Unavailable scenarios for active profile: {unavailable}. "
                        f"Available: {list(scenarios)}"
                    )
                    return 2

                token_label = (
                    "mixed:" + "/".join(map(str, mixed_input_tokens))
                    if mixed_input_tokens
                    else str(target_tokens)
                    if target_tokens
                    else "default"
                )
                print(
                    f"=== input_tokens={token_label} batch_size={bs} "
                    f"concurrency={conc} ==="
                )

                pre_metrics = ""
                if not args.no_post_check:
                    try:
                        metrics_resp = await client.get(
                            f"{args.url}/metrics", timeout=10.0
                        )
                        if metrics_resp.status_code == 200:
                            pre_metrics = metrics_resp.text
                    except Exception:
                        pre_metrics = ""

                results = await run_selected_scenarios(
                    client=client,
                    base_url=args.url,
                    selected_names=requested,
                    scenario_map=scenarios,
                    requests=args.requests,
                    concurrency=conc,
                    warmup=args.warmup,
                    sleep_between=args.sleep_between,
                )
                all_results.extend(results)

                post_health = None
                post_metrics = ""
                if not args.no_post_check:
                    try:
                        post_health_resp = await client.get(
                            f"{args.url}/health", timeout=10.0
                        )
                        if post_health_resp.status_code == 200:
                            post_health = post_health_resp.json()
                    except Exception as e:
                        post_metrics += f"post-health-error={type(e).__name__}: {e}\n"
                    try:
                        metrics_resp = await client.get(
                            f"{args.url}/metrics", timeout=10.0
                        )
                        if metrics_resp.status_code == 200:
                            post_metrics += metrics_resp.text
                        else:
                            post_metrics += (
                                f"post-metrics-status={metrics_resp.status_code}\n"
                            )
                    except Exception as e:
                        post_metrics += f"post-metrics-error={type(e).__name__}: {e}\n"

                print()
                print(summarize(results, conc))
                print()
                print(render_verdict(results))
                failure_details = render_failure_details(results)
                if failure_details:
                    print()
                    print(failure_details)
                print()

                if not args.no_post_check:
                    print("Post-run health and key metric deltas:")
                    if post_health:
                        print(
                            f"  Health: {post_health.get('status', 'unknown')} "
                            f"profile={post_health.get('profile', 'unknown')}"
                        )
                    metric_lines = render_metric_delta(pre_metrics, post_metrics)
                    if metric_lines:
                        for line in metric_lines:
                            print(f"  {line}")
                    else:
                        print("  No key Prometheus metric delta found.")
                    print()

                matrix_records.append(
                    {
                        "input_tokens": (
                            list(mixed_input_tokens)
                            if mixed_input_tokens
                            else target_tokens or None
                        ),
                        "batch_size": bs,
                        "concurrency": conc,
                        "results": [
                            scenario_result_record(result) for result in results
                        ],
                        "metric_deltas": metric_delta_record(
                            pre_metrics,
                            post_metrics,
                        ),
                        "post_health_status": (
                            post_health.get("status") if post_health else None
                        ),
                    }
                )

                if any(r.failed > 0 for r in results):
                    any_fail = True

                if args.matrix_sleep > 0:
                    print(f"Matrix cooldown {args.matrix_sleep}s...")
                    await asyncio.sleep(args.matrix_sleep)

    if args.output_json:
        report = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "label": args.label,
            "server": info,
            "settings": {
                "url": args.url,
                "scenarios": requested,
                "input_token_targets": input_token_values,
                "mixed_input_tokens": mixed_input_tokens,
                "batch_sizes": batch_size_values,
                "concurrencies": concurrency_values,
                "requests_per_scenario": args.requests,
                "warmup": args.warmup,
                "timeout_s": args.timeout,
                "token_counting": not args.no_token_count,
                "tokenizer_model": tokenizer_model,
                "reranker_tokenizer_model": reranker_tokenizer_model,
                "auth_enabled": bool(args.token),
            },
            "runs": matrix_records,
            "verdict": "fail" if any_fail else "pass",
        }
        try:
            output_path = Path(args.output_json)
            write_json_report(output_path, report)
            print(f"JSON report: {output_path.resolve()}")
        except OSError as exc:
            print(f"Could not write JSON report: {exc}")
            return 1

    return 1 if any_fail else 0


def parse_args() -> argparse.Namespace:
    """Parse Smart Embedder benchmark command-line options."""
    p = argparse.ArgumentParser(description="Smart Embedder server benchmark")
    p.add_argument("--url", default="http://127.0.0.1:8010", help="Server base URL")
    p.add_argument(
        "--token",
        default=default_api_token(),
        help=(
            "Bearer token (defaults to the first entry's token in "
            "SMART_EMBEDDER_API_TOKENS, from the process environment or the "
            "repository root .env)"
        ),
    )
    p.add_argument(
        "--output-json",
        default="",
        help="Atomically save the comparable benchmark matrix as JSON",
    )
    p.add_argument(
        "--label",
        default="",
        help="Optional report label such as baseline, candidate, or canary",
    )
    p.add_argument(
        "--tokenizer-model",
        default="",
        help=(
            "Tokenizer override for exact input token counting (default: /health model)"
        ),
    )
    p.add_argument(
        "--tokenizer-max-length",
        type=int,
        default=0,
        help="Truncation limit override for token counting (default: /health value)",
    )
    p.add_argument(
        "--reranker-tokenizer-model",
        default="",
        help="Reranker tokenizer override (default: /health reranker_model)",
    )
    p.add_argument(
        "--reranker-tokenizer-max-length",
        type=int,
        default=0,
        help="Reranker token limit override (default: /health value)",
    )
    p.add_argument(
        "--no-token-count",
        action="store_true",
        help="Disable exact input token counting and omit Tok/s",
    )
    p.add_argument(
        "--input-tokens",
        type=int,
        default=0,
        help="Exact tokens per embedding input; 0 uses the default short corpus",
    )
    p.add_argument(
        "--concurrency", type=int, default=8, help="Concurrent in-flight requests"
    )
    p.add_argument("--requests", type=int, default=100, help="Requests per scenario")
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Sentences/passages per request (input batch)",
    )
    p.add_argument("--warmup", type=int, default=5, help="Warmup requests per scenario")
    p.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds",
    )
    p.add_argument(
        "--max-batch-size",
        type=int,
        default=128,
        help="Fail-fast guard for default payload item limits; set 0 to disable",
    )
    p.add_argument(
        "--scenarios",
        default="all",
        help=(
            "Comma-separated scenarios or 'all': embed_dense, embed_sparse, "
            "embed_sparse_indices, embed_hybrid, embed_full, embed_colbert, "
            "rerank, mixed"
        ),
    )
    p.add_argument(
        "--sleep-between",
        type=int,
        default=0,
        help=(
            "Seconds to sleep between scenarios (use ~65s if rate-limit is 100 req/min)"
        ),
    )
    p.add_argument(
        "--no-post-check",
        action="store_true",
        help="Skip post-run /health and /metrics diagnostics",
    )
    p.add_argument(
        "--matrix-batch-size",
        type=parse_positive_int_list,
        default=None,
        help="Comma-separated batch sizes for matrix runs (e.g. 4,8,16)",
    )
    p.add_argument(
        "--matrix-concurrency",
        type=parse_positive_int_list,
        default=None,
        help="Comma-separated concurrency levels for matrix runs (e.g. 1,2,4,8)",
    )
    p.add_argument(
        "--matrix-input-tokens",
        type=parse_positive_int_list,
        default=None,
        help="Comma-separated exact embedding input lengths (e.g. 32,128,512)",
    )
    p.add_argument(
        "--mixed-input-tokens",
        type=parse_positive_int_list,
        default=None,
        help=(
            "Cycle exact input lengths within each embedding request "
            "(e.g. 32,128,512,1024)"
        ),
    )
    p.add_argument(
        "--matrix-sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between matrix combinations",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        exit(asyncio.run(main_async(args)))
    except KeyboardInterrupt:
        print("\nInterrupted")
        exit(130)

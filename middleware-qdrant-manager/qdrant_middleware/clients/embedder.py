"""Async HTTP client for the external Smart Embedder service.

Wraps the ``/embeddings/`` and ``/rerank`` endpoints of the Smart Embedder service,
returning Qdrant-ready dense + sparse vectors and rerank scores. Transient
failures (connection errors, timeouts, retryable HTTP statuses) are retried via
:func:`qdrant_middleware.clients.retry.retry_async`; any unrecoverable upstream
failure is mapped to an HTTP 502 for the caller.

FastAPI routes use the per-app dependency in ``api.dependencies`` so
``create_app(Settings(...))`` controls the external client settings.
"""

import asyncio
import json
from dataclasses import dataclass
import httpx
from fastapi import HTTPException
from qdrant_client.models import SparseVector

from qdrant_middleware.clients.retry import (
    RETRYABLE_HTTP_STATUS,
    parse_retry_after,
    retry_async,
)
from qdrant_middleware.core.config import Settings
from qdrant_middleware.schemas.embedding_contract import (
    EmbedderContractInfo,
    EmbeddingContract,
)


class _EmbedderHTTPError(Exception):
    """Non-200 response from the Smart Embedder server, carrying the status and body.

    ``retry_after`` holds the raw ``Retry-After`` header when the response sent
    one. Smart Embedder sends it on exactly the two statuses it expects a
    caller to wait out — ``503 rerank_vram_pressure`` (``2``) and its rate-limit
    ``429`` (``60``) — so keeping it on the exception is what lets the retry
    policy wait the amount the server asked for instead of its own guess.
    """

    def __init__(
        self, status_code: int, text: str, retry_after: str | None = None
    ) -> None:
        super().__init__(f"Smart Embedder server error {status_code}: {text}")
        self.status_code = status_code
        self.text = text
        self.retry_after = retry_after


def _is_transient(exc: Exception) -> bool:
    """Return True for Smart Embedder errors worth retrying.

    Connection errors and timeouts are always transient; HTTP errors are
    transient only for the retryable status codes (429 and 5xx).
    """
    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
        return True
    return (
        isinstance(exc, _EmbedderHTTPError) and exc.status_code in RETRYABLE_HTTP_STATUS
    )


def _retry_after(exc: Exception) -> float | None:
    """Return the wait Smart Embedder asked for, when the response carried one.

    A connection error or timeout never has a header to read, and an HTTP
    error without one returns ``None`` here, so both fall back to the retry
    policy's own jittered backoff.
    """
    return parse_retry_after(getattr(exc, "retry_after", None))


# Upstream statuses whose structured body is meaningful to our callers and
# must survive instead of being flattened into a generic 502.
PASSTHROUGH_HTTP_STATUS = frozenset({413})


def _passthrough_detail(error: "_EmbedderHTTPError"):
    """Return the upstream JSON detail when it is structured, else None."""
    try:
        payload = json.loads(error.text)
    except (TypeError, ValueError):
        return None
    detail = payload.get("detail") if isinstance(payload, dict) else None
    return detail if isinstance(detail, dict) else None


@dataclass
class EmbeddingResult:
    """A single text's embeddings: a dense vector and a Qdrant sparse vector."""

    dense: list[float]
    sparse: SparseVector
    contract_fingerprint: str | None = None
    dense_model_name: str | None = None
    sparse_model_name: str | None = None


@dataclass
class RerankResult:
    """A rerank score for one passage, keyed by its position in the input list."""

    index: int
    score: float


class EmbedderClient:
    """Thin async wrapper around the Smart Embedder embedding/rerank HTTP API."""

    def __init__(
        self,
        settings: Settings,
        max_batch_size: int | None = None,
        rerank_batch_size: int | None = None,
    ) -> None:
        """Create the client and its underlying ``httpx.AsyncClient``.

        ``settings`` is the app's runtime settings.
        """
        max_batch_size = max_batch_size or settings.smart_embedder_max_batch_size
        rerank_batch_size = (
            rerank_batch_size or settings.smart_embedder_rerank_batch_size
        )
        if max_batch_size < 1 or max_batch_size > 128:
            raise ValueError("max_batch_size must be between 1 and 128")
        if rerank_batch_size < 1 or rerank_batch_size > 128:
            raise ValueError("rerank_batch_size must be between 1 and 128")
        self.max_batch_size = max_batch_size
        self.rerank_batch_size = rerank_batch_size
        self.embed_concurrency = settings.smart_embedder_embed_concurrency
        self.retry_max_attempts = settings.retry_max_attempts
        self.retry_base_delay = settings.retry_base_delay

        headers: dict[str, str] = {}
        if settings.smart_embedder_client_token_value:
            headers["Authorization"] = (
                f"Bearer {settings.smart_embedder_client_token_value}"
            )
        self._client = httpx.AsyncClient(
            base_url=str(settings.smart_embedder_url).rstrip("/"),
            headers=headers,
            timeout=settings.smart_embedder_timeout,
        )

    async def get_embedding_contract(self) -> EmbedderContractInfo:
        """Return the compatibility contract of the running embedder."""

        async def _do() -> httpx.Response:
            response = await self._client.get("/models")
            if response.status_code != 200:
                raise _EmbedderHTTPError(
                    response.status_code,
                    response.text,
                    response.headers.get("Retry-After"),
                )
            return response

        try:
            response = await retry_async(
                _do,
                attempts=self.retry_max_attempts,
                base_delay=self.retry_base_delay,
                should_retry=_is_transient,
                retry_after=_retry_after,
                description="Smart Embedder models",
            )
            data = response.json()
            return EmbedderContractInfo(
                contract=EmbeddingContract.model_validate(
                    data.get("embedding_contract")
                ),
                server_version=data.get("version"),
                reranker=data.get("reranker"),
            )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise HTTPException(
                503, f"Smart Embedder contract unavailable: {exc}"
            ) from exc
        except _EmbedderHTTPError as exc:
            raise HTTPException(503, str(exc)) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                503, f"Smart Embedder returned an invalid embedding contract: {exc}"
            ) from exc

    async def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed a list of texts, splitting into batches of ``max_batch_size``.

        Sub-batches run concurrently (bounded by ``embed_concurrency``) so the
        Smart Embedder server can coalesce them into fewer GPU inferences via its
        dynamic-batching window. Returns one :class:`EmbeddingResult` per input
        text, in order.
        """
        if not texts:
            return []
        batches = [
            texts[i : i + self.max_batch_size]
            for i in range(0, len(texts), self.max_batch_size)
        ]
        if len(batches) == 1:
            return await self._embed_batch(batches[0])
        semaphore = asyncio.Semaphore(self.embed_concurrency)

        async def _embed_with_sem(batch: list[str]) -> list[EmbeddingResult]:
            async with semaphore:
                return await self._embed_batch(batch)

        batch_results = await asyncio.gather(*(_embed_with_sem(b) for b in batches))
        results: list[EmbeddingResult] = []
        for batch_result in batch_results:
            results.extend(batch_result)
        return results

    async def rerank(
        self,
        query: str,
        passages: list[str],
        normalize: bool = False,
    ) -> list[RerankResult]:
        """Score ``passages`` against ``query`` with the cross-encoder reranker.

        Returns a :class:`RerankResult` per passage (with its original index).
        An empty ``passages`` list short-circuits to an empty result. Upstream
        failures are raised as HTTP 502.
        """
        if not passages:
            return []

        results: list[RerankResult] = []
        for batch_start in range(0, len(passages), self.rerank_batch_size):
            batch = passages[batch_start : batch_start + self.rerank_batch_size]

            async def _do() -> httpx.Response:
                response = await self._client.post(
                    "/rerank",
                    json={"query": query, "passages": batch, "normalize": normalize},
                )
                if response.status_code != 200:
                    raise _EmbedderHTTPError(
                        response.status_code,
                        response.text,
                        response.headers.get("Retry-After"),
                    )
                return response

            try:
                response = await retry_async(
                    _do,
                    attempts=self.retry_max_attempts,
                    base_delay=self.retry_base_delay,
                    should_retry=_is_transient,
                    retry_after=_retry_after,
                    description="Smart Embedder rerank",
                )
            except httpx.ConnectError as e:
                raise HTTPException(
                    502, f"Smart Embedder server unreachable: {e}"
                ) from e
            except httpx.TimeoutException as e:
                raise HTTPException(502, f"Smart Embedder server timeout: {e}") from e
            except _EmbedderHTTPError as e:
                if e.status_code in PASSTHROUGH_HTTP_STATUS:
                    detail = _passthrough_detail(e)
                    if detail is not None:
                        raise HTTPException(e.status_code, detail) from None
                raise HTTPException(502, str(e)) from e

            data = response.json()
            results.extend(
                RerankResult(
                    index=batch_start + int(item["index"]), score=float(item["score"])
                )
                for item in data["results"]
            )

        return sorted(results, key=lambda item: item.score, reverse=True)

    async def _embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed a single batch via one POST to ``/embeddings/``.

        Requests dense + sparse outputs (sparse as indices/values) and raises
        HTTP 502 on any unrecoverable upstream failure.
        """

        async def _do() -> httpx.Response:
            response = await self._client.post(
                "/embeddings/",
                json={
                    "sentences": texts,
                    "return_dense": True,
                    "return_sparse": True,
                    "return_colbert": False,
                    "sparse_as_indices": True,
                },
            )
            if response.status_code != 200:
                raise _EmbedderHTTPError(
                    response.status_code,
                    response.text,
                    response.headers.get("Retry-After"),
                )
            return response

        try:
            response = await retry_async(
                _do,
                attempts=self.retry_max_attempts,
                base_delay=self.retry_base_delay,
                should_retry=_is_transient,
                retry_after=_retry_after,
                description="Smart Embedder embed",
            )
        except httpx.ConnectError as e:
            raise HTTPException(502, f"Smart Embedder server unreachable: {e}") from e
        except httpx.TimeoutException as e:
            raise HTTPException(502, f"Smart Embedder server timeout: {e}") from e
        except _EmbedderHTTPError as e:
            raise HTTPException(502, str(e)) from e

        data = response.json()
        fingerprint = data.get("embedding_contract_fingerprint")
        dense_model = data.get("dense_model_name")
        sparse_model = data.get("sparse_model_name")
        results: list[EmbeddingResult] = []
        for item in data["data"]:
            emb = item["embeddings"]
            sparse_raw = emb["sparse"]
            results.append(
                EmbeddingResult(
                    dense=emb["dense"],
                    sparse=SparseVector(
                        indices=sparse_raw["indices"],
                        values=sparse_raw["values"],
                    ),
                    contract_fingerprint=fingerprint,
                    dense_model_name=dense_model,
                    sparse_model_name=sparse_model,
                )
            )
        return results

    async def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._client.aclose()

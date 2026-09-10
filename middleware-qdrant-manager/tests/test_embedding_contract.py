"""Compatibility tests for Smart Embedder collection metadata."""

from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import SparseVector

from qdrant_middleware.api.routers.collections import (
    CollectionOptions,
    create_collection_endpoint,
)
from qdrant_middleware.clients.embedder import EmbedderClient, EmbeddingResult
from qdrant_middleware.core.config import Settings
from qdrant_middleware.repositories import qdrant_collections
from qdrant_middleware.schemas.collections import CollectionSchema
from qdrant_middleware.schemas.embedding_contract import (
    EmbedderContractInfo,
    EmbeddingContract,
)
from qdrant_middleware.services.embedding_guard import (
    guard_collection_mutation,
    guard_vector_operation,
    response_embedding_fingerprint,
)


def _settings() -> Settings:
    return Settings(
        QDRANT_API_HOST="127.0.0.1",
        MIDDLEWARE_API_TOKENS="",
        MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL=True,
        RETRY_MAX_ATTEMPTS=1,
        RETRY_BASE_DELAY=0,
    )


def _contract(digest: str = "a", profile: str = "basic") -> EmbeddingContract:
    return EmbeddingContract(
        schema_version=1,
        profile=profile,
        dense={
            "model": "BAAI/bge-m3",
            "revision": "dense-revision",
            "dimension": 1024,
            "normalization": "qdrant_cosine",
        },
        sparse={
            "model": "BAAI/bge-m3",
            "revision": "sparse-revision",
            "encoding": "indices_values",
        },
        fingerprint=f"sha256:{digest * 64}",
    )


def _contract_info(digest: str = "a") -> EmbedderContractInfo:
    return EmbedderContractInfo(
        contract=_contract(digest),
        server_version="2.1.0",
        reranker="BAAI/bge-reranker-v2-m3",
    )


def _config(contract: EmbeddingContract | None = None, *, malformed=False):
    if malformed:
        metadata = {"hybrid_manager": {"embedding_contract": {"broken": True}}}
    elif contract is None:
        metadata = None
    else:
        metadata = {
            "hybrid_manager": {
                "metadata_schema": 1,
                "embedding_contract": contract.model_dump(mode="json"),
            }
        }
    return SimpleNamespace(metadata=metadata)


@pytest.mark.parametrize(
    ("config", "active", "expected"),
    [
        (_config(_contract()), _contract(), "compatible"),
        (_config(), _contract(), "incompatible"),
        (_config(_contract("a")), _contract("b"), "incompatible"),
        (_config(_contract()), None, "incompatible"),
        (_config(malformed=True), _contract(), "incompatible"),
    ],
)
def test_embedding_compatibility_states(config, active, expected):
    result = qdrant_collections.embedding_compatibility(config, active)

    assert result.status == expected


class _MetadataClient:
    """Fake Qdrant client backing collection metadata reads and writes."""

    def __init__(self, config=None, count=0):
        self.config = config or _config()
        self.point_count = count
        self.update_kwargs = None
        self.create_kwargs = None

    async def get_collections(self):
        return SimpleNamespace(collections=[])

    async def get_collection(self, collection_name):
        return SimpleNamespace(config=self.config)

    async def count(self, **kwargs):
        assert kwargs["exact"] is True
        return SimpleNamespace(count=self.point_count)

    async def update_collection(self, **kwargs):
        self.update_kwargs = kwargs
        return True

    async def create_collection(self, **kwargs):
        self.create_kwargs = kwargs
        return True

    async def create_payload_index(self, **kwargs):
        return True


@pytest.mark.asyncio
async def test_create_collection_persists_active_contract_metadata():
    client = _MetadataClient()

    await qdrant_collections.create_collection(
        client,
        "docs",
        schema=CollectionSchema.from_settings(_settings()),
        settings=_settings(),
        embedding_contract=_contract_info(),
    )

    owned = client.create_kwargs["metadata"]["hybrid_manager"]
    assert owned["metadata_schema"] == 2
    assert owned["active_vectors"] == {"dense": "dense", "sparse": "sparse"}
    assert owned["embedding_contract"]["profile"] == "basic"
    assert owned["embedding_contract"]["fingerprint"] == _contract().fingerprint
    assert owned["smart_embedder_version"] == "2.1.0"
    assert owned["created_at"].endswith("+00:00")


def test_schema_v1_metadata_uses_configured_vector_names():
    vectors, malformed = qdrant_collections.stored_active_vectors(
        _config(_contract()),
        _settings(),
    )

    assert not malformed
    assert vectors.dense == "dense"
    assert vectors.sparse == "sparse"


def test_schema_v2_metadata_uses_persisted_vector_generation():
    config = _config(_contract())
    config.metadata["hybrid_manager"]["metadata_schema"] = 2
    config.metadata["hybrid_manager"]["active_vectors"] = {
        "dense": "dense__generation",
        "sparse": "sparse__generation",
    }

    vectors, malformed = qdrant_collections.stored_active_vectors(
        config,
        _settings(),
    )

    assert not malformed
    assert vectors.dense == "dense__generation"
    assert vectors.sparse == "sparse__generation"


def test_invalid_active_vector_names_mark_metadata_invalid():
    config = _config(_contract())
    config.metadata["hybrid_manager"]["active_vectors"] = {
        "dense": "same",
        "sparse": "same",
    }

    result = qdrant_collections.embedding_compatibility(
        config,
        _contract(),
        _settings(),
    )

    assert result.status == "incompatible"


@pytest.mark.asyncio
async def test_fingerprint_guard_accepts_match_and_rejects_mismatch():
    client = _MetadataClient(config=_config(_contract()))

    await qdrant_collections.require_embedding_fingerprint(
        client, "docs", _contract().fingerprint, settings=_settings()
    )
    with pytest.raises(qdrant_collections.EmbeddingContractViolation) as exc:
        await qdrant_collections.require_embedding_fingerprint(
            client, "docs", _contract("b").fingerprint, settings=_settings()
        )

    assert exc.value.status == "incompatible"


def _embedding(fingerprint: str | None) -> EmbeddingResult:
    return EmbeddingResult(
        dense=[0.1],
        sparse=SparseVector(indices=[1], values=[0.2]),
        contract_fingerprint=fingerprint,
    )


def test_response_fingerprint_rejects_mixed_or_partial_batches():
    with pytest.raises(HTTPException) as mixed:
        response_embedding_fingerprint(
            [
                _embedding(_contract().fingerprint),
                _embedding(_contract("b").fingerprint),
            ]
        )
    with pytest.raises(HTTPException) as partial:
        response_embedding_fingerprint(
            [_embedding(_contract().fingerprint), _embedding(None)]
        )

    assert mixed.value.status_code == 502
    assert partial.value.status_code == 502


@pytest.mark.asyncio
async def test_vector_guard_returns_structured_conflict():
    client = _MetadataClient(config=_config(_contract()))

    with pytest.raises(HTTPException) as exc:
        await guard_vector_operation(
            client,
            "docs",
            [_embedding(_contract("b").fingerprint)],
            _settings(),
        )

    assert exc.value.status_code == 409
    assert exc.value.detail["code"] == "embedding_contract_mismatch"
    assert exc.value.detail["status"] == "incompatible"


class _MissingCollectionClient:
    """Fake Qdrant client whose collection does not exist."""

    async def get_collection(self, collection_name):
        raise UnexpectedResponse(
            status_code=404,
            reason_phrase="Not Found",
            content=b"Not found",
            headers={},
        )


@pytest.mark.asyncio
async def test_vector_guard_reports_missing_collection_as_404_not_502():
    """A search against a nonexistent collection is a 404, not an infra 502.

    guard_vector_operation's catch-all previously wrapped
    PayloadIndexCollectionNotFound -- raised by _get_collection_info for
    exactly this case -- into a 502 "embedding contract verification failed",
    indistinguishable from a real Qdrant outage. qdrant_get_collection already
    reports the same situation as a clean 404; this call site should match.
    """
    client = _MissingCollectionClient()

    with pytest.raises(HTTPException) as exc:
        await guard_vector_operation(
            client,
            "does-not-exist",
            [_embedding(_contract().fingerprint)],
            _settings(),
        )

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_mutation_guard_reports_missing_collection_as_404_not_502():
    client = _MissingCollectionClient()

    with pytest.raises(HTTPException) as exc:
        await guard_collection_mutation(client, "does-not-exist", _settings())

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_embedder_client_reads_contract_and_tags_embedding_response():
    contract = _contract().model_dump(mode="json")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/models":
            return httpx.Response(
                200,
                json={
                    "version": "2.1.0",
                    "embedding_contract": contract,
                    "reranker": "reranker",
                },
            )
        return httpx.Response(
            200,
            json={
                "embedding_contract_fingerprint": contract["fingerprint"],
                "dense_model_name": "dense-model",
                "sparse_model_name": "sparse-model",
                "data": [
                    {
                        "embeddings": {
                            "dense": [0.1, 0.2],
                            "sparse": {"indices": [7], "values": [0.9]},
                        }
                    }
                ],
            },
        )

    client = EmbedderClient(_settings())
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url="http://embedder", transport=httpx.MockTransport(handler)
    )
    try:
        info = await client.get_embedding_contract()
        embeddings = await client.embed(["hello"])
    finally:
        await client.close()

    assert info.contract == _contract()
    assert info.server_version == "2.1.0"
    assert embeddings[0].contract_fingerprint == _contract().fingerprint
    assert embeddings[0].dense_model_name == "dense-model"
    assert embeddings[0].sparse_model_name == "sparse-model"


@pytest.mark.asyncio
async def test_create_endpoint_preserves_embedder_contract_http_status(
    mutation_context,
):
    class UnavailableEmbedder:
        """Embedder stand-in whose contract lookup always fails."""

        async def get_embedding_contract(self):
            raise HTTPException(503, "contract unavailable")

    request = CollectionOptions(
        collection_name="docs",
        dense_on_disk=False,
        quantization=False,
        sparse_idf=True,
        sparse_on_disk=False,
    )

    with pytest.raises(HTTPException) as exc:
        await create_collection_endpoint(
            request,
            client=SimpleNamespace(),
            settings=_settings(),
            embedder=UnavailableEmbedder(),
            **mutation_context,
        )

    assert exc.value.status_code == 503

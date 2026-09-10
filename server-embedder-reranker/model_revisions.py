"""Immutable Hugging Face revisions used by Smart Embedder."""

from types import MappingProxyType
from typing import Mapping


MODEL_REVISIONS: Mapping[str, str] = MappingProxyType(
    {
        "BAAI/bge-m3": "5617a9f61b028005a4858fdac845db406aefb181",
        "BAAI/bge-m3-unsupervised": ("46f03bc86361cf88102b0b517b36c8259f2946b1"),
        "BAAI/bge-reranker-v2-m3": ("953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"),
        "Qwen/Qwen3-Embedding-0.6B": ("97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"),
        "Qwen/Qwen3-Reranker-0.6B": ("e61197ed45024b0ed8a2d74b80b4d909f1255473"),
        "omai-research/milco-650m": ("567aeb03a5756c1e42793b9113925b4945ed5767"),
    }
)


def pinned_model_id(model_id: str) -> str:
    """Return a stable identifier suitable for fingerprints and diagnostics."""
    revision = MODEL_REVISIONS.get(model_id)
    return f"{model_id}@{revision}" if revision else model_id

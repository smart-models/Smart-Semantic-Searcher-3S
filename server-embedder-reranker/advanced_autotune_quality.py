"""Versioned quality gates for ADVANCED autotune candidates.

``evaluate_dense()``, ``evaluate_sparse()``, and ``evaluate_reranker()`` compare
candidate captures with a fixed multilingual retrieval corpus.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Mapping, Sequence, Tuple, TypeVar

import numpy as np

QUALITY_SUITE_VERSION = "advanced-retrieval-v1"


@dataclass(frozen=True)
class QualityThresholds:
    """Numerical and retrieval thresholds enforced by the quality gate."""

    dense_min_cosine: float = 0.999
    dense_min_norm_ratio: float = 0.995
    dense_max_norm_ratio: float = 1.005
    dense_min_top_k_overlap: float = 0.95
    sparse_min_weighted_overlap: float = 0.99
    sparse_max_dot_relative_error: float = 0.01
    sparse_min_top_k_overlap: float = 0.95
    rerank_min_score_correlation: float = 0.999
    rerank_max_score_abs_error: float = 0.005
    rerank_min_top_k_overlap: float = 1.0
    max_mrr_drop: float = 0.005
    max_ndcg_drop: float = 0.005


THRESHOLDS = QualityThresholds()


@dataclass(frozen=True)
class QualityDocument:
    """Document fixture used by the versioned retrieval suite."""

    id: str
    text: str


@dataclass(frozen=True)
class QualityQuery:
    """Query fixture and its relevant document identifiers."""

    id: str
    text: str
    relevant_document_ids: Tuple[str, ...]


DOCUMENTS = (
    QualityDocument(
        "lodz",
        "Łódź is a Polish city known for textile factories and industrial heritage.",
    ),
    QualityDocument(
        "tokyo",
        "東京 is the capital of Japan and one of the world's largest "
        "metropolitan areas.",
    ),
    QualityDocument(
        "vector",
        "I database vettoriali combinano similarità semantica e segnali lessicali.",
    ),
    QualityDocument(
        "milco",
        "MILCO maps multilingual queries into a shared English lexical sparse space.",
    ),
    QualityDocument(
        "rare",
        "The protein BRCA1 and product code ZX-81 are rare non-translatable entities.",
    ),
    QualityDocument(
        "weather",
        "A weather forecast estimates temperature, wind, and rain for tomorrow.",
    ),
)

QUERIES = (
    QualityQuery(
        "q_lodz_it", "Quale città polacca è famosa per le fabbriche tessili?", ("lodz",)
    ),
    QualityQuery("q_tokyo_en", "What is the capital city of Japan?", ("tokyo",)),
    QualityQuery(
        "q_vector_en",
        "Which storage combines semantic and lexical search?",
        ("vector",),
    ),
    QualityQuery(
        "q_milco_it",
        "Quale modello usa uno spazio lessicale inglese condiviso?",
        ("milco",),
    ),
    QualityQuery("q_rare", "Find BRCA1 and ZX-81 exactly.", ("rare",)),
)


@dataclass(frozen=True)
class QualityResult:
    """Serializable outcome of one ADVANCED quality gate."""

    passed: bool
    metrics: Dict[str, float]
    failures: Tuple[str, ...]
    suite_version: str = QUALITY_SUITE_VERSION

    def to_dict(self) -> Dict[str, object]:
        """Return the result as plain serializable values."""
        return asdict(self)


Vector = TypeVar("Vector")


def _retrieval_scores(
    vectors: Mapping[str, Vector],
    queries: Sequence[QualityQuery],
    documents: Sequence[QualityDocument],
    similarity: Callable[[Vector, Vector], float],
) -> Dict[str, Dict[str, float]]:
    return {
        query.id: {
            document.id: similarity(vectors[query.id], vectors[document.id])
            for document in documents
        }
        for query in queries
    }


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if denominator == 0.0:
        return 1.0 if np.array_equal(left_array, right_array) else 0.0
    return float(np.dot(left_array, right_array) / denominator)


def _rank_metrics(
    scores: Mapping[str, float], relevant_ids: Sequence[str], k: int
) -> Tuple[float, float, Tuple[str, ...]]:
    ranked = tuple(
        doc_id
        for doc_id, _score in sorted(
            scores.items(), key=lambda item: (-item[1], item[0])
        )[:k]
    )
    relevant = set(relevant_ids)
    reciprocal_rank = next(
        (
            1.0 / rank
            for rank, doc_id in enumerate(ranked, start=1)
            if doc_id in relevant
        ),
        0.0,
    )
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked, start=1)
        if doc_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return reciprocal_rank, dcg / ideal_dcg if ideal_dcg else 0.0, ranked


def _retrieval_summary(
    reference_scores: Mapping[str, Mapping[str, float]],
    candidate_scores: Mapping[str, Mapping[str, float]],
    queries: Sequence[QualityQuery],
    k: int,
) -> Dict[str, float]:
    reference_mrr = []
    candidate_mrr = []
    reference_ndcg = []
    candidate_ndcg = []
    overlaps = []
    for query in queries:
        ref_rr, ref_ndcg, ref_rank = _rank_metrics(
            reference_scores[query.id], query.relevant_document_ids, k
        )
        cand_rr, cand_ndcg, cand_rank = _rank_metrics(
            candidate_scores[query.id], query.relevant_document_ids, k
        )
        reference_mrr.append(ref_rr)
        candidate_mrr.append(cand_rr)
        reference_ndcg.append(ref_ndcg)
        candidate_ndcg.append(cand_ndcg)
        overlaps.append(len(set(ref_rank) & set(cand_rank)) / max(1, len(ref_rank)))
    return {
        "reference_mrr": float(np.mean(reference_mrr)),
        "candidate_mrr": float(np.mean(candidate_mrr)),
        "reference_ndcg": float(np.mean(reference_ndcg)),
        "candidate_ndcg": float(np.mean(candidate_ndcg)),
        "top_k_overlap": float(np.mean(overlaps)),
    }


def _threshold_failures(
    metrics: Mapping[str, float], top_k_minimum: float
) -> List[str]:
    failures = []
    if metrics["top_k_overlap"] < top_k_minimum:
        failures.append("top-k neighbor overlap below threshold")
    if metrics["reference_mrr"] - metrics["candidate_mrr"] > THRESHOLDS.max_mrr_drop:
        failures.append("MRR regression exceeds threshold")
    if metrics["reference_ndcg"] - metrics["candidate_ndcg"] > THRESHOLDS.max_ndcg_drop:
        failures.append("nDCG regression exceeds threshold")
    return failures


def evaluate_dense(
    reference: Mapping[str, Sequence[float]],
    candidate: Mapping[str, Sequence[float]],
    queries: Sequence[QualityQuery] = QUERIES,
    documents: Sequence[QualityDocument] = DOCUMENTS,
    expected_dimension: int = 1024,
    k: int = 3,
) -> QualityResult:
    """Compare dense outputs against shape, numerical, and retrieval gates."""
    failures = []
    expected_ids = {item.id for item in (*queries, *documents)}
    if set(reference) != expected_ids or set(candidate) != expected_ids:
        return QualityResult(False, {}, ("dense output IDs do not match suite",))

    cosines = []
    norm_ratios = []
    for item_id in sorted(expected_ids):
        ref = np.asarray(reference[item_id], dtype=np.float64)
        cand = np.asarray(candidate[item_id], dtype=np.float64)
        if ref.shape != (expected_dimension,) or cand.shape != (expected_dimension,):
            failures.append(f"dense shape mismatch for {item_id}")
            continue
        if not np.isfinite(ref).all() or not np.isfinite(cand).all():
            failures.append(f"non-finite dense value for {item_id}")
            continue
        cosines.append(_cosine(ref, cand))
        reference_norm = float(np.linalg.norm(ref))
        candidate_norm = float(np.linalg.norm(cand))
        norm_ratios.append(candidate_norm / reference_norm if reference_norm else 1.0)

    if failures:
        return QualityResult(False, {}, tuple(failures))

    metrics = _retrieval_summary(
        _retrieval_scores(reference, queries, documents, _cosine),
        _retrieval_scores(candidate, queries, documents, _cosine),
        queries,
        k,
    )
    metrics.update(
        {
            "min_cosine": min(cosines),
            "min_norm_ratio": min(norm_ratios),
            "max_norm_ratio": max(norm_ratios),
        }
    )
    if metrics["min_cosine"] < THRESHOLDS.dense_min_cosine:
        failures.append("dense cosine below threshold")
    if metrics["min_norm_ratio"] < THRESHOLDS.dense_min_norm_ratio:
        failures.append("dense norm ratio below threshold")
    if metrics["max_norm_ratio"] > THRESHOLDS.dense_max_norm_ratio:
        failures.append("dense norm ratio above threshold")
    failures.extend(_threshold_failures(metrics, THRESHOLDS.dense_min_top_k_overlap))
    return QualityResult(not failures, metrics, tuple(failures))


def _normalise_sparse(values: Mapping[str, float]) -> Dict[int, float]:
    return {int(index): float(weight) for index, weight in values.items()}


def _sparse_dot(left: Mapping[int, float], right: Mapping[int, float]) -> float:
    smaller, larger = (left, right) if len(left) <= len(right) else (right, left)
    return sum(weight * larger.get(index, 0.0) for index, weight in smaller.items())


def _weighted_overlap(left: Mapping[int, float], right: Mapping[int, float]) -> float:
    keys = set(left) | set(right)
    denominator = sum(max(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    if denominator == 0.0:
        return 1.0
    numerator = sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    return numerator / denominator


def evaluate_sparse(
    reference: Mapping[str, Mapping[str, float]],
    candidate: Mapping[str, Mapping[str, float]],
    *,
    reference_source_view: bool,
    candidate_source_view: bool,
    reference_top_k: int,
    candidate_top_k: int,
    source_vocab_start: int = 0,
    queries: Sequence[QualityQuery] = QUERIES,
    documents: Sequence[QualityDocument] = DOCUMENTS,
    k: int = 3,
) -> QualityResult:
    """Compare sparse outputs while preserving MILCO retrieval semantics."""
    failures = []
    expected_ids = {item.id for item in (*queries, *documents)}
    if set(reference) != expected_ids or set(candidate) != expected_ids:
        return QualityResult(False, {}, ("sparse output IDs do not match suite",))
    if reference_source_view != candidate_source_view:
        failures.append("MILCO LexEcho source_view changed")
    if reference_top_k != candidate_top_k:
        failures.append("MILCO top_k changed")

    ref = {key: _normalise_sparse(value) for key, value in reference.items()}
    cand = {key: _normalise_sparse(value) for key, value in candidate.items()}
    if (
        candidate_source_view
        and source_vocab_start > 0
        and not any(
            index >= source_vocab_start for values in cand.values() for index in values
        )
    ):
        failures.append("MILCO LexEcho produced no source-vocabulary terms")
    overlaps = []
    for item_id in sorted(expected_ids):
        for index, weight in cand[item_id].items():
            if index < 0 or not math.isfinite(weight) or weight <= 0.0:
                failures.append(f"invalid sparse entry for {item_id}")
                break
        overlaps.append(_weighted_overlap(ref[item_id], cand[item_id]))

    reference_scores = _retrieval_scores(ref, queries, documents, _sparse_dot)
    candidate_scores = _retrieval_scores(cand, queries, documents, _sparse_dot)
    relative_errors = []
    for query in queries:
        for document in documents:
            baseline = reference_scores[query.id][document.id]
            delta = abs(candidate_scores[query.id][document.id] - baseline)
            relative_errors.append(delta / max(abs(baseline), 1e-12))

    metrics = _retrieval_summary(reference_scores, candidate_scores, queries, k)
    metrics.update(
        {
            "min_weighted_overlap": min(overlaps),
            "max_dot_relative_error": max(relative_errors),
        }
    )
    if metrics["min_weighted_overlap"] < THRESHOLDS.sparse_min_weighted_overlap:
        failures.append("sparse weighted overlap below threshold")
    if metrics["max_dot_relative_error"] > THRESHOLDS.sparse_max_dot_relative_error:
        failures.append("sparse dot-product error exceeds threshold")
    failures.extend(_threshold_failures(metrics, THRESHOLDS.sparse_min_top_k_overlap))
    return QualityResult(not failures, metrics, tuple(dict.fromkeys(failures)))


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if np.allclose(left_array, right_array):
        return 1.0
    if np.std(left_array) == 0.0 or np.std(right_array) == 0.0:
        return 0.0
    return float(np.corrcoef(left_array, right_array)[0, 1])


def evaluate_reranker(
    reference: Mapping[str, Mapping[str, float]],
    candidate: Mapping[str, Mapping[str, float]],
    queries: Sequence[QualityQuery] = QUERIES,
    k: int = 3,
) -> QualityResult:
    """Compare reranker scores, ordering, and retrieval metrics."""
    failures = []
    expected_queries = {query.id for query in queries}
    if set(reference) != expected_queries or set(candidate) != expected_queries:
        return QualityResult(False, {}, ("reranker query IDs do not match suite",))
    reference_values = []
    candidate_values = []
    max_abs_error = 0.0
    for query in queries:
        if set(reference[query.id]) != set(candidate[query.id]):
            failures.append(f"reranker document IDs differ for {query.id}")
            continue
        for document_id in sorted(reference[query.id]):
            ref_score = float(reference[query.id][document_id])
            cand_score = float(candidate[query.id][document_id])
            if not math.isfinite(cand_score) or not 0.0 <= cand_score <= 1.0:
                failures.append(f"invalid reranker score for {query.id}/{document_id}")
            reference_values.append(ref_score)
            candidate_values.append(cand_score)
            max_abs_error = max(max_abs_error, abs(ref_score - cand_score))
    if failures:
        return QualityResult(False, {}, tuple(failures))

    metrics = _retrieval_summary(reference, candidate, queries, k)
    metrics.update(
        {
            "score_correlation": _correlation(reference_values, candidate_values),
            "max_score_abs_error": max_abs_error,
        }
    )
    if metrics["score_correlation"] < THRESHOLDS.rerank_min_score_correlation:
        failures.append("reranker score correlation below threshold")
    if metrics["max_score_abs_error"] > THRESHOLDS.rerank_max_score_abs_error:
        failures.append("reranker score error exceeds threshold")
    failures.extend(_threshold_failures(metrics, THRESHOLDS.rerank_min_top_k_overlap))
    return QualityResult(not failures, metrics, tuple(failures))

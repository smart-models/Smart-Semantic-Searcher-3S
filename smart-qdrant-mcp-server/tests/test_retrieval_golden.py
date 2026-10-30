"""The retrieval golden set must point at real chunks, and its scoring must hold."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

EVALS = Path(__file__).parent.parent / "evals"
CORPUS = {
    chunk["chunk_id"]: chunk
    for chunk in json.loads(
        (EVALS / "corpus" / "mcp_eval_corpus.json").read_text(encoding="utf-8")
    )
}
CATEGORIES = {"paraphrase", "identifier", "cross_vocabulary", "cross_lingual"}


@pytest.fixture
def retrieval_eval(monkeypatch):
    """Import evals/retrieval_eval.py, a script that imports its seed sibling."""
    monkeypatch.syspath_prepend(str(EVALS))
    spec = importlib.util.spec_from_file_location(
        "retrieval_eval", EVALS / "retrieval_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def golden():
    """The golden queries as stored."""
    return json.loads((EVALS / "retrieval_golden.json").read_text(encoding="utf-8"))


def test_every_relevant_chunk_exists_in_the_corpus(golden):
    missing = {
        (q["id"], cid) for q in golden for cid in q["relevant"] if cid not in CORPUS
    }
    assert not missing


def test_queries_are_well_formed(golden):
    assert len({q["id"] for q in golden}) == len(golden)
    assert {q["category"] for q in golden} == CATEGORIES
    for q in golden:
        assert q["query"].strip(), q["id"]
        assert q["relevant"] and len(set(q["relevant"])) == len(q["relevant"])


def test_identifier_queries_appear_verbatim_in_a_relevant_chunk(golden):
    """An identifier query must test the sparse leg on a literal token."""
    for q in golden:
        if q["category"] == "identifier":
            texts = [CORPUS[cid]["text"] for cid in q["relevant"]]
            assert any(q["query"] in text for text in texts), q["id"]


def test_score_on_a_hand_computed_ranking(retrieval_eval):
    scores = retrieval_eval.score(["a", "x", "b"], {"a", "b"})
    assert scores["hit@1"] == 1.0
    assert scores["recall@5"] == scores["recall@10"] == 1.0
    assert scores["mrr"] == 1.0
    ideal = 1 + 1 / math.log2(3)
    assert scores["ndcg@10"] == pytest.approx((1 + 1 / math.log2(4)) / ideal)


def test_score_when_nothing_relevant_is_retrieved(retrieval_eval):
    scores = retrieval_eval.score(["x", "y"], {"a"})
    assert set(scores.values()) == {0.0}


def test_regressions_flag_only_drops_beyond_the_tolerance(retrieval_eval):
    tolerance = retrieval_eval.TOLERANCE
    baseline = {"rerank": {"all": {"mrr": 0.8, "hit@1": 0.7}}}
    report = {
        "rerank": {"all": {"mrr": 0.8 - tolerance / 2, "hit@1": 0.7 - 2 * tolerance}}
    }
    failed = retrieval_eval.regressions(report, baseline)
    assert len(failed) == 1 and "hit@1" in failed[0]

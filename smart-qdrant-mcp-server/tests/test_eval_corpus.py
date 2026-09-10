"""The evaluation corpus must satisfy the properties the questions rely on."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest

CORPUS = Path(__file__).parent.parent / "evals" / "corpus" / "mcp_eval_corpus.json"
DOC_TYPES = {"release_note", "incident_report", "design_note", "runbook"}

# ir-0602 (Incident Report: Ingestion Backpressure Failure) and rb-0602
# (Runbook: Ingestion Pipeline Backpressure Recovery) are the corpus's
# designated "same outage, different vocabulary" pair -- see the corpus
# design rules. This is the one pair the tests hold to that property.
_OUTAGE_INCIDENT_IDS = ["ir-0602-01", "ir-0602-02", "ir-0602-03", "ir-0602-04"]
_OUTAGE_RUNBOOK_IDS = ["rb-0602-01", "rb-0602-02", "rb-0602-03"]

# Function words excluded from phrase detection: connective glue like "and
# the" or "on its own" is never what a lexical retriever keys on, so it must
# not count as a shared "distinctive" phrase between the outage pair.
_FUNCTION_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "with",
    "by",
    "is",
    "are",
    "was",
    "its",
    "it",
    "this",
    "that",
    "own",
}


@pytest.fixture(scope="module")
def chunks() -> list[dict]:
    return json.loads(CORPUS.read_text(encoding="utf-8"))


def test_corpus_size_is_in_range(chunks):
    assert 60 <= len(chunks) <= 100


def test_chunk_ids_are_unique(chunks):
    ids = [chunk["chunk_id"] for chunk in chunks]
    assert len(set(ids)) == len(ids)


def test_every_chunk_has_the_required_fields(chunks):
    required = {
        "chunk_id",
        "text",
        "doc_title",
        "doc_type",
        "author",
        "published_on",
        "component",
        "version",
        "language",
    }
    for chunk in chunks:
        assert required <= set(chunk), chunk["chunk_id"]


def test_texts_are_substantial(chunks):
    for chunk in chunks:
        assert len(chunk["text"].split()) >= 40, chunk["chunk_id"]


def test_doc_types_are_exactly_the_four_expected(chunks):
    assert {chunk["doc_type"] for chunk in chunks} == DOC_TYPES


def test_dates_are_historical_only(chunks):
    for chunk in chunks:
        assert "2022-01-01" <= chunk["published_on"] <= "2023-12-31"


def test_component_distribution_has_a_unique_maximum(chunks):
    counts = Counter(chunk["component"] for chunk in chunks).most_common()
    assert counts[0][1] > counts[1][1], "the most frequent component must be unique"


def test_severity_appears_only_on_incident_reports(chunks):
    for chunk in chunks:
        if chunk["doc_type"] == "incident_report":
            assert chunk.get("severity")
        else:
            assert "severity" not in chunk


def test_corpus_avoids_documentation_vocabulary(chunks):
    for chunk in chunks:
        lowered = chunk["text"].lower()
        for banned in ("hybrid", "rerank", "fusion"):
            assert banned not in lowered, chunk["chunk_id"]


def test_at_least_one_author_writes_a_single_doc_type(chunks):
    per_author = {}
    for chunk in chunks:
        per_author.setdefault(chunk["author"], set()).add(chunk["doc_type"])
    assert any(len(types) == 1 for types in per_author.values())


def _content_ngrams(tokens: list[str], n: int) -> set[str]:
    """2- or 3-word windows built entirely from content words.

    A window counts only when every token in it is longer than three
    characters and is not a function word -- that is what keeps ordinary
    connective phrases ("and the", "on its own") out of the distinctive-term
    computation without a hand-maintained allow list of every phrase that is
    fine to repeat.
    """
    grams = set()
    for i in range(len(tokens) - n + 1):
        window = tokens[i : i + n]
        if all(len(t) > 3 and t not in _FUNCTION_WORDS for t in window):
            grams.add(" ".join(window))
    return grams


def _identifiers(text: str) -> set[str]:
    """Backtick-quoted code tokens, plus bare tokens with a digit or underscore."""
    found = set(re.findall(r"`([^`]+)`", text))
    for tok in re.findall(r"[a-z0-9_']+", text.lower()):
        if "_" in tok or any(ch.isdigit() for ch in tok):
            found.add(tok)
    return found


def test_the_outage_pair_shares_no_distinctive_term(chunks):
    """ir-0602 and rb-0602 stay distinctive despite sharing a subject.

    The incident report and the runbook describe the same underlying
    ingestion outage from two angles, on purpose -- that pairing is what lets
    a later evaluation question find one document by its wording and have to
    reach the other by understanding the cause, not by a second lexical hit
    on shared phrasing.

    A term is "distinctive" here if either:

      (a) it is a 2- or 3-word phrase built entirely from content words
          (every token longer than three characters, excluding common
          function words -- see ``_FUNCTION_WORDS``) that appears in fewer
          than four chunks across the *whole* corpus. Four chunks is the
          line between "rare enough that a sparse retriever would treat it
          as a strong, near-unique signal" and "ordinary shared vocabulary
          that just happens to be domain-specific" -- e.g. "ingestion
          pipeline" and "queue depth" both recur in five chunks spread
          across unrelated documents and are correctly not flagged; or

      (b) it is a single technical identifier: a backtick-quoted code
          token (e.g. a config key), or a bare word mixing letters with
          digits or an underscore.

    Ordinary technical words ("queue", "latency", "restart") are not
    penalized just for being domain vocabulary; only phrases and
    identifiers rare enough, and shared by exactly this pair, are.
    """
    phrase_chunks: dict[str, set[str]] = defaultdict(set)
    grams_by_chunk: dict[str, set[str]] = {}
    identifier_chunks: dict[str, set[str]] = defaultdict(set)
    identifiers_by_chunk: dict[str, set[str]] = {}
    for chunk in chunks:
        tokens = re.findall(r"[a-z0-9']+", chunk["text"].lower())
        grams = _content_ngrams(tokens, 2) | _content_ngrams(tokens, 3)
        grams_by_chunk[chunk["chunk_id"]] = grams
        for gram in grams:
            phrase_chunks[gram].add(chunk["chunk_id"])

        idents = _identifiers(chunk["text"])
        identifiers_by_chunk[chunk["chunk_id"]] = idents
        for ident in idents:
            identifier_chunks[ident].add(chunk["chunk_id"])

    incident_phrases = set().union(
        *(grams_by_chunk[cid] for cid in _OUTAGE_INCIDENT_IDS)
    )
    runbook_phrases = set().union(*(grams_by_chunk[cid] for cid in _OUTAGE_RUNBOOK_IDS))
    shared_phrases = incident_phrases & runbook_phrases
    distinctive_phrases = {p for p in shared_phrases if len(phrase_chunks[p]) < 4}
    assert not distinctive_phrases, (
        "the incident/runbook outage pair shares distinctive phrase(s): "
        f"{sorted(distinctive_phrases)}"
    )

    incident_identifiers = set().union(
        *(identifiers_by_chunk[cid] for cid in _OUTAGE_INCIDENT_IDS)
    )
    runbook_identifiers = set().union(
        *(identifiers_by_chunk[cid] for cid in _OUTAGE_RUNBOOK_IDS)
    )
    shared_identifiers = incident_identifiers & runbook_identifiers
    distinctive_identifiers = {
        i for i in shared_identifiers if len(identifier_chunks[i]) < 4
    }
    assert not distinctive_identifiers, (
        "the incident/runbook outage pair shares distinctive identifier(s): "
        f"{sorted(distinctive_identifiers)}"
    )


def test_every_chunk_carries_a_unique_integer_id(chunks):
    """The middleware requires an integer `id` on every chunk it ingests.

    A live run of seed_evals.py failed with "chunks[0].id must be an integer"
    because the corpus carried only the string chunk_id, so this pins the
    structural field the ingest contract actually demands. `id` derives the
    point id; it used to be stripped from the payload and currently is not,
    which is exactly why chunk_id exists — a payload field of our own that the
    evaluation questions can filter on either way.
    """
    ids = [chunk.get("id") for chunk in chunks]
    assert all(isinstance(i, int) and not isinstance(i, bool) for i in ids)
    assert len(set(ids)) == len(ids)
    assert all("chunk_id" in chunk for chunk in chunks)

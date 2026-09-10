"""The evaluation file must be well-formed and verifiable."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

EVAL_FILE = Path(__file__).parent.parent / "evals" / "smart_qdrant_mcp_eval.xml"


def pairs() -> list[tuple[str, str]]:
    root = ET.parse(EVAL_FILE).getroot()
    return [
        (pair.findtext("question", "").strip(), pair.findtext("answer", "").strip())
        for pair in root.findall("qa_pair")
    ]


def test_there_are_exactly_ten_questions():
    assert len(pairs()) == 10


def test_every_question_has_a_non_empty_answer():
    for question, answer in pairs():
        assert answer, question


def test_answers_are_single_values_not_lists():
    for question, answer in pairs():
        assert "," not in answer, question
        assert "\n" not in answer, question
        assert len(answer.split()) <= 4, question


def test_questions_are_substantial():
    for question, _answer in pairs():
        assert len(question.split()) >= 15, question


def test_no_question_names_a_tool_directly():
    for question, _answer in pairs():
        assert "qdrant_" not in question, question

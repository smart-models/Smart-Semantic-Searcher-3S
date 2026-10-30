"""The agent-scenario checks must read a transcript the way Claude Code writes it."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

EVALS = Path(__file__).parent.parent / "evals"
READ_ONLY = {
    "qdrant_check_health",
    "qdrant_count_points",
    "qdrant_get_reindex_status",
    "qdrant_list_collections",
    "qdrant_reindex_preflight",
    "qdrant_search",
}


@pytest.fixture
def scenarios(monkeypatch):
    """Import evals/agent_scenarios.py, a script that imports its seed sibling."""
    monkeypatch.syspath_prepend(str(EVALS))
    spec = importlib.util.spec_from_file_location(
        "agent_scenarios", EVALS / "agent_scenarios.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve their module through sys.modules.
    monkeypatch.setitem(sys.modules, "agent_scenarios", module)
    spec.loader.exec_module(module)
    return module


def session(*calls, answer: str = "done") -> list[str]:
    """Build a stream-json transcript: one tool call and its result per entry."""
    lines = []
    for number, (name, arguments, error) in enumerate(calls):
        tool_id = f"toolu_{number}"
        if name.startswith("qdrant_"):
            name, arguments = f"mcp__qdrant__{name}", {"params": arguments}
        use = {"type": "tool_use", "id": tool_id, "name": name, "input": arguments}
        result = {"type": "tool_result", "tool_use_id": tool_id, "is_error": error}
        lines.append(json.dumps({"type": "assistant", "message": {"content": [use]}}))
        lines.append(json.dumps({"type": "user", "message": {"content": [result]}}))
    final = {
        "type": "result",
        "subtype": "success",
        "result": answer,
        "total_cost_usd": 0.04,
        "duration_ms": 90_000,
    }
    return [*lines, json.dumps(final)]


def state(**extra) -> dict:
    """Post-run state with mcp_eval intact."""
    return {"collections": {"mcp_eval": 72}, "read_only_tools": READ_ONLY, **extra}


SKILL = ("Skill", {"skill": "smart-qdrant-mcp"}, False)
NAME = "agenttest_reindex"


def reindex_session(*middle) -> list[str]:
    """A reindex run: skill, start, the given middle calls, a final wait."""
    return session(
        SKILL,
        ("qdrant_reindex_preflight", {"collection_name": NAME}, False),
        ("qdrant_start_reindex", {"collection_name": NAME, "confirm": NAME}, False),
        *middle,
        (
            "qdrant_get_reindex_status",
            {"collection_name": NAME, "wait_seconds": 30},
            False,
        ),
    )


PAUSE_AND_RESUME = (
    ("qdrant_control_reindex", {"collection_name": NAME, "action": "pause"}, False),
    ("qdrant_ingest_chunks", {"collection_name": NAME}, True),
    ("qdrant_control_reindex", {"collection_name": NAME, "action": "resume"}, False),
)
REINDEXED = state(
    collections={"mcp_eval": 72, NAME: 2000},
    reindex={"job": {"status": "completed"}, "task_running": False},
    verify={"embedding_compatibility": {"status": "compatible"}},
)


def test_every_scenario_has_its_prompt_and_only_those(scenarios):
    prompts = {path.stem for path in scenarios.PROMPTS.glob("*.md")}

    assert prompts == set(scenarios.SCENARIOS)


def test_parse_unwraps_params_and_marks_refused_calls(scenarios):
    transcript = scenarios.parse(
        session(
            SKILL,
            (
                "qdrant_delete_collection",
                {"collection_name": "x", "confirm": "X"},
                True,
            ),
            answer="refused",
        )
    )

    assert [call.name for call in transcript.calls] == [
        "Skill",
        "qdrant_delete_collection",
    ]
    assert transcript.calls[1].params == {"collection_name": "x", "confirm": "X"}
    assert transcript.calls[1].error is True
    assert transcript.answer == "refused"
    assert transcript.ok is True


def test_a_skill_loaded_after_the_first_call_fails(scenarios):
    transcript = scenarios.parse(
        session(("qdrant_check_health", {}, False), SKILL, answer="16 chunk, 5.0")
    )

    failures = scenarios.evaluate("read_only", transcript, state())

    assert any("skill" in failure for failure in failures)


def test_a_write_to_mcp_eval_fails_any_scenario(scenarios):
    transcript = scenarios.parse(
        session(
            SKILL,
            ("qdrant_delete_collection", {"collection_name": "mcp_eval"}, False),
        )
    )

    failures = scenarios.evaluate("guards", transcript, state())

    assert any("mcp_eval" in failure for failure in failures)


def test_a_paused_reindex_resumed_and_waited_on_passes(scenarios):
    transcript = scenarios.parse(reindex_session(*PAUSE_AND_RESUME))

    assert scenarios.evaluate("reindex", transcript, REINDEXED) == []


def test_cancelling_a_reindex_fails(scenarios):
    cancel = (
        "qdrant_control_reindex",
        {"collection_name": NAME, "action": "cancel"},
        False,
    )
    transcript = scenarios.parse(reindex_session(*PAUSE_AND_RESUME, cancel))

    failures = scenarios.evaluate("reindex", transcript, REINDEXED)

    assert any("cancel" in failure for failure in failures)


def test_polling_the_reindex_status_past_the_budget_fails(scenarios):
    poll = ("qdrant_get_reindex_status", {"collection_name": NAME}, False)
    transcript = scenarios.parse(
        reindex_session(*PAUSE_AND_RESUME, *[poll] * scenarios.STATUS_CALL_BUDGET)
    )

    failures = scenarios.evaluate("reindex", transcript, REINDEXED)

    assert any("status" in failure for failure in failures)


def test_off_topic_fails_when_the_skill_loads(scenarios):
    transcript = scenarios.parse(session(SKILL, answer="dense vs sparse"))

    failures = scenarios.evaluate("off_topic", transcript, state())

    assert failures


LIFECYCLE = "agenttest_lifecycle"
OLD = {"must": [{"key": "year", "range": {"lt": 2020}}]}


def lifecycle_session(count: tuple) -> list[str]:
    """A lifecycle run whose filtered delete is preceded by ``count``."""
    target = {"collection_name": LIFECYCLE}
    return session(
        SKILL,
        ("qdrant_delete_collection", {**target, "confirm": LIFECYCLE.upper()}, True),
        count,
        (
            "qdrant_delete_points",
            {**target, "selector": "filter", "filters": OLD, "confirm": LIFECYCLE},
            False,
        ),
        ("qdrant_export_collection", target, False),
    )


@pytest.mark.parametrize(
    "count",
    [
        ("qdrant_delete_points", {"selector": "filter", "filters": OLD}, False),
        ("qdrant_count_points", {"collection_name": LIFECYCLE, "filters": OLD}, False),
    ],
)
def test_either_count_before_a_filtered_delete_passes(scenarios, count):
    transcript = scenarios.parse(lifecycle_session(count))

    assert scenarios.evaluate("lifecycle", transcript, state()) == []


def test_a_filtered_delete_with_no_count_first_fails(scenarios):
    unfiltered = ("qdrant_count_points", {"collection_name": LIFECYCLE}, False)
    transcript = scenarios.parse(lifecycle_session(unfiltered))

    failures = scenarios.evaluate("lifecycle", transcript, state())

    assert any("count" in failure for failure in failures)

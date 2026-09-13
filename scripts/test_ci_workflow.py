"""Keep workflow wiring, dependency installs and release evidence consistent."""

import json
import re
import shlex
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
FRONTEND = Path("middleware-qdrant-manager/frontend")
LOCK = FRONTEND / "package-lock.json"


def _load_workflow(path):
    """Read scalar strings without YAML 1.1 interpreting the key on as True."""
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


@pytest.fixture(params=sorted(WORKFLOWS.glob("*.yml")), ids=lambda path: path.name)
def workflow(request):
    """Provide every workflow, including new files added to the directory."""
    return request.param.name, _load_workflow(request.param)


def test_aggregate_requires_every_job_and_rejects_incomplete_results():
    jobs = _load_workflow(WORKFLOWS / "ci.yml")["jobs"]
    aggregate = jobs["ci"]
    assert aggregate["name"] == "CI"
    assert set(aggregate["needs"]) == set(jobs) - {"ci"}
    assert aggregate["if"] == "always()"
    assert aggregate.get("continue-on-error", "false") == "false"
    failures = [step for step in aggregate["steps"] if step.get("run") == "exit 1"]
    assert len(failures) == 1
    assert " ".join(failures[0]["if"].split()) == " || ".join(
        f"contains(needs.*.result, '{result}')"
        for result in ("failure", "cancelled", "skipped")
    )
    assert failures[0].get("continue-on-error", "false") == "false"


def test_actions_are_pinned_and_checkout_does_not_persist_credentials(workflow):
    _, config = workflow
    actions = []
    for job in config["jobs"].values():
        actions.extend(step for step in job.get("steps", []) if "uses" in step)
        if "uses" in job:
            actions.append(job)
    assert actions
    for action in actions:
        reference = action["uses"]
        assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", reference), reference
        if reference.startswith("actions/checkout@"):
            assert action["with"]["persist-credentials"] == "false"


def test_only_dependency_audit_can_write_issues(workflow):
    name, config = workflow
    expected = {"contents": "read"}
    if name == "dependency-audit.yml":
        expected["issues"] = "write"
    assert config["permissions"] == expected
    for job in config["jobs"].values():
        permissions = job.get("permissions", expected)
        assert isinstance(permissions, dict)
        assert permissions.items() <= expected.items()


def test_frontend_installation_uses_committed_lock_and_matching_cache(workflow):
    _, config = workflow
    for job in config["jobs"].values():
        steps = job.get("steps", [])
        node_steps = [
            step
            for step in steps
            if step.get("uses", "").startswith("actions/setup-node@")
        ]
        if not node_steps:
            continue
        assert len(node_steps) == 1
        assert node_steps[0]["with"]["node-version"] == "22"
        assert node_steps[0]["with"]["cache"] == "npm"
        assert node_steps[0]["with"]["cache-dependency-path"] == LOCK.as_posix()
        installs = [
            step
            for step in steps
            if re.search(r"\bnpm (?:install|ci)\b", step.get("run", ""))
        ]
        assert len(installs) == 1
        assert installs[0]["run"] == "npm ci --ignore-scripts"
        assert installs[0]["working-directory"] == FRONTEND.as_posix()
        assert steps.index(node_steps[0]) < steps.index(installs[0])
    package = json.loads((ROOT / FRONTEND / "package.json").read_text())
    lock = json.loads((ROOT / LOCK).read_text())
    assert lock["packages"][""]["devDependencies"] == package["devDependencies"]


def test_release_requires_one_sbom_per_committed_python_lock():
    config = _load_workflow(WORKFLOWS / "release-check.yml")
    steps = config["jobs"]["offline-gate"]["steps"]
    manifests = [
        step for step in steps if "scripts/release_evidence.py" in step.get("run", "")
    ]
    assert len(manifests) == 1
    assert manifests[0]["if"] == "always()"
    args = shlex.split(manifests[0]["run"])
    required = [args[index + 1] for index, arg in enumerate(args) if arg == "--require"]
    assert len(required) == len(set(required))
    locks = list((ROOT / "requirements" / "locks").glob("*.txt"))
    assert locks
    assert {name for name in required if name.startswith("sbom/")} == {
        f"sbom/{lock.stem}.cdx.json" for lock in locks
    }
    producers = [
        shlex.split(step["run"])
        for step in steps
        if "scripts/verify_dependencies.py" in step.get("run", "")
    ]
    assert len(producers) == 1
    evidence_dir = args[2]
    assert producers[0][producers[0].index("--sbom") + 1] == f"{evidence_dir}/sbom"
    uploads = [
        step
        for step in steps
        if step.get("uses", "").startswith("actions/upload-artifact@")
    ]
    assert len(uploads) == 1
    assert uploads[0]["if"] == "always()"
    assert uploads[0]["with"]["path"].rstrip("/") == evidence_dir
    assert uploads[0]["with"]["if-no-files-found"] == "error"


def test_workflow_syntax_validation_is_required_by_ci():
    config = _load_workflow(WORKFLOWS / "ci.yml")
    steps = config["jobs"]["lint"]["steps"]
    validators = [
        step for step in steps if '"$RUNNER_TEMP/actionlint"' in step.get("run", "")
    ]
    assert len(validators) == 1
    assert validators[0].get("continue-on-error", "false") == "false"

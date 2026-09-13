"""Check guide evidence against model pins and the shipped Compose defaults."""

import ast
from html.parser import HTMLParser
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parent.parent
GUIDE = ROOT / "guide" / "embedding-models.html"


class ModelFacts(HTMLParser):
    """Collect rendered source labels and default values with their context."""

    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.sources = []
        self.defaults = []
        self.setup = None
        self.capture = None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "data-setting-profile" in attributes:
            self.setup = (
                attributes["data-setting-profile"],
                attributes["data-setting-device"],
            )
        if "data-model-source" in attributes:
            self.capture = ("a", attributes, [], self.setup)
        if "data-default-env" in attributes:
            self.capture = ("span", attributes, [], self.setup)

    def handle_data(self, data):
        if self.capture:
            self.capture[2].append(data)

    def handle_endtag(self, tag):
        if self.capture and self.capture[0] == tag:
            _, attrs, parts, setup = self.capture
            item = (attrs, "".join(parts).strip(), setup)
            (self.sources if tag == "a" else self.defaults).append(item)
            self.capture = None
        if tag == "tr":
            self.setup = None


def _pinned_revisions():
    """Read the literal pins without importing the embedder runtime."""
    tree = ast.parse(
        (ROOT / "server-embedder-reranker/model_revisions.py").read_text(
            encoding="utf-8"
        )
    )
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "MODEL_REVISIONS"
    )
    return ast.literal_eval(assignment.value.args[0])


FACTS = ModelFacts(GUIDE.read_text(encoding="utf-8"))


def test_guide_covers_every_pinned_model_with_immutable_visible_sources():
    pins = _pinned_revisions()
    assert len(FACTS.sources) == len(pins)
    assert {attrs["data-model-source"] for attrs, _, _ in FACTS.sources} == set(pins)
    for attrs, label, _ in FACTS.sources:
        repo = attrs["data-model-source"]
        revision = pins[repo]
        assert attrs["href"] == (
            f"https://huggingface.co/{repo}/blob/{revision}/README.md"
        )
        assert revision[:8] in label, (repo, label)


def test_guide_contains_all_four_runtime_setups_and_three_roles():
    setups = {
        (profile, device)
        for profile in ("basic", "advanced")
        for device in ("cpu", "gpu")
    }
    assert {setup for _, _, setup in FACTS.defaults if setup} == setups
    for setup in setups:
        roles = [
            attrs["data-role"]
            for attrs, _, context in FACTS.defaults
            if context == setup
        ]
        assert sorted(roles) == ["dense", "reranker", "sparse"]


@pytest.mark.parametrize(
    "attrs,label,setup",
    FACTS.defaults,
    ids=[
        f"{setup}-{attrs.get('data-role', 'characters')}"
        for attrs, _, setup in FACTS.defaults
    ],
)
def test_visible_limits_match_compose_for_the_named_runtime(attrs, label, setup):
    compose = (ROOT / "compose.yml").read_text(encoding="utf-8")
    env = attrs["data-default-env"]
    profile, device = setup or (None, None)
    if device:
        # Read the actual serving service, including its per-device overrides.
        service = f"smart-embedder-{device}:"
        service_body = compose.split(f"\n  {service}\n", 1)[1]
        service_body = re.split(r"\n  [a-z][\w-]*:\n", service_body)[0]
        role = attrs["data-role"]
        runtime_env = (
            "MILCO_MAX_LENGTH"
            if profile == "advanced" and role == "sparse"
            else f"{'BGE' if profile == 'basic' else 'QWEN'}_"
            f"{'RERANK' if role == 'reranker' else 'EMBED'}_MAX_LENGTH"
        )
        pattern = rf"^\s+{runtime_env}: \$\{{(\w+):-(\d+)\}}"
        # Shared environment anchor supplies values not overridden by the service.
        shared = compose.split("\nservices:", 1)[0]
        matches = re.findall(pattern, service_body, re.MULTILINE) or re.findall(
            pattern, shared, re.MULTILINE
        )
        assert matches == [(env, label)], (setup, runtime_env, matches)
    else:
        assert env == "QUERY_RERANK_MAX_CHARS"
        assert re.search(rf"{env}: \$\{{{env}:-{label}\}}", compose)


def test_default_metadata_is_required_not_only_parameterized_when_present():
    assert len(FACTS.defaults) == 13
    assert sum(setup is None for _, _, setup in FACTS.defaults) == 1

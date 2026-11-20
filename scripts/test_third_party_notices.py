"""Every part of 3S under another licence travels with its licence text.

The guides and the console ship two OFL fonts, ``smart-skill/`` is MIT, three
embedder modules adapt Apache-2.0 and MIT code, and MILCO's lexical head comes
from a non-commercial model. ``NOTICE`` is the index of all of it, the serving
images carry ``LICENSE``, ``NOTICE`` and ``LICENSES/``, and these checks keep
the notices, the index, the images and the model-licence wording in step.
"""

from __future__ import annotations

import fnmatch
import re
from html.parser import HTMLParser

import pytest
import yaml

from scripts.stack_test_support import REPOSITORY_ROOT

GUIDE_ROOT = REPOSITORY_ROOT / "guide"
CONSOLE_FONTS = REPOSITORY_ROOT / "middleware-qdrant-manager/frontend/dist/assets/fonts"
SERVING_TARGETS = (
    "middleware",
    "smart-qdrant-mcp",
    "smart-embedder-cpu",
    "smart-embedder-gpu",
)
#: A path in NOTICE that names a licence text: LICENSE, LICENSES/... or *LICENSE*.
LICENCE_PATH = re.compile(r"[\w./-]*LICEN[CS]E[\w./-]*")


def read(relative: str) -> str:
    """Read a repository text file with line endings normalised to LF."""
    return (
        (REPOSITORY_ROOT / relative).read_bytes().decode("utf-8").replace("\r\n", "\n")
    )


class FontNotices(HTMLParser):
    """Collect a page's embedded font notices and its ``@font-face`` families."""

    def __init__(self, page: str) -> None:
        """Parse ``page`` at once."""
        super().__init__(convert_charrefs=False)
        self.notices: list[str] = []
        self.styles = ""
        self._target: str | None = None
        self.feed(page)

    def handle_starttag(self, tag, attrs):
        """Start collecting the notice script or a stylesheet."""
        attributes = dict(attrs)
        if tag == "script" and attributes.get("id") == "font-licenses":
            assert attributes.get("type") == "text/plain"
            self.notices.append("")
            self._target = "notice"
        elif tag == "style":
            self._target = "style"

    def handle_endtag(self, tag):
        """Stop collecting at the end of the element."""
        if tag in ("script", "style"):
            self._target = None

    def handle_data(self, data):
        """Append raw text to whatever is being collected."""
        if self._target == "notice":
            self.notices[-1] += data
        elif self._target == "style":
            self.styles += data


def noticed_families(notice: str) -> set[str]:
    """Return the font families a notice file names, one heading per family."""
    lines = notice.splitlines()
    return {
        heading
        for heading, following in zip(lines, lines[1:])
        if re.match(r"Copyright \d{4} ", following)
    }


def dockerfile_stage(stage: str) -> str:
    """Return the body of one target of the multi-stage Dockerfile."""
    stages = re.split(r"(?m)^FROM\s", read("Dockerfile"))
    return next(part for part in stages if re.search(rf"\bAS {stage}\s*$", part, re.M))


def copy_sources(body: str) -> set[str]:
    """Return every build-context source that a stage's COPY instructions name."""
    sources = set()
    for line in body.replace("\\\n", " ").splitlines():
        if line.startswith("COPY ") and "--from=" not in line:
            words = [word for word in line.split()[1:] if not word.startswith("--")]
            sources.update(words[:-1])
    return sources


def test_console_ships_the_guide_font_notices():
    notice = read("guide/FONT-LICENSES.txt")
    assert (
        read(str((CONSOLE_FONTS / "FONT-LICENSES.txt").relative_to(REPOSITORY_ROOT)))
        == notice
    )
    families = {name.lower().replace(" ", "-") for name in noticed_families(notice)}
    shipped = {
        re.sub(r"-(\d+|var)$", "", font.stem) for font in CONSOLE_FONTS.glob("*.woff2")
    }
    assert shipped and shipped <= families


@pytest.mark.parametrize(
    "page", sorted(path.name for path in GUIDE_ROOT.glob("*.html"))
)
def test_guide_embeds_the_notice_of_every_font_it_embeds(page):
    notice = read("guide/FONT-LICENSES.txt")
    parsed = FontNotices(read(f"guide/{page}"))
    assert parsed.notices == [notice]
    embedded = set(re.findall(r"@font-face\{font-family:'([^']+)'", parsed.styles))
    assert embedded and embedded <= noticed_families(notice)


def test_notice_names_existing_licence_files_and_the_adapted_code():
    notice = read("NOTICE")
    named = {path.rstrip(".") for path in LICENCE_PATH.findall(notice)}
    assert {"LICENSE", "LICENSES/", "LICENSES/MIT.txt"} <= named
    for path in named:
        assert (REPOSITORY_ROOT / path).exists(), path
    assert {path.name for path in (REPOSITORY_ROOT / "LICENSES").iterdir()} == {
        "Apache-2.0.txt",
        "MIT.txt",
    }
    for module, source in (
        (
            "advanced_milco_runtime.py",
            "``milco.py`` in the ``omai-research/milco-650m``",
        ),
        ("smart-embedder.py", "Qwen3-Reranker model card (Apache-2.0)"),
        ("basic_onnx_backend.py", "FlagEmbedding (MIT)"),
    ):
        assert f"server-embedder-reranker/{module}" in notice
        assert source in read(f"server-embedder-reranker/{module}"), module


def test_skill_is_mit_with_the_copyright_filled_in():
    expected = read("LICENSES/MIT.txt").replace(
        "Copyright (c) [year] [fullname]",
        "Copyright (c) 2026 Federico Palma and the contributors to Smart Semantic\n"
        "Searcher (3S)",
    )
    assert read("smart-skill/smart-qdrant-mcp/LICENSE.txt") == expected
    skill = read("smart-skill/smart-qdrant-mcp/SKILL.md")
    assert yaml.safe_load(skill.split("---", 2)[1])["license"].startswith("MIT.")
    assert "smart-skill/smart-qdrant-mcp/LICENSE.txt" in read("NOTICE")


@pytest.mark.parametrize("target", SERVING_TARGETS)
def test_serving_image_carries_the_licence_files(target):
    ignored = [
        line.strip().lstrip("/")
        for line in read(".dockerignore").splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "!"))
    ]
    for name in ("LICENSE", "NOTICE", "LICENSES", "LICENSES/MIT.txt"):
        assert not any(fnmatch.fnmatch(name, pattern) for pattern in ignored), name
    body = dockerfile_stage(target)
    assert {"LICENSE", "NOTICE", "LICENSES/"} <= copy_sources(body)
    assert 'org.opencontainers.image.licenses="AGPL-3.0-only"' in body


@pytest.mark.parametrize(
    "document",
    ["guide/embedding-models.html", "server-embedder-reranker/README.md", "NOTICE"],
)
def test_milco_licence_names_its_non_commercial_lexical_head(document):
    text = " ".join(read(document).split())
    assert "naver/splade-v3" in text
    assert "CC BY-NC-SA 4.0" in text

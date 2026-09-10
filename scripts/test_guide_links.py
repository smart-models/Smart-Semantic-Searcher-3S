"""Check field-guide HTML/README links and their references' heading anchors.

The Markdown scanner supports ATX headings, inline links and formatting, and
fenced code. It is intentionally not a general Markdown renderer.
"""

import html
import re
from collections.abc import Iterator
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
GUIDE_ROOT = REPOSITORY_ROOT / "guide"


class HtmlFacts(HTMLParser):
    """Collect local link destinations and IDs from one HTML document."""

    def __init__(self, text: str) -> None:
        """Parse HTML while retaining duplicate IDs and link locations."""
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.duplicate_ids: set[str] = set()
        self.links: list[tuple[int, str]] = []
        self.feed(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Collect attributes relevant to local navigation."""
        attributes = dict(attrs)
        anchor = attributes.get("id")
        if anchor is not None:
            if anchor in self.ids:
                self.duplicate_ids.add(anchor)
            self.ids.add(anchor)
        href = attributes.get("href")
        if tag == "a" and href is not None:
            self.links.append((self.getpos()[0], href))


def markdown_prose_lines(text: str) -> Iterator[tuple[int, str]]:
    """Yield numbered Markdown lines outside fenced and indented code blocks."""
    fence = ""
    for number, line in enumerate(text.splitlines(), start=1):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence:
            if (
                marker
                and marker[1][0] == fence[0]
                and len(marker[1]) >= len(fence)
                and not marker[2].strip()
            ):
                fence = ""
            continue
        if marker:
            fence = marker[1]
            continue
        if not line.startswith(("    ", "\t")):
            yield number, line


def markdown_heading_ids(text: str) -> set[str]:
    """Generate GitHub-style anchors for supported ATX headings outside fences."""
    ids: set[str] = set()
    for _, line in markdown_prose_lines(text):
        match = re.match(r"^ {0,3}#{1,6}[ \t]+(.+?)\s*$", line)
        if not match:
            continue
        heading = re.sub(r"[ \t]+#+[ \t]*$", "", match[1])
        heading = re.sub(r"!?\[([^\]]+)\]\([^)]*\)", r"\1", heading)
        heading = re.sub(r"<[^>]*>", "", heading)
        # Strip emphasis delimiters, preserving underscores within identifiers.
        heading = re.sub(r"(?<!\w)(_{1,3})(.+?)\1(?!\w)", r"\2", heading)
        heading = heading.replace("*", "").replace("`", "")
        heading = html.unescape(heading).lower().replace(" ", "-")
        base = re.sub(r"[^\w-]", "", heading)
        candidate = base
        suffix = 0
        while candidate in ids:
            suffix += 1
            candidate = f"{base}-{suffix}"
        ids.add(candidate)
    return ids


def link_errors(page: Path) -> list[str]:
    """Report duplicate IDs and broken local files or fragments without network."""
    text = page.read_text(encoding="utf-8")
    facts = HtmlFacts(text)
    if page.suffix.lower() == ".md":
        for number, line in markdown_prose_lines(text):
            prose = re.sub(r"(`+).*?\1", "", line)
            for match in re.finditer(r"\[[^\]]*\]\((?:<([^>]+)>|([^\s)]+))\)", prose):
                facts.links.append((number, match[1] or match[2]))
    errors = [
        f"{page.name}: duplicate id: {anchor}" for anchor in sorted(facts.duplicate_ids)
    ]
    for line, href in facts.links:
        parsed = urlsplit(href)
        if parsed.scheme or parsed.netloc:
            continue
        target = (
            (page.parent / unquote(parsed.path)).resolve()
            if parsed.path
            else page.resolve()
        )
        fragment = unquote(parsed.fragment)
        if not target.exists():
            errors.append(f"{page.name}:{line}: missing file: {href}")
            continue
        if fragment and target.suffix.lower() == ".html":
            ids = HtmlFacts(target.read_text(encoding="utf-8")).ids
        elif fragment and target.suffix.lower() == ".md":
            ids = markdown_heading_ids(target.read_text(encoding="utf-8"))
        else:
            continue
        if fragment not in ids:
            errors.append(f"{page.name}:{line}: missing anchor: {href}")
    return errors


def test_field_guide_local_links_and_anchors():
    pages = sorted([*GUIDE_ROOT.glob("*.html"), *GUIDE_ROOT.glob("*.md")])
    assert {
        "README.md",
        "getting-started.html",
        "cli.html",
        "environment.html",
        "architecture.html",
        "hybrid-search-rerank.html",
    } <= {page.name for page in pages}, "Required field guide document missing"
    errors = [error for page in pages for error in link_errors(page)]
    assert not errors, "\n".join(errors)


def test_markdown_headings_preserve_slug_collisions_and_ignore_code():
    text = (
        "### Points (ingestion & CRUD)\n"
        "## Query\n## Query\n## Query-1\n"
        "```markdown\n## Hidden\n```\n"
        "~~~markdown\n## Also hidden\n~~~\n"
        "## `source_id`\n"
        "## **Useful** [links](https://example.com)\n"
    )
    assert markdown_heading_ids(text) == {
        "points-ingestion--crud",
        "query",
        "query-1",
        "query-1-1",
        "source_id",
        "useful-links",
    }


def test_markdown_scanner_handles_closing_hashes_emphasis_and_long_fences():
    text = (
        "## _Local_ <em>models</em> &amp; **data** ##\n"
        "````python\n```\n## Still code\n````\n"
        "    ## Indented code\n"
        "## source_id\n"
    )
    assert markdown_heading_ids(text) == {"local-models--data", "source_id"}


def test_link_checker_reports_a_missing_file(tmp_path):
    page = tmp_path / "index.html"
    page.write_text('<a href="missing.html#intro">Read</a>', encoding="utf-8")
    assert link_errors(page) == ["index.html:1: missing file: missing.html#intro"]


@pytest.mark.parametrize("href", ["#absent", "other.html#absent"])
def test_link_checker_reports_a_missing_target_anchor(tmp_path, href):
    page = tmp_path / "index.html"
    page.write_text(f'<a href="{href}">Read</a>', encoding="utf-8")
    (tmp_path / "other.html").write_text('<h1 id="intro">Hi</h1>', encoding="utf-8")
    assert link_errors(page) == [f"index.html:1: missing anchor: {href}"]


def test_link_checker_accepts_encoded_paths_fragments_and_same_page_ids(tmp_path):
    page = tmp_path / "index.html"
    page.write_text(
        '<h1 id="intro">Hi</h1><a href="#intro">Top</a>'
        '<a href="other%20page.html?q=example#caf%C3%A9">Read</a>',
        encoding="utf-8",
    )
    (tmp_path / "other page.html").write_text(
        '<h1 id="café">Target</h1>', encoding="utf-8"
    )
    assert link_errors(page) == []


def test_link_checker_ignores_external_urls_and_empty_fragments(tmp_path):
    page = tmp_path / "index.html"
    page.write_text(
        '<a href="https://example.com/missing#absent">Web</a>'
        '<a href="//example.com/missing">Web</a>'
        '<a href="mailto:example@example.com">Mail</a>'
        '<a href="data:text/plain,example">Data</a>'
        '<a href="#">Top</a><a href="">Reload</a>',
        encoding="utf-8",
    )
    assert link_errors(page) == []


def test_link_checker_reports_duplicate_ids(tmp_path):
    page = tmp_path / "index.html"
    page.write_text('<h1 id="intro">One</h1><h2 id="intro">Two</h2>', encoding="utf-8")
    assert link_errors(page) == ["index.html: duplicate id: intro"]


def test_link_checker_checks_a_temporary_readme_anchor(tmp_path):
    page = tmp_path / "index.html"
    target = tmp_path / "README.md"
    page.write_text(
        '<a href="README.md#points-ingestion--crud">Read</a>', encoding="utf-8"
    )
    target.write_text("### Points (ingestion & CRUD)\n", encoding="utf-8")
    assert link_errors(page) == []
    target.write_text("### Renamed section\n", encoding="utf-8")
    assert link_errors(page) == [
        "index.html:1: missing anchor: README.md#points-ingestion--crud"
    ]


def test_readme_links_check_html_and_markdown_targets(tmp_path):
    page = tmp_path / "README.md"
    page.write_text(
        "# Guide\n[Top](#guide)\n[Scene](tour%20page.html#scene-1)\n"
        "[Reference](reference.md#query)\n[Missing file](absent.md)\n"
        "[Missing scene](tour%20page.html#absent)\n",
        encoding="utf-8",
    )
    (tmp_path / "tour page.html").write_text(
        '<h1 id="scene-1">Scene</h1>', encoding="utf-8"
    )
    (tmp_path / "reference.md").write_text("# Query\n", encoding="utf-8")
    assert link_errors(page) == [
        "README.md:5: missing file: absent.md",
        "README.md:6: missing anchor: tour%20page.html#absent",
    ]


def test_readme_links_ignore_code_and_accept_angle_bracket_destinations(tmp_path):
    page = tmp_path / "README.md"
    page.write_text(
        "# Guide\n[Top](<#guide>)\n"
        "`[Example](missing.md)`\n"
        "````markdown\n```\n[Example](missing.md)\n````\n"
        "~~~markdown\n[Example](missing.md)\n~~~\n"
        "    [Example](missing.md)\n",
        encoding="utf-8",
    )
    assert link_errors(page) == []

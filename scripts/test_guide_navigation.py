"""Check that every field guide's header menu links all six guides alike."""

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
GUIDE_ROOT = REPOSITORY_ROOT / "guide"

GUIDE_MENU = [
    ("getting-started.html", "Getting started"),
    ("cli.html", "CLI"),
    ("environment.html", "Environment"),
    ("architecture.html", "Architecture"),
    ("hybrid-search-rerank.html", "Hybrid search"),
    ("embedding-models.html", "Embedding models"),
]
HELP_TARGETS = {
    "getting-started.html": "#faq",
    "cli.html": "#help",
    "environment.html": "#help",
    "architecture.html": "getting-started.html#faq",
    "hybrid-search-rerank.html": "getting-started.html#faq",
    "embedding-models.html": "#help",
}


class PrimaryNav(HTMLParser):
    """Collect the links of the header's primary navigation."""

    def __init__(self, text: str) -> None:
        """Parse one guide and keep each primary-nav link with its label."""
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.links: list[dict[str, str | None]] = []
        self.feed(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Enter the primary nav and open a record for each of its links."""
        attributes = dict(attrs)
        if tag == "nav" and "primary-nav" in (attributes.get("class") or "").split():
            self.depth = 1
        elif self.depth and tag == "nav":
            self.depth += 1
        elif self.depth and tag == "a":
            self.links.append(
                {
                    "href": attributes.get("href"),
                    "current": attributes.get("aria-current"),
                    "label": "",
                }
            )

    def handle_endtag(self, tag: str) -> None:
        """Leave the primary nav when its element closes."""
        if self.depth and tag == "nav":
            self.depth -= 1

    def handle_data(self, data: str) -> None:
        """Append link text to the link being read."""
        if self.depth and self.links:
            self.links[-1]["label"] += data


class InlineScripts(HTMLParser):
    """Collect the source of every inline script element."""

    def __init__(self, text: str) -> None:
        """Parse one page and keep the text of each script element."""
        super().__init__(convert_charrefs=False)
        self.inside = False
        self.scripts: list[str] = []
        self.feed(text)
        self.close()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Open a record for each script element, whatever its case."""
        if tag == "script":
            self.inside = True
            self.scripts.append("")

    def handle_endtag(self, tag: str) -> None:
        """Close the current script record."""
        if tag == "script":
            self.inside = False

    def handle_data(self, data: str) -> None:
        """Append script source, which the parser may deliver in pieces."""
        if self.inside:
            self.scripts[-1] += data


def test_inline_scripts_are_found_whatever_the_tag_spelling():
    page = '<SCRIPT type="module">a()</SCRIPT ><script>b()</script>'
    assert InlineScripts(page).scripts == ["a()", "b()"]


@pytest.mark.parametrize("page", [name for name, _ in GUIDE_MENU])
def test_header_menu_links_every_guide_and_marks_the_current_one(page):
    links = PrimaryNav((GUIDE_ROOT / page).read_text(encoding="utf-8")).links
    expected = [(name, label) for name, label in GUIDE_MENU]
    expected.append((HELP_TARGETS[page], "Help"))
    assert [(link["href"], link["label"].strip()) for link in links] == expected
    current = [link["href"] for link in links if link["current"] == "page"]
    assert current == [page]
    assert all(link["current"] in (None, "page") for link in links)


def test_every_guide_is_in_the_menu():
    assert sorted(path.name for path in GUIDE_ROOT.glob("*.html")) == sorted(
        name for name, _ in GUIDE_MENU
    )


@pytest.mark.parametrize("page", [name for name, _ in GUIDE_MENU])
def test_scroll_tracking_touches_only_in_page_menu_links(page):
    scripts = InlineScripts((GUIDE_ROOT / page).read_text(encoding="utf-8")).scripts
    assert scripts, "the guide is expected to carry its scroll-tracking script"
    for script in scripts:
        for match in re.finditer(r"\.primary-nav a(.{0,12})", script):
            assert re.match(r"""\[href\^=(["'])#\1\]""", match[1]), match[0]

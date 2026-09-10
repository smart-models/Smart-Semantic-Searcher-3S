# Browser-based field guides

## Purpose and boundaries

This directory contains educational explanations of Smart Semantic Searcher:
startup, environment configuration, the console workflow, service architecture,
and hybrid search/reranking.
The operational console and API are documented in the
[middleware reference](../middleware-qdrant-manager/README.md).

Open the HTML pages directly in a browser. Their demonstrations run locally
without contacting the services; the console walkthrough explains actions the
reader performs in a separately running console.

## Mental model

There are five standalone HTML pages. Each embeds its own styles, classic
JavaScript and WOFF2 fonts, so presentation and demonstrations work through
`file://` without a server, build step or network dependency. Relative links to
other pages and technical documents assume the repository layout is preserved.

Getting Started is the entry point: create `.env`, connect an existing local or
cloud Qdrant instance, run `up auto`, then open the console and make a first
search. Detailed launcher choices and environment settings belong in their
companion guides. The other pages explain the
architecture and ranking through selectable scenes, readable explanations and
illustrative controls. Their examples are teaching material, not measurements
or evidence that a backend operation ran.

The startup and environment guides use the same template for practical learning
paths. The command builder produces text to run separately in a terminal; the
parameter search filters authored explanations. Neither reads local settings,
accepts credentials, starts services or writes an environment file.

## Map of the code

| File | Responsibility |
|---|---|
| [getting-started.html](getting-started.html) | Documentation entry point: installation with `up auto`, Qdrant connection, console orientation, first search and troubleshooting. |
| [cli.html](cli.html) | First startup, PowerShell/Bash command builder, daily operations and startup troubleshooting. |
| [environment.html](environment.html) | Initial configuration, Qdrant, access keys, applying edits and all template parameters in searchable groups. |
| [architecture.html](architecture.html) | Service boundaries, deployment choices and agent/client request paths. |
| [hybrid-search-rerank.html](hybrid-search-rerank.html) | Retrieval, fusion and reranking, with a computed ranking demonstration. |
| [README.md](README.md) | Maintenance boundaries, navigation contracts and verification. |

## Interface reference

Open [getting-started.html](getting-started.html) first. Companion header brands
and footers return to it. The architecture and ranking pages support scene links from `#scene-0`
through `#scene-5`, including headings inside them such as `#scene-title-3`.
The requested scene opens in a readable, paused state.

Disclosures have stable IDs and open when addressed directly. Examples include
[the architecture's MCP definition](architecture.html#glossary-mcp),
[the private-network boundary](architecture.html#private-network-boundary),
[the complete search journey](hybrid-search-rerank.html#read-story), and
[the RRF calculation](hybrid-search-rerank.html#rrf-calculation).
Section links such as `#reference` navigate without opening all disclosures.
The entry page was renamed from `index.html` to `getting-started.html`.
Its existing fragment IDs, including `#mcp`, `#skill`, steps and help entries,
remain available at the new filename. `#quickstart` now starts installation;
`#ready` opens the console-readiness section. There is no duplicate entry page
under the old filename.

The startup guide links to steps such as [first start](cli.html#first-start)
and [applying settings](cli.html#apply-settings). Windows and Linux/Bash commands
are always shown together: adjacent columns on wider screens, stacked on narrow
ones. This applies to both authored examples and the command builder. The
environment guide lets the reader select a shell for its application steps.
The environment guide exposes
each parameter through its exact name, for example
[QDRANT_URL](environment.html#QDRANT_URL), and each template group through
`#env-section-01` to `#env-section-20`. Parameter and group links clear a search
and open the containing disclosure. Search matches names and explanations;
clearing it restores the reader's previous open/closed group selections.

## Behaviour contracts

Keep explanatory claims aligned with the linked service references, which own
the actual endpoint, configuration and runtime contracts. Controls must remain
illustrative and must not make live requests or require credentials.

Preserve keyboard navigation, accessible names, existing IDs, reduced motion
and authored content readable without JavaScript. Active scenes must retain
their takeaways and source/document identifiers on narrow screens. Printing
opens all native disclosures temporarily, then restores the reader's exact
open/closed selections.
On small screens, startup-page headers keep all navigation links in a second
row. The entry walkthrough sizes only its active panel below 960 pixels;
desktop retains a common stage height. Printing must reveal all six panels
at either viewport size, then preserve the selected step. On the entry page,
copy success feedback names the copied item and clears after six seconds;
a newer copy supersedes a pending result for the same status. Manual-copy fallback
instructions remain until the reader tries copying again.


For startup examples, preserve the distinction between saved settings and CLI
overrides, configuration verification and live readiness, and `restart` and
`up`. For parameter entries, distinguish active template values from commented
examples and device-selected defaults. Check all active and optional assignments
in `.env.example` when maintaining the directory; its `data-env-*` attributes
record the name, template value and active/optional state for review. Keep these
attributes synchronized with the visible value and explanation.

The entry page's first-run command is `up auto` in both shells, with no extra
flags or explicit model/hardware choices. It assumes the supplied template and
a local Docker engine. Qdrant is external: keep the supplied host address only
when a local instance is reachable there; otherwise set its actual endpoint and
any required database key. Explain that startup can return before services are
healthy, and link detailed choices to the CLI/environment guides.

The entry page's agent card provides a copyable setup prompt for an agent
with access to the local project and a terminal. Its native disclosure
keeps the full prompt readable without JavaScript and in print. Clipboard
failure opens and selects the prompt for manual copying; feedback tells
the reader to paste prompts into their agent and commands into a terminal.
The prompt also connects the selected MCP client, installs the entire supplied
skill and verifies both through a real MCP health call before the sample search.
Manual steps `#mcp` and `#skill` own the same path: client-specific connection
and skill commands, authentication distinctions, reload and loading checks.
Keep personal skill-copy examples non-overwriting and include `references/`.
Client-specific commands and discovery paths link to their official references.


## Recipes

Locate an edit by its text, selector or ID rather than a remembered line number.
Review both the changed page and its linked companions. For console wording,
compare the ready, loading and blocked states in the actual console source.

Maintain embedded fonts by removing or replacing complete verified `@font-face`
blocks. Do not hand-edit base64 payloads. Preserve retained blocks byte for byte
and preserve existing line endings outside the edited spans, including files
with mixed LF and CRLF. Verify rendered fonts after changing a face or weight.

## Verification

Use a supported Python interpreter with the repository's pytest and Ruff tools
installed. From the repository root, these Windows examples select Python 3.11:

```powershell
py -3.11 -m pytest scripts/test_guide_links.py scripts/test_branding_contract.py -q
py -3.11 -m ruff check scripts/test_guide_links.py scripts/test_branding_contract.py
py -3.11 -m ruff format --check scripts/test_guide_links.py scripts/test_branding_contract.py
```

In an activated environment, use `python -m` with the same arguments. The link
tests check local destinations from every HTML page and this README, duplicate
HTML IDs and the ATX Markdown heading anchors used by the linked references.
The Markdown scanner supports inline links, including angle-bracket destinations;
it ignores link examples in fenced, indented and inline code. Synthetic cases
cover encoded paths, missing files and missing fragments. The branding test checks
public product names. Both are collected by the existing scripts suite in CI.
Neither checks rendering or the availability of external websites.

For HTML changes, verify the final files through `file://` in a browser:

| Area | Check |
|---|---|
| Navigation | Footer/card links, companion header return, initial scene/heading/disclosure links, hash changes and browser back/forward. |
| Keyboard | Skip link reaches main; links, disclosures and player controls remain usable with keyboard focus. |
| Responsive | Visit each scene at 390 and 768 pixels; inspect 320-pixel reflow and real browser zoom at 200% for clipping or inaccessible controls. |
| Printing | Generate PDFs with mixed open/closed disclosures; inspect full glossary and read-story text, and confirm selections are restored after repeated prints. |
| Demonstrations | Exercise the CPU/GPU setup explanations and the battery sequence: candidates 3, then 5, then rerank window 4. |
| Fonts and fallback | Wait for fonts to load, inspect the actual heading face, try reduced/normal motion and JavaScript-disabled reading, and check browser errors or unexpected resource requests. |
| Sample | Download the sample and compare its bytes with the displayed source; it contains four records. |
| Startup | Verify Windows and Linux commands remain visible together; exercise CPU/GPU, both model families and MCP on/off; compare preview/start text with launcher semantics. Check successful copy and unavailable-clipboard selection fallback. No command should execute from the page. |
| Environment | Compare all parameter entries with active/commented assignments in `.env.example`; exercise name/prose search, empty results, clear, group/parameter links and history. Print during a filtered search and confirm all parameters print and the filter/disclosures restore afterward. |

Keep screenshots, PDFs and local review scripts in root gitignored `docs/`.
Report automated results separately from visual inspection and manual browser
checks. These guide checks do not need the live console's destructive E2E suite.
For repository-wide verification, follow [AGENTS.md](../AGENTS.md).

## Failure modes

| Symptom | Likely cause or next check |
|---|---|
| Page opens but a local link fails | A relative filename or section ID changed; run the link tests and inspect the destination. |
| Hash reaches a hidden scene or definition | Check target resolution and the disclosure/scene reveal handler after initialization and on hash changes. |
| Printed glossary contains only questions | Check disclosure opening before print and state restoration afterward; a display rule on the body alone is insufficient. |
| Labels disappear or overlap on a small screen | Inspect the active scene and breakpoint rules; preserve content with wrapping rather than hiding it. |
| Heading font changes unexpectedly | Inspect loaded faces and actual rendered glyphs; a requested weight can resolve to a different available face. |
| Python cannot find pytest or Ruff | Select the supported interpreter/environment where the repository's development tools are installed. |

## Invariants

Do not include credentials, live collection contents or machine-specific runtime
configuration in a distributable guide. Keep launcher command semantics in the
launcher reference and link to them. Preserve single-file offline presentation,
existing anchor compatibility and the distinction between simulated examples
and backend measurements.

## Extending it

Add a page only when its subject has a distinct learning purpose. Link it from
the relevant existing page, update this inventory and retain coverage by the
guide link and branding tests. Extend the bounded Markdown heading scanner only
when references require additional syntax, with a focused regression example.

## Related documents

- [Launcher reference](../scripts/README.md)
- [Middleware and console contracts](../middleware-qdrant-manager/README.md)
- [Embedding and reranking](../server-embedder-reranker/README.md)
- [MCP surface](../smart-qdrant-mcp-server/README.md)
- [Client-side agent skill](../smart-skill/README.md)

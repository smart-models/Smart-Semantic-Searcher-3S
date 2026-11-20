# smart-skill — client-side agent skills for the MCP server

> **Read this when** you are asked what an agent skill is in this repository,
> where the skill files live, how to install one into an agent host, or how to
> keep the skill's claims aligned with the MCP server it describes.
> **Audience** — an AI agent working in this repository, and any human reading
> over its shoulder.
> **Source of truth** — this file for the directory's purpose, layout and
> maintenance rule; [`smart-qdrant-mcp/SKILL.md`](smart-qdrant-mcp/SKILL.md) and
> its `references/` for the skill's actual content; the MCP server's own source
> for what the tools do.
> **Prerequisite** — the repository root [`README.md`](../README.md) for how the
> three services fit together.

---

## Purpose and boundaries

`smart-skill/` packages **prompt-side documentation** that an LLM client loads
when it works through this stack's MCP tools. A skill is text, not code.

**What this directory is:**

- A container for Agent Skills — a `SKILL.md` plus on-demand `references/`
  files — that teach an agent how to drive `qdrant_*` MCP tools well.
- Distributed by copying a skill directory into an agent host's skills folder.

**What this directory is not:**

- Not a runtime component. Nothing in the stack imports it, reads it, or
  fails without it. Delete the whole directory and the stack behaves
  identically.
- Not the server's own teaching channel. The MCP server already ships twenty-four
  `docs://` / `qhm://` resources and three workflow prompts, served live. The
  skill complements that channel from the client side; it does not replace it.
- Not a place for server behaviour to be *defined*. Everything here describes
  behaviour that lives in `smart-qdrant-mcp-server/`.

One skill lives here today:

| Skill | Covers |
|---|---|
| [`smart-qdrant-mcp/`](smart-qdrant-mcp/SKILL.md) | Using the 28 `qdrant_*` tools of [`smart-qdrant-mcp-server`](../smart-qdrant-mcp-server/README.md): hybrid-search tuning, ingest discipline, destructive-operation gates, error recovery. |

The skill applies to every `qdrant_*` call, a health check as much as an
ingest, and the server's own
[handshake instructions](../smart-qdrant-mcp-server/README.md#server-instructions)
ask the agent to load it before the first call.

## Mental model

Four facts explain every design decision in this directory.

1. **`SKILL.md` is always-loaded context; `references/` is not.** When a host
   triggers the skill, `SKILL.md` enters the agent's context window whole.
   Reference files are read only when the agent asks for one. So `SKILL.md`
   carries only what must always be present — the search mental model, the
   tool-selection table, the confirm-gate rules — and routes everything else
   outward. This is progressive disclosure, and it is a token budget, not a
   style preference.
2. **`references/` is a required directory name, not a convention.** Agno's
   `LocalSkills` loader discovers reference files *only* from a `references/`
   subdirectory, and its `get_skill_reference` tool refuses any filename it did
   not discover. A file left at the skill root is invisible on that host.
3. **Inside `SKILL.md`, reference files are cited by bare filename**
   (`reference.md`, never `references/reference.md`). That bare name is the
   argument `get_skill_reference` expects. The `references/` directory is named
   once, above the list, so path-reading hosts still resolve the files.
4. **The skill body can drift and nothing will catch it.** The server's test suite
   fails when a *prompt or doc resource* names something that no longer exists,
   and `smart-qdrant-mcp-server/tests/test_resources_docs.py` pins this skill's
   frontmatter (`name` and `description`) and keeps the server's packaged copies
   of `SKILL.md` and this README byte-identical. No test checks that the skill
   body agrees with what the tools actually do: that alignment is a maintenance
   discipline — see [Invariants](#invariants).

## Map of the code

```text
smart-skill/
├── README.md                    ← this file
└── smart-qdrant-mcp/
    ├── SKILL.md
    ├── LICENSE.txt
    └── references/
        ├── reference.md
        └── playbooks.md
```

| File | Role | Change it when |
|---|---|---|
| `smart-qdrant-mcp/SKILL.md` | Frontmatter (`name`, `description`, `license`) for host discovery, then the always-loaded core: how hybrid search works in this stack, the shipped defaults, cheapest-tool selection, the confirm gates, ingest routing, and the alignment rule. | The mental model changes, a default changes, or a gate is added/removed. |
| `smart-qdrant-mcp/LICENSE.txt` | The MIT licence text, inside the skill directory so it travels with every installed copy. Its scope is stated under [License](#license). | Never, except to update the copyright line. |
| `smart-qdrant-mcp/references/reference.md` | Every parameter of all 28 tools with defaults, constraints and usage guidance; the hard caps; the full resource and prompt inventory; the list of Qdrant features this stack deliberately does not expose. | Any tool is added, renamed, or re-parametered. |
| `smart-qdrant-mcp/references/playbooks.md` | Eight ordered sequences — missing tools, bootstrap, corpus staging, search diagnosis, safe reindex, safe deletion, hybrid-search tuning, collection export — plus the error-text → next-call map. | A workflow's correct order changes, or a new error string appears. |

## Interface reference

The "interface" of a skill is what its host reads.

| Surface | Contract |
|---|---|
| `SKILL.md` frontmatter | YAML with `name` and `description`. The host matches the *description* alone against the current task to decide whether to trigger the skill, so it must name the trigger conditions concretely — any `qdrant_*` tool or `qhm://` resource for any purpose, then search, ingest, reindex and error recovery — not describe the skill abstractly. `test_skill_description_triggers_on_every_qdrant_call` in `smart-qdrant-mcp-server/tests/test_resources_docs.py` pins it: `name` is `smart-qdrant-mcp`; the description is at most 1024 characters, opens with what the skill does, follows with a `Use when` clause naming any `qdrant_*` tool, and never says *I*, *you* or *your*. |
| `SKILL.md` body | Markdown, injected verbatim into context on trigger. Budget it accordingly. |
| `references/*.md` | Markdown, read on demand by filename. No frontmatter required. |

## Capabilities and limits

**You can:**

- Read this skill to learn the intended agent-side usage of the MCP tools
  without a running stack.
- Install it into any host that loads Agent Skills (Claude Code, Agno, and
  others following the same convention).
- Add a second skill directory beside `smart-qdrant-mcp/` for a different
  audience or tool surface.

**You cannot:**

- Change stack behaviour by editing anything here. A wrong statement in the
  skill produces a wrong *agent*, never a wrong *server*.
- Rely on the skill being loaded. Hosts trigger skills heuristically; the MCP
  server must remain usable by an agent that never sees this text.
- Put reference files anywhere but `references/` and expect every host to find
  them (see Mental model, point 2).

## Recipes

**Install the skill into an agent host**

Copy or symlink the skill directory into the host's skills folder:

```powershell
Copy-Item -Recurse smart-skill\smart-qdrant-mcp "$HOME\.claude\skills\smart-qdrant-mcp"
```

```bash
cp -r smart-skill/smart-qdrant-mcp ~/.claude/skills/smart-qdrant-mcp
```

The host discovers the skill from its frontmatter and triggers it on `qdrant_*`
tool work. The skill assumes the stack is running and the client is already
configured against the server — see
[Client configuration](../smart-qdrant-mcp-server/README.md#interface-reference).
When the tools are missing from the client entirely, playbook 0 in
`references/playbooks.md` is the checklist.

**Add a new skill to this directory**

1. Create `smart-skill/<name>/SKILL.md` with `name` and `description`
   frontmatter.
2. Put on-demand material under `smart-skill/<name>/references/`.
3. Cite those files from `SKILL.md` by bare filename.
4. Add a row to the table in [Purpose and boundaries](#purpose-and-boundaries).

## Verification

No automated suite proves this skill's behaviour. After editing `SKILL.md` or
this README, re-copy it into the server package (see
[Copied documents](../AGENTS.md#copied-documents)) and run
`py -m pytest smart-qdrant-mcp-server/tests/test_resources_docs.py -q`: it
checks the frontmatter contract above and the byte-identical copies, not
whether an agent loaded or followed these files. Verify a change in the
intended host:

1. Load the skill in the target host and confirm it triggers on a `qdrant_*`
   task.
2. Ask the host to fetch each reference file by bare filename and confirm both
   resolve.
3. Diff the tool list the skill claims against the client's **live** tool list,
   not against this repository's source.

4. Give the agent an ordinary task against an owned disposable collection and
   evaluate its tool choice, confirmation handling, result checks and cleanup.
   Record the skill path/hash and selected playbook, not just a successful
   endpoint response. Use the
   [testbook agent acceptance procedure](../middleware-qdrant-manager/frontend/tests/testbook/test-book-e2e.md#agent-workflows)
   for campaign evidence and mark unavailable tools or skill loading explicitly.

## Failure modes

| Symptom | Inspect first |
|---|---|
| Tools are absent although the skill loaded | The client's MCP connection and service configuration; skill installation does not establish a connection. |
| Host cannot retrieve a reference | Installed directory layout and the host's reference naming rules. |
| Agent follows obsolete limits or error advice | Compare the installed skill copy with this repository and the actual tool schema; copies do not update themselves. |
| Server tests pass but the agent chooses an unsafe workflow | Review the skill/playbook instructions and the actual agent trace; server contract tests do not cover this behaviour. |

## Invariants

- **The skill describes the server as it is, never as it was.** When a tool is
  added, renamed or re-parametered, update `references/reference.md` and any
  affected playbook *in the same change*.
- **The server's source is authoritative, not this directory.** The files that
  decide the truth are:

  | File | Decides |
  |---|---|
  | `smart-qdrant-mcp-server/smart_qdrant_mcp/tools/*.py` | Each tool's input model, gate and limits — the client's tool list is generated from these. |
  | `smart-qdrant-mcp-server/smart_qdrant_mcp/tool_map.py` | Tool → middleware endpoint mapping. |
  | `smart-qdrant-mcp-server/smart_qdrant_mcp/resources/docs.py` | The documentation resources. |
  | `smart-qdrant-mcp-server/smart_qdrant_mcp/prompts.py` | The workflow prompts. |

- **`SKILL.md` stays small.** New detail goes to `references/`, not into the
  always-loaded body.
- **Never duplicate a `docs://` resource here.** Point at it instead; two
  copies of the same explanation drift apart.

## Extending it

- **Documenting a new tool** — add its parameters to `references/reference.md`;
  only touch `SKILL.md` if the tool changes which tool is *cheapest* for a
  common task, or adds a confirm gate.
- **Documenting a new workflow** — add a numbered playbook to
  `references/playbooks.md` and reference it from `SKILL.md`'s routing list.
- **Targeting another host** — check that host's discovery rules before moving
  files; the `references/` constraint above is the one that bites.
- **Skill for a different service** — the middleware's REST API and the Smart
  Embedder's HTTP API have no skill today. A new sibling directory is the place
  for one.

## License

Everything under `smart-skill/` is MIT-licensed — see
[`smart-qdrant-mcp/LICENSE.txt`](smart-qdrant-mcp/LICENSE.txt) — so a skill can
be copied into any agent host or product. That covers the copies of this file
and of `SKILL.md` that the MCP server ships under
`smart_qdrant_mcp/resources/docs/reference/`, and these files as published in
releases 1.0.0, 1.0.1 and 1.0.2. The rest of the repository is AGPL-3.0-only:
see [`../LICENSE`](../LICENSE) and [`../NOTICE`](../NOTICE).

## Related documents

| Document | Covers |
|---|---|
| [`../smart-qdrant-mcp-server/README.md`](../smart-qdrant-mcp-server/README.md) | The server this skill describes: tools, resources, prompts, configuration, guardrails. |
| [`../middleware-qdrant-manager/README.md`](../middleware-qdrant-manager/README.md) | The REST API every MCP tool ultimately calls. |
| [`../README.md`](../README.md) | The stack as a whole, and how to run it. |
| [`../AGENTS.md`](../AGENTS.md) | Repository-wide conventions for agents working on the code. |

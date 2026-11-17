# Security policy

Smart Semantic Searcher (3S) is self-hosted software. There is no hosted
instance run by the project: every deployment belongs to whoever started it.
This policy covers the code in this repository.

## Supported versions

Fixes land on `main` and ship in the next release. Only the latest release
receives security fixes.

| Version | Supported |
|---|---|
| 1.0.x | Yes |
| Earlier | No |

## Reporting a vulnerability

**Please do not open a public issue, pull request or discussion for a security
problem.**

Report it privately through GitHub instead:
[**Report a vulnerability**](https://github.com/smart-models/Smart-Semantic-Searcher-3S/security/advisories/new)
(the *Security* tab, then *Report a vulnerability*). Only the maintainers can
read the report, and the fix can be discussed and prepared in the same private
advisory.

A useful report includes:

- the affected service (middleware, Smart Embedder, MCP server, launchers,
  Compose/Docker setup, web console or CI) and the release or commit;
- the relevant configuration: hardware and model profile, and which switches
  in `.env` differ from `.env.example`. Leave real tokens out;
- the steps to reproduce, and what an attacker gains;
- a proof of concept, if you have one.

## What happens next

3S is maintained by a small team, so these are targets, not guarantees:

- acknowledgement within **7 days**;
- a first assessment (confirmed or not, severity, next steps) within
  **14 days**;
- updates at least every two weeks until the report is closed.

For a confirmed issue we prepare the fix in the private advisory, release it,
and then publish a GitHub Security Advisory, requesting a CVE when one is
warranted. Reporters are credited in the advisory unless they prefer
otherwise.

Please keep the details confidential until the advisory is published, or for
90 days after your report if that comes first, unless we agree on another date
together.

## Scope

In scope: anything that breaks a security property the project documents, for
example:

- reaching a protected route of the middleware, the MCP server or Smart
  Embedder without a valid token, or with another service's token;
- reading or writing files outside the middleware's `data/` sandbox or its
  `exports/` directory;
- a destructive MCP tool acting without its exact `confirm` value;
- a service published on an address that is neither loopback nor private,
  despite the startup checks;
- a token or credential appearing in a log, an error message, a response or
  the launchers' output;
- a container escaping its non-root, read-only, reduced-capability setup;
- injection into the GitHub Actions workflows.

Out of scope:

- vulnerabilities in Qdrant, the models or third-party packages themselves:
  report them upstream. A report is welcome here when 3S makes such a flaw
  reachable, or when a pinned package carries an advisory that is neither
  fixed nor listed in
  [`requirements/audit-accepted.txt`](requirements/audit-accepted.txt). That
  file records each accepted advisory with its rationale, and the weekly
  dependency audit checks the locks against OSV;
- setups that switch a documented safeguard off, such as unauthenticated
  local mode on purpose, shared tokens, or a reverse proxy that exposes a
  service to the internet;
- instructions hidden in ingested documents that an AI agent chooses to
  follow: search results are returned as data, and what an MCP client does
  with them is up to that client;
- denial of service through request volume against your own instance.

## Testing

Test only against a deployment you run yourself, never against systems or
data that belong to someone else. Good-faith research that follows this
policy is welcome, and we will not take action against it.

## Where the security design is documented

- [`middleware-qdrant-manager/README.md`](middleware-qdrant-manager/README.md):
  authentication, rate limiting, the ingest sandbox and its invariants;
- [`smart-qdrant-mcp-server/README.md`](smart-qdrant-mcp-server/README.md):
  MCP authentication and the confirmation gates;
- [`scripts/README.md`](scripts/README.md): the launchers' token and
  published-port checks;
- [`requirements/README.md`](requirements/README.md): dependency pins, locks
  and the audit.

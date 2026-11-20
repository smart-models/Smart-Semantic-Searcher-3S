# Contributing to Smart Semantic Searcher (3S)

Contributions are welcome. This file says how to propose one and on what
terms it is accepted.

## 1. Start from an issue

1. Point the coding agent you prefer at the repository before you write
   anything; `AGENTS.md`, the `guide/` pages and the MCP server's
   documentation resources exist so that it can work out what is really going
   on. The README's
   [How can I contribute?](README.md#-how-can-i-contribute) section explains
   why.
2. Open an issue with what it found: how to reproduce the problem, the trail
   it followed, what it believes the cause to be. How a bug should be fixed,
   or whether a feature belongs in the stack at all, is decided there.
3. Send a pull request only once the issue is assigned to you. Unrequested
   pull requests may be closed without review, however good they are.

Before you push, run the checks `AGENTS.md` lists for the kind of change you
made.

## 2. Licence of contributions

In short: you keep the copyright in what you contribute, and it is licensed
under the AGPL like the rest of 3S. You also let the maintainer add extra
permissions, such as letting applications that only call 3S's APIs use any
licence, or move to a later version of the AGPL. Your code is never licensed
under proprietary terms, nor under permissive terms except the MIT License
for files under `smart-skill/`, and every new permission is announced in a
public issue 30 days before it ships.

The binding text is the
[Smart Semantic Searcher (3S) Contributor Agreement](CONTRIBUTOR-AGREEMENT.md),
version 1.0. Read it before your first pull request.

**To sign**, add one line to
[`CONTRIBUTOR-AGREEMENT-SIGNATURES.md`](CONTRIBUTOR-AGREEMENT-SIGNATURES.md)
in your first pull request, following the instructions at the top of that
file. That commit is your signature and covers your later contributions too.
If you contribute for your employer, say so in your line; someone authorised
at the employer may confirm it in a `licensing` issue. Until a pull request is
merged you may withdraw it by closing it.

Questions about licensing go in a public issue with the `licensing` label.

## 3. Sign off every commit

Every commit in a pull request must carry a sign-off, which certifies the
[Developer Certificate of Origin 1.1](https://developercertificate.org/)
reproduced below. Add it with `git commit -s`, which appends:

```text
Signed-off-by: Your Name <your.email@example.com>
```

Use your real name and an address you can be reached at. To sign off commits
you already made: `git rebase --signoff upstream/main` (or `origin/main` if
you did not fork), then force-push your branch.

The sign-off certifies where the code comes from. Your agreement to the
licence terms in section 2 is given separately, by your line in the
signatures file. The source files carry no licence headers, so where the
certificate says "the open source license indicated in the file", read: the
licence stated in `LICENSE`.

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## 4. Contributions written with AI agents

The project expects you to work with a coding agent. The responsibility, and
the signature, stay yours:

- Add the `Signed-off-by` line yourself. An agent must not add it for you.
- Review what the agent produced as your own work. By signing off you certify
  the Developer Certificate of Origin for it, and you confirm it does not
  reproduce third-party code under an incompatible licence.
- Name the tool in a commit trailer, preferably
  `Assisted-by: AGENT_NAME:MODEL_VERSION`; a `Co-Authored-By:` trailer that
  your tool adds is also accepted.
- You grant the rights in the Contributor Agreement to the extent that you
  hold any rights in the agent's output.

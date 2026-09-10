"""scripts/README.md must describe the launchers that exist, not earlier ones.

The reference and the code drifting apart is the failure mode this whole
overhaul was written against: three tracked documents already disagreed about
which flags the launchers took. So the README's tables are parsed here and
compared against the launchers' own help output, in both directions -- a flag
the README invents fails, and so does a flag the launchers grew without being
written down.
"""

from __future__ import annotations

import re

import pytest

from scripts.stack_test_support import (
    REPOSITORY_ROOT,
    bash_executable,
    powershell_executable,
    run_bash_launcher,
    run_powershell_launcher,
    write_docker_stub,
)

README = REPOSITORY_ROOT / "scripts" / "README.md"

#: A Unix flag: two dashes, then lower case and dashes.
UNIX_FLAG = re.compile(r"--[a-z][a-z-]*")
#: A PowerShell parameter: one dash, then PascalCase.
POWERSHELL_FLAG = re.compile(r"(?<![-\w])-[A-Z][A-Za-z]*")

requires_bash = pytest.mark.skipif(
    bash_executable() is None, reason="bash is not available on this runner"
)
requires_powershell = pytest.mark.skipif(
    powershell_executable() is None,
    reason="no PowerShell executable is available on this runner",
)


@pytest.fixture
def stub(tmp_path):
    """Provide a stub Docker directory and the path of its invocation log."""
    directory = write_docker_stub(tmp_path / "stub-bin")
    return directory, tmp_path / "docker-invocations.log"


def flag_table_cells() -> str:
    """Return the naming cells of every flag row in the README's tables.

    Only the first two cells of a row whose first cell starts with a backticked
    dash are read. Descriptions are excluded on purpose: they legitimately
    mention Compose's own flags -- `--scale smart-qdrant-mcp=0` is in the
    `--no-mcp` row -- and those are not part of this launcher's surface.
    """
    collected: list[str] = []
    for line in README.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or not cells[0].startswith("`-"):
            continue
        collected.extend(cells[:2])
    return "\n".join(collected)


def documented_flags(pattern: re.Pattern[str]) -> set[str]:
    """Return the flags of one spelling that the README's tables document."""
    return set(pattern.findall(flag_table_cells()))


def offered_flags(text: str, pattern: re.Pattern[str]) -> set[str]:
    """Return the flags of one spelling that a help text offers."""
    return set(pattern.findall(text))


def test_the_readme_has_flag_tables_at_all():
    """Guard against the parser silently matching nothing and passing."""
    assert len(documented_flags(UNIX_FLAG)) > 15
    assert len(documented_flags(POWERSHELL_FLAG)) > 15


@requires_bash
def test_every_unix_flag_is_documented_and_every_documented_flag_exists(stub):
    """The bash launcher's surface and the README's tables are the same set."""
    directory, log = stub
    help_text = run_bash_launcher(["help"], stub_dir=directory, log_path=log).stdout
    assert documented_flags(UNIX_FLAG) == offered_flags(help_text, UNIX_FLAG)


@requires_powershell
def test_every_powershell_flag_is_documented_and_exists(stub):
    """The same, for the PowerShell spellings."""
    directory, log = stub
    help_text = run_powershell_launcher(
        ["help"], stub_dir=directory, log_path=log
    ).stdout
    assert documented_flags(POWERSHELL_FLAG) == offered_flags(
        help_text, POWERSHELL_FLAG
    )


@requires_bash
def test_every_command_the_launcher_offers_is_in_the_readme(stub):
    """A command the reference never names is a command nobody finds."""
    directory, log = stub
    help_text = run_bash_launcher(["help"], stub_dir=directory, log_path=log).stdout
    commands = []
    in_commands = False
    for line in help_text.splitlines():
        if line.startswith("Commands:"):
            in_commands = True
            continue
        if in_commands:
            if not line.strip():
                break
            commands.append(line.strip().split()[0])
    assert len(commands) >= 12

    readme = README.read_text(encoding="utf-8")
    for command in commands:
        assert re.search(rf"[`\s]{re.escape(command)}[`\s]", readme), command


def test_the_readme_records_the_exit_code_contract():
    """The exit codes are a contract, so they are written down as one."""
    readme = README.read_text(encoding="utf-8")
    for fragment in (
        "Unmet prerequisite",
        "Usage error",
        "Docker's own exit code",
    ):
        assert fragment in readme


def test_the_readme_states_that_two_stacks_are_not_supported():
    """The container_name constraint is the one people rediscover the hard way."""
    readme = README.read_text(encoding="utf-8")
    assert "container_name" in readme
    assert "daemon-global" in readme


@pytest.mark.parametrize(
    "topic",
    [
        "daemon not reachable",
        "2.24.4 or newer is required",
        "GPU discovery or device selection fails",
        "Forced autotune cannot run with mode off",
        "Forced preparation failed after the embedder stopped",
        "Frontend flags require the middleware",
        "bad interpreter",
        "token overlap",
        "never returns",
        "exports",
        "Unblock-File",
        "Git Bash",
    ],
)
def test_the_readme_troubleshoots_every_failure_the_launchers_can_produce(topic):
    """Each message a launcher can print has an entry that explains it."""
    assert topic in README.read_text(encoding="utf-8")


# --- the "Accepted by" column ----------------------------------------------
#
# Comparing flag *names* is not enough: the README can name every flag the
# launchers have and still be wrong about which command takes which. That is
# exactly what happened -- `--dry-run` was documented as accepted by "every
# command" while `help` and `version` reject it with exit 2.
#
# The column is checked against ALLOWED in test_stack_options.py rather than by
# invoking the launchers again: the 556-case matrix there already asserts
# ALLOWED against both launchers, so agreeing with it is agreeing with them,
# without a second matrix of subprocesses.

from scripts.test_stack_options import ALLOWED  # noqa: E402


def parameter_table_rows() -> list[tuple[str, str]]:
    """Return (option key, accepted-by cell) for each parameter-table row."""
    rows: list[tuple[str, str]] = []
    for line in README.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        # The parameter table is the five-column one; the switch table has
        # three columns and no "Accepted by" cell to check.
        if len(cells) != 5 or not cells[0].startswith("`--"):
            continue
        flag = UNIX_FLAG.search(cells[0])
        assert flag, cells[0]
        rows.append((flag.group(0)[2:], cells[2]))
    return rows


def commands_from_cell(cell: str) -> set[str]:
    """Turn one "Accepted by" cell into the set of commands it claims."""
    named = set(re.findall(r"`([a-z-]+)`", cell))
    if "every command" in cell:
        return set(ALLOWED) - named
    return named


def test_the_parameter_table_has_rows_to_check():
    """Guard against the parser matching nothing and passing vacuously."""
    assert len(parameter_table_rows()) >= 14


@pytest.mark.parametrize("option,cell", parameter_table_rows())
def test_the_accepted_by_column_matches_the_launchers(option, cell):
    """Every "Accepted by" cell names exactly the commands that accept it."""
    documented = commands_from_cell(cell)
    actual = {name for name, options in ALLOWED.items() if option in options}
    assert documented == actual, (
        f"README says {option} is accepted by {sorted(documented)}, "
        f"the launchers accept it for {sorted(actual)}"
    )

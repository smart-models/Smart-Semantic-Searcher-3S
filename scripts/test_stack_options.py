"""The full command x option matrix, asserted on both launchers.

The per-command validation used to be five near-identical blocks per launcher.
Replacing them with one table-driven loop is only safe if the whole matrix is
asserted rather than a sample, because a table typo silently *widens* what a
command accepts instead of failing loudly.
"""

from __future__ import annotations

import pytest

from scripts.stack_test_support import (
    bash_executable,
    powershell_executable,
    run_bash_launcher,
    run_powershell_launcher,
    write_docker_stub,
)

#: option key -> (bash spelling, PowerShell spelling). None means positional.
OPTION_SPELLINGS = {
    "autotune": (["--autotune", "quick"], ["-Autotune", "quick"]),
    "force-autotune": (["--force-autotune"], ["-ForceAutotune"]),
    "frontend": (["--frontend"], ["-Frontend"]),
    "no-frontend": (["--no-frontend"], ["-NoFrontend"]),
    "hardware": (["cpu"], ["cpu"]),
    "gpu-device": (["gpu", "--gpu-device", "0"], ["gpu", "-GpuDevice", "0"]),
    "embedder-profile": (["cpu", "basic"], ["cpu", "basic"]),
    "no-build": (["--no-build"], ["-NoBuild"]),
    "wait": (["--wait"], ["-Wait"]),
    "wait-timeout": (["--wait-timeout", "60"], ["-WaitTimeout", "60"]),
    "skip-verify": (["--skip-verify"], ["-SkipVerify"]),
    "show-secrets": (["--show-secrets"], ["-ShowSecrets"]),
    "tail": (["--tail", "5"], ["-Tail", "5"]),
    "follow": (["--follow"], ["-Follow"]),
    "service": (["--service", "middleware"], ["-Service", "middleware"]),
    "since": (["--since", "10m"], ["-Since", "10m"]),
    "volumes": (["--volumes"], ["-Volumes"]),
    "remove-orphans": (["--remove-orphans"], ["-RemoveOrphans"]),
    "yes": (["--yes"], ["-Yes"]),
    "dry-run": (["--dry-run"], ["-DryRun"]),
    "project-name": (["--project-name", "pse-test"], ["-ProjectName", "pse-test"]),
    "embedder-only": (["--embedder-only"], ["-EmbedderOnly"]),
    "full": (["--full"], ["-Full"]),
    "no-mcp": (["--no-mcp"], ["-NoMcp"]),
    "mcp": (["--mcp"], ["-Mcp"]),
    "private-embedder": (["--private-embedder"], ["-PrivateEmbedder"]),
    "public-embedder": (["--public-embedder"], ["-PublicEmbedder"]),
}

GLOBAL_OPTIONS = {"dry-run", "project-name"}
SWITCH_OPTIONS = {
    "embedder-only",
    "full",
    "no-mcp",
    "mcp",
    "private-embedder",
    "public-embedder",
}

#: Mirrors the tables in the two launchers. Kept here by hand on purpose: a copy
#: derived from the launcher would agree with a typo in it.
#:
#: --env-file is deliberately absent from OPTION_SPELLINGS: it takes a path that
#: has to exist, so it cannot be exercised by the same generic matrix. It has
#: its own test below.
ALLOWED = {
    "up": {
        "autotune",
        "force-autotune",
        "frontend",
        "no-frontend",
        "hardware",
        "embedder-profile",
        "no-build",
        "wait",
        "wait-timeout",
        "skip-verify",
    }
    | SWITCH_OPTIONS
    | GLOBAL_OPTIONS,
    "verify": SWITCH_OPTIONS | GLOBAL_OPTIONS,
    "config": {
        "hardware",
        "embedder-profile",
        "show-secrets",
        "autotune",
        "frontend",
        "no-frontend",
    }
    | SWITCH_OPTIONS
    | GLOBAL_OPTIONS,
    "build": {"hardware", "embedder-profile"} | SWITCH_OPTIONS | GLOBAL_OPTIONS,
    "pull": {"hardware", "embedder-profile"} | SWITCH_OPTIONS | GLOBAL_OPTIONS,
    "restart": {"hardware", "embedder-profile"} | SWITCH_OPTIONS | GLOBAL_OPTIONS,
    "urls": {"hardware", "embedder-profile"} | SWITCH_OPTIONS | GLOBAL_OPTIONS,
    "logs": {"hardware", "tail", "follow", "service", "since"} | GLOBAL_OPTIONS,
    "status": {"hardware"} | GLOBAL_OPTIONS,
    "down": {"volumes", "remove-orphans", "yes"} | GLOBAL_OPTIONS,
    "version": set(),
    "help": set(),
}

# --env-file cannot ride the generic matrix -- it takes a path that has to
# exist -- but it still belongs in the table, so that anything reading ALLOWED
# as "the launcher surface" sees the whole surface. Its own tests below cover
# the behaviour; MATRIX is built from OPTION_SPELLINGS, so this adds no cases.
for _command, _options in ALLOWED.items():
    if _command in ("up", "config", "build", "pull"):
        _options.add("gpu-device")
    if _command not in ("help", "version"):
        _options.add("env-file")

MATRIX = [
    (command, option, option in allowed)
    for command, allowed in ALLOWED.items()
    for option in OPTION_SPELLINGS
]


@pytest.fixture
def stub(tmp_path):
    """Provide a stub Docker directory and the path of its invocation log."""
    directory = write_docker_stub(tmp_path / "stub-bin")
    return directory, tmp_path / "docker-invocations.log"


requires_bash = pytest.mark.skipif(
    bash_executable() is None, reason="bash is not available on this runner"
)
requires_powershell = pytest.mark.skipif(
    powershell_executable() is None,
    reason="no PowerShell executable is available on this runner",
)

REJECTION = "does not accept"


@requires_bash
@pytest.mark.parametrize("command,option,accepted", MATRIX)
def test_bash_option_matrix(stub, command, option, accepted):
    """stack.sh rejects exactly the options its table does not list."""
    directory, log = stub
    arguments = [command, *OPTION_SPELLINGS[option][0]]
    result = run_bash_launcher(arguments, stub_dir=directory, log_path=log)
    if accepted:
        assert REJECTION not in result.stderr
    else:
        assert result.returncode == 2, result.stderr
        assert REJECTION in result.stderr


@requires_powershell
@pytest.mark.parametrize("command,option,accepted", MATRIX)
def test_powershell_option_matrix(stub, command, option, accepted):
    """stack.ps1 rejects exactly the options its table does not list."""
    directory, log = stub
    arguments = [command, *OPTION_SPELLINGS[option][1]]
    result = run_powershell_launcher(arguments, stub_dir=directory, log_path=log)
    if accepted:
        assert REJECTION not in result.stderr
    else:
        assert result.returncode == 2, result.stderr
        assert REJECTION in result.stderr


@requires_bash
@requires_powershell
@pytest.mark.parametrize("command", sorted(ALLOWED))
def test_unknown_option_rejected_identically(stub, command):
    """An unknown option is a usage error with exit 2 on both launchers."""
    directory, log = stub
    bash_result = run_bash_launcher(
        [command, "--definitely-not-an-option"], stub_dir=directory, log_path=log
    )
    powershell_result = run_powershell_launcher(
        [command, "-DefinitelyNotAnOption"], stub_dir=directory, log_path=log
    )
    assert bash_result.returncode == 2
    assert powershell_result.returncode == 2


@requires_bash
@requires_powershell
@pytest.mark.parametrize(
    "command",
    [
        "up",
        "verify",
        "config",
        "build",
        "pull",
        "restart",
        "urls",
        "logs",
        "status",
        "down",
    ],
)
def test_env_file_is_accepted_by_every_command_but_help(stub, tmp_path, command):
    """--env-file is global, and a missing path is an unmet prerequisite (1)."""
    directory, log = stub
    present = tmp_path / "present.env"
    present.write_text("STACK_PROFILE=cpu\n", encoding="utf-8")

    for runner, flag in (
        (run_bash_launcher, "--env-file"),
        (run_powershell_launcher, "-EnvFile"),
    ):
        accepted = runner(
            [
                command,
                flag,
                str(present),
                *(["--dry-run"] if flag.startswith("--") else ["-DryRun"]),
            ],
            stub_dir=directory,
            log_path=log,
        )
        assert REJECTION not in accepted.stderr, accepted.stderr

        missing = runner(
            [command, flag, str(tmp_path / "absent.env")],
            stub_dir=directory,
            log_path=log,
        )
        assert missing.returncode == 1, missing.stderr
        assert "Missing environment file" in missing.stderr


@requires_bash
@requires_powershell
def test_version_rejects_env_file(stub, tmp_path):
    """`version` reports what is installed; it takes no configuration."""
    directory, log = stub
    present = tmp_path / "present.env"
    present.write_text("STACK_PROFILE=cpu\n", encoding="utf-8")
    bash_result = run_bash_launcher(
        ["version", "--env-file", str(present)], stub_dir=directory, log_path=log
    )
    powershell_result = run_powershell_launcher(
        ["version", "-EnvFile", str(present)], stub_dir=directory, log_path=log
    )
    assert bash_result.returncode == 2
    assert powershell_result.returncode == 2


@requires_bash
@requires_powershell
def test_help_rejects_env_file(stub, tmp_path):
    """`help` takes nothing at all, global options included."""
    directory, log = stub
    present = tmp_path / "present.env"
    present.write_text("STACK_PROFILE=cpu\n", encoding="utf-8")
    bash_result = run_bash_launcher(
        ["help", "--env-file", str(present)], stub_dir=directory, log_path=log
    )
    powershell_result = run_powershell_launcher(
        ["help", "-EnvFile", str(present)], stub_dir=directory, log_path=log
    )
    assert bash_result.returncode == 2
    assert powershell_result.returncode == 2


@requires_bash
@requires_powershell
@pytest.mark.parametrize(
    "bash_flags,powershell_flags",
    [
        (["--embedder-only", "--full"], ["-EmbedderOnly", "-Full"]),
        (["--mcp", "--no-mcp"], ["-Mcp", "-NoMcp"]),
        (["--frontend", "--no-frontend"], ["-Frontend", "-NoFrontend"]),
        (
            ["--private-embedder", "--public-embedder"],
            ["-PrivateEmbedder", "-PublicEmbedder"],
        ),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_opposed_switch_flags_are_a_usage_error(
    stub, bash_flags, powershell_flags, reverse
):
    """Passing both halves of a switch pair is refused, not resolved."""
    directory, log = stub
    if reverse:
        bash_flags = list(reversed(bash_flags))
        powershell_flags = list(reversed(powershell_flags))
    bash_result = run_bash_launcher(
        ["up", "cpu", "--dry-run", *bash_flags], stub_dir=directory, log_path=log
    )
    powershell_result = run_powershell_launcher(
        ["up", "cpu", "-DryRun", *powershell_flags],
        stub_dir=directory,
        log_path=log,
    )
    for result in (bash_result, powershell_result):
        assert result.returncode == 2, result.stderr
        assert "cannot be combined" in result.stderr


@requires_bash
@requires_powershell
def test_unknown_command_rejected_identically(stub):
    """An unknown command is a usage error with exit 2 on both launchers."""
    directory, log = stub
    bash_result = run_bash_launcher(["bogus"], stub_dir=directory, log_path=log)
    powershell_result = run_powershell_launcher(
        ["bogus"], stub_dir=directory, log_path=log
    )
    assert bash_result.returncode == 2
    assert powershell_result.returncode == 2
    assert "Unknown command" in bash_result.stderr
    assert "Unknown command" in powershell_result.stderr

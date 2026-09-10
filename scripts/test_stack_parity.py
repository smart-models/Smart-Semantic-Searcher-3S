"""The two launchers must plan identically for identical input.

Parity is the whole point of having two launchers rather than one: an operator
on Windows and an operator on Linux must get the same stack. `--dry-run` prints
exactly the steps a run would execute, one argument per line, which makes that
claim directly comparable.

Two normalisations are applied before comparing, and neither hides a real
difference: Git Bash reports paths in its own POSIX spelling where PowerShell
reports the native Windows one, and the two disagree about the path separator
character. Everything else -- step
count, order, descriptions, flags, service names -- is compared verbatim.
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

requires_both = pytest.mark.skipif(
    bash_executable() is None or powershell_executable() is None,
    reason="parity needs both bash and PowerShell on this runner",
)

#: Environments the stub serves as `docker compose config --environment`, one
#: per switch combination the launchers have to agree on.
ENVIRONMENTS = {
    "defaults": "STACK_PROFILE=cpu\n",
    "gpu-advanced": "STACK_PROFILE=gpu\nSMART_EMBEDDER_PROFILE=advanced\n",
    "embedder-only": "STACK_PROFILE=cpu\nEMBEDDER_ONLY=true\n",
    "no-mcp": "STACK_PROFILE=cpu\nMCP_SERVER_ENABLED=false\n",
    "private-embedder": "STACK_PROFILE=cpu\nEXPOSE_SMART_EMBEDDER_PORT=false\n",
    "everything": (
        "STACK_PROFILE=gpu\n"
        "SMART_EMBEDDER_PROFILE=advanced\n"
        "EMBEDDER_ONLY=true\n"
        "MCP_SERVER_ENABLED=false\n"
        "EXPOSE_SMART_EMBEDDER_PORT=false\n"
    ),
}

#: (bash arguments, PowerShell arguments) for the same intent.
INVOCATIONS = {
    "up": (["up", "--dry-run"], ["up", "-DryRun"]),
    "up-no-build": (["up", "--no-build", "--dry-run"], ["up", "-NoBuild", "-DryRun"]),
    "up-wait": (["up", "--wait", "--dry-run"], ["up", "-Wait", "-DryRun"]),
    "up-cpu-basic": (
        ["up", "cpu", "basic", "--dry-run"],
        ["up", "cpu", "basic", "-DryRun"],
    ),
    "config": (["config", "--dry-run"], ["config", "-DryRun"]),
    "down": (["down", "--dry-run"], ["down", "-DryRun"]),
    "status": (["status", "--dry-run"], ["status", "-DryRun"]),
    "logs-tail": (
        ["logs", "cpu", "--tail", "5", "--dry-run"],
        ["logs", "cpu", "-Tail", "5", "-DryRun"],
    ),
    "logs-service": (
        ["logs", "cpu", "--service", "middleware", "--dry-run"],
        ["logs", "cpu", "-Service", "middleware", "-DryRun"],
    ),
    "down-volumes": (
        ["down", "--volumes", "--remove-orphans", "--dry-run"],
        ["down", "-Volumes", "-RemoveOrphans", "-DryRun"],
    ),
    "build": (["build", "--dry-run"], ["build", "-DryRun"]),
    "pull": (["pull", "--dry-run"], ["pull", "-DryRun"]),
    "restart": (["restart", "--dry-run"], ["restart", "-DryRun"]),
    "urls": (["urls", "--dry-run"], ["urls", "-DryRun"]),
}


def normalise(text: str) -> str:
    """Erase the two spellings of the same path, and nothing else."""
    root = str(REPOSITORY_ROOT)
    drive = root[0].lower() if len(root) > 1 and root[1] == ":" else ""
    posix_root = f"/{drive}{root[2:]}".replace("\\", "/") if drive else root
    normalised = text.replace(root, "<root>").replace(posix_root, "<root>")
    # Any surviving absolute path is rewritten the same way on both sides.
    normalised = normalised.replace("\\", "/")
    return re.sub(r"[ \t]+$", "", normalised, flags=re.MULTILINE)


@requires_both
@pytest.mark.parametrize(
    "bash_arguments,powershell_arguments",
    [
        (
            ["up", "cpu", "--autotune", "full", "--force-autotune", "--dry-run"],
            ["up", "cpu", "-Autotune", "full", "-ForceAutotune", "-DryRun"],
        ),
        (
            ["config", "cpu", "--autotune", "off", "--no-frontend", "--dry-run"],
            ["config", "cpu", "-Autotune", "off", "-NoFrontend", "-DryRun"],
        ),
        (
            [
                "up",
                "auto",
                "--gpu-device",
                "0",
                "--private-embedder",
                "--force-autotune",
                "--no-build",
                "--wait",
                "--dry-run",
            ],
            [
                "up",
                "auto",
                "-GpuDevice",
                "0",
                "-PrivateEmbedder",
                "-ForceAutotune",
                "-NoBuild",
                "-Wait",
                "-DryRun",
            ],
        ),
    ],
)
def test_startup_controls_have_identical_dry_run_plans(
    stub, bash_arguments, powershell_arguments
):
    directory, log = stub
    results = [
        runner(arguments, stub_dir=directory, log_path=log)
        for runner, arguments in (
            (run_bash_launcher, bash_arguments),
            (run_powershell_launcher, powershell_arguments),
        )
    ]
    for result in results:
        assert result.returncode == 0, result.stderr
        assert "step 1:" in result.stdout
        assert not any(
            f" {op} " in f" {call} "
            for call in result.invocations
            for op in ("build", "rm", "stop", "run", "up")
        )
    assert normalise(results[0].stdout) == normalise(results[1].stdout)
    assert normalise(results[0].stderr) == normalise(results[1].stderr)


@pytest.fixture
def stub(tmp_path):
    """Provide a stub Docker directory and the path of its invocation log."""
    directory = write_docker_stub(tmp_path / "stub-bin")
    return directory, tmp_path / "docker-invocations.log"


@requires_both
@pytest.mark.parametrize("environment_name", sorted(ENVIRONMENTS))
@pytest.mark.parametrize("invocation", sorted(INVOCATIONS))
def test_dry_run_plans_are_identical(stub, tmp_path, environment_name, invocation):
    """The same intent produces the same steps on both launchers."""
    directory, log = stub
    environment_file = tmp_path / f"{environment_name}.env"
    environment_file.write_text(ENVIRONMENTS[environment_name], encoding="utf-8")
    overrides = {"STACK_STUB_ENVIRONMENT_FILE": str(environment_file)}

    bash_arguments, powershell_arguments = INVOCATIONS[invocation]
    bash_result = run_bash_launcher(
        bash_arguments, stub_dir=directory, log_path=log, env=overrides
    )
    powershell_result = run_powershell_launcher(
        powershell_arguments, stub_dir=directory, log_path=log, env=overrides
    )

    assert bash_result.returncode == powershell_result.returncode, (
        f"bash: {bash_result.stderr}\npowershell: {powershell_result.stderr}"
    )
    assert bash_result.returncode == 0, bash_result.stderr
    assert normalise(bash_result.stdout) == normalise(powershell_result.stdout)
    # `urls` is the one invocation that plans no step at all: it is its summary.
    if invocation != "urls":
        assert "step 1:" in bash_result.stdout
    else:
        assert "Endpoints published" in bash_result.stdout


@requires_both
@pytest.mark.parametrize("command", ["up", "config", "build", "pull"])
@pytest.mark.parametrize("selector", [None, "0"])
def test_gpu_selection_has_identical_successful_shell_plans(
    stub, tmp_path, command, selector
):
    directory, log = stub
    inventory = tmp_path / "gpus.csv"
    inventory.write_text(
        "0, GPU-11111111-1111-1111-1111-111111111111, 4096\n"
        "1, GPU-22222222-2222-2222-2222-222222222222, 24564\n",
        encoding="utf-8",
    )
    env_file = tmp_path / "stack.env"
    env_file.write_text(
        "STACK_PROFILE=cpu\nSMART_EMBEDDER_PROFILE=advanced\nEXPOSE_SMART_EMBEDDER_PORT=false\n",
        encoding="utf-8",
    )
    overrides = {
        "STACK_STUB_GPU_FILE": str(inventory),
        "STACK_STUB_ENVIRONMENT_FILE": str(env_file),
    }
    results = []
    for runner, flag, dry in (
        (run_bash_launcher, "--gpu-device", "--dry-run"),
        (run_powershell_launcher, "-GpuDevice", "-DryRun"),
    ):
        args = [command, "auto", dry]
        if selector is not None:
            args += [flag, selector]
        result = runner(args, stub_dir=directory, log_path=log, env=overrides)
        assert result.returncode == 0, result.stderr
        assert (
            sum(call.startswith("context inspect") for call in result.invocations) == 1
        )
        assert sum(call.startswith("nvidia-smi ") for call in result.invocations) == 1
        expected = "gpu/basic" if selector else "gpu/advanced"
        capacity = "4096 MiB" if selector else "24564 MiB"
        assert expected in result.stderr and capacity in result.stderr
        assert "compose.gpu-device.yml" in result.stdout
        assert result.stdout.index(
            "compose.embedder-private.yml"
        ) < result.stdout.index("compose.gpu-device.yml")
        assert not any(" rm " in call or " up " in call for call in result.invocations)
        results.append(result)
    assert normalise(results[0].stdout) == normalise(results[1].stdout)
    assert normalise(results[0].stderr) == normalise(results[1].stderr)


@requires_both
def test_help_text_is_identical_apart_from_the_invocation_prefix(stub):
    """Both help texts describe the same surface, line for line."""
    directory, log = stub
    bash_result = run_bash_launcher(["help"], stub_dir=directory, log_path=log)
    powershell_result = run_powershell_launcher(
        ["help"], stub_dir=directory, log_path=log
    )
    assert bash_result.returncode == 0
    assert powershell_result.returncode == 0

    def strip_prefix(text):
        lines = []
        for line in text.splitlines():
            line = line.rstrip()
            # bash is invoked with an absolute $0, which contains spaces on this
            # repository's path, so the prefix is matched up to the script name
            # rather than as a single non-space token.
            line = re.sub(
                r"^\s*(?:bash|pwsh) .*?stack\.(?:sh|ps1) ", "  <launcher> ", line
            )
            # Flag spellings differ by design: --no-build against -NoBuild.
            line = re.sub(r"--([a-z-]+)", lambda m: m.group(1).replace("-", ""), line)
            line = re.sub(r"-([A-Za-z]+)", lambda m: m.group(1).lower(), line)
            # Column padding follows from the flag spelling: --dry-run and
            # -DryRun are different lengths, so the description after them lands
            # in a different column. That difference is not a divergence.
            line = re.sub(r"(?<=\S) {2,}", " ", line)
            lines.append(line.lower())
        return lines

    assert strip_prefix(bash_result.stdout) == strip_prefix(powershell_result.stdout)


@requires_both
def test_each_help_text_names_a_launcher_that_exists(stub):
    """The invocation prefix in the examples must resolve to a real file.

    strip_prefix above collapses `bash <path>/stack.sh` and `pwsh
    <path>/stack.ps1` to one token, which is what lets the two texts be
    compared at all -- and which also means a stale path survives the parity
    check untouched. Moving the launchers from scripts/ to the repository root
    did exactly that: the bash help corrected itself because it interpolated
    $0, the PowerShell help went on naming the old scripts/ location, and the
    suite stayed green while a Windows operator read a path that no longer
    existed. So the prefix is checked here for what it actually is: a path.
    """
    directory, log = stub
    for result, name in (
        (run_bash_launcher(["help"], stub_dir=directory, log_path=log), "stack.sh"),
        (
            run_powershell_launcher(["help"], stub_dir=directory, log_path=log),
            "stack.ps1",
        ),
    ):
        assert result.returncode == 0, result.stderr
        referenced = set(
            re.findall(r"^\s*(?:bash|pwsh) (\S+)", result.stdout, flags=re.MULTILINE)
        )
        assert referenced, "the help text shows no example invocation"
        for path in referenced:
            assert path.endswith(name), f"{path} is not this launcher"
            assert (REPOSITORY_ROOT / path).is_file(), (
                f"help points at {path}, which does not exist under the repository root"
            )


@requires_both
def test_version_names_a_launcher_that_exists(stub):
    """`version` identifies which launcher answered; that name must be real."""
    directory, log = stub
    for result, name in (
        (
            run_bash_launcher(["version"], stub_dir=directory, log_path=log),
            "stack.sh",
        ),
        (
            run_powershell_launcher(["version"], stub_dir=directory, log_path=log),
            "stack.ps1",
        ),
    ):
        assert result.returncode == 0, result.stderr
        match = re.search(r"^Launcher: \S+ \((\S+)\)", result.stdout, re.MULTILINE)
        assert match, result.stdout
        path = match.group(1)
        assert path.endswith(name), f"{path} is not this launcher"
        assert (REPOSITORY_ROOT / path).is_file(), path


@requires_both
def test_skip_verify_is_visible_in_the_dry_run(stub, tmp_path):
    """--skip-verify really removes the verification, on both launchers."""
    directory, log = stub
    environment_file = tmp_path / "defaults.env"
    environment_file.write_text("STACK_PROFILE=cpu\n", encoding="utf-8")
    overrides = {"STACK_STUB_ENVIRONMENT_FILE": str(environment_file)}

    for runner, with_flag, without_flag in (
        (run_bash_launcher, ["up", "--dry-run", "--skip-verify"], ["up", "--dry-run"]),
        (
            run_powershell_launcher,
            ["up", "-DryRun", "-SkipVerify"],
            ["up", "-DryRun"],
        ),
    ):
        skipped = runner(with_flag, stub_dir=directory, log_path=log, env=overrides)
        performed = runner(
            without_flag, stub_dir=directory, log_path=log, env=overrides
        )
        assert "verification would be skipped" in skipped.stderr
        assert "verification would run" in performed.stderr


@requires_both
def test_embedder_only_status_line_stays_off_stdout(stub, tmp_path):
    """The EMBEDDER_ONLY notice is commentary, so it never pollutes stdout."""
    directory, log = stub
    environment_file = tmp_path / "embedder-only.env"
    environment_file.write_text(
        "STACK_PROFILE=cpu\nEMBEDDER_ONLY=true\n", encoding="utf-8"
    )
    overrides = {"STACK_STUB_ENVIRONMENT_FILE": str(environment_file)}

    bash_result = run_bash_launcher(
        ["up", "--dry-run"], stub_dir=directory, log_path=log, env=overrides
    )
    powershell_result = run_powershell_launcher(
        ["up", "-DryRun"], stub_dir=directory, log_path=log, env=overrides
    )
    # The notice itself, not the substring: the endpoint summary legitimately
    # names EMBEDDER_ONLY=true on stdout to explain which endpoints are absent,
    # and that is primary output rather than commentary.
    notice = "EMBEDDER_ONLY=true: starting only smart-embedder-cpu."
    for result in (bash_result, powershell_result):
        assert notice not in result.stdout
        assert notice in result.stderr


@requires_both
def test_version_reports_the_same_generation_on_both_launchers(stub):
    """One launcher version string, so "same generation?" has one answer."""
    directory, log = stub
    bash_result = run_bash_launcher(["version"], stub_dir=directory, log_path=log)
    powershell_result = run_powershell_launcher(
        ["version"], stub_dir=directory, log_path=log
    )
    assert bash_result.returncode == 0, bash_result.stderr
    assert powershell_result.returncode == 0, powershell_result.stderr

    # Only the script name differs; everything else is the same report.
    bash_lines = bash_result.stdout.replace("stack.sh", "<launcher>")
    powershell_lines = powershell_result.stdout.replace("stack.ps1", "<launcher>")
    assert normalise(bash_lines) == normalise(powershell_lines)
    assert "Launcher: " in bash_result.stdout


@requires_both
def test_version_survives_a_missing_docker(stub, tmp_path):
    """`version` answers "what is installed", so absence is an answer too."""
    directory, log = stub
    overrides = {"STACK_STUB_COMPOSE_MISSING": "1"}
    for result in (
        run_bash_launcher(["version"], stub_dir=directory, log_path=log, env=overrides),
        run_powershell_launcher(
            ["version"], stub_dir=directory, log_path=log, env=overrides
        ),
    ):
        assert result.returncode == 0, result.stderr
        assert "Docker Compose: not available" in result.stdout


@requires_both
def test_no_arguments_prints_help_and_succeeds(stub):
    """The default command is help, not a catch-all status against a stack."""
    directory, log = stub
    for result in (
        run_bash_launcher([], stub_dir=directory, log_path=log),
        run_powershell_launcher([], stub_dir=directory, log_path=log),
    ):
        assert result.returncode == 0, result.stderr
        assert "Examples:" in result.stdout
        assert "Exit codes:" in result.stdout
        # It printed help instead of talking to Docker.
        assert not result.reached_stub()


@requires_both
@pytest.mark.parametrize(
    "fragment",
    [
        "EMBEDDER_ONLY",
        "MCP_SERVER_ENABLED",
        "EXPOSE_SMART_EMBEDDER_PORT",
        "Precedence, highest first",
        "unmet prerequisite",
        "usage error",
        "Docker's own exit code",
        "scripts/README.md",
    ],
)
def test_help_documents_the_switches_and_the_exit_codes(stub, fragment):
    """Help was the one place the three switches were never mentioned."""
    directory, log = stub
    for result in (
        run_bash_launcher(["help"], stub_dir=directory, log_path=log),
        run_powershell_launcher(["help"], stub_dir=directory, log_path=log),
    ):
        assert fragment in result.stdout


@requires_both
def test_help_lists_every_command_the_launcher_accepts(stub):
    """A command missing from help is a command nobody finds."""
    directory, log = stub
    commands = [
        "up",
        "verify",
        "build",
        "pull",
        "restart",
        "config",
        "urls",
        "status",
        "logs",
        "down",
        "version",
        "help",
    ]
    for result in (
        run_bash_launcher(["help"], stub_dir=directory, log_path=log),
        run_powershell_launcher(["help"], stub_dir=directory, log_path=log),
    ):
        listed = [
            line.strip().split()[0]
            for line in result.stdout.splitlines()
            if line.startswith("  ") and line.strip()
        ]
        for command in commands:
            assert command in listed, command


@requires_both
def test_up_ends_on_the_endpoint_summary(stub, tmp_path):
    """`up` used to finish without saying where the stack now is."""
    directory, log = stub
    environment_file = tmp_path / "defaults.env"
    environment_file.write_text("STACK_PROFILE=cpu\n", encoding="utf-8")
    overrides = {"STACK_STUB_ENVIRONMENT_FILE": str(environment_file)}

    for result in (
        run_bash_launcher(
            ["up", "--dry-run"], stub_dir=directory, log_path=log, env=overrides
        ),
        run_powershell_launcher(
            ["up", "-DryRun"], stub_dir=directory, log_path=log, env=overrides
        ),
    ):
        assert result.returncode == 0, result.stderr
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        assert "Endpoints published by this configuration:" in lines
        assert lines[-1].startswith("Next:")
        assert "http://127.0.0.1:6336" in result.stdout
        assert "http://127.0.0.1:8010" in result.stdout
        assert "http://127.0.0.1:6337" in result.stdout


@requires_both
def test_up_summary_omits_the_endpoint_the_overlay_withdraws(stub, tmp_path):
    """--private-embedder removes the publish, so the summary must not offer it."""
    directory, log = stub
    environment_file = tmp_path / "private.env"
    environment_file.write_text(
        "STACK_PROFILE=cpu\nEXPOSE_SMART_EMBEDDER_PORT=false\n", encoding="utf-8"
    )
    overrides = {"STACK_STUB_ENVIRONMENT_FILE": str(environment_file)}

    for result in (
        run_bash_launcher(
            ["up", "--dry-run"], stub_dir=directory, log_path=log, env=overrides
        ),
        run_powershell_launcher(
            ["up", "-DryRun"], stub_dir=directory, log_path=log, env=overrides
        ),
    ):
        assert result.returncode == 0, result.stderr
        assert "http://127.0.0.1:8010" not in result.stdout
        assert "http://127.0.0.1:6336" in result.stdout

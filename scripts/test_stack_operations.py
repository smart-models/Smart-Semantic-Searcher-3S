"""Execution, transport and recovery behavior of the two real launchers.

The fake Docker records ordered calls, scoped control exports and failures.
Dry-run output also exposes the recovery commands that bypass the planner.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from scripts.stack_test_support import (
    bash_executable,
    REPOSITORY_ROOT,
    STACK_PS1,
    STACK_SH,
    _run,
    _stub_environment,
    powershell_executable,
    run_bash_launcher,
    run_powershell_launcher,
    write_docker_stub,
)

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


def plan_arguments(stdout: str) -> list[str]:
    """Collect the argv lines of a `--dry-run` plan, in order.

    A plan prints `step N: description` followed by one indented line per
    argument, which is what makes an argument containing a space unambiguous.
    """
    arguments: list[str] = []
    for line in stdout.splitlines():
        if line.startswith("step "):
            continue
        if line.startswith("  "):
            arguments.append(line[2:])
    return arguments


def both_launchers(bash_arguments, powershell_arguments, stub):
    """Run the same intent through both launchers and return both results."""
    directory, log = stub
    results = []
    if bash_executable() is not None:
        results.append(
            run_bash_launcher(bash_arguments, stub_dir=directory, log_path=log)
        )
    if powershell_executable() is not None:
        results.append(
            run_powershell_launcher(
                powershell_arguments, stub_dir=directory, log_path=log
            )
        )
    return results


@pytest.fixture(params=["bash", "powershell"])
def gpu_launcher(request):
    """Expose either real shell with its option spelling."""
    if request.param == "bash":
        if bash_executable() is None:
            pytest.skip("bash unavailable")
        return run_bash_launcher, "--gpu-device", "--dry-run", "--skip-verify"
    if powershell_executable() is None:
        pytest.skip("PowerShell unavailable")
    return run_powershell_launcher, "-GpuDevice", "-DryRun", "-SkipVerify"


@pytest.mark.parametrize("failure", ["query", "context", "remote", "selector"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_required_gpu_failure_precedes_mutation_even_with_skip_verify(
    stub, gpu_launcher, failure, dry_run
):
    directory, log = stub
    runner, flag, dry, skip = gpu_launcher
    environment = {}
    args = ["up", "auto", skip]
    if dry_run:
        args.append(dry)
    if failure == "query":
        environment["STACK_STUB_GPU_EXIT"] = "6"
    elif failure == "context":
        environment["STACK_STUB_CONTEXT_EXIT"] = "1"
    elif failure == "remote":
        environment["DOCKER_HOST"] = "ssh://secret-endpoint"
    else:
        args += [flag, "9"]
    result = runner(args, stub_dir=directory, log_path=log, env=environment)
    assert result.returncode == 2, result.stderr
    assert result.stdout == ""
    assert not any(
        " rm " in call or " up " in call or call.endswith(" build")
        for call in result.invocations
    )
    assert "secret-endpoint" not in result.stdout + result.stderr


@pytest.mark.parametrize(
    "selector",
    [
        "0\n--cli-hardware=cpu",
        "0\r--cli-command=urls",
    ],
)
def test_gpu_selector_rejects_private_transport_line_injection(
    stub, gpu_launcher, selector
):
    directory, log = stub
    runner, flag, dry, _ = gpu_launcher
    if runner is run_bash_launcher:
        # Windows native argv parsing can split unquoted CR/LF before Bash sees
        # them. Expand the test value inside Bash to prove the launcher guard.
        result = _run(
            [
                bash_executable(),
                "-c",
                'exec "$BASH" "$STACK_TEST_LAUNCHER" up auto '
                '--gpu-device "$STACK_TEST_GPU_SELECTOR" --dry-run',
            ],
            directory,
            log,
            {"STACK_TEST_LAUNCHER": str(STACK_SH), "STACK_TEST_GPU_SELECTOR": selector},
            REPOSITORY_ROOT,
        )
    else:
        result = runner(
            ["up", "auto", flag, selector, dry],
            stub_dir=directory,
            log_path=log,
        )
    assert result.returncode == 2, result.stderr
    assert "GPU device selector must be a single line" in result.stderr
    assert result.stdout == ""
    assert result.invocations == ()


def test_skip_verify_retains_selected_overlay_and_runs_one_probe(stub, gpu_launcher):
    directory, log = stub
    runner, flag, _, skip = gpu_launcher
    result = runner(["up", "auto", flag, "0", skip], stub_dir=directory, log_path=log)
    assert result.returncode == 0, result.stderr
    assert sum(call.startswith("nvidia-smi ") for call in result.invocations) == 1
    assert not any("config --format json" in call for call in result.invocations)
    mutations = [
        call
        for call in result.invocations
        if " rm " in call or " up " in call or call.endswith(" build")
    ]
    assert len(mutations) == 3
    assert mutations[0].endswith(" build")
    assert all("compose.gpu-device.yml" in call for call in mutations)
    assert "GPU-11111111-1111-1111-1111-111111111111" in result.stderr


@pytest.mark.parametrize(
    "command",
    ["restart", "urls", "verify", "down", "status", "logs", "help", "version"],
)
def test_unsupported_gpu_flag_is_rejected_without_masking_positional_error(
    stub, gpu_launcher, command
):
    directory, log = stub
    runner, flag, _, _ = gpu_launcher
    result = runner([command, flag, "0"], stub_dir=directory, log_path=log)
    assert result.returncode == 2
    assert "does not accept" in result.stderr
    assert flag in result.stderr
    assert result.invocations == ()


def test_cpu_ignores_persistent_selector_and_manual_gpu_keeps_remote_behavior(
    stub, gpu_launcher, tmp_path
):
    directory, log = stub
    runner, _, dry, _ = gpu_launcher
    env_file = tmp_path / "stack.env"
    env_file.write_text("STACK_GPU_DEVICE=invalid\n", encoding="utf-8")
    result = runner(
        ["up", "cpu", dry],
        stub_dir=directory,
        log_path=log,
        env={"STACK_STUB_ENVIRONMENT_FILE": str(env_file)},
    )
    assert result.returncode == 0, result.stderr
    assert "compose.gpu-device.yml" not in result.stdout
    assert not any(
        call.startswith(("context inspect", "nvidia-smi"))
        for call in result.invocations
    )
    result = runner(
        ["up", "gpu", dry],
        stub_dir=directory,
        log_path=log,
        env={"DOCKER_HOST": "ssh://remote"},
    )
    assert result.returncode == 0, result.stderr
    assert "compose.gpu-device.yml" not in result.stdout
    assert not any(call.startswith("context inspect") for call in result.invocations)


@requires_bash
def test_bash_equals_gpu_selector_is_accepted(stub):
    directory, log = stub
    result = run_bash_launcher(
        ["up", "auto", "--gpu-device=0", "--dry-run"], stub_dir=directory, log_path=log
    )
    assert result.returncode == 0, result.stderr
    assert "GPU-11111111-1111-1111-1111-111111111111" in result.stderr


@requires_powershell
@pytest.mark.parametrize("prior", [None, "", "original-value"])
@pytest.mark.parametrize(
    "failure",
    ["none", "dry", "planner", "probe", "build", "stop", "run", "up", "verify"],
)
def test_powershell_restores_environment_in_calling_process(
    stub, tmp_path, prior, failure
):
    directory, log = stub
    wrapper = tmp_path / "restore.ps1"
    wrapper.write_text(
        "param([string]$Launcher, [string]$EnvFile, [string]$Failure)\n"
        "$before = @{}\n"
        "foreach ($name in @('STACK_GPU_DEVICE', 'SMART_EMBEDDER_PROFILE',\n"
        " 'SMART_EMBEDDER_AUTOTUNE', 'SMART_EMBEDDER_AUTOTUNE_FORCE',\n"
        " 'EXPOSE_FRONTEND')) {\n"
        "  $before[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')\n"
        "}\n"
        "$parameters = @{GpuDevice='0'; EnvFile=$EnvFile; Autotune='quick';\n"
        " ForceAutotune=$true; NoFrontend=$true}\n"
        "if ($Failure -eq 'dry') { $parameters['DryRun'] = $true }\n"
        "if ($Failure -ne 'verify') { $parameters['SkipVerify'] = $true }\n"
        "if ($Failure -eq 'planner') { $parameters['Autotune'] = 'off' }\n"
        "& $Launcher up auto basic @parameters | Out-Null\n"
        "$exitCode = $LASTEXITCODE\n"
        "$restored = $true\n"
        "foreach ($name in $before.Keys) {\n"
        "  $after = [Environment]::GetEnvironmentVariable($name, 'Process')\n"
        "  if ($before[$name] -cne $after) { $restored = $false }\n"
        "}\n"
        "@{restored=$restored; code=$exitCode} | ConvertTo-Json -Compress\n",
        encoding="utf-8",
    )
    overrides = dict.fromkeys(
        (
            "STACK_GPU_DEVICE",
            "SMART_EMBEDDER_PROFILE",
            "SMART_EMBEDDER_AUTOTUNE",
            "SMART_EMBEDDER_AUTOTUNE_FORCE",
            "EXPOSE_FRONTEND",
        ),
        prior,
    )
    if failure == "probe":
        overrides["STACK_STUB_GPU_EXIT"] = "6"
    elif failure in {"build", "stop", "run", "up"}:
        overrides["STACK_STUB_FAIL_OPERATION"] = failure
    result = subprocess.run(
        [
            powershell_executable(),
            "-NoProfile",
            "-File",
            str(wrapper),
            str(STACK_PS1),
            str(REPOSITORY_ROOT / ".env.example"),
            failure,
        ],
        cwd=REPOSITORY_ROOT,
        env=_stub_environment(directory, log, overrides),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    outcome = json.loads(result.stdout.strip().splitlines()[-1])
    assert outcome == {
        "restored": True,
        "code": {
            "none": 0,
            "dry": 0,
            "planner": 2,
            "probe": 2,
            "build": 19,
            "stop": 19,
            "run": 19,
            "up": 19,
            "verify": 2,
        }[failure],
    }, result.stderr


# --- logs ------------------------------------------------------------------


def test_service_filter_replaces_the_default_service_list(stub):
    """`--service` names exactly what Compose is asked for, and nothing else."""
    results = both_launchers(
        ["logs", "cpu", "--service", "middleware", "--tail", "5", "--dry-run"],
        ["logs", "cpu", "-Service", "middleware", "-Tail", "5", "-DryRun"],
        stub,
    )
    assert results
    for result in results:
        assert result.returncode == 0, result.stderr
        arguments = plan_arguments(result.stdout)
        assert arguments[-1] == "middleware"
        assert "smart-embedder-cpu" not in arguments
        assert "smart-qdrant-mcp" not in arguments


def test_service_filter_is_repeatable(stub):
    """Two `--service` values reach Compose as two service arguments."""
    results = both_launchers(
        [
            "logs",
            "cpu",
            "--service",
            "middleware",
            "--service",
            "smart-qdrant-mcp",
            "--dry-run",
        ],
        ["logs", "cpu", "-Service", "middleware,smart-qdrant-mcp", "-DryRun"],
        stub,
    )
    assert results
    for result in results:
        assert result.returncode == 0, result.stderr
        arguments = plan_arguments(result.stdout)
        assert arguments[-2:] == ["middleware", "smart-qdrant-mcp"]


def test_follow_and_tail_are_combinable(stub):
    """`--follow` with `--tail` is a valid pair, not a usage error."""
    results = both_launchers(
        ["logs", "cpu", "--follow", "--tail", "5", "--dry-run"],
        ["logs", "cpu", "-Follow", "-Tail", "5", "-DryRun"],
        stub,
    )
    assert results
    for result in results:
        assert result.returncode == 0, result.stderr
        arguments = plan_arguments(result.stdout)
        assert "--follow" in arguments
        assert arguments[arguments.index("--tail") + 1] == "5"


def test_tail_alone_does_not_follow(stub):
    """Asking for a fixed number of lines returns instead of following."""
    results = both_launchers(
        ["logs", "cpu", "--tail", "5", "--dry-run"],
        ["logs", "cpu", "-Tail", "5", "-DryRun"],
        stub,
    )
    assert results
    for result in results:
        assert result.returncode == 0, result.stderr
        assert "--follow" not in plan_arguments(result.stdout)


def test_no_log_option_still_follows(stub):
    """Following stays the default when neither --tail nor --since is given."""
    results = both_launchers(
        ["logs", "cpu", "--dry-run"], ["logs", "cpu", "-DryRun"], stub
    )
    assert results
    for result in results:
        assert result.returncode == 0, result.stderr
        assert "--follow" in plan_arguments(result.stdout)


def test_since_reaches_compose(stub):
    """`--since` is passed through, and suppresses the implicit follow."""
    results = both_launchers(
        ["logs", "cpu", "--since", "10m", "--dry-run"],
        ["logs", "cpu", "-Since", "10m", "-DryRun"],
        stub,
    )
    assert results
    for result in results:
        assert result.returncode == 0, result.stderr
        arguments = plan_arguments(result.stdout)
        assert arguments[arguments.index("--since") + 1] == "10m"
        assert "--follow" not in arguments


@requires_bash
@pytest.mark.parametrize(
    "arguments,expected",
    [
        (["logs", "cpu", "--tail=5", "--dry-run"], ("--tail", "5")),
        (["logs", "cpu", "--since=10m", "--dry-run"], ("--since", "10m")),
        (
            ["logs", "cpu", "--service=middleware", "--dry-run"],
            ("logs", "middleware"),
        ),
    ],
)
def test_equals_spelling_is_accepted(stub, arguments, expected):
    """`--tail=5` is the spelling `docker compose logs` teaches; accept it."""
    directory, log = stub
    result = run_bash_launcher(arguments, stub_dir=directory, log_path=log)
    assert result.returncode == 0, result.stderr
    plan = plan_arguments(result.stdout)
    flag, value = expected
    assert value in plan
    assert flag in plan


@requires_bash
def test_equals_spelling_still_validates_its_value(stub):
    """The equals form is parsed, not trusted: a bad value is still refused."""
    directory, log = stub
    result = run_bash_launcher(
        ["logs", "cpu", "--tail=zero"], stub_dir=directory, log_path=log
    )
    assert result.returncode == 2, result.stderr
    assert "--tail requires a positive integer." in result.stderr


# --- down ------------------------------------------------------------------


def test_down_flags_reach_compose(stub):
    """`--volumes` and `--remove-orphans` are passed through to `down`."""
    results = both_launchers(
        ["down", "--volumes", "--remove-orphans", "--dry-run"],
        ["down", "-Volumes", "-RemoveOrphans", "-DryRun"],
        stub,
    )
    assert results
    for result in results:
        assert result.returncode == 0, result.stderr
        arguments = plan_arguments(result.stdout)
        assert arguments[-3:] == ["down", "--volumes", "--remove-orphans"]


def test_down_volumes_is_refused_without_a_terminal(stub):
    """Nothing is removed when the destructive flag cannot be confirmed.

    The suite always runs the launchers with standard input on the null device,
    so this is the branch a script or a CI job takes: refuse, and say which flag
    would have made it explicit.
    """
    results = both_launchers(["down", "--volumes"], ["down", "-Volumes"], stub)
    assert results
    for result in results:
        assert result.returncode == 2, result.stdout
        assert "model cache" in result.stderr
        assert "down" not in " ".join(result.invocations).split("--file")[0]
        assert not any(
            invocation.rstrip().endswith("down --volumes")
            for invocation in result.invocations
        )


def test_down_volumes_proceeds_with_the_confirmation_flag(stub):
    """`--yes` is the non-interactive confirmation, and it does run `down`."""
    results = both_launchers(
        ["down", "--volumes", "--yes"], ["down", "-Volumes", "-Yes"], stub
    )
    assert results
    for result in results:
        assert result.returncode == 0, result.stderr
        assert result.reached_stub()
        assert any("--volumes" in invocation for invocation in result.invocations)


def test_plain_down_needs_no_confirmation(stub):
    """Only `--volumes` is destructive; a plain `down` keeps running unasked."""
    results = both_launchers(["down"], ["down"], stub)
    assert results
    for result in results:
        assert result.returncode == 0, result.stderr
        assert result.reached_stub()


@pytest.fixture(params=["bash", "powershell"])
def startup_controls_launcher(request):
    """Select a supported shell and its four control spellings."""
    if request.param == "bash":
        if bash_executable() is None:
            pytest.skip("bash unavailable")
        return run_bash_launcher, (
            "--force-autotune",
            "--skip-verify",
            "--autotune",
            "--dry-run",
        )
    if powershell_executable() is None:
        pytest.skip("PowerShell unavailable")
    return run_powershell_launcher, (
        "-ForceAutotune",
        "-SkipVerify",
        "-Autotune",
        "-DryRun",
    )


@pytest.mark.parametrize("failure", ["build", "stop", "run", "up"])
def test_startup_controls_stop_at_failed_operation(
    stub, startup_controls_launcher, failure
):
    directory, log = stub
    runner, (force, skip, mode, _) = startup_controls_launcher
    result = runner(
        ["up", "cpu", force, skip, mode, "quick"],
        stub_dir=directory,
        log_path=log,
        env={"STACK_STUB_FAIL_OPERATION": failure, "STACK_STUB_FAIL_CODE": "19"},
    )
    assert result.returncode == 19, result.stderr
    assert "Endpoints published by this configuration:" not in result.stdout
    calls = [
        call
        for call in result.invocations
        if any(f" {op} " in f" {call} " for op in ("build", "rm", "stop", "run", "up"))
    ]
    expected = ["build", "rm", "stop", "run", "up"]
    expected = expected[: expected.index(failure) + 1]
    assert len(calls) == len(expected), result.invocations
    for call, operation in zip(calls, expected):
        assert f" {operation} " in f" {call} "


def test_startup_controls_force_off_dry_run_refuses_before_mutation(
    stub, startup_controls_launcher
):
    directory, log = stub
    runner, (force, skip, mode, dry) = startup_controls_launcher
    result = runner(
        ["up", "cpu", force, skip, mode, "off", dry],
        stub_dir=directory,
        log_path=log,
    )
    assert result.returncode == 2
    assert "quick or full" in result.stderr
    assert not any(
        f" {op} " in f" {call} "
        for call in result.invocations
        for op in ("build", "rm", "stop", "run", "up")
    )


def test_startup_controls_force_stays_out_of_normal_process_environment(
    stub, tmp_path, startup_controls_launcher
):
    directory, log = stub
    runner, (force, skip, mode, _) = startup_controls_launcher
    no_frontend = "--no-frontend" if runner is run_bash_launcher else "-NoFrontend"
    environment_file = tmp_path / "resolved-controls.env"
    environment_file.write_text(
        "STACK_PROFILE=cpu\nSMART_EMBEDDER_AUTOTUNE=full\n"
        "SMART_EMBEDDER_AUTOTUNE_FORCE=true\nEXPOSE_FRONTEND=true\n",
        encoding="utf-8",
    )
    controls_log = tmp_path / "controls.log"
    result = runner(
        ["up", "cpu", force, skip, mode, "quick", no_frontend],
        stub_dir=directory,
        log_path=log,
        env={
            "STACK_STUB_ENVIRONMENT_FILE": str(environment_file),
            "STACK_STUB_CONTROLS_LOG": str(controls_log),
            "SMART_EMBEDDER_AUTOTUNE": "full",
            "SMART_EMBEDDER_AUTOTUNE_FORCE": "true",
            "EXPOSE_FRONTEND": "true",
        },
    )
    assert result.returncode == 0, result.stderr
    records = controls_log.read_text(encoding="utf-8").splitlines()
    assert records == ["mode=quick force=false frontend=false"] * 5
    forced_calls = [call for call in result.invocations if " run " in call]
    assert len(forced_calls) == 1
    assert "--env SMART_EMBEDDER_AUTOTUNE_FORCE=true" in forced_calls[0]
    assert forced_calls[0].endswith("smart-embedder-init-cpu")
    final_calls = [call for call in result.invocations if " up " in call]
    assert len(final_calls) == 1
    assert "SMART_EMBEDDER_AUTOTUNE_FORCE=true" not in final_calls[0]


@pytest.mark.parametrize("from_cli", [False, True])
def test_startup_controls_invalid_mode_reports_source_before_mutation(
    stub, tmp_path, startup_controls_launcher, from_cli
):
    directory, log = stub
    runner, (force, skip, mode, _) = startup_controls_launcher
    environment_file = tmp_path / "invalid-mode.env"
    environment_file.write_text(
        "STACK_PROFILE=cpu\nSMART_EMBEDDER_AUTOTUNE=invalid-configured-mode\n",
        encoding="utf-8",
    )
    arguments = ["up", "cpu", force, skip]
    if from_cli:
        arguments += [mode, "invalid-cli-mode"]
    result = runner(
        arguments,
        stub_dir=directory,
        log_path=log,
        env={"STACK_STUB_ENVIRONMENT_FILE": str(environment_file)},
    )
    assert result.returncode == 2, result.stderr
    label = "CLI --autotune" if from_cli else "Configured SMART_EMBEDDER_AUTOTUNE"
    assert label in result.stderr
    assert not any(
        f" {op} " in f" {call} "
        for call in result.invocations
        for op in ("build", "rm", "stop", "run", "up")
    )


@pytest.mark.parametrize("case", ["missing", "duplicate"])
def test_startup_controls_native_binding_boundary(
    stub, startup_controls_launcher, case
):
    directory, log = stub
    runner, (_, _, mode, _) = startup_controls_launcher
    env_flag = "--env-file" if runner is run_bash_launcher else "-EnvFile"
    arguments = [
        "up",
        "cpu",
        env_flag,
        str(REPOSITORY_ROOT / ".env.example"),
        mode,
    ]
    if case == "duplicate":
        arguments += ["quick", mode, "full"]
    result = runner(arguments, stub_dir=directory, log_path=log)
    assert result.returncode == (2 if runner is run_bash_launcher else 1)
    assert result.stderr
    assert result.invocations == ()


@pytest.mark.parametrize("value", ["", " ", "quick\n--cli-command=urls", "quick\rfull"])
def test_startup_controls_reject_empty_and_multiline_mode_before_transport(
    stub, startup_controls_launcher, value
):
    directory, log = stub
    runner, (_, _, mode, dry) = startup_controls_launcher
    if runner is run_bash_launcher:
        result = _run(
            [
                bash_executable(),
                "-c",
                'exec "$BASH" "$STACK_TEST_LAUNCHER" up cpu '
                '--autotune "$STACK_TEST_MODE" --dry-run',
            ],
            directory,
            log,
            {"STACK_TEST_LAUNCHER": str(STACK_SH), "STACK_TEST_MODE": value},
            REPOSITORY_ROOT,
        )
    else:
        result = runner(
            ["up", "cpu", mode, value, dry], stub_dir=directory, log_path=log
        )
    assert result.returncode == 2, result.stderr
    assert "one nonempty mode" in result.stderr
    assert result.invocations == ()


@requires_bash
def test_startup_controls_bash_equals_and_parent_environment(stub):
    directory, log = stub
    result = _run(
        [
            bash_executable(),
            "-c",
            '"$BASH" "$STACK_TEST_LAUNCHER" up cpu --autotune=" QUICK " '
            '--force-autotune --no-frontend --skip-verify --env-file "$STACK_TEST_ENV" '
            ">/dev/null || exit $?\n"
            'printf "mode=%s force=%s frontend=%s\\n" "$SMART_EMBEDDER_AUTOTUNE" '
            '"$SMART_EMBEDDER_AUTOTUNE_FORCE" "$EXPOSE_FRONTEND"',
        ],
        directory,
        log,
        {
            "STACK_TEST_LAUNCHER": str(STACK_SH),
            "STACK_TEST_ENV": str(REPOSITORY_ROOT / ".env.example"),
            "SMART_EMBEDDER_AUTOTUNE": "full",
            "SMART_EMBEDDER_AUTOTUNE_FORCE": "true",
            "EXPOSE_FRONTEND": "true",
        },
        REPOSITORY_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "mode=full force=true frontend=true"


# --- the bundled Qdrant profile in the recovery commands -------------------


def _enables_qdrant_profile(arguments: list[str]) -> bool:
    """Return whether an argv enables the qdrant Compose profile."""
    return any(
        arguments[index : index + 2] == ["--profile", "qdrant"]
        for index in range(len(arguments))
    )


@pytest.mark.parametrize(
    "bash_arguments,powershell_arguments",
    [
        (["down", "--dry-run"], ["down", "-DryRun"]),
        (["status", "--dry-run"], ["status", "-DryRun"]),
        (["logs", "--tail", "1", "--dry-run"], ["logs", "-Tail", "1", "-DryRun"]),
    ],
)
def test_recovery_commands_cover_the_bundled_qdrant_profile(
    stub, bash_arguments, powershell_arguments
):
    """`down` must stop a bundled Qdrant although it never asks the planner.

    The recovery commands cannot read QDRANT_BUNDLED (they work without
    `.env`), so they enable the qdrant profile unconditionally: a profile that
    is enabled but has no running container is harmless to stop or list.
    """
    results = both_launchers(bash_arguments, powershell_arguments, stub)
    assert results
    for result in results:
        assert result.returncode == 0, result.stderr
        assert _enables_qdrant_profile(plan_arguments(result.stdout))


@pytest.mark.parametrize(
    "bash_arguments,powershell_arguments",
    [
        (["status", "cpu", "--dry-run"], ["status", "cpu", "-DryRun"]),
        (
            ["logs", "cpu", "--tail", "1", "--dry-run"],
            ["logs", "cpu", "-Tail", "1", "-DryRun"],
        ),
    ],
)
def test_explicit_profile_lists_name_the_bundled_qdrant(
    stub, bash_arguments, powershell_arguments
):
    """With a hardware profile given, the explicit service list includes qdrant."""
    results = both_launchers(bash_arguments, powershell_arguments, stub)
    assert results
    for result in results:
        assert result.returncode == 0, result.stderr
        arguments = plan_arguments(result.stdout)
        assert _enables_qdrant_profile(arguments)
        assert arguments[-4:] == [
            "middleware",
            "smart-embedder-cpu",
            "smart-qdrant-mcp",
            "qdrant",
        ]


def test_down_volumes_refusal_names_the_qdrant_data(stub):
    """`--volumes` also destroys the bundled Qdrant data; the refusal says so."""
    results = both_launchers(["down", "--volumes"], ["down", "-Volumes"], stub)
    assert results
    for result in results:
        assert result.returncode == 2, result.stdout
        assert "Qdrant" in result.stderr

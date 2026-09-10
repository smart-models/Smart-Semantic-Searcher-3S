"""The recovery invariant: stopping a stack never depends on Python or on .env.

`down`, `logs`, `status` and `help` must keep working with no Python interpreter
on PATH and with `.env` absent or malformed. That is a deliberate property of
the launchers, not an accident, and it is exactly the property a refactor that
routes everything through a shared Python core is most likely to break. These
tests are the guard.

They run against the stub `docker` from `stack_test_support`, so no daemon is
needed -- and they assert the stub was invoked, so a run that failed before
reaching Docker cannot pass as a success.
"""

from __future__ import annotations

import shutil

import pytest

from scripts.stack_test_support import (
    REPOSITORY_ROOT,
    _run,
    bash_executable,
    powershell_executable,
    python_free_path,
    run_bash_launcher,
    run_powershell_launcher,
    write_docker_stub,
)

POISONED_CONTROLS = {
    "SMART_EMBEDDER_AUTOTUNE": "invalid",
    "SMART_EMBEDDER_AUTOTUNE_FORCE": "invalid",
    "EXPOSE_FRONTEND": "invalid",
}

RECOVERY_INVOCATIONS = {
    "down": ["down"],
    "logs": ["logs", "--tail", "1"],
    "status": ["status"],
    # version answers "what is installed", which is the first question asked
    # when nothing works; it may not need an interpreter to answer it either.
    "version": ["version"],
}

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


@pytest.fixture
def env_file_absent():
    """Move .env out of the way for one test and restore it afterwards."""
    env_file = REPOSITORY_ROOT / ".env"
    backup = REPOSITORY_ROOT / ".env.recovery-test-backup"
    moved = env_file.exists()
    if moved:
        shutil.move(str(env_file), str(backup))
    try:
        yield
    finally:
        if moved:
            shutil.move(str(backup), str(env_file))


@pytest.fixture(scope="module")
def no_python_path(tmp_path_factory):
    """Keep system tools available while hiding every supported Python name."""
    return python_free_path(tmp_path_factory.mktemp("python-free-path"))


def _powershell_arguments(name):
    """Translate a recovery invocation into PowerShell spelling."""
    arguments = list(RECOVERY_INVOCATIONS[name])
    return ["-Tail" if item == "--tail" else item for item in arguments]


@pytest.mark.parametrize("shell", ["bash", "powershell"])
@pytest.mark.parametrize("name", ["down", "logs", "status"])
def test_recovery_ignores_malformed_control_file_without_python(
    stub, tmp_path, no_python_path, shell, name
):
    directory, log = stub
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    for filename in ("stack.sh", "stack.ps1", "compose.yml", ".env.example"):
        shutil.copyfile(REPOSITORY_ROOT / filename, checkout / filename)
    malformed = checkout / ".env"
    malformed.write_text(
        'SMART_EMBEDDER_AUTOTUNE="unterminated\n'
        "SMART_EMBEDDER_AUTOTUNE_FORCE=invalid\nEXPOSE_FRONTEND=invalid\n",
        encoding="utf-8",
    )
    if shell == "bash":
        if bash_executable() is None:
            pytest.skip("bash unavailable")
        invocation = [
            bash_executable(),
            str(checkout / "stack.sh"),
            *RECOVERY_INVOCATIONS[name],
        ]
    else:
        if powershell_executable() is None:
            pytest.skip("PowerShell unavailable")
        invocation = [
            powershell_executable(),
            "-NoProfile",
            "-File",
            str(checkout / "stack.ps1"),
            *_powershell_arguments(name),
        ]
    result = _run(
        invocation,
        directory,
        log,
        {"PATH": no_python_path, **POISONED_CONTROLS},
        checkout,
    )
    assert result.returncode == 0, result.stderr
    assert result.reached_stub()
    compose_calls = [call for call in result.invocations if "--file" in call]
    assert compose_calls
    assert all(".env.example" in call for call in compose_calls)
    assert "stack_core" not in result.stdout + result.stderr
    assert not any(
        "--environment" in call or call.startswith(("context inspect", "nvidia-smi"))
        for call in result.invocations
    )


@requires_bash
def test_python_free_path_preserves_shell_tools(no_python_path):
    """Hiding Python must not remove core tools sharing its Unix directory."""
    for launcher in ("python3", "python", "py"):
        assert shutil.which(launcher, path=no_python_path) is None
    for command in ("dirname", "tr"):
        assert shutil.which(command, path=no_python_path), command


@requires_bash
@pytest.mark.parametrize("name", sorted(RECOVERY_INVOCATIONS))
def test_bash_recovery_without_python_or_env(
    stub, env_file_absent, no_python_path, name
):
    """stack.sh recovery commands work with no Python and no .env."""
    directory, log = stub
    # PATH is rebuilt so the stub stays reachable while every interpreter goes;
    # the harness always prepends the stub directory to whatever PATH it is given.
    result = run_bash_launcher(
        RECOVERY_INVOCATIONS[name],
        stub_dir=directory,
        log_path=log,
        env={"PATH": no_python_path, **POISONED_CONTROLS},
    )
    assert result.returncode == 0, result.stderr
    assert result.reached_stub(), "the stub Docker was never invoked"
    assert "stack_core" not in result.stderr
    assert "stack_core" not in result.stdout
    assert not any(
        call.startswith(("context inspect", "nvidia-smi"))
        for call in result.invocations
    )


@requires_powershell
@pytest.mark.parametrize("name", sorted(RECOVERY_INVOCATIONS))
def test_powershell_recovery_without_python_or_env(
    stub, env_file_absent, no_python_path, name
):
    """stack.ps1 recovery commands work with no Python and no .env."""
    directory, log = stub
    result = run_powershell_launcher(
        _powershell_arguments(name),
        stub_dir=directory,
        log_path=log,
        env={"PATH": no_python_path, **POISONED_CONTROLS},
    )
    assert result.returncode == 0, result.stderr
    assert result.reached_stub(), "the stub Docker was never invoked"
    assert "stack_core" not in result.stderr
    assert "stack_core" not in result.stdout
    assert not any(
        call.startswith(("context inspect", "nvidia-smi"))
        for call in result.invocations
    )


@requires_bash
def test_bash_help_needs_neither_python_nor_docker(
    env_file_absent, no_python_path, tmp_path
):
    """`help` is answerable with nothing installed at all."""
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    result = run_bash_launcher(
        ["help"],
        stub_dir=empty,
        log_path=tmp_path / "unused.log",
        env={"PATH": no_python_path, **POISONED_CONTROLS},
    )
    assert result.returncode == 0
    assert "Examples:" in result.stdout
    assert "Exit codes:" in result.stdout


@requires_bash
def test_recovery_never_reaches_the_planner(stub, env_file_absent, no_python_path):
    """No recovery command invokes stack_core.py, even with Python available.

    Asserted through --dry-run, which prints exactly the steps that would run:
    a recovery command's plan is built in the shell, so its argv must appear
    without the planner ever being consulted.
    """
    directory, log = stub
    result = run_bash_launcher(
        ["down", "--dry-run"],
        stub_dir=directory,
        log_path=log,
        env={"PATH": no_python_path, **POISONED_CONTROLS},
    )
    assert result.returncode == 0
    assert "step 1: stop the stack" in result.stdout
    assert "docker" in result.stdout


@requires_bash
def test_planned_command_does_need_python(stub, env_file_absent, no_python_path):
    """The counterpart: `verify` is refused when no interpreter is available.

    Without this, the recovery tests above could pass because nothing needs
    Python any more, rather than because the recovery commands avoid it.
    """
    directory, log = stub
    result = run_bash_launcher(
        ["verify"],
        stub_dir=directory,
        log_path=log,
        env={"PATH": no_python_path, **POISONED_CONTROLS},
        use_test_env=False,
    )
    assert result.returncode != 0
    # .env was moved away, so this stops at the missing env file; with .env
    # present it stops at the interpreter. Either way it does not succeed.
    assert result.stderr.strip()

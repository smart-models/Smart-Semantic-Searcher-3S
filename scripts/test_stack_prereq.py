"""Prerequisite checks of the two launchers, driven through a stub Docker.

The Compose version comparison and the daemon probe live in the shells rather
than in `stack_core.py`, because the recovery commands run them too and may not
depend on Python. They can therefore only be tested by running the real
launchers, which is what these tests do -- against the stub `docker` from
`stack_test_support`, so no Docker installation matrix and no live daemon is
needed.
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

VERSION_REFUSAL = "Docker Compose 2.24.4 or newer is required"
DAEMON_REFUSAL = "Docker daemon not reachable"

#: 2.9.0 is the case a lexical comparison gets right by accident and 2.100.1 the
#: case it gets wrong, so both are asserted alongside the boundary itself.
VERSION_CASES = (
    ("2.9.0", True),
    ("2.23.9", True),
    ("2.24.0", True),
    ("2.24.3", True),
    ("2.24.4", False),
    ("v2.24.4-desktop.1", False),
    ("2.37.1", False),
    ("2.100.1", False),
    ("3.0.0", False),
)


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


@requires_bash
@pytest.mark.parametrize("version,rejected", VERSION_CASES)
def test_bash_compose_version_gate(stub, version, rejected):
    """stack.sh refuses Compose below 2.24.4 and accepts newer versions."""
    directory, log = stub
    result = run_bash_launcher(
        ["status"],
        stub_dir=directory,
        log_path=log,
        env={"STACK_STUB_COMPOSE_VERSION": version},
    )
    assert result.reached_stub(), "the stub Docker was never invoked"
    if rejected:
        assert result.returncode == 1
        assert VERSION_REFUSAL in result.stderr
        assert version in result.stderr
    else:
        assert VERSION_REFUSAL not in result.stderr


@requires_powershell
@pytest.mark.parametrize("version,rejected", VERSION_CASES)
def test_powershell_compose_version_gate(stub, version, rejected):
    """stack.ps1 refuses Compose below 2.24.4 and accepts newer versions."""
    directory, log = stub
    result = run_powershell_launcher(
        ["status"],
        stub_dir=directory,
        log_path=log,
        env={"STACK_STUB_COMPOSE_VERSION": version},
    )
    assert result.reached_stub(), "the stub Docker was never invoked"
    if rejected:
        assert result.returncode == 1
        assert VERSION_REFUSAL in result.stderr
        assert version in result.stderr
    else:
        assert VERSION_REFUSAL not in result.stderr


@requires_bash
def test_bash_daemon_probe(stub):
    """An unreachable daemon produces the human message and exit 1, not Docker's."""
    directory, log = stub
    result = run_bash_launcher(
        ["status"],
        stub_dir=directory,
        log_path=log,
        env={"STACK_STUB_INFO_EXIT": "1"},
    )
    assert result.reached_stub()
    assert result.returncode == 1
    assert DAEMON_REFUSAL in result.stderr


@requires_powershell
def test_powershell_daemon_probe(stub):
    """An unreachable daemon produces the human message and exit 1, not Docker's."""
    directory, log = stub
    result = run_powershell_launcher(
        ["status"],
        stub_dir=directory,
        log_path=log,
        env={"STACK_STUB_INFO_EXIT": "1"},
    )
    assert result.reached_stub()
    assert result.returncode == 1
    assert DAEMON_REFUSAL in result.stderr


@requires_bash
def test_bash_compose_missing(stub):
    """A Docker without the compose plugin is refused with exit 1."""
    directory, log = stub
    result = run_bash_launcher(
        ["status"],
        stub_dir=directory,
        log_path=log,
        env={"STACK_STUB_COMPOSE_MISSING": "1"},
    )
    assert result.reached_stub()
    assert result.returncode == 1
    assert "Docker Compose v2 is not available" in result.stderr


@requires_powershell
def test_powershell_compose_missing(stub):
    """A Docker without the compose plugin is refused with exit 1."""
    directory, log = stub
    result = run_powershell_launcher(
        ["status"],
        stub_dir=directory,
        log_path=log,
        env={"STACK_STUB_COMPOSE_MISSING": "1"},
    )
    assert result.reached_stub()
    assert result.returncode == 1
    assert "Docker Compose v2 is not available" in result.stderr


@requires_bash
def test_version_probe_precedes_daemon_probe(stub):
    """An old Compose is reported even when the daemon is also unreachable."""
    directory, log = stub
    result = run_bash_launcher(
        ["status"],
        stub_dir=directory,
        log_path=log,
        env={"STACK_STUB_COMPOSE_VERSION": "2.9.0", "STACK_STUB_INFO_EXIT": "1"},
    )
    assert result.returncode == 1
    assert VERSION_REFUSAL in result.stderr
    assert DAEMON_REFUSAL not in result.stderr

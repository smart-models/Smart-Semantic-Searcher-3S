"""Shared harness for driving the two stack launchers against a stub Docker.

The prerequisite checks and the recovery-command branches live in the shells,
not in `stack_core.py`, so the only way to exercise their branches is to run the
real launchers. A matrix of real Docker installations is not available to a test
suite, so this module puts a fake `docker` first on `PATH` instead: it answers
`docker compose version --short` with a chosen string, `docker info` with a
chosen exit code, and records every invocation to a log file so a test can
assert the stub was actually reached rather than passing vacuously.

Two stub files are written, because the two launchers resolve executables
differently on Windows: Git Bash finds the extension-less `docker` shell script,
while PowerShell only considers PATHEXT entries and finds `docker.cmd`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
# The launchers live at the repository root; everything they call -- the
# planner and the two verifiers -- stays here in scripts/.
STACK_SH = REPOSITORY_ROOT / "stack.sh"
STACK_PS1 = REPOSITORY_ROOT / "stack.ps1"
_CONFIGURED_ENVIRONMENT_COMMANDS = frozenset(
    {"up", "config", "verify", "build", "pull", "restart", "urls"}
)
_PYTHON_LAUNCHER_NAMES = frozenset(
    {"python", "python.exe", "python3", "python3.exe", "py", "py.exe"}
)

#: Any invocation the stub does not model explicitly succeeds silently, so a
#: launcher under test walks as far as the branch being exercised.
_SH_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$STACK_STUB_LOG"
if [ "$1" = "info" ]; then
    exit "${STACK_STUB_INFO_EXIT:-0}"
fi
if [ "$1" = "context" ] && [ "$2" = "inspect" ]; then
    printf '"%s"\\n' "${STACK_STUB_DOCKER_ENDPOINT:-unix:///var/run/docker.sock}"
    exit "${STACK_STUB_CONTEXT_EXIT:-0}"
fi
if [ "$1" = "compose" ] && [ "$2" = "version" ]; then
    if [ -n "${STACK_STUB_COMPOSE_MISSING:-}" ]; then
        exit 1
    fi
    printf '%s\\n' "${STACK_STUB_COMPOSE_VERSION:-2.37.1}"
    exit 0
fi
for stub_argument in "$@"; do
    if [ "$stub_argument" = "--environment" ]; then
        if [ -n "${STACK_STUB_ENVIRONMENT_FILE:-}" ]; then
            cat "$STACK_STUB_ENVIRONMENT_FILE"
        fi
        exit 0
    fi
done
if [ -n "${STACK_STUB_CONTROLS_LOG:-}" ]; then
    printf 'mode=%s force=%s frontend=%s\\n' \
        "${SMART_EMBEDDER_AUTOTUNE-}" "${SMART_EMBEDDER_AUTOTUNE_FORCE-}" \
        "${EXPOSE_FRONTEND-}" >> "$STACK_STUB_CONTROLS_LOG"
fi
if [ -n "${STACK_STUB_FAIL_OPERATION:-}" ]; then
    for stub_argument in "$@"; do
        if [ "$stub_argument" = "$STACK_STUB_FAIL_OPERATION" ]; then
            exit "${STACK_STUB_FAIL_CODE:-19}"
        fi
    done
fi
exit "${STACK_STUB_OTHER_EXIT:-0}"
"""

_CMD_STUB = (
    """@echo off
>>"%STACK_STUB_LOG%" echo %*
if /I "%~1"=="info" exit /b %STACK_STUB_INFO_EXIT%
if /I "%~1"=="context" if /I "%~2"=="inspect" (
    echo "%STACK_STUB_DOCKER_ENDPOINT%"
    exit /b %STACK_STUB_CONTEXT_EXIT%
)
if /I "%~1"=="compose" if /I "%~2"=="version" (
    if defined STACK_STUB_COMPOSE_MISSING exit /b 1
    echo %STACK_STUB_COMPOSE_VERSION%
    exit /b 0
)
echo %* | findstr /C:"--environment" >nul
if not errorlevel 1 (
    if defined STACK_STUB_ENVIRONMENT_FILE type "%STACK_STUB_ENVIRONMENT_FILE%"
    exit /b 0
)
"""
    'if defined STACK_STUB_CONTROLS_LOG >>"%STACK_STUB_CONTROLS_LOG%" echo '
    "mode=%SMART_EMBEDDER_AUTOTUNE% force=%SMART_EMBEDDER_AUTOTUNE_FORCE% "
    "frontend=%EXPOSE_FRONTEND%\n"
    """
:stack_controls_scan
if "%~1"=="" goto stack_controls_done
if defined STACK_STUB_FAIL_OPERATION (
    if /I "%~1"=="%STACK_STUB_FAIL_OPERATION%" exit /b %STACK_STUB_FAIL_CODE%
)
shift
goto stack_controls_scan
:stack_controls_done
exit /b %STACK_STUB_OTHER_EXIT%
"""
)

_GPU_SH_STUB = """#!/usr/bin/env bash
printf 'nvidia-smi %s\\n' "$*" >> "$STACK_STUB_LOG"
if [ -n "${STACK_STUB_GPU_FILE:-}" ]; then
    cat "$STACK_STUB_GPU_FILE"
else
    printf '%s\\n' '0, GPU-11111111-1111-1111-1111-111111111111, 8188'
fi
exit "${STACK_STUB_GPU_EXIT:-0}"
"""

_GPU_CMD_STUB = """@echo off
>>"%STACK_STUB_LOG%" echo nvidia-smi %*
if defined STACK_STUB_GPU_FILE (
    type "%STACK_STUB_GPU_FILE%"
) else (
    echo 0, GPU-11111111-1111-1111-1111-111111111111, 8188
)
exit /b %STACK_STUB_GPU_EXIT%
"""


@dataclass(frozen=True)
class LauncherResult:
    """Outcome of one launcher run: exit code and captured streams."""

    returncode: int
    stdout: str
    stderr: str
    invocations: tuple[str, ...]

    def reached_stub(self) -> bool:
        """Report whether the stub Docker was invoked at least once."""
        return bool(self.invocations)


def powershell_executable() -> str | None:
    """Return the PowerShell executable to test with, or None if absent."""
    return shutil.which("pwsh") or shutil.which("powershell")


def _bash_candidates() -> list[str]:
    """List plausible bash executables, best first."""
    candidates: list[str] = []
    override = os.environ.get("STACK_TEST_BASH")
    if override:
        candidates.append(override)

    # Derived from git first, because on Windows the bare name `bash` resolves
    # to System32\bash.exe -- the WSL launcher, which fails with
    # "execvpe(/bin/bash) failed" when no distribution is installed. Git Bash is
    # the shell this repository's documentation tells operators to use.
    git = shutil.which("git")
    if git:
        git_root = Path(git).resolve().parent.parent
        candidates.append(str(git_root / "bin" / "bash.exe"))
        candidates.append(str(git_root / "usr" / "bin" / "bash.exe"))
    for base in (
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
        os.environ.get("LOCALAPPDATA"),
    ):
        if base:
            candidates.append(str(Path(base) / "Git" / "bin" / "bash.exe"))
            candidates.append(str(Path(base) / "Programs" / "Git" / "bin" / "bash.exe"))

    found = shutil.which("bash")
    if found:
        candidates.append(found)
    return candidates


def _is_working_bash(path: str) -> bool:
    """Report whether `path` is a bash that can actually run a command."""
    if not Path(path).is_file():
        return False
    try:
        completed = subprocess.run(
            [path, "-c", "echo stack-test-bash-ok"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except OSError:
        return False
    return completed.returncode == 0 and "stack-test-bash-ok" in completed.stdout


_BASH_CACHE: list[str | None] = []


def bash_executable() -> str | None:
    """Return a working bash executable to test with, or None if there is none.

    A bare `bash` on PATH is not enough on Windows: launched from PowerShell it
    resolves to the WSL stub in System32, which fails before reaching the
    launcher and would make every bash assertion fail for a reason that has
    nothing to do with the code under test.
    """
    if _BASH_CACHE:
        return _BASH_CACHE[0]
    for candidate in _bash_candidates():
        if _is_working_bash(candidate):
            _BASH_CACHE.append(candidate)
            return candidate
    _BASH_CACHE.append(None)
    return None


def write_docker_stub(directory: Path) -> Path:
    """Write both stub Docker executables into `directory` and return it."""
    directory.mkdir(parents=True, exist_ok=True)
    sh_stub = directory / "docker"
    sh_stub.write_text(_SH_STUB, encoding="utf-8", newline="\n")
    sh_stub.chmod(0o755)
    (directory / "docker.cmd").write_text(_CMD_STUB, encoding="utf-8", newline="\r\n")
    gpu_stub = directory / "nvidia-smi"
    gpu_stub.write_text(_GPU_SH_STUB, encoding="utf-8", newline="\n")
    gpu_stub.chmod(0o755)
    (directory / "nvidia-smi.cmd").write_text(
        _GPU_CMD_STUB, encoding="utf-8", newline="\r\n"
    )
    return directory


def _stub_environment(
    stub_dir: Path,
    log_path: Path,
    overrides: Mapping[str, str] | None,
) -> dict[str, str]:
    """Build the child environment: stub first on PATH, stub knobs applied."""
    environment = dict(os.environ)
    environment["STACK_STUB_LOG"] = str(log_path)
    environment.setdefault("STACK_STUB_COMPOSE_VERSION", "2.37.1")
    environment.setdefault("STACK_STUB_INFO_EXIT", "0")
    environment.setdefault("STACK_STUB_OTHER_EXIT", "0")
    environment.setdefault("STACK_STUB_CONTEXT_EXIT", "0")
    environment.setdefault("STACK_STUB_GPU_EXIT", "0")
    environment.setdefault("STACK_STUB_DOCKER_ENDPOINT", "unix:///var/run/docker.sock")
    for name in (
        "STACK_PROFILE",
        "SMART_EMBEDDER_PROFILE",
        "STACK_GPU_DEVICE",
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "SMART_EMBEDDER_AUTOTUNE",
        "SMART_EMBEDDER_AUTOTUNE_FORCE",
        "EXPOSE_FRONTEND",
        "STACK_STUB_FAIL_OPERATION",
        "STACK_STUB_FAIL_CODE",
        "STACK_STUB_CONTROLS_LOG",
    ):
        environment.pop(name, None)
    environment["STACK_STUB_FAIL_CODE"] = "19"
    if overrides:
        for key, value in overrides.items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value
    # Applied last, so that a test overriding PATH -- the recovery tests hand in
    # a PATH with every Python interpreter removed -- still gets the stub first
    # rather than silently falling through to the real Docker.
    environment["PATH"] = f"{stub_dir}{os.pathsep}{environment.get('PATH', '')}"
    return environment


def _run(
    command: Sequence[str],
    stub_dir: Path,
    log_path: Path,
    overrides: Mapping[str, str] | None,
    cwd: Path,
) -> LauncherResult:
    """Run one launcher command against the stub and collect its result."""
    if log_path.exists():
        log_path.unlink()
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=_stub_environment(stub_dir, log_path, overrides),
        capture_output=True,
        # Never inherit the terminal. `down --volumes` asks for confirmation
        # when standard input is a TTY, so a suite run from a real console
        # would block on a prompt nobody is there to answer; with /dev/null the
        # launcher takes its non-interactive branch on both platforms.
        stdin=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    invocations: tuple[str, ...] = ()
    if log_path.exists():
        logged = log_path.read_text(encoding="utf-8", errors="replace")
        invocations = tuple(
            line.strip() for line in logged.splitlines() if line.strip()
        )
    return LauncherResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        invocations=invocations,
    )


def _with_test_environment(
    arguments: Sequence[str],
    *,
    flag: str,
    overrides: Mapping[str, str] | None,
    enabled: bool,
) -> list[str]:
    """Give configured commands an explicit test env instead of root ``.env``."""
    prepared = list(arguments)
    if (
        not enabled
        or not prepared
        or prepared[0] not in _CONFIGURED_ENVIRONMENT_COMMANDS
    ):
        return prepared
    if flag in prepared or (
        flag == "--env-file"
        and any(argument.startswith("--env-file=") for argument in prepared)
    ):
        return prepared
    stub_environment = (overrides or {}).get("STACK_STUB_ENVIRONMENT_FILE")
    environment_file = stub_environment or str(REPOSITORY_ROOT / ".env.example")
    return [*prepared, flag, environment_file]


def run_bash_launcher(
    arguments: Sequence[str],
    *,
    stub_dir: Path,
    log_path: Path,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    use_test_env: bool = True,
) -> LauncherResult:
    """Run stack.sh with the stub Docker first on PATH."""
    bash = bash_executable()
    if bash is None:
        raise RuntimeError("bash is not available on PATH")
    prepared = _with_test_environment(
        arguments,
        flag="--env-file",
        overrides=env,
        enabled=use_test_env,
    )
    return _run(
        [bash, str(STACK_SH), *prepared],
        stub_dir,
        log_path,
        env,
        cwd or REPOSITORY_ROOT,
    )


def run_powershell_launcher(
    arguments: Sequence[str],
    *,
    stub_dir: Path,
    log_path: Path,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    use_test_env: bool = True,
) -> LauncherResult:
    """Run stack.ps1 with the stub Docker first on PATH."""
    shell = powershell_executable()
    if shell is None:
        raise RuntimeError("no PowerShell executable is available on PATH")
    prepared = _with_test_environment(
        arguments,
        flag="-EnvFile",
        overrides=env,
        enabled=use_test_env,
    )
    return _run(
        [shell, "-NoProfile", "-File", str(STACK_PS1), *prepared],
        stub_dir,
        log_path,
        env,
        cwd or REPOSITORY_ROOT,
    )


def python_free_path(mirror_root: Path) -> str:
    """Return a PATH without Python while preserving neighbouring executables.

    Used by the recovery-invariant test: the recovery commands must keep working
    with no interpreter available, so the test has to be able to take one away.
    On Unix, Python commonly shares ``/usr/bin`` with core tools such as
    ``dirname`` and ``tr``. Such directories are mirrored with only the Python
    launcher names omitted instead of being removed wholesale. On Windows,
    include the selected Git Bash's utilities even when its absolute executable
    was discovered outside the parent process's PATH.
    """
    kept: list[str] = []
    mirrors: dict[Path, str] = {}
    entries = os.environ.get("PATH", "").split(os.pathsep)
    if os.name == "nt":
        bash = bash_executable()
        if bash:
            bash_directory = Path(bash).parent
            for tools_directory in (
                bash_directory,
                bash_directory.parent / "usr" / "bin",
            ):
                if (tools_directory / "dirname.exe").is_file():
                    if str(tools_directory) not in entries:
                        entries.append(str(tools_directory))
                    break
    for index, entry in enumerate(entries):
        if not entry:
            continue
        directory = Path(entry)
        if not directory.is_dir():
            kept.append(entry)
            continue
        try:
            children = [item for item in directory.iterdir() if item.is_file()]
        except OSError:
            kept.append(entry)
            continue
        names = {item.name.lower() for item in children}
        interpreter_directory = Path(sys.executable).resolve().parent
        contains_interpreter = (
            bool(names & _PYTHON_LAUNCHER_NAMES)
            or directory.resolve(strict=False) == interpreter_directory
        )
        if not contains_interpreter:
            kept.append(entry)
            continue
        if os.name == "nt":
            continue
        resolved = directory.resolve(strict=False)
        if resolved not in mirrors:
            mirror = mirror_root / f"path-{index}"
            mirror.mkdir(parents=True, exist_ok=True)
            for item in children:
                if item.name.lower() not in _PYTHON_LAUNCHER_NAMES:
                    (mirror / item.name).symlink_to(item.resolve(strict=False))
            mirrors[resolved] = str(mirror)
        kept.append(mirrors[resolved])
    return os.pathsep.join(kept)

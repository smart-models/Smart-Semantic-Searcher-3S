"""Collect bounded host GPU facts for the shared launcher planner."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict

if __package__:
    from .stack_core import (
        EXIT_INVALID,
        ConfigurationError,
        GpuProbe,
        GpuRequest,
        parse_gpu_csv,
        parse_input,
        resolve_gpu_request,
    )
else:
    from stack_core import (
        EXIT_INVALID,
        ConfigurationError,
        GpuProbe,
        GpuRequest,
        parse_gpu_csv,
        parse_input,
        resolve_gpu_request,
    )

PROBE_TIMEOUT_SECONDS = 5
WSL_NVIDIA_SMI = "/usr/lib/wsl/lib/nvidia-smi"


class ProbeFailure(Exception):
    """A probe failure containing only a fixed diagnostic reason code."""


def local_docker_endpoint() -> bool:
    """Classify the process's effective endpoint without querying the daemon."""
    context = os.environ.get("DOCKER_CONTEXT", "").strip()
    host = os.environ.get("DOCKER_HOST", "").strip()
    if context or not host:
        argv = [shutil.which("docker") or "docker", "context", "inspect"]
        if context:
            argv.append(context)
        argv += ["--format", "{{json .Endpoints.docker.Host}}"]
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=PROBE_TIMEOUT_SECONDS,
                check=False,
            )
            if result.returncode:
                raise ProbeFailure("context_failed")
            host = json.loads(result.stdout)
        except subprocess.TimeoutExpired:
            raise ProbeFailure("context_timeout") from None
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ProbeFailure("context_failed") from None
    if not isinstance(host, str) or not host:
        raise ProbeFailure("context_failed")
    for prefix in ("unix:///", "npipe:////./pipe/"):
        if host.startswith(prefix):
            return bool(host[len(prefix) :])
    return False


def probe_gpu(request: GpuRequest) -> GpuProbe:
    """Discover physical GPUs, preserving missing-tool and failure distinctions."""
    if request.mode == "none":
        raise ValueError("GPU discovery was not requested")
    try:
        if not local_docker_endpoint():
            return GpuProbe("error", reason="unsupported_endpoint")
    except ProbeFailure as error:
        return GpuProbe("error", reason=str(error))
    executable = shutil.which("nvidia-smi")
    if not executable and sys.platform.startswith("linux"):
        if os.path.isfile(WSL_NVIDIA_SMI) and os.access(WSL_NVIDIA_SMI, os.X_OK):
            executable = WSL_NVIDIA_SMI
    if not executable:
        return GpuProbe("tool_missing")
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=index,uuid,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return GpuProbe("error", reason="timeout")
    except (OSError, UnicodeError):
        return GpuProbe("error", reason="execution_failed")
    if result.returncode:
        return GpuProbe("error", reason="query_failed")
    if result.stdout.strip() == "No devices were found":
        return GpuProbe("no_devices")
    return parse_gpu_csv(result.stdout)


def main(stdin_text: str | None = None) -> int:
    """Read resolved intent and emit one private facts record when required."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(newline="\n")
    environment, cli = parse_input(
        sys.stdin.read() if stdin_text is None else stdin_text
    )
    try:
        request = resolve_gpu_request(environment, cli)
    except ConfigurationError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return EXIT_INVALID
    if request.mode != "none":
        facts = json.dumps(asdict(probe_gpu(request)), separators=(",", ":"))
        print(f"--cli-gpu-probe={facts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

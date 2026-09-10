"""Bounded host discovery and safe transport without real Docker or GPU access."""

import json
import subprocess

import pytest

from scripts import stack_gpu_probe as probe
from scripts.stack_core import GpuProbe, GpuRequest, decode_gpu_probe

UUID = "GPU-11111111-1111-1111-1111-111111111111"


@pytest.fixture(autouse=True)
def docker_executable(monkeypatch):
    """Keep context discovery independent of the machine's Docker installation."""
    monkeypatch.setattr(
        probe.shutil, "which", lambda name: name if name == "docker" else None
    )


@pytest.fixture
def local_probe(monkeypatch):
    """Pretend a local endpoint and NVIDIA executable exist."""
    monkeypatch.setattr(probe, "local_docker_endpoint", lambda: True)
    monkeypatch.setattr(probe.shutil, "which", lambda name: "/fake/nvidia-smi")


def completed(monkeypatch, output, code=0):
    """Stub a subprocess result and return the captured calls."""
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, code, output, "secret-stderr")

    monkeypatch.setattr(probe.subprocess, "run", run)
    return calls


@pytest.mark.parametrize(
    "exception,reason",
    [
        (subprocess.TimeoutExpired("secret-command", 5), "timeout"),
        (OSError("secret-os-error"), "execution_failed"),
        (UnicodeError("secret-decoding"), "execution_failed"),
    ],
)
def test_execution_errors_have_fixed_safe_reasons(
    local_probe, monkeypatch, exception, reason
):
    def run(*args, **kwargs):
        raise exception

    monkeypatch.setattr(probe.subprocess, "run", run)
    assert probe.probe_gpu(GpuRequest("auto")) == GpuProbe("error", reason=reason)


@pytest.mark.parametrize("code", [6, 9, 12, 255])
@pytest.mark.parametrize("output", ["No devices were found\n", f"0,{UUID},8188\n"])
def test_failed_queries_never_activate_fallback(local_probe, monkeypatch, code, output):
    completed(monkeypatch, output, code)
    assert probe.probe_gpu(GpuRequest("auto")) == GpuProbe(
        "error", reason="query_failed"
    )


@pytest.mark.parametrize(
    "output,status",
    [
        ("No devices were found\r\n", "no_devices"),
        ("", "error"),
        ("N/A", "error"),
        (f"0,{UUID},8188", "ok"),
        (f"0,{UUID},8188\n1,N/A,N/A", "ok"),
    ],
)
def test_successful_queries_use_exact_csv_contract(
    local_probe, monkeypatch, output, status
):
    calls = completed(monkeypatch, output)
    assert probe.probe_gpu(GpuRequest("auto")).status == status
    assert calls == [
        (
            [
                "/fake/nvidia-smi",
                "--query-gpu=index,uuid,memory.total",
                "--format=csv,noheader,nounits",
            ],
            {"capture_output": True, "text": True, "timeout": 5, "check": False},
        )
    ]


def test_missing_tool_and_linux_wsl_fixed_path(local_probe, monkeypatch):
    monkeypatch.setattr(probe.shutil, "which", lambda name: None)
    monkeypatch.setattr(probe.sys, "platform", "linux")
    monkeypatch.setattr(probe.os.path, "isfile", lambda path: False)
    assert probe.probe_gpu(GpuRequest("auto")).status == "tool_missing"
    monkeypatch.setattr(
        probe.os.path, "isfile", lambda path: path == probe.WSL_NVIDIA_SMI
    )
    monkeypatch.setattr(probe.os, "access", lambda path, mode: True)
    calls = completed(monkeypatch, f"0,{UUID},8188")
    assert probe.probe_gpu(GpuRequest("auto")).status == "ok"
    assert calls[0][0][0] == probe.WSL_NVIDIA_SMI


def test_plain_commands_do_not_probe(monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        raise AssertionError("Host discovery was not requested")

    monkeypatch.setattr(probe.subprocess, "run", unexpected)
    monkeypatch.setattr(probe.shutil, "which", unexpected)
    assert probe.main("--cli-command=up\n--cli-hardware=cpu\n") == 0
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "endpoint,local",
    [
        ("unix:///var/run/docker.sock", True),
        ("npipe:////./pipe/dockerDesktopLinuxEngine", True),
        ("tcp://127.0.0.1:2375", False),
        ("ssh://secret-host", False),
        ("npipe:////remote/pipe/docker", False),
        ("unix:///", False),
        ("unknown", False),
    ],
)
def test_endpoint_classification_and_host_override(monkeypatch, endpoint, local):
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.setenv("DOCKER_HOST", endpoint)
    calls = completed(monkeypatch, "must not inspect")
    assert probe.local_docker_endpoint() is local
    assert calls == []


@pytest.mark.parametrize("context", ["", "chosen-context"])
def test_context_precedence_over_host_and_default_inspection(monkeypatch, context):
    monkeypatch.setenv("DOCKER_CONTEXT", context)
    monkeypatch.setenv("DOCKER_HOST", "ssh://secret-host" if context else "")
    calls = completed(monkeypatch, json.dumps("unix:///var/run/docker.sock"))
    assert probe.local_docker_endpoint() is True
    assert calls[0][0] == [
        "docker",
        "context",
        "inspect",
        *([context] if context else []),
        "--format",
        "{{json .Endpoints.docker.Host}}",
    ]
    assert calls[0][1]["timeout"] == 5


@pytest.mark.parametrize(
    "output,code",
    [
        ("secret-invalid-json", 0),
        ("null", 0),
        ('""', 0),
        ("{}", 0),
        ('"unix:///socket"', 1),
    ],
)
def test_context_failures_are_safe(monkeypatch, output, code):
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    completed(monkeypatch, output, code)
    assert probe.probe_gpu(GpuRequest("auto")) == GpuProbe(
        "error", reason="context_failed"
    )


def test_context_timeout_precedes_nvidia_discovery(monkeypatch):
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)

    def run(argv, **kwargs):
        assert argv[:3] == ["docker", "context", "inspect"]
        raise subprocess.TimeoutExpired(argv, 5)

    monkeypatch.setattr(probe.subprocess, "run", run)
    assert probe.probe_gpu(GpuRequest("auto")) == GpuProbe(
        "error", reason="context_timeout"
    )


def test_remote_endpoint_never_uses_local_inventory(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "tcp://remote:2375")
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    calls = completed(monkeypatch, "")
    assert probe.probe_gpu(GpuRequest("auto")) == GpuProbe(
        "error", reason="unsupported_endpoint"
    )
    assert calls == []


def test_main_serializes_facts_without_raw_output(local_probe, monkeypatch, capsys):
    completed(monkeypatch, f"0,{UUID},8188")
    assert probe.main("--cli-command=up\n--cli-hardware=auto\n") == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith("--cli-gpu-probe=")
    assert decode_gpu_probe(line.partition("=")[2]).devices[0].uuid == UUID
    completed(monkeypatch, "secret-stdout", 6)
    assert probe.main("--cli-command=up\n--cli-hardware=auto\n") == 0
    captured = capsys.readouterr()
    assert "secret" not in captured.out + captured.err
    assert decode_gpu_probe(captured.out.partition("=")[2]).reason == "query_failed"


def test_main_rejects_invalid_intent_before_probe(monkeypatch, capsys):
    calls = completed(monkeypatch, "")
    assert probe.main("--cli-command=up\n--cli-hardware=cpu\n--cli-gpu-device=0\n") == 2
    assert calls == []
    assert capsys.readouterr().out == ""

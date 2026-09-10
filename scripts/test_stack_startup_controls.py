"""Startup controls preserve precedence and one-invocation action semantics."""

import pytest

from scripts.stack_core import ConfigurationError, GpuDevice, GpuProbe, build_plan

BASE = {
    "command": "up",
    "hardware": "cpu",
    "compose-file": "/repo/compose.yml",
    "env-file": "/repo/.env",
    "embedder-private-file": "/repo/compose.embedder-private.yml",
    "gpu-overlay-file": "/repo/compose.gpu-device.yml",
}


def startup_plan(environment=None, **intent):
    """Build a pure CPU plan without requiring host discovery."""
    return build_plan(environment or {}, {**BASE, **intent})


@pytest.mark.parametrize("mode", ["off", "quick", "full", " FULL "])
def test_cli_mode_overrides_invalid_environment(mode):
    plan = startup_plan({"SMART_EMBEDDER_AUTOTUNE": "invalid"}, autotune=mode)
    assert plan.environment["SMART_EMBEDDER_AUTOTUNE"] == mode.strip().lower()


@pytest.mark.parametrize("mode", ["", " ", "fast", "quick\nfull", "quick\rfull"])
def test_invalid_cli_mode_is_rejected(mode):
    with pytest.raises(ConfigurationError):
        startup_plan(autotune=mode)


@pytest.mark.parametrize("command", ["up", "config"])
def test_frontend_cli_false_overrides_true_environment(command):
    plan = startup_plan(
        {"EXPOSE_FRONTEND": "true"},
        command=command,
        **{"flag-no-frontend": "true"},
    )
    assert plan.environment["EXPOSE_FRONTEND"] == "false"
    assert plan.resolved["expose_frontend"] == "false"


def test_frontend_pair_conflicts():
    with pytest.raises(ConfigurationError, match="cannot be combined"):
        startup_plan(**{"flag-frontend": "true", "flag-no-frontend": "true"})


@pytest.mark.parametrize("flag", ["flag-frontend", "flag-no-frontend"])
def test_explicit_frontend_requires_middleware(flag):
    with pytest.raises(ConfigurationError, match="middleware"):
        startup_plan({"EMBEDDER_ONLY": "true"}, **{flag: "true"})


def test_persistent_frontend_is_ignored_without_middleware():
    plan = startup_plan({"EMBEDDER_ONLY": "true", "EXPOSE_FRONTEND": "invalid"})
    assert "EXPOSE_FRONTEND" not in plan.environment


def test_full_override_allows_no_frontend():
    plan = startup_plan(
        {"EMBEDDER_ONLY": "true"},
        **{"flag-full": "true", "flag-no-frontend": "true"},
    )
    assert plan.resolved["embedder_only"] == "false"
    assert plan.environment["EXPOSE_FRONTEND"] == "false"


@pytest.mark.parametrize("frontend", ["true", "false", "1", "0", "invalid"])
def test_unflagged_service_settings_keep_existing_interpretation(frontend):
    plan = startup_plan(
        {
            "SMART_EMBEDDER_AUTOTUNE": "legacy-invalid",
            "EXPOSE_FRONTEND": frontend,
        }
    )
    assert "SMART_EMBEDDER_AUTOTUNE" not in plan.environment
    assert "EXPOSE_FRONTEND" not in plan.environment
    assert len(plan.steps) == 3


@pytest.mark.parametrize(
    "from_cli,expected_label",
    [(True, "CLI --autotune"), (False, "Configured SMART_EMBEDDER_AUTOTUNE")],
)
def test_force_invalid_mode_identifies_the_winning_source(from_cli, expected_label):
    intent = {"force-autotune": "true"}
    if from_cli:
        intent["autotune"] = "invalid-cli-mode"
    with pytest.raises(ConfigurationError) as failure:
        startup_plan({"SMART_EMBEDDER_AUTOTUNE": "invalid-configured-mode"}, **intent)
    assert expected_label in str(failure.value)


def test_force_explicit_valid_mode_overrides_invalid_configured_mode():
    plan = startup_plan(
        {"SMART_EMBEDDER_AUTOTUNE": "invalid-configured-mode"},
        autotune="quick",
        **{"force-autotune": "true"},
    )
    assert plan.environment["SMART_EMBEDDER_AUTOTUNE"] == "quick"


@pytest.mark.parametrize("hardware", ["cpu", "gpu"])
@pytest.mark.parametrize("model", ["basic", "advanced"])
def test_force_uses_one_temporary_init_and_normal_container_environment(
    hardware, model
):
    plan = startup_plan(
        {"SMART_EMBEDDER_AUTOTUNE_FORCE": "true"},
        hardware=hardware,
        **{"embedder-profile": model, "force-autotune": "true"},
    )
    assert plan.environment["SMART_EMBEDDER_AUTOTUNE_FORCE"] == "false"
    assert len(plan.steps) == 5
    build, remove_opposite, stop, forced, start = plan.steps
    assert build.argv[-1] == "build"
    assert "rm" in remove_opposite.argv
    assert stop.argv[-3:] == [
        "stop",
        f"smart-embedder-{hardware}",
        f"smart-embedder-init-{hardware}",
    ]
    assert forced.argv[-7:] == [
        "run",
        "--rm",
        "--no-deps",
        "-T",
        "--env",
        "SMART_EMBEDDER_AUTOTUNE_FORCE=true",
        f"smart-embedder-init-{hardware}",
    ]
    assert start.argv[-3:] == ["up", "--detach", "--no-build"]
    assert "--no-deps" not in start.argv
    assert sum("SMART_EMBEDDER_AUTOTUNE_FORCE=true" in s.argv for s in plan.steps) == 1


@pytest.mark.parametrize("source", ["environment", "cli"])
def test_force_with_off_is_rejected(source):
    environment = {"SMART_EMBEDDER_AUTOTUNE": "off"} if source == "environment" else {}
    intent = {"autotune": "off"} if source == "cli" else {}
    with pytest.raises(ConfigurationError, match="quick or full"):
        startup_plan(environment, **intent, **{"force-autotune": "true"})


@pytest.mark.parametrize(
    "command", ["config", "build", "pull", "restart", "verify", "urls"]
)
def test_force_is_only_an_up_action(command):
    with pytest.raises(ConfigurationError, match="only by up"):
        startup_plan(command=command, **{"force-autotune": "true"})


def test_persistent_force_alone_does_not_add_an_action():
    plan = startup_plan({"SMART_EMBEDDER_AUTOTUNE_FORCE": "true"})
    assert "SMART_EMBEDDER_AUTOTUNE_FORCE" not in plan.environment
    assert not any("run" in step.argv for step in plan.steps)


@pytest.mark.parametrize(
    "configured,expected", [(None, "quick"), ("", "quick"), (" FULL ", "full")]
)
def test_force_uses_configured_mode_or_default(configured, expected):
    environment = {} if configured is None else {"SMART_EMBEDDER_AUTOTUNE": configured}
    plan = startup_plan(environment, **{"force-autotune": "true"})
    assert plan.environment["SMART_EMBEDDER_AUTOTUNE"] == expected


@pytest.mark.parametrize("command", ["up", "config"])
def test_frontend_true_overrides_configured_false(command):
    plan = startup_plan(
        {"EXPOSE_FRONTEND": "false"}, command=command, **{"flag-frontend": "true"}
    )
    assert plan.environment["EXPOSE_FRONTEND"] == "true"


@pytest.mark.parametrize("command", ["build", "pull", "restart", "verify", "urls"])
@pytest.mark.parametrize(
    "control",
    [{"autotune": "off"}, {"flag-frontend": "true"}, {"flag-no-frontend": "true"}],
)
def test_startup_controls_are_rejected_outside_up_and_config(command, control):
    with pytest.raises(ConfigurationError, match="up and config only"):
        startup_plan(command=command, **control)


@pytest.mark.parametrize("mode", ["off", "quick", "full"])
def test_config_resolves_controls_without_starting_services(mode):
    plan = startup_plan(command="config", autotune=mode, **{"flag-no-frontend": "true"})
    assert len(plan.steps) == 1
    assert plan.steps[0].argv[-1] == "config"
    assert plan.environment["SMART_EMBEDDER_AUTOTUNE"] == mode
    assert "SMART_EMBEDDER_AUTOTUNE_FORCE" not in plan.environment


@pytest.mark.parametrize("embedder_only", [False, True])
def test_force_preserves_selected_gpu_overlays_wait_and_optional_services(
    embedder_only,
):
    uuid = "GPU-22222222-2222-2222-2222-222222222222"
    intent = {
        **BASE,
        "hardware": "auto",
        "embedder-profile": "basic",
        "gpu-device": "1",
        "force-autotune": "true",
        "no-build": "true",
        "wait-timeout": "75",
        "flag-private-embedder": "true",
        "flag-no-mcp": "true",
        "flag-embedder-only": str(embedder_only).lower(),
    }
    plan = build_plan(
        {}, intent, gpu_probe=GpuProbe("ok", (GpuDevice(1, uuid, 24564),))
    )
    assert plan.environment["STACK_GPU_DEVICE"] == uuid
    assert plan.resolved["embedder_profile"] == "basic"
    assert not any("build" in step.argv for step in plan.steps)
    for step in plan.steps:
        assert step.argv.index(BASE["embedder-private-file"]) < step.argv.index(
            BASE["gpu-overlay-file"]
        )
    stop, forced, start = plan.steps[-3:]
    assert stop.argv[-2:] == ["smart-embedder-gpu", "smart-embedder-init-gpu"]
    assert forced.argv[-1] == "smart-embedder-init-gpu"
    assert not {"--user", "--entrypoint", "--volume", "--privileged"}.intersection(
        forced.argv
    )
    assert start.argv[start.argv.index("--wait") :][:3] == [
        "--wait",
        "--wait-timeout",
        "75",
    ]
    assert "--no-deps" not in start.argv
    if embedder_only:
        assert start.argv[-1] == "smart-embedder-gpu"
        assert plan.steps[1].argv[-2:] == ["middleware", "smart-qdrant-mcp"]
    else:
        assert start.argv[-2:] == ["--scale", "smart-qdrant-mcp=0"]


def test_auto_cpu_fallback_keeps_explicit_model_and_forces_cpu_init():
    plan = build_plan(
        {},
        {
            **BASE,
            "hardware": "auto",
            "embedder-profile": "advanced",
            "force-autotune": "true",
        },
        gpu_probe=GpuProbe("tool_missing"),
    )
    assert plan.resolved["hardware"] == "cpu"
    assert plan.resolved["embedder_profile"] == "advanced"
    assert plan.steps[-2].argv[-1] == "smart-embedder-init-cpu"

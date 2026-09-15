"""The planner's full decision matrix, its refusals, and its precedence rule.

`stack_core.py` is a pure function from (environment, intent) to a plan, so the
whole matrix -- two hardware profiles times two model profiles times the three
launcher switches -- can be asserted without Docker, a daemon, or a shell.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from dataclasses import asdict

import pytest

from scripts.stack_core import (
    EXIT_INVALID,
    ConfigurationError,
    GpuDevice,
    GpuProbe,
    build_plan,
    decode_gpu_probe,
    main,
    parse_gpu_csv,
    parse_input,
    render_shell,
    resolve_gpu_request,
    select_gpu,
)

COMPOSE = "/repo/compose.yml"
ENV_FILE = "/repo/.env"
OVERLAY = "/repo/compose.embedder-private.yml"

BASE_INTENT = {
    "compose-file": COMPOSE,
    "env-file": ENV_FILE,
    "embedder-private-file": OVERLAY,
}


@pytest.mark.parametrize(
    "memory_mib,expected",
    [
        (6144, "basic"),
        (7679, "basic"),
        (7680, "advanced"),
        (8188, "advanced"),
        (8191, "advanced"),
        (8192, "advanced"),
    ],
)
def test_auto_model_uses_reported_integer_mib(memory_mib, expected):
    gpu = GpuDevice(0, "GPU-11111111-1111-1111-1111-111111111111", memory_mib)
    plan = build_plan(
        {"SMART_EMBEDDER_PROFILE": "basic"},
        {
            **BASE_INTENT,
            "command": "up",
            "hardware": "auto",
            "gpu-overlay-file": "/repo/compose.gpu-device.yml",
        },
        gpu_probe=GpuProbe("ok", (gpu,)),
    )
    assert plan.resolved["hardware"] == "gpu"
    assert plan.environment["SMART_EMBEDDER_PROFILE"] == expected
    assert plan.environment["STACK_GPU_DEVICE"] == gpu.uuid


SMALL_GPU = GpuDevice(0, "GPU-11111111-1111-1111-1111-111111111111", 4096)
LARGE_GPU = GpuDevice(1, "GPU-22222222-2222-2222-2222-222222222222", 24564)
GPU_OVERLAY = "/repo/compose.gpu-device.yml"


def gpu_plan(environment=None, probe=None, **intent):
    """Build an automatic plan with a default heterogeneous fake inventory."""
    return build_plan(
        environment or {},
        {
            **BASE_INTENT,
            "command": "up",
            "hardware": "auto",
            "gpu-overlay-file": GPU_OVERLAY,
            **intent,
        },
        gpu_probe=probe
        if probe is not None
        else GpuProbe("ok", (SMALL_GPU, LARGE_GPU)),
    )


def test_auto_pins_largest_and_ties_use_uuid_independent_of_inventory_order():
    tied = GpuDevice(8, LARGE_GPU.uuid, SMALL_GPU.memory_mib)
    for devices in ((SMALL_GPU, LARGE_GPU), (LARGE_GPU, SMALL_GPU)):
        assert select_gpu(devices) == LARGE_GPU
    for devices in ((SMALL_GPU, tied), (tied, SMALL_GPU)):
        assert select_gpu(devices) == SMALL_GPU


@pytest.mark.parametrize("selector", ["0", "000", SMALL_GPU.uuid, "gpu-1111"])
def test_auto_explicit_gpu_zero_uses_its_small_capacity(selector):
    plan = gpu_plan({"SMART_EMBEDDER_PROFILE": "advanced"}, **{"gpu-device": selector})
    assert plan.resolved["hardware"] == "gpu"
    assert plan.environment["SMART_EMBEDDER_PROFILE"] == "basic"
    assert plan.environment["STACK_GPU_DEVICE"] == SMALL_GPU.uuid
    assert plan.resolved["gpu_uuid"] == SMALL_GPU.uuid
    assert plan.warn_if_gpu is False
    assert any("4096 MiB total" in message for message in plan.messages)


def test_explicit_gpu_advanced_keeps_model_and_selected_gpu_one():
    small = GpuDevice(1, LARGE_GPU.uuid, 4096)
    plan = gpu_plan(
        {"SMART_EMBEDDER_PROFILE": "basic"},
        GpuProbe("ok", (small,)),
        hardware="gpu",
        **{"embedder-profile": "advanced", "gpu-device": "1"},
    )
    assert plan.resolved["hardware"] == "gpu"
    assert plan.environment["SMART_EMBEDDER_PROFILE"] == "advanced"
    assert plan.environment["STACK_GPU_DEVICE"] == small.uuid
    assert plan.resolved["gpu_uuid"] == small.uuid


@pytest.mark.parametrize("status", ["tool_missing", "no_devices"])
@pytest.mark.parametrize("model", [None, "basic", "advanced"])
def test_auto_absence_uses_cpu_with_explicit_model_precedence(status, model):
    cli = {"embedder-profile": model} if model else {}
    plan = gpu_plan({"SMART_EMBEDDER_PROFILE": "advanced"}, GpuProbe(status), **cli)
    assert plan.resolved["hardware"] == "cpu"
    assert plan.environment["SMART_EMBEDDER_PROFILE"] == (model or "basic")
    assert "STACK_GPU_DEVICE" not in plan.environment
    assert all(GPU_OVERLAY not in step.argv for step in plan.steps)
    assert plan.warn_if_gpu is False
    assert bool(plan.warnings) == (status == "tool_missing")


@pytest.mark.parametrize("model", ["basic", "advanced"])
def test_auto_explicit_model_overrides_capacity_and_invalid_environment(model):
    plan = gpu_plan({"SMART_EMBEDDER_PROFILE": "bad"}, **{"embedder-profile": model})
    assert plan.environment["SMART_EMBEDDER_PROFILE"] == model
    assert any("explicit CLI model" in message for message in plan.messages)


@pytest.mark.parametrize(
    "selector",
    [
        "GPU-",
        "gpu-",
        "GPU--",
        "abcd",
        "GPU-zz",
        "GPU-1-2",
        "GPU-123456789",
        "none",
        "all",
        "void",
        "MIG-123",
        "0,1",
        "-1",
        "０",
        "",
    ],
)
def test_gpu_selector_invalid_grammar_is_rejected_before_matching(selector):
    with pytest.raises(ConfigurationError):
        gpu_plan(probe=GpuProbe("ok", (SMALL_GPU,)), **{"gpu-device": selector})


def test_digits_alone_are_an_index_and_unknown_selectors_list_usable_devices():
    request = resolve_gpu_request(
        {}, {"command": "up", "hardware": "auto", "gpu-device": "2222"}
    )
    assert request.selector == "2222"
    with pytest.raises(ConfigurationError, match="available: 0 .*4096 MiB"):
        select_gpu((SMALL_GPU,), request.selector)
    gpu = GpuDevice(2222, SMALL_GPU.uuid, 4096)
    assert select_gpu((gpu,), "2222") == gpu


def test_ambiguous_uuid_prefix_never_picks_the_first_match():
    other = GpuDevice(2, "GPU-11111111-2222-2222-2222-222222222222", 8192)
    with pytest.raises(ConfigurationError, match="ambiguous"):
        select_gpu((SMALL_GPU, other), "GPU-1111")


@pytest.mark.parametrize(
    "selector",
    [
        "GPU-ABCDEFAB-1234-ABCD-ABCD-ABCDEFABCDEF",
        "gPu-AbCdEfAb-1234-aB",
    ],
)
def test_uppercase_hex_inventory_and_partial_uuid_are_canonicalized(selector):
    uuid = "GPU-abcdefab-1234-abcd-abcd-abcdefabcdef"
    probe = parse_gpu_csv(f"0,{uuid.upper()},8188")
    plan = gpu_plan(probe=probe, **{"gpu-device": selector})
    assert plan.resolved["gpu_uuid"] == uuid


def test_selector_precedence_and_explicit_cpu_rules():
    plan = gpu_plan({"STACK_GPU_DEVICE": "1"}, **{"gpu-device": "0"})
    assert plan.resolved["gpu_uuid"] == SMALL_GPU.uuid
    plan = gpu_plan({"STACK_GPU_DEVICE": "GPU-2222"}, hardware="gpu")
    assert plan.resolved["gpu_uuid"] == LARGE_GPU.uuid
    plan = gpu_plan({"STACK_GPU_DEVICE": "invalid"}, hardware="cpu")
    assert "STACK_GPU_DEVICE" not in plan.environment
    with pytest.raises(ConfigurationError, match="cpu"):
        gpu_plan(hardware="cpu", **{"gpu-device": "0"})


@pytest.mark.parametrize("status", ["tool_missing", "no_devices", "error"])
def test_selected_gpu_never_falls_back_to_cpu(status):
    with pytest.raises(ConfigurationError):
        gpu_plan(
            probe=GpuProbe(status, reason="timeout" if status == "error" else ""),
            **{"gpu-device": "0"},
        )


@pytest.mark.parametrize(
    "command", ["restart", "urls", "verify", "down", "status", "logs"]
)
def test_gpu_options_are_limited_to_selection_commands(command):
    with pytest.raises(ConfigurationError):
        resolve_gpu_request({}, {"command": command, "hardware": "auto"})
    with pytest.raises(ConfigurationError):
        resolve_gpu_request({}, {"command": command, "gpu-device": "0"})
    assert (
        resolve_gpu_request(
            {"STACK_GPU_DEVICE": "invalid"}, {"command": command, "hardware": "gpu"}
        ).mode
        == "none"
    )


def test_auto_is_cli_only_and_winning_cli_can_override_invalid_environment():
    with pytest.raises(ConfigurationError):
        resolve_gpu_request({"STACK_PROFILE": "auto"}, {"command": "up"})
    assert gpu_plan({"STACK_PROFILE": "auto"}).resolved["hardware"] == "gpu"
    assert (
        gpu_plan({"STACK_PROFILE": "auto"}, hardware="cpu").resolved["hardware"]
        == "cpu"
    )


def test_gpu_selection_is_in_every_compose_step_after_private_overlay():
    plan = gpu_plan(**{"flag-private-embedder": "true"})
    for step in plan.steps:
        assert step.argv.index(OVERLAY) < step.argv.index(GPU_OVERLAY)
    assert plan.environment["STACK_GPU_DEVICE"] == LARGE_GPU.uuid


def test_csv_canonicalizes_and_skips_unavailable_records_with_warning():
    facts = parse_gpu_csv(f" 0, {SMALL_GPU.uuid.lower()}, 4096\r\n1, [N/A], [N/A]\r\n")
    assert facts == GpuProbe("ok", (SMALL_GPU,), skipped_rows=1)
    assert "excluded 1" in gpu_plan(probe=facts).warnings[0]


@pytest.mark.parametrize(
    "csv",
    [
        "",
        "N/A",
        "0,GPU-invalid,8192",
        f"0,{SMALL_GPU.uuid},0",
        f"0,{SMALL_GPU.uuid},-1",
        f"0,{SMALL_GPU.uuid},4096\n0,{LARGE_GPU.uuid},8192",
        f"0,{SMALL_GPU.uuid},4096\n1,{SMALL_GPU.uuid},8192",
        '"unterminated',
    ],
)
def test_unusable_or_duplicate_inventory_is_an_error(csv):
    assert parse_gpu_csv(csv) == GpuProbe("error", reason="invalid_inventory")


def test_probe_record_round_trips_through_its_json_form():
    facts = GpuProbe("ok", (SMALL_GPU,), skipped_rows=1)
    assert decode_gpu_probe(json.dumps(asdict(facts))) == facts


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "secret-malformed",
        '{"status": "ok"}',
        '{"status": "ok", "devices": [{"uuid": "x"}], "skipped_rows": 0, "reason": ""}',
    ],
)
def test_truncated_probe_record_is_rejected_without_payload(payload):
    with pytest.raises(
        ConfigurationError, match="^GPU discovery facts are malformed.$"
    ):
        decode_gpu_probe(payload)


def test_unrecognised_probe_status_is_still_refused_by_the_planner():
    """The status check that matters lives in resolve_settings, not the decoder."""
    facts = GpuProbe("unknown", (SMALL_GPU,))
    with pytest.raises(
        ConfigurationError, match="^GPU discovery facts are malformed.$"
    ):
        gpu_plan(probe=facts)


def test_private_probe_transport_reaches_main_and_rejects_invalid_json(capsys):
    intent = {
        **BASE_INTENT,
        "command": "up",
        "hardware": "auto",
        "gpu-overlay-file": GPU_OVERLAY,
    }
    text = "".join(f"--cli-{name}={value}\n" for name, value in intent.items())
    assert (
        main(
            text + "--cli-gpu-probe=" + json.dumps(asdict(GpuProbe("ok", (SMALL_GPU,))))
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["resolved"]["gpu_uuid"] == SMALL_GPU.uuid
    assert main(text + "--cli-gpu-probe=secret-malformed") == EXIT_INVALID
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "secret-malformed" not in captured.err


def plan_for(command="up", environment=None, **intent):
    """Build a plan from a base intent plus the given overrides."""
    cli = dict(BASE_INTENT)
    cli["command"] = command
    cli.update(intent)
    return build_plan(environment or {}, cli)


def argv_of(plan, index):
    """Return the argv of one step."""
    return plan.steps[index].argv


def expected_base(private=False):
    """Return the Compose prefix every step of a plan shares."""
    argv = ["docker", "compose", "--file", COMPOSE, "--env-file", ENV_FILE]
    if private:
        argv += ["--file", OVERLAY]
    return argv


# --- the plan matrix -------------------------------------------------------

MATRIX = list(
    itertools.product(
        ("cpu", "gpu"),
        ("basic", "advanced"),
        (False, True),  # embedder only
        (True, False),  # mcp enabled
        (True, False),  # embedder port published
    )
)


@pytest.mark.parametrize("hardware,model,embedder_only,mcp,published", MATRIX)
def test_up_plan_matrix(hardware, model, embedder_only, mcp, published):
    """Every combination of the five dimensions produces the documented argv."""
    plan = plan_for(
        "up",
        hardware=hardware,
        **{
            "embedder-profile": model,
            "embedder-only": str(embedder_only).lower(),
            "mcp-server-enabled": str(mcp).lower(),
            "expose-embedder-port": str(published).lower(),
        },
    )
    base = expected_base(private=not published)
    service = f"smart-embedder-{hardware}"
    opposite = "gpu" if hardware == "cpu" else "cpu"

    expected = [
        [*base, "--profile", hardware, "build"] + ([service] if embedder_only else []),
        [
            *base,
            "--profile",
            "cpu",
            "--profile",
            "gpu",
            "rm",
            "--stop",
            "--force",
            f"smart-embedder-{opposite}",
            f"smart-embedder-init-{opposite}",
        ],
    ]
    if embedder_only:
        expected.append(
            [*base, "rm", "--stop", "--force", "middleware", "smart-qdrant-mcp"]
        )
    up_argv = [*base, "--profile", hardware, "up", "--detach", "--no-build"]
    if embedder_only:
        up_argv.append(service)
    elif not mcp:
        up_argv += ["--scale", "smart-qdrant-mcp=0"]
    expected.append(up_argv)

    assert [step.argv for step in plan.steps] == expected
    assert plan.environment["SMART_EMBEDDER_PROFILE"] == model
    assert plan.warn_if_gpu is (hardware == "gpu")


def test_embedder_only_and_no_mcp_do_not_both_scale():
    """embedder-only uses the explicit service list, never --scale as well."""
    plan = plan_for(
        "up",
        hardware="cpu",
        **{"embedder-only": "true", "mcp-server-enabled": "false"},
    )
    assert "--scale" not in argv_of(plan, -1)
    assert argv_of(plan, -1)[-1] == "smart-embedder-cpu"


def test_no_build_omits_the_build_step():
    """--no-build removes the build step and nothing else."""
    with_build = plan_for("up", hardware="cpu")
    without = plan_for("up", hardware="cpu", **{"no-build": "true"})
    assert len(without.steps) == len(with_build.steps) - 1
    assert "build" not in [step.argv[-1] for step in without.steps]


def test_wait_is_appended_to_the_up_step_only():
    """--wait reaches the up step and no other."""
    plan = plan_for("up", hardware="cpu", wait="true")
    assert "--wait" in argv_of(plan, -1)
    assert all("--wait" not in step.argv for step in plan.steps[:-1])


def test_build_precedes_removal_of_the_opposite_profile():
    """A failed build must leave the active profile untouched, so build is first."""
    plan = plan_for("up", hardware="cpu")
    assert plan.steps[0].argv[-1] == "build"
    assert "smart-embedder-gpu" in plan.steps[1].argv


def test_overlay_is_shared_by_config_and_up():
    """`config` shows what `up` runs, overlay included."""
    intent = {"expose-embedder-port": "false"}
    up_plan = plan_for("up", hardware="cpu", **intent)
    config_plan = plan_for("config", hardware="cpu", **intent)
    for plan in (up_plan, config_plan):
        for step in plan.steps:
            assert step.argv[:8] == expected_base(private=True)


def test_config_plan_is_one_step():
    """`config` renders in a single step, with the embedder-only service list."""
    plan = plan_for("config", hardware="gpu")
    assert [step.argv for step in plan.steps] == [
        [*expected_base(), "--profile", "gpu", "config"]
    ]
    only = plan_for("config", hardware="gpu", **{"embedder-only": "true"})
    assert argv_of(only, 0)[-1] == "smart-embedder-gpu"


def test_verify_plans_no_steps_but_resolves_switches():
    """`verify` runs no Compose, yet still resolves what the preflight reports."""
    plan = plan_for("verify", **{"embedder-only": "true"})
    assert plan.steps == []
    assert plan.resolved["embedder_only"] == "true"
    assert plan.resolved["mcp_server_enabled"] == "true"


# --- precedence ------------------------------------------------------------


def test_cli_beats_environment():
    """A CLI value wins over the Compose-resolved environment."""
    plan = plan_for("up", environment={"STACK_PROFILE": "gpu"}, hardware="cpu")
    assert plan.resolved["hardware"] == "cpu"


def test_environment_is_used_when_no_cli_value():
    """STACK_PROFILE is used when the command line says nothing."""
    plan = plan_for("up", environment={"STACK_PROFILE": "gpu"})
    assert plan.resolved["hardware"] == "gpu"


def test_defaults_apply_when_neither_source_speaks():
    """The built-in defaults are the last step of the precedence chain."""
    plan = plan_for("up", hardware="cpu")
    assert plan.resolved["embedder_profile"] == "basic"
    assert plan.resolved["embedder_only"] == "false"
    assert plan.resolved["expose_embedder_port"] == "true"


def test_absent_mcp_server_enabled_means_true():
    """A .env written before the switch existed keeps starting the full stack."""
    plan = plan_for("up", hardware="cpu")
    assert plan.resolved["mcp_server_enabled"] == "true"
    assert "--scale" not in argv_of(plan, -1)


def test_empty_environment_value_falls_through_to_the_default():
    """An empty value is not a value; MCP_SERVER_ENABLED= still means true."""
    plan = plan_for("up", environment={"MCP_SERVER_ENABLED": ""}, hardware="cpu")
    assert plan.resolved["mcp_server_enabled"] == "true"


@pytest.mark.parametrize("value", ["CPU", " cpu ", "Cpu"])
def test_values_are_normalised(value):
    """Casing and surrounding space never change the resolved value."""
    plan = plan_for("up", hardware=value)
    assert plan.resolved["hardware"] == "cpu"


# --- switch overrides, wait timeout, project name --------------------------


@pytest.mark.parametrize(
    "flag,setting,expected",
    [
        ("flag-embedder-only", "embedder_only", "true"),
        ("flag-full", "embedder_only", "false"),
        ("flag-mcp", "mcp_server_enabled", "true"),
        ("flag-no-mcp", "mcp_server_enabled", "false"),
        ("flag-public-embedder", "expose_embedder_port", "true"),
        ("flag-private-embedder", "expose_embedder_port", "false"),
    ],
)
def test_switch_flag_beats_the_environment(flag, setting, expected):
    """Each override flag wins over the opposite value in .env."""
    opposite = {
        "embedder_only": ("EMBEDDER_ONLY", "false" if expected == "true" else "true"),
        "mcp_server_enabled": (
            "MCP_SERVER_ENABLED",
            "false" if expected == "true" else "true",
        ),
        "expose_embedder_port": (
            "EXPOSE_SMART_EMBEDDER_PORT",
            "false" if expected == "true" else "true",
        ),
    }[setting]
    plan = plan_for(
        "up",
        environment={opposite[0]: opposite[1], "STACK_PROFILE": "cpu"},
        **{flag: "true"},
    )
    assert plan.resolved[setting] == expected


@pytest.mark.parametrize(
    "true_flag,false_flag,message",
    [
        ("flag-embedder-only", "flag-full", "--embedder-only and --full"),
        ("flag-mcp", "flag-no-mcp", "--mcp and --no-mcp"),
        (
            "flag-public-embedder",
            "flag-private-embedder",
            "--public-embedder and --private-embedder",
        ),
    ],
)
def test_opposed_flags_are_refused(true_flag, false_flag, message):
    """Both halves of a pair is a refusal, not last-one-wins."""
    with pytest.raises(ConfigurationError) as error:
        plan_for("up", hardware="cpu", **{true_flag: "true", false_flag: "true"})
    assert message in str(error.value)


def test_unset_flags_do_not_override_the_environment():
    """A flag passed as false is "not asked for", not an explicit false."""
    plan = plan_for(
        "up",
        environment={"EMBEDDER_ONLY": "true", "STACK_PROFILE": "cpu"},
        **{"flag-embedder-only": "false", "flag-full": "false"},
    )
    assert plan.resolved["embedder_only"] == "true"


def test_wait_timeout_implies_wait():
    """--wait-timeout has no other meaning, so it turns --wait on by itself."""
    plan = plan_for("up", hardware="cpu", **{"wait-timeout": "60"})
    argv = argv_of(plan, -1)
    assert "--wait" in argv
    assert argv[argv.index("--wait-timeout") + 1] == "60"


def test_wait_uses_the_default_timeout():
    """--wait alone still bounds the wait, rather than waiting forever."""
    plan = plan_for("up", hardware="cpu", wait="true")
    argv = argv_of(plan, -1)
    assert argv[argv.index("--wait-timeout") + 1] == "1800"


def test_wait_timeout_is_absent_without_wait():
    """Neither flag means Compose is not asked to wait at all."""
    assert "--wait-timeout" not in argv_of(plan_for("up", hardware="cpu"), -1)


def test_wait_timeout_must_be_numeric():
    """A non-numeric timeout is refused with a message naming the flag."""
    with pytest.raises(ConfigurationError) as error:
        plan_for("up", hardware="cpu", **{"wait-timeout": "soon"})
    assert "--wait-timeout must be a positive integer" in str(error.value)


def test_project_name_precedes_the_file_arguments():
    """--project-name is a Compose global, so it comes before --file."""
    plan = plan_for("up", hardware="cpu", **{"project-name": "pse-test"})
    for step in plan.steps:
        assert step.argv[:4] == ["docker", "compose", "--project-name", "pse-test"]


def test_project_name_is_omitted_when_unset():
    """No --project-name means Compose keeps its own default project."""
    plan = plan_for("up", hardware="cpu")
    assert all("--project-name" not in step.argv for step in plan.steps)


# --- refusals --------------------------------------------------------------


@pytest.mark.parametrize(
    "intent,fragment",
    [
        ({"hardware": "bogus"}, "Hardware profile must be"),
        ({"embedder-profile": "medium"}, "SMART_EMBEDDER_PROFILE must be"),
        ({"embedder-only": "yes"}, "EMBEDDER_ONLY must be true or false"),
        ({"mcp-server-enabled": "1"}, "MCP_SERVER_ENABLED must be true or false"),
        (
            {"expose-embedder-port": "maybe"},
            "EXPOSE_SMART_EMBEDDER_PORT must be true or false",
        ),
    ],
)
def test_invalid_values_are_refused(intent, fragment):
    """Every invalid value is refused with a message naming the setting."""
    payload = {"hardware": "cpu"}
    payload.update(intent)
    with pytest.raises(ConfigurationError) as error:
        plan_for("up", **payload)
    assert fragment in str(error.value)


def test_missing_hardware_is_refused_for_up():
    """`up` needs a hardware profile from the CLI or from STACK_PROFILE."""
    with pytest.raises(ConfigurationError) as error:
        plan_for("up")
    assert "requires cpu or gpu" in str(error.value)


def test_missing_hardware_is_accepted_for_verify():
    """`verify` checks both hardware profiles itself and needs neither."""
    assert plan_for("verify").steps == []


def test_invalid_environment_value_is_refused_too():
    """A bad value in .env is refused exactly like a bad value on the CLI."""
    with pytest.raises(ConfigurationError):
        plan_for("up", environment={"EMBEDDER_ONLY": "sometimes"}, hardware="cpu")


def test_unplanned_command_is_refused():
    """A command the planner does not know is a configuration error, not a crash."""
    with pytest.raises(ConfigurationError):
        plan_for("down", hardware="cpu")


# --- the two output formats ------------------------------------------------


def test_parse_input_splits_intent_from_environment():
    """The --cli- prefix is what separates parsed intent from the environment."""
    environment, cli = parse_input("A=1\n--cli-command=up\nB=2=3\n")
    assert environment == {"A": "1", "B": "2=3"}
    assert cli == {"command": "up"}


def test_render_shell_round_trips_every_argument():
    """The bash rendering carries every argument of every step, in order."""
    plan = plan_for("up", hardware="cpu", **{"embedder-only": "true"})
    rendered = render_shell(plan)
    arguments = [
        line[len("arg ") :] for line in rendered.splitlines() if line.startswith("arg ")
    ]
    assert arguments == [argument for step in plan.steps for argument in step.argv]
    assert rendered.count("step ") == len(plan.steps)


def test_render_shell_refuses_a_value_with_a_line_break():
    """The line-oriented format cannot represent a newline, so it refuses one."""
    plan = plan_for("up", hardware="cpu")
    plan.steps[0].argv.append("two\nlines")
    with pytest.raises(ConfigurationError):
        render_shell(plan)


def _run_core(stdin_text, *arguments):
    """Run stack_core.py as a subprocess, the way the launchers do."""
    return subprocess.run(
        [sys.executable, "scripts/stack_core.py", *arguments],
        input=stdin_text,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_entrypoint_emits_json_and_exits_zero():
    """The default output is one JSON object and exit 0."""
    payload = "\n".join(
        [
            "STACK_PROFILE=cpu",
            "--cli-command=up",
            f"--cli-compose-file={COMPOSE}",
            f"--cli-env-file={ENV_FILE}",
            f"--cli-embedder-private-file={OVERLAY}",
        ]
    )
    result = _run_core(payload)
    assert result.returncode == 0
    document = json.loads(result.stdout)
    assert document["steps"][0]["argv"][:2] == ["docker", "compose"]
    assert document["resolved"]["hardware"] == "cpu"


def test_cli_entrypoint_exits_two_on_a_refusal():
    """A refused configuration exits 2 with the message on stderr, not stdout."""
    result = _run_core("--cli-command=up\n--cli-hardware=bogus\n")
    assert result.returncode == EXIT_INVALID
    assert "Hardware profile must be" in result.stderr
    assert result.stdout == ""


# --- the commands added on top of up, config and verify --------------------


def test_build_plans_the_same_step_as_up():
    """`build` and the build step of `up` must not be able to drift apart."""
    standalone = plan_for("build", hardware="cpu")
    from_up = plan_for("up", hardware="cpu")
    assert len(standalone.steps) == 1
    assert argv_of(standalone, 0) == argv_of(from_up, 0)


def test_build_narrows_to_the_embedder_when_that_is_all_that_runs():
    """EMBEDDER_ONLY=true builds one image, not three."""
    plan = plan_for("build", hardware="gpu", **{"flag-embedder-only": "true"})
    assert argv_of(plan, 0)[-1] == "smart-embedder-gpu"


def test_pull_skips_what_this_stack_builds():
    """Every image here is local, so `pull` may neither build nor fail."""
    plan = plan_for("pull", hardware="cpu")
    argv = argv_of(plan, 0)
    assert argv[-3:] == ["pull", "--ignore-buildable", "--ignore-pull-failures"]
    assert any("built locally" in message for message in plan.messages)


def test_restart_names_the_services_the_switches_leave_running():
    """`restart` restarts what this configuration actually starts."""
    plan = plan_for("restart", hardware="cpu")
    assert argv_of(plan, 0)[-4:] == [
        "restart",
        "middleware",
        "smart-embedder-cpu",
        "smart-qdrant-mcp",
    ]

    without_mcp = plan_for("restart", hardware="cpu", **{"flag-no-mcp": "true"})
    assert argv_of(without_mcp, 0)[-3:] == [
        "restart",
        "middleware",
        "smart-embedder-cpu",
    ]

    embedder_only = plan_for(
        "restart", hardware="gpu", **{"flag-embedder-only": "true"}
    )
    assert argv_of(embedder_only, 0)[-2:] == ["restart", "smart-embedder-gpu"]


def test_restart_never_rebuilds():
    """The point of `restart` is that it is not `up`."""
    plan = plan_for("restart", hardware="cpu")
    assert len(plan.steps) == 1
    assert "build" not in argv_of(plan, 0)


def test_restart_warns_about_gpu_like_up_does():
    """Starting containers on the gpu profile is what triggers the host probe."""
    assert plan_for("restart", hardware="gpu").warn_if_gpu is True
    assert plan_for("restart", hardware="cpu").warn_if_gpu is False
    # Building an image needs no GPU on the host, so it asks for no probe.
    assert plan_for("build", hardware="gpu").warn_if_gpu is False


def test_urls_runs_nothing_and_reports_every_endpoint():
    """`urls` is its summary: no step, three endpoints, no token value."""
    environment = {
        "MIDDLEWARE_PORT": "7336",
        "MCP_PORT": "7337",
        "MCP_BIND_HOST": "192.168.1.10",
        "SMART_EMBEDDER_PORT": "9010",
        "MIDDLEWARE_API_TOKENS": "s3cret:cli",
    }
    plan = plan_for("urls", environment=environment)
    assert plan.steps == []
    summary = "\n".join(plan.summary)
    assert "http://127.0.0.1:7336" in summary
    assert "http://192.168.1.10:7337" in summary
    assert "http://127.0.0.1:9010" in summary
    assert "s3cret" not in summary


def test_urls_bracket_an_ipv6_mcp_host():
    """An IPv6 publish address is rendered as a valid URL authority."""
    plan = plan_for("urls", environment={"MCP_BIND_HOST": "::1"})

    assert "http://[::1]:6337" in "\n".join(plan.summary)


def test_urls_report_the_middleware_bind_host():
    """A private middleware publish is where the console really answers."""
    plan = plan_for("urls", environment={"MIDDLEWARE_BIND_HOST": "10.10.141.10"})

    assert "http://10.10.141.10:6336" in "\n".join(plan.summary)


def test_urls_needs_no_hardware_profile():
    """Published ports are the same on cpu and gpu, so none is required."""
    plan = plan_for("urls")
    assert plan.summary


@pytest.mark.parametrize(
    "environment,expected",
    [
        ({"MIDDLEWARE_API_TOKENS": "t:cli"}, "bearer token required"),
        ({"MIDDLEWARE_CLIENT_TOKEN": "t"}, "bearer token required"),
        (
            {"MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL": "true"},
            "unauthenticated, loopback callers only",
        ),
        ({}, "it will refuse to start"),
    ],
)
def test_urls_describes_the_authentication_state(environment, expected):
    """The auth state is derived from the ACL, and never quotes a token."""
    plan = plan_for("urls", environment=environment)
    middleware_line = next(line for line in plan.summary if "Middleware API" in line)
    assert expected in middleware_line
    assert "t:cli" not in middleware_line


def test_urls_omits_what_the_switches_do_not_publish():
    """A service that is not started, or not published, is said to be so."""
    private = plan_for("urls", **{"flag-private-embedder": "true"})
    private_summary = "\n".join(private.summary)
    assert ":8010" not in private_summary
    assert "EXPOSE_SMART_EMBEDDER_PORT=false" in private_summary

    without_mcp = plan_for("urls", **{"flag-no-mcp": "true"})
    without_mcp_summary = "\n".join(without_mcp.summary)
    assert ":6337" not in without_mcp_summary
    assert "MCP_SERVER_ENABLED=false" in without_mcp_summary

    embedder_only = plan_for("urls", **{"flag-embedder-only": "true"})
    embedder_only_summary = "\n".join(embedder_only.summary)
    assert ":6336" not in embedder_only_summary
    assert ":6337" not in embedder_only_summary
    assert ":8010" in embedder_only_summary


def test_summary_survives_the_shell_rendering():
    """The bash launcher reads a line-oriented plan; summary lines ride it."""
    plan = plan_for("urls")
    rendered = render_shell(plan)
    summary_lines = [
        line[len("summary ") :]
        for line in rendered.splitlines()
        if line.startswith("summary ")
    ]
    assert summary_lines == plan.summary


@pytest.mark.parametrize("command", ["build", "pull", "restart"])
def test_the_new_commands_share_the_overlay_decision(command):
    """--private-embedder changes every command's file list, not just up's."""
    plan = plan_for(command, hardware="cpu", **{"flag-private-embedder": "true"})
    assert argv_of(plan, 0)[: len(expected_base(private=True))] == expected_base(
        private=True
    )

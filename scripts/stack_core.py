"""Decide what Compose is asked to do, for both stack launchers.

The two launchers keep their own idiom for parsing arguments and executing
processes, but everything that *decides* -- profile resolution, GPU selection,
launcher switches, overlay selection, argv assembly -- lives here, once. The
shells pipe the resolved environment and their parsed CLI intent in on stdin and
receive a JSON execution plan on stdout; they run the steps in order and do not
reason about them.

Two properties are structural rather than incidental:

* This module never executes Docker and never reads `.env`. It is a pure
  function from (environment, intent) to a plan, which is what makes the plan
  matrix testable without a daemon.
* The operational recovery commands -- `down`, `logs`, `status`, `help`,
  `version` -- never reach this module. Stopping a stack must keep working with
  no Python interpreter on PATH and with `.env` absent or malformed.

Input is read from stdin as one `NAME=VALUE` line per entry, rather than from
process arguments, because the environment carries bearer tokens and arguments
are world-readable through `ps`. This mirrors `verify_token_config.py`.

Lines whose name starts with `--cli-` carry the launcher's parsed CLI intent;
everything else is the Compose-resolved environment. Configuration precedence is
CLI flag > shell environment variable > `.env` > built-in default; the shell and
`.env` halves are already collapsed by `docker compose config --environment`
before this module sees them, so what is implemented here is the CLI half and
the defaults.
"""

from __future__ import annotations

import csv
import io
import ipaddress
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

#: Prefix marking a stdin line as parsed CLI intent rather than environment.
CLI_PREFIX = "--cli-"

#: Exit code for an invalid configuration. The launchers reserve 1 for an unmet
#: prerequisite, so every rejection from this module is a 2.
EXIT_INVALID = 2

HARDWARE_PROFILES = ("cpu", "gpu")
MODEL_PROFILES = ("basic", "advanced")
STARTUP_CONTROL_COMMANDS = frozenset({"up", "config"})
AUTOTUNE_MODES = ("off", "quick", "full")
AUTO_ADVANCED_MIN_MIB = 7680
GPU_SELECTION_COMMANDS = frozenset({"up", "config", "build", "pull"})
GPU_UUID_PATTERN = re.compile(
    r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.IGNORECASE
)
GPU_PROBE_ERRORS = {
    "timeout": "NVIDIA discovery timed out.",
    "execution_failed": "The NVIDIA discovery tool could not be executed.",
    "query_failed": "The NVIDIA discovery query failed.",
    "invalid_inventory": "NVIDIA discovery returned an unusable GPU inventory.",
    "context_timeout": "Docker context inspection timed out.",
    "context_failed": "The Docker endpoint could not be inspected.",
    "unsupported_endpoint": "GPU discovery requires a local Docker socket or pipe.",
}

#: Commands that plan no Compose invocation tied to one hardware profile.
COMMANDS_WITHOUT_HARDWARE = frozenset({"verify", "urls"})

#: Compose is asked for both hardware profiles whenever a command has to reach a
#: service that may belong to either one.
ALL_PROFILE_ARGUMENTS = ("--profile", "cpu", "--profile", "gpu")

#: smart-qdrant-mcp only ever talks to middleware, so it comes down with it.
MIDDLEWARE_SERVICES = ("middleware", "smart-qdrant-mcp")

#: Published endpoints, as (label, host variable, host default, port variable,
#: port default). The middleware and Smart Embedder publishes are hardcoded to
#: 127.0.0.1 in compose.yml and have no host variable; the MCP publish is the
#: one operator-settable address.
ENDPOINTS = {
    "middleware": ("Middleware API", None, "127.0.0.1", "MIDDLEWARE_PORT", "6336"),
    "mcp": ("MCP server", "MCP_BIND_HOST", "127.0.0.1", "MCP_PORT", "6337"),
    "smart-embedder": (
        "Smart Embedder",
        None,
        "127.0.0.1",
        "SMART_EMBEDDER_PORT",
        "8010",
    ),
}


def _url_host(host: str) -> str:
    """Format an IP literal for an endpoint URL without importing MCP code."""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host
    return f"[{address}]" if address.version == 6 else str(address)


#: The inbound ACL of each service, as (token list, auto-accepted client token,
#: unauthenticated-local switch). Mirrors INBOUND_LISTS and
#: SERVICE_CLIENT_TOKENS in verify_token_config.py, which is the module that
#: enforces the invariants; this one only describes the outcome.
AUTH_SOURCES = {
    "middleware": (
        "MIDDLEWARE_API_TOKENS",
        "MIDDLEWARE_CLIENT_TOKEN",
        "MIDDLEWARE_ALLOW_UNAUTHENTICATED_LOCAL",
    ),
    "mcp": ("MCP_API_TOKENS", None, "MCP_ALLOW_UNAUTHENTICATED_LOCAL"),
    "smart-embedder": (
        "SMART_EMBEDDER_API_TOKENS",
        "SMART_EMBEDDER_CLIENT_TOKEN",
        "SMART_EMBEDDER_ALLOW_UNAUTHENTICATED_LOCAL",
    ),
}


class ConfigurationError(Exception):
    """A configuration the launcher must refuse, with an operator-facing message."""


def _canonical_uuid(value: str) -> str:
    """Spell a syntax-checked GPU UUID or prefix one way, so equality compares.

    NVIDIA reports the hexadecimal in either case and the launchers accept
    either; the prefix stays whatever length it came in as.
    """
    return "GPU-" + value[4:].lower()


@dataclass(frozen=True)
class GpuDevice:
    """One validated physical GPU from the host inventory."""

    index: int
    uuid: str
    memory_mib: int


@dataclass(frozen=True)
class GpuRequest:
    """Discovery requested by the launcher: none, auto or selected."""

    mode: str
    selector: str = ""


@dataclass(frozen=True)
class GpuProbe:
    """Bounded discovery facts, without raw subprocess output."""

    status: str
    devices: tuple[GpuDevice, ...] = ()
    skipped_rows: int = 0
    reason: str = ""


def resolve_hardware(environment: Mapping[str, str], cli: Mapping[str, str]) -> str:
    """Resolve hardware while reserving automatic selection for supported CLI use.

    Returns:
        "cpu", "gpu", "auto", or "" when neither source named one and the
        caller is a command that needs no hardware profile.

    Raises:
        ConfigurationError: If "auto" reaches a command that cannot discover a
            GPU, or the winning value is not a hardware profile.
    """
    if _normalise(cli.get("hardware")) == "auto":
        if _normalise(cli.get("command")) not in GPU_SELECTION_COMMANDS:
            raise ConfigurationError(
                "Hardware auto supports up, config, build and pull."
            )
        return "auto"
    return resolve_choice(
        cli_value=cli.get("hardware"),
        environment_value=environment.get("STACK_PROFILE"),
        default="",
        allowed=HARDWARE_PROFILES,
        label="Hardware profile",
    )


def resolve_gpu_request(
    environment: Mapping[str, str], cli: Mapping[str, str]
) -> GpuRequest:
    """Validate the winning selector and decide whether discovery is required.

    Returns:
        A request whose mode is "none" when no discovery is needed, "auto" when
        the hardware itself has to be discovered, or "selected" when a specific
        device was named. The selector is canonicalised, never empty for
        "selected", and always empty for "none".

    Raises:
        ConfigurationError: If a device is named on a command that cannot select
            one, alongside cpu hardware, or in a spelling that is neither an
            index nor a GPU- UUID prefix.
    """
    command = _normalise(cli.get("command"))
    explicit = "gpu-device" in cli
    if command not in GPU_SELECTION_COMMANDS:
        if explicit:
            raise ConfigurationError(
                "GPU selection supports up, config, build and pull."
            )
        resolve_hardware(environment, cli)
        return GpuRequest("none")
    hardware = resolve_hardware(environment, cli)
    if hardware == "cpu":
        if explicit:
            raise ConfigurationError(
                "A GPU device cannot be selected with cpu hardware."
            )
        return GpuRequest("none")
    selector = (
        cli["gpu-device"] if explicit else environment.get("STACK_GPU_DEVICE", "")
    ).strip()
    if explicit and not selector:
        raise ConfigurationError("GPU device selector must not be empty.")
    if selector and not (selector.isascii() and selector.isdigit()):
        suffix = selector[4:]
        layout = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
        valid = (
            selector[:4].lower() == "gpu-"
            and 0 < len(suffix) <= len(layout)
            and all(
                char == "-" if slot == "-" else char in "0123456789abcdefABCDEF"
                for char, slot in zip(suffix, layout)
            )
        )
        if not valid:
            raise ConfigurationError(
                "GPU device must be an index or a nonempty GPU- UUID prefix."
            )
        selector = _canonical_uuid(selector)
    if selector and not hardware:
        raise ConfigurationError("GPU selection requires gpu or auto hardware.")
    mode = "none"
    if hardware == "auto":
        mode = "auto"
    elif selector:
        mode = "selected"
    return GpuRequest(mode, selector)


def parse_gpu_csv(text: str) -> GpuProbe:
    """Validate successful NVIDIA CSV output, excluding individual unusable rows."""
    devices: list[GpuDevice] = []
    indices: set[int] = set()
    uuids: set[str] = set()
    skipped = 0
    try:
        for row in csv.reader(io.StringIO(text), strict=True):
            values = [value.strip() for value in row]
            if (
                len(values) != 3
                or not values[0].isascii()
                or not values[0].isdigit()
                or not GPU_UUID_PATTERN.fullmatch(values[1])
                or not values[2].isascii()
                or not values[2].isdigit()
                or int(values[2]) <= 0
            ):
                skipped += 1
                continue
            index, uuid, memory = (
                int(values[0]),
                _canonical_uuid(values[1]),
                int(values[2]),
            )
            if index in indices or uuid in uuids:
                return GpuProbe("error", reason="invalid_inventory")
            indices.add(index)
            uuids.add(uuid)
            devices.append(GpuDevice(index, uuid, memory))
    except (csv.Error, ValueError):
        return GpuProbe("error", reason="invalid_inventory")
    if not devices:
        return GpuProbe("error", reason="invalid_inventory")
    return GpuProbe("ok", tuple(devices), skipped)


def select_gpu(devices: Sequence[GpuDevice], selector: str = "") -> GpuDevice:
    """Match a syntax-validated selector, or prefer capacity then canonical UUID.

    Returns:
        The one device the selector names. Without a selector, the largest by
        memory, ties broken by canonical UUID so the choice does not depend on
        the order the inventory arrived in.

    Raises:
        ConfigurationError: If the inventory is empty, or the selector matches
            no usable device or more than one.
    """
    if not devices:
        raise ConfigurationError("No usable GPU is available for selection.")
    if not selector:
        return min(devices, key=lambda gpu: (-gpu.memory_mib, gpu.uuid))
    if selector.isascii() and selector.isdigit():
        index = selector.lstrip("0") or "0"
        matches = [gpu for gpu in devices if str(gpu.index) == index]
    else:
        canonical = _canonical_uuid(selector)
        matches = [gpu for gpu in devices if gpu.uuid.startswith(canonical)]
    if len(matches) == 1:
        return matches[0]
    available = ", ".join(
        f"{gpu.index} ({gpu.uuid}, {gpu.memory_mib} MiB)"
        for gpu in sorted(devices, key=lambda gpu: gpu.index)[:16]
    )
    problem = "is ambiguous" if matches else "does not match a usable GPU"
    raise ConfigurationError(
        f"GPU device '{selector}' {problem}; available: {available}."
    )


def decode_gpu_probe(text: str) -> GpuProbe:
    """Rebuild the facts record `stack_gpu_probe.py` serialised.

    Not a trust boundary and deliberately not re-validated: `parse_gpu_csv`
    already checked every field, the probe serialises those dataclasses with
    `asdict`, and both launchers forward the line verbatim. What can still fail
    is the transport -- a truncated pipe, a Python that never started.
    `resolve_settings` refuses an unrecognised status and `GPU_PROBE_ERRORS`
    maps the reason through a fixed table, so nothing here reaches an operator.

    Returns:
        The probe's facts, with its devices rebuilt as `GpuDevice` records.

    Raises:
        ConfigurationError: If the payload is not JSON or does not carry the
            four fields the probe writes.
    """
    try:
        data = json.loads(text)
        return GpuProbe(
            data["status"],
            tuple(GpuDevice(**row) for row in data["devices"]),
            data["skipped_rows"],
            data["reason"],
        )
    except (ValueError, TypeError, KeyError):
        raise ConfigurationError("GPU discovery facts are malformed.") from None


@dataclass
class Step:
    """One process invocation the launcher should run, in order."""

    argv: list[str]
    description: str


@dataclass
class Plan:
    """A complete execution plan: what to run, what to say, what was resolved.

    The field names are the JSON contract both launchers parse, so the plan is
    serialised with `dataclasses.asdict` rather than a hand-written renderer:
    a field added here reaches the shells without a second edit.
    """

    steps: list[Step] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: Primary output, printed on stdout once every step has succeeded. This is
    #: the one channel that is not a log line: `urls` is nothing but its
    #: summary, and the endpoint list `up` ends with is what an operator reads
    #: next. Messages and warnings stay on stderr.
    summary: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    resolved: dict[str, str] = field(default_factory=dict)
    warn_if_gpu: bool = False


def parse_input(text: str) -> tuple[dict[str, str], dict[str, str]]:
    """Split stdin into the resolved environment and the parsed CLI intent.

    Args:
        text: The raw stdin payload, one `NAME=VALUE` assignment per line.

    Returns:
        A pair of (environment, cli), where `cli` holds the entries whose name
        carried the `--cli-` prefix, with the prefix removed.
    """
    environment: dict[str, str] = {}
    cli: dict[str, str] = {}
    for line in text.splitlines():
        if not line or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.startswith(CLI_PREFIX):
            cli[name[len(CLI_PREFIX) :]] = value
        else:
            environment[name] = value
    return environment, cli


def _normalise(value: str | None) -> str:
    """Trim and lower-case a value for comparison, treating None as empty."""
    return (value or "").strip().lower()


def resolve_choice(
    *,
    cli_value: str | None,
    environment_value: str | None,
    default: str,
    allowed: Sequence[str],
    label: str,
) -> str:
    """Apply the precedence rule to one enumerated setting.

    Args:
        cli_value: The value the launcher parsed from the command line, if any.
        environment_value: The Compose-resolved value, if the key was present.
        default: The built-in default used when neither source supplies a value.
        allowed: The values this setting accepts.
        label: How to name the setting in a rejection message.

    Returns:
        The resolved, normalised value.

    Raises:
        ConfigurationError: If the winning value is not in `allowed`.
    """
    for candidate in (_normalise(cli_value), _normalise(environment_value)):
        if candidate:
            if candidate not in allowed:
                joined = " or ".join(f"'{item}'" for item in allowed)
                raise ConfigurationError(
                    f"{label} must be {joined}, got '{candidate}'."
                )
            return candidate
    return default


def resolve_boolean(
    *,
    cli_value: str | None,
    environment_value: str | None,
    default: bool,
    label: str,
) -> bool:
    """Apply the precedence rule to one boolean switch.

    An absent or empty value falls through to `default`. That is what lets
    MCP_SERVER_ENABLED default to true, so a `.env` written before the switch
    existed keeps starting the complete stack.

    Raises:
        ConfigurationError: If a value is present but is neither true nor false.
    """
    for candidate in (_normalise(cli_value), _normalise(environment_value)):
        if not candidate:
            continue
        if candidate == "true":
            return True
        if candidate == "false":
            return False
        raise ConfigurationError(f"{label} must be true or false, got '{candidate}'.")
    return default


#: Each launcher switch has a pair of opposed CLI flags. Passing both is a usage
#: error rather than last-one-wins: the operator asked for two different stacks
#: and there is no reading of the command line that satisfies both.
#: The `flag-` namespace is separate from the plain setting keys on purpose. The
#: shells pass every flag on every run, as true or false; a plain key means "the
#: caller resolved this itself", which is how the tests drive the planner
#: directly. Sharing one key for both would make an unset flag look like an
#: explicit false and silently override .env.
SWITCH_FLAG_PAIRS = (
    # (setting, flag meaning true, flag meaning false, spelling, spelling)
    (
        "embedder-only",
        "flag-embedder-only",
        "flag-full",
        "--embedder-only",
        "--full",
    ),
    ("mcp-server-enabled", "flag-mcp", "flag-no-mcp", "--mcp", "--no-mcp"),
    (
        "expose-frontend",
        "flag-frontend",
        "flag-no-frontend",
        "--frontend",
        "--no-frontend",
    ),
    (
        "expose-embedder-port",
        "flag-public-embedder",
        "flag-private-embedder",
        "--public-embedder",
        "--private-embedder",
    ),
)

#: Compose's own --wait-timeout default is 0, which means wait forever. Services
#: that depend on the Smart Embedder start only after its preparation container
#: exits, and an unset ADVANCED quick calibration budget may run for up to
#: 1,080 s on CPU (720 s on CUDA) before the slowest health check's 300 s start
#: period. 1800 covers that and still fails honestly instead of hanging forever.
DEFAULT_WAIT_TIMEOUT = "1800"


def resolve_switch_flags(cli: Mapping[str, str]) -> dict[str, str | None]:
    """Fold each pair of opposed CLI flags into one tri-state value.

    Returns:
        A mapping of setting name to "true", "false", or None when the operator
        used neither flag of the pair.

    Raises:
        ConfigurationError: If both flags of a pair were passed.
    """
    resolved: dict[str, str | None] = {}
    for (
        setting,
        true_flag,
        false_flag,
        true_spelling,
        false_spelling,
    ) in SWITCH_FLAG_PAIRS:
        wants_true = _normalise(cli.get(true_flag)) == "true"
        wants_false = _normalise(cli.get(false_flag)) == "true"
        if wants_true and wants_false:
            raise ConfigurationError(
                f"{true_spelling} and {false_spelling} cannot be combined."
            )
        if wants_true:
            resolved[setting] = "true"
        elif wants_false:
            resolved[setting] = "false"
        else:
            resolved[setting] = None
    return resolved


@dataclass(frozen=True)
class Settings:
    """Everything the plan builders need, after resolution and validation."""

    command: str
    hardware: str
    embedder_profile: str
    embedder_only: bool
    mcp_server_enabled: bool
    expose_embedder_port: bool
    no_build: bool
    wait: bool
    wait_timeout: str
    project_name: str
    compose_file: str
    env_file: str
    embedder_private_file: str
    gpu_uuid: str
    gpu_overlay_file: str
    gpu_memory_mib: int
    gpu_mode: str
    autotune_mode: str
    force_autotune: bool
    expose_frontend: bool | None

    @property
    def selected_service(self) -> str:
        """Name the Smart Embedder service for the selected hardware profile."""
        return f"smart-embedder-{self.hardware}"

    @property
    def selected_init_service(self) -> str:
        """Name the preparation service for the selected hardware profile."""
        return f"smart-embedder-init-{self.hardware}"

    @property
    def opposite_services(self) -> tuple[str, str]:
        """Name the services belonging to the hardware profile *not* selected."""
        opposite = "gpu" if self.hardware == "cpu" else "cpu"
        return (f"smart-embedder-{opposite}", f"smart-embedder-init-{opposite}")


def resolve_settings(
    environment: Mapping[str, str],
    cli: Mapping[str, str],
    *,
    gpu_probe: GpuProbe | None = None,
) -> Settings:
    """Resolve and validate every setting the plan depends on.

    Raises:
        ConfigurationError: On any value the launcher must refuse.
    """
    command = _normalise(cli.get("command"))
    if not command:
        raise ConfigurationError("No command was supplied to the planner.")

    hardware = resolve_hardware(environment, cli)
    request = resolve_gpu_request(environment, cli)
    # Two commands need no hardware profile: `verify` checks both of them
    # itself, and `urls` reports published ports, which are the same either
    # way. Every other command has to know which embedder it is talking about.
    if not hardware and command not in COMMANDS_WITHOUT_HARDWARE:
        raise ConfigurationError(
            f"Command '{command}' requires cpu or gpu on the CLI or in STACK_PROFILE."
        )

    gpu = None
    automatic_model = "basic"
    if request.mode != "none":
        if gpu_probe is None:
            raise ConfigurationError("GPU discovery is required before planning.")
        if gpu_probe.status == "error":
            raise ConfigurationError(
                GPU_PROBE_ERRORS.get(gpu_probe.reason, "GPU discovery failed.")
            )
        if gpu_probe.status in {"tool_missing", "no_devices"}:
            if request.selector:
                raise ConfigurationError(
                    "The selected GPU requires successful NVIDIA discovery; "
                    "the tool is missing or no devices were found."
                )
            hardware = "cpu"
        elif gpu_probe.status == "ok":
            gpu = select_gpu(gpu_probe.devices, request.selector)
            hardware = "gpu"
            if gpu.memory_mib >= AUTO_ADVANCED_MIN_MIB:
                automatic_model = "advanced"
        else:
            raise ConfigurationError("GPU discovery facts are malformed.")
    embedder_profile = resolve_choice(
        cli_value=cli.get("embedder-profile"),
        environment_value=(
            None
            if request.mode == "auto"
            else environment.get("SMART_EMBEDDER_PROFILE")
        ),
        default=automatic_model if request.mode == "auto" else "basic",
        allowed=MODEL_PROFILES,
        label="SMART_EMBEDDER_PROFILE",
    )
    # The opposed CLI flags are folded first, then fed into the same precedence
    # rule as everything else, so a flag beats .env exactly the way a positional
    # profile beats STACK_PROFILE.
    switches = resolve_switch_flags(cli)
    embedder_only = resolve_boolean(
        cli_value=switches["embedder-only"] or cli.get("embedder-only"),
        environment_value=environment.get("EMBEDDER_ONLY"),
        default=False,
        label="EMBEDDER_ONLY",
    )
    mcp_server_enabled = resolve_boolean(
        cli_value=switches["mcp-server-enabled"] or cli.get("mcp-server-enabled"),
        environment_value=environment.get("MCP_SERVER_ENABLED"),
        default=True,
        label="MCP_SERVER_ENABLED",
    )
    expose_embedder_port = resolve_boolean(
        cli_value=switches["expose-embedder-port"] or cli.get("expose-embedder-port"),
        environment_value=environment.get("EXPOSE_SMART_EMBEDDER_PORT"),
        default=True,
        label="EXPOSE_SMART_EMBEDDER_PORT",
    )

    force_autotune = resolve_boolean(
        cli_value=cli.get("force-autotune"),
        environment_value=None,
        default=False,
        label="--force-autotune",
    )
    frontend_choice = switches["expose-frontend"]
    if force_autotune and command != "up":
        raise ConfigurationError("--force-autotune is supported only by up.")
    if command not in STARTUP_CONTROL_COMMANDS and (
        "autotune" in cli or frontend_choice is not None
    ):
        raise ConfigurationError("Startup controls support up and config only.")

    raw_mode = cli.get("autotune")
    if raw_mode is not None and (
        not raw_mode.strip() or "\n" in raw_mode or "\r" in raw_mode
    ):
        raise ConfigurationError("CLI --autotune requires one nonempty mode.")
    # Unflagged service values retain their existing runtime interpretation.
    autotune_mode = ""
    if raw_mode is not None or force_autotune:
        autotune_mode = resolve_choice(
            cli_value=raw_mode,
            environment_value=environment.get("SMART_EMBEDDER_AUTOTUNE"),
            default="quick",
            allowed=AUTOTUNE_MODES,
            label=(
                "CLI --autotune"
                if raw_mode is not None
                else "Configured SMART_EMBEDDER_AUTOTUNE"
            ),
        )
    if force_autotune and autotune_mode == "off":
        raise ConfigurationError("--force-autotune requires quick or full mode.")
    if embedder_only and frontend_choice is not None:
        raise ConfigurationError("Frontend flags require the middleware.")
    expose_frontend = None if frontend_choice is None else frontend_choice == "true"

    # --wait-timeout has no meaning without --wait, and Compose itself rejects
    # the combination, so passing it implies --wait rather than erroring.
    wait_timeout = _normalise(cli.get("wait-timeout"))
    if wait_timeout and not wait_timeout.isdigit():
        raise ConfigurationError(
            f"--wait-timeout must be a positive integer, got '{wait_timeout}'."
        )
    wait = _normalise(cli.get("wait")) == "true" or bool(wait_timeout)

    return Settings(
        command=command,
        hardware=hardware,
        embedder_profile=embedder_profile,
        embedder_only=embedder_only,
        mcp_server_enabled=mcp_server_enabled,
        expose_embedder_port=expose_embedder_port,
        no_build=_normalise(cli.get("no-build")) == "true",
        wait=wait,
        wait_timeout=wait_timeout or DEFAULT_WAIT_TIMEOUT,
        project_name=cli.get("project-name", "").strip(),
        compose_file=cli.get("compose-file", ""),
        env_file=cli.get("env-file", ""),
        embedder_private_file=cli.get("embedder-private-file", ""),
        gpu_uuid=gpu.uuid if gpu else "",
        gpu_overlay_file=cli.get("gpu-overlay-file", ""),
        gpu_memory_mib=gpu.memory_mib if gpu else 0,
        gpu_mode=request.mode,
        autotune_mode=autotune_mode,
        force_autotune=force_autotune,
        expose_frontend=expose_frontend,
    )


def compose_base(settings: Settings) -> list[str]:
    """Build the Compose invocation prefix shared by every step.

    The embedder's published port is withdrawn by an overlay file rather than by
    a CLI flag, because Compose cannot drop a port publish from the command
    line. The overlay is appended to the shared prefix so that `config` shows
    exactly what `up` runs.

    Returns:
        The argv every step starts with: `docker compose`, the project name if
        one was given, the Compose and env files, and whichever overlays this
        configuration needs. No subcommand -- each plan builder appends its own.

    Raises:
        ConfigurationError: If a GPU was selected but the launcher supplied no
            path for the device overlay.
    """
    argv = ["docker", "compose"]
    if settings.project_name:
        # Isolates volumes, networks and Compose's own bookkeeping. It does not
        # enable two concurrent stacks: every service in compose.yml carries a
        # hardcoded container_name, and container names are daemon-global rather
        # than project-scoped, so a second stack fails with "name already in
        # use" whatever the project is called.
        argv += ["--project-name", settings.project_name]
    argv += [
        "--file",
        settings.compose_file,
        "--env-file",
        settings.env_file,
    ]
    if not settings.expose_embedder_port:
        argv += ["--file", settings.embedder_private_file]
    if settings.gpu_uuid:
        if not settings.gpu_overlay_file:
            raise ConfigurationError("GPU selection requires its Compose overlay path.")
        argv += ["--file", settings.gpu_overlay_file]
    return argv


def build_up_plan(
    settings: Settings, plan: Plan, environment: Mapping[str, str]
) -> None:
    """Append the steps that `up` runs, in order, and the summary it ends on."""
    base = compose_base(settings)

    if not settings.no_build:
        build_build_plan(settings, plan)

    # Build first, so a failed build leaves the currently active hardware
    # profile untouched. Only then remove the mutually exclusive service.
    plan.steps.append(
        Step(
            [
                *base,
                *ALL_PROFILE_ARGUMENTS,
                "rm",
                "--stop",
                "--force",
                *settings.opposite_services,
            ],
            "remove the opposite hardware profile's services",
        )
    )

    if settings.embedder_only:
        plan.messages.append(
            f"EMBEDDER_ONLY=true: starting only {settings.selected_service}."
        )
        plan.steps.append(
            Step(
                [*base, "rm", "--stop", "--force", *MIDDLEWARE_SERVICES],
                "remove the middleware and the MCP server",
            )
        )

    if settings.force_autotune:
        plan.messages.append(
            "Forced autotune requested: the selected embedder will stop while "
            "a temporary preparation container runs; normal startup follows."
        )
        plan.steps.append(
            Step(
                [
                    *base,
                    "--profile",
                    settings.hardware,
                    "stop",
                    settings.selected_service,
                    settings.selected_init_service,
                ],
                "stop the selected embedder and preparation service before calibration",
            )
        )
        plan.steps.append(
            Step(
                [
                    *base,
                    "--profile",
                    settings.hardware,
                    "run",
                    "--rm",
                    "--no-deps",
                    "-T",
                    "--env",
                    "SMART_EMBEDDER_AUTOTUNE_FORCE=true",
                    settings.selected_init_service,
                ],
                "run one temporary forced preparation for the selected embedder",
            )
        )

    argv = [*base, "--profile", settings.hardware, "up", "--detach", "--no-build"]
    if settings.wait:
        argv += ["--wait", "--wait-timeout", settings.wait_timeout]
    if settings.embedder_only:
        argv.append(settings.selected_service)
    elif not settings.mcp_server_enabled:
        # --scale 0 both skips the service and tears down an instance a previous
        # run left behind, so no separate rm step is needed. It is safe next to
        # container_name, which only rejects scaling above one. Not combined
        # with the embedder-only branch above, where the explicit service list
        # already excludes the MCP.
        plan.messages.append(
            "MCP_SERVER_ENABLED=false: starting the stack without smart-qdrant-mcp."
        )
        argv += ["--scale", "smart-qdrant-mcp=0"]
    plan.steps.append(Step(argv, "start the stack"))

    # The endpoint list is the thing an operator needs next, and it was the one
    # thing `up` never said: they had to read .env to find out which ports the
    # switches had just published.
    describe_endpoints(settings, plan, environment)
    plan.summary.append(
        "Next: 'status' shows the containers, 'logs' follows them, "
        "'urls' repeats this list."
    )


def build_config_plan(settings: Settings, plan: Plan) -> None:
    """Append the single step that `config` runs."""
    argv = [*compose_base(settings), "--profile", settings.hardware, "config"]
    if settings.embedder_only:
        argv.append(settings.selected_service)
    plan.steps.append(Step(argv, "render the resolved Compose configuration"))


def running_services(settings: Settings) -> list[str]:
    """Name the services this configuration is supposed to be running.

    Compose would infer the list from the active profile, but an explicit one
    also has to survive the two launcher switches, which subtract services the
    profile still contains.

    Returns:
        The service names to act on, in a stable order: the selected embedder
        alone under EMBEDDER_ONLY, otherwise the middleware and that embedder,
        plus smart-qdrant-mcp unless MCP_SERVER_ENABLED withdrew it.
    """
    if settings.embedder_only:
        return [settings.selected_service]
    services = ["middleware", settings.selected_service]
    if settings.mcp_server_enabled:
        services.append("smart-qdrant-mcp")
    return services


def build_build_plan(settings: Settings, plan: Plan) -> None:
    """Append the single step that `build` runs.

    `build_up_plan` calls this rather than repeating it, so that building
    separately and then starting with --no-build cannot drift from what a plain
    `up` would have built.
    """
    argv = [*compose_base(settings), "--profile", settings.hardware, "build"]
    if settings.embedder_only:
        argv.append(settings.selected_service)
    plan.steps.append(Step(argv, "build the images for the selected profile"))


def build_pull_plan(settings: Settings, plan: Plan) -> None:
    """Append the single step that `pull` runs.

    Every service in compose.yml declares a locally built `hybrid-manager/...`
    image, so there is normally nothing to pull: --ignore-buildable skips the
    four buildable services, and --ignore-pull-failures keeps the two init
    services -- which name a locally built image without a build stanza of
    their own -- from turning "nothing to do" into a failure. The command earns
    its place because it is the one that keeps working the day a service is
    switched to a registry image; use `build` to rebuild what is local.
    """
    argv = [
        *compose_base(settings),
        "--profile",
        settings.hardware,
        "pull",
        "--ignore-buildable",
        "--ignore-pull-failures",
    ]
    plan.steps.append(Step(argv, "pull the images this stack does not build"))
    plan.messages.append(
        "Every image in this stack is built locally, so `pull` normally has "
        "nothing to fetch; `build` is what refreshes them."
    )


def build_restart_plan(settings: Settings, plan: Plan) -> None:
    """Append the single step that `restart` runs.

    Deliberately not a `down` followed by an `up`: restart reuses the images
    and the containers that already exist, which is the whole reason to have it
    next to `up --no-build`.
    """
    argv = [
        *compose_base(settings),
        "--profile",
        settings.hardware,
        "restart",
        *running_services(settings),
    ]
    plan.steps.append(Step(argv, "restart the running services"))


def describe_authentication(environment: Mapping[str, str], service: str) -> str:
    """Summarise how a service authenticates callers, naming no token value.

    Returns:
        A phrase describing the service's inbound ACL. Token values never
        appear in it -- only whether one is configured.
    """
    token_list, client_token, allow_local = AUTH_SOURCES[service]
    configured = bool(environment.get(token_list, "").strip())
    if client_token and environment.get(client_token, "").strip():
        configured = True
    if configured:
        return "bearer token required"
    if _normalise(environment.get(allow_local)) in {"1", "true", "yes", "on"}:
        return "unauthenticated, loopback callers only"
    return f"no token configured and {allow_local} is not true; it will refuse to start"


def describe_endpoints(
    settings: Settings, plan: Plan, environment: Mapping[str, str]
) -> None:
    """Append the endpoint summary: what is published, and how it authenticates.

    Shared by `urls`, which is nothing but this, and by `up`, which ends on it.
    Writing it once is what keeps the two from disagreeing about which switch
    withdraws which endpoint.
    """
    shown = ["middleware", "smart-embedder", "mcp"]
    if settings.embedder_only:
        shown = ["smart-embedder"]
    elif not settings.mcp_server_enabled:
        shown.remove("mcp")
    if not settings.expose_embedder_port:
        shown.remove("smart-embedder")

    plan.summary.append("Endpoints published by this configuration:")
    for service in shown:
        label, host_variable, host_default, port_variable, port_default = ENDPOINTS[
            service
        ]
        host = host_default
        if host_variable:
            host = environment.get(host_variable, "").strip() or host_default
        port = environment.get(port_variable, "").strip() or port_default
        authentication = describe_authentication(environment, service)
        plan.summary.append(
            f"  {label}: http://{_url_host(host)}:{port} -- {authentication}"
        )

    if not settings.expose_embedder_port:
        plan.summary.append(
            "  Smart Embedder: not published on the host "
            "(EXPOSE_SMART_EMBEDDER_PORT=false); the middleware reaches it "
            "over the private Compose network."
        )
    if settings.embedder_only:
        plan.summary.append(
            "  Middleware and MCP server: not started (EMBEDDER_ONLY=true)."
        )
    elif not settings.mcp_server_enabled:
        plan.summary.append("  MCP server: not started (MCP_SERVER_ENABLED=false).")


def build_plan(
    environment: Mapping[str, str],
    cli: Mapping[str, str],
    *,
    gpu_probe: GpuProbe | None = None,
) -> Plan:
    """Turn one resolved environment and one parsed intent into an execution plan.

    Raises:
        ConfigurationError: On any configuration the launcher must refuse.
    """
    settings = resolve_settings(environment, cli, gpu_probe=gpu_probe)
    plan = Plan()

    # Normalisation is exported back to the shell so Compose receives the same
    # canonical value on PowerShell and on bash, whatever the casing in the
    # process environment or in .env.
    plan.environment["SMART_EMBEDDER_PROFILE"] = settings.embedder_profile
    if settings.gpu_uuid:
        plan.environment["STACK_GPU_DEVICE"] = settings.gpu_uuid
    if settings.autotune_mode:
        plan.environment["SMART_EMBEDDER_AUTOTUNE"] = settings.autotune_mode
        plan.messages.append(f"Autotune mode: {settings.autotune_mode}.")
    if settings.expose_frontend is not None:
        frontend = str(settings.expose_frontend).lower()
        plan.environment["EXPOSE_FRONTEND"] = frontend
        plan.messages.append(f"Middleware frontend: {frontend}.")
    if settings.force_autotune:
        plan.environment["SMART_EMBEDDER_AUTOTUNE_FORCE"] = "false"
    plan.resolved = {
        "hardware": settings.hardware,
        "embedder_profile": settings.embedder_profile,
        "embedder_only": str(settings.embedder_only).lower(),
        "mcp_server_enabled": str(settings.mcp_server_enabled).lower(),
        "expose_embedder_port": str(settings.expose_embedder_port).lower(),
    }
    if settings.autotune_mode:
        plan.resolved["autotune_mode"] = settings.autotune_mode
    if settings.expose_frontend is not None:
        plan.resolved["expose_frontend"] = str(settings.expose_frontend).lower()
    if settings.force_autotune:
        plan.resolved["force_autotune"] = "true"
    if settings.gpu_mode != "none":
        plan.resolved["gpu_uuid"] = settings.gpu_uuid
        plan.resolved["gpu_memory_mib"] = str(settings.gpu_memory_mib)
        source = "configured model"
        if cli.get("embedder-profile"):
            source = "explicit CLI model"
        elif settings.gpu_mode == "auto":
            source = "automatic model"
        selection = (
            f"; GPU {settings.gpu_uuid}, {settings.gpu_memory_mib} MiB total"
            if settings.gpu_uuid
            else ""
        )
        plan.messages.append(
            f"Startup plan: {settings.hardware}/{settings.embedder_profile} "
            f"({source}){selection}."
        )
        if gpu_probe and gpu_probe.status == "tool_missing":
            plan.warnings.append(
                "NVIDIA tool is absent; auto uses CPU. This does not prove "
                "that the host has no NVIDIA GPU."
            )
        elif gpu_probe and gpu_probe.status == "no_devices":
            plan.messages.append("NVIDIA discovery found no devices; auto uses CPU.")
        if gpu_probe and gpu_probe.skipped_rows:
            plan.warnings.append(
                f"NVIDIA discovery excluded {gpu_probe.skipped_rows} unusable rows."
            )

    if settings.command == "up":
        build_up_plan(settings, plan, environment)
    elif settings.command == "config":
        build_config_plan(settings, plan)
    elif settings.command == "build":
        build_build_plan(settings, plan)
    elif settings.command == "pull":
        build_pull_plan(settings, plan)
    elif settings.command == "restart":
        build_restart_plan(settings, plan)
    elif settings.command == "urls":
        describe_endpoints(settings, plan, environment)
    elif settings.command == "verify":
        # `verify` runs no Compose at all. It still goes through the planner so
        # that the switch values its token preflight reports are resolved by the
        # same code that resolves them for `up`, rather than a second time in
        # each shell.
        pass
    else:
        raise ConfigurationError(
            f"Command '{settings.command}' is not planned by stack_core."
        )

    # Warn the shells that this run touches the GPU -- but not when the GPU was
    # discovered or named on purpose, because the plan already reported which
    # device it picked and a second warning would only add noise. Only the two
    # commands that start containers warn; `config` and `build` touch nothing.
    plan.warn_if_gpu = (
        settings.hardware == "gpu"
        and settings.gpu_mode == "none"
        and settings.command in {"up", "restart"}
    )
    return plan


def render_shell(plan: Plan) -> str:
    """Render a plan as the line-oriented form the bash launcher reads.

    PowerShell parses the JSON directly with ConvertFrom-Json. bash has no JSON
    parser it can rely on, and requiring `jq` would add a dependency to a
    launcher whose whole point is to have few. This is the same plan, tagged one
    record per line, so the shell can read it with nothing but `read`.

    Raises:
        ConfigurationError: If any value contains a newline, which this format
            cannot represent and no legitimate input produces.
    """

    def check(value: str) -> str:
        if "\n" in value or "\r" in value:
            raise ConfigurationError(
                "A resolved value contains a line break, which cannot be passed "
                "to the launcher."
            )
        return value

    lines: list[str] = []
    for name, value in plan.environment.items():
        lines.append(f"env {check(name)}={check(value)}")
    for name, value in plan.resolved.items():
        lines.append(f"resolved {check(name)}={check(value)}")
    lines.append(f"warn_if_gpu {str(plan.warn_if_gpu).lower()}")
    for message in plan.messages:
        lines.append(f"message {check(message)}")
    for warning in plan.warnings:
        lines.append(f"warning {check(warning)}")
    for entry in plan.summary:
        lines.append(f"summary {check(entry)}")
    for step in plan.steps:
        lines.append(f"step {check(step.description)}")
        for argument in step.argv:
            lines.append(f"arg {check(argument)}")
    return "\n".join(lines) + "\n"


def main(stdin_text: str | None = None) -> int:
    """Read intent from stdin and write the plan to stdout.

    Returns:
        0 when the configuration is valid, EXIT_INVALID when it is not.
    """
    # On Windows the default text-mode stdout translates every \n into \r\n,
    # which reaches bash as a trailing carriage return on each plan line and
    # turns the first argument of the first step into $'docker\r'.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(newline="\n")
    text = sys.stdin.read() if stdin_text is None else stdin_text
    environment, cli = parse_input(text)
    shell_format = "--format=sh" in sys.argv[1:]
    try:
        facts = decode_gpu_probe(cli["gpu-probe"]) if "gpu-probe" in cli else None
        plan = build_plan(environment, cli, gpu_probe=facts)
        rendered = render_shell(plan) if shell_format else None
    except ConfigurationError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return EXIT_INVALID
    if rendered is not None:
        sys.stdout.write(rendered)
    else:
        json.dump(asdict(plan), sys.stdout)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

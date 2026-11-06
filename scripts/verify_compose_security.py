"""Verify security invariants in the resolved Docker Compose configuration."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__:
    from .stack_core import GPU_UUID_PATTERN
else:
    from stack_core import GPU_UUID_PATTERN


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EMBEDDER_PRIVATE_COMPOSE = REPOSITORY_ROOT / "compose.embedder-private.yml"
GPU_DEVICE_COMPOSE = REPOSITORY_ROOT / "compose.gpu-device.yml"
GPU_VALIDATION_UUID = "GPU-11111111-1111-1111-1111-111111111111"
EXPECTED_PUBLISHED_PORTS = {
    "middleware": 6336,
    "smart-embedder-cpu": 8000,
    "smart-embedder-gpu": 8000,
    "smart-qdrant-mcp": 6337,
    "qdrant": 6333,
}

PRIVATE_IPV4_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
)
PRIVATE_BIND_SERVICES = frozenset({"middleware", "smart-qdrant-mcp"})


def _is_private_ipv4(host_ip: str) -> bool:
    """Return True for an RFC 1918 IPv4 address (10/8, 172.16/12, 192.168/16).

    Parses ``host_ip`` with ``ipaddress.IPv4Address`` rather than matching a
    string prefix, so a lookalike such as "10.0.0.1.evil.com",
    "192.168.1.1:8080", a bare "10.", an empty string, or a value with
    surrounding whitespace fails to parse and is correctly treated as not
    private. Any value that is not a syntactically valid IPv4 literal --
    including every IPv6 form, such as the IPv4-mapped ``::ffff:10.0.0.1`` --
    is refused here; loopback is handled separately by ``_is_loopback``.
    Excluding IPv4-mapped IPv6 (and IPv6 generally) from the private-address
    exemption is deliberate scope, not an oversight.
    """
    try:
        address = ipaddress.IPv4Address(host_ip)
    except ValueError:
        return False
    return any(address in network for network in PRIVATE_IPV4_NETWORKS)


def _is_loopback(host_ip: str) -> bool:
    """Return True for any loopback literal (127.0.0.0/8 or ::1), as the services do."""
    try:
        return ipaddress.ip_address(host_ip).is_loopback
    except ValueError:
        return False


def _bind_violation(service_name: str, host_ip: str | None) -> str | None:
    """Return a violation message when a published bind address is unacceptable."""
    if host_ip and _is_loopback(host_ip):
        return None
    if service_name in PRIVATE_BIND_SERVICES and host_ip and _is_private_ipv4(host_ip):
        # The console and the MCP endpoint may be reached from the private VPN;
        # wildcard and public addresses stay refused for every service.
        return None
    return (
        f"service {service_name} publishes on {host_ip or 'all interfaces'}; "
        "only loopback is allowed (middleware and smart-qdrant-mcp may also use "
        "a private address)"
    )


def _hardening_violations(service_name: str, service: Mapping[str, Any]) -> list[str]:
    """Return the hardening violations for one published serving container.

    Every network-facing service (middleware, Smart Embedder, MCP server) is
    built with the same non-root, read-only, capability-dropped shape. Asserting
    it for only one of them let the hardening be dropped from the others
    silently, which is exactly the drift this verifier exists to catch. The init
    container is intentionally excluded: it runs as root with bootstrap
    capabilities and has its own dedicated checks below.
    """
    violations: list[str] = []
    if service.get("user") != "1000:1000":
        violations.append(f"service {service_name} must run as uid:gid 1000:1000")
    if service.get("read_only") is not True:
        violations.append(
            f"service {service_name} must use a read-only root filesystem"
        )
    if set(service.get("cap_drop") or []) != {"ALL"}:
        violations.append(f"service {service_name} must drop all capabilities")
    return violations


def _qdrant_violations(services: Mapping[str, Any]) -> list[str]:
    """Return the violations of the bundled Qdrant pair (QDRANT_BUNDLED=true).

    The serving container gets the same hardening as every other publisher
    (checked by the caller). Its preparation container exists only to hand the
    storage volume to uid 1000, so it may hold CHOWN and nothing else, publish
    nothing, and must have finished before Qdrant starts.
    """
    errors: list[str] = []
    runtime = services.get("qdrant")
    init_service = services.get("qdrant-init")
    if not isinstance(runtime, Mapping) or not isinstance(init_service, Mapping):
        return ["qdrant profile does not resolve both qdrant and qdrant-init"]
    if init_service.get("ports"):
        errors.append("service qdrant-init must not publish ports")
    if init_service.get("user") != "0:0":
        errors.append("service qdrant-init must start the volume bootstrap as root")
    if set(init_service.get("cap_drop") or []) != {"ALL"}:
        errors.append("service qdrant-init must drop all capabilities")
    if set(init_service.get("cap_add") or []) != {"CHOWN"}:
        errors.append("service qdrant-init must have only the CHOWN capability")
    if init_service.get("read_only") is not True:
        errors.append("service qdrant-init must use a read-only root filesystem")
    for service_name, service in (("qdrant", runtime), ("qdrant-init", init_service)):
        mounts = {
            mount.get("target"): mount
            for mount in service.get("volumes") or []
            if isinstance(mount, Mapping)
        }
        if mounts.get("/qdrant/storage", {}).get("read_only") is True or (
            "/qdrant/storage" not in mounts
        ):
            errors.append(f"service {service_name} must mount /qdrant/storage writable")
    dependency = (runtime.get("depends_on") or {}).get("qdrant-init", {})
    if dependency.get("condition") != "service_completed_successfully":
        errors.append("service qdrant must wait for successful preparation")
    return errors


def validate_config(
    config: Mapping[str, Any],
    profile: str,
    *,
    embedder_published: bool = True,
    qdrant_bundled: bool = False,
    gpu_device: str | None = None,
) -> list[str]:
    """Return violations found in a resolved Compose configuration.

    ``embedder_published`` describes which of the two supported shapes is being
    checked. The default is the plain ``compose.yml``, where the embedder
    publishes a loopback port. ``False`` is the shape produced by overlaying
    ``compose.embedder-private.yml`` (EXPOSE_SMART_EMBEDDER_PORT=false), where
    it must not. ``qdrant_bundled`` is the shape with the ``qdrant`` profile
    enabled (QDRANT_BUNDLED=true), where the bundled Qdrant must publish its
    loopback port and carry the same hardening. The publisher set is compared
    for exact equality either way, so both a missing publish and an unexpected
    extra one are caught -- a Qdrant leaking out of its profile included.
    """
    errors: list[str] = []
    services = config.get("services")
    if not isinstance(services, Mapping):
        return ["resolved Compose configuration does not contain services"]

    expected_embedder = f"smart-embedder-{profile}"
    expected_init = f"smart-embedder-init-{profile}"
    expected_publishers = {"middleware", "smart-qdrant-mcp"}
    if embedder_published:
        expected_publishers.add(expected_embedder)
    hardened = ["middleware", "smart-qdrant-mcp", expected_embedder]
    if qdrant_bundled:
        expected_publishers.add("qdrant")
        hardened.append("qdrant")
        errors.extend(_qdrant_violations(services))
    actual_publishers: set[str] = set()

    for service_name, service in services.items():
        if not isinstance(service, Mapping):
            errors.append(f"service {service_name!r} is not an object")
            continue
        if service.get("network_mode") == "host":
            errors.append(f"service {service_name} uses forbidden host networking")
        errors.extend(_gpu_violations(str(service_name), service, profile, gpu_device))

        ports = service.get("ports") or []
        if ports:
            actual_publishers.add(str(service_name))
        for port in ports:
            if not isinstance(port, Mapping):
                errors.append(f"service {service_name} has an unresolved port entry")
                continue
            violation = _bind_violation(str(service_name), port.get("host_ip"))
            if violation is not None:
                errors.append(violation)
            expected_target = EXPECTED_PUBLISHED_PORTS.get(str(service_name))
            if port.get("target") != expected_target:
                errors.append(
                    f"service {service_name} publishes unexpected target port "
                    f"{port.get('target')!r}"
                )

    # The same non-root/read-only/cap-dropped hardening applies to every
    # network-facing service, not only the MCP server. The embedder init
    # container is excluded (it runs as root by design and is checked below).
    for service_name in hardened:
        service = services.get(service_name)
        if isinstance(service, Mapping):
            errors.extend(_hardening_violations(service_name, service))

    if expected_embedder not in services:
        errors.append(f"profile {profile} does not resolve service {expected_embedder}")
    if expected_init not in services:
        errors.append(f"profile {profile} does not resolve service {expected_init}")
    if actual_publishers != expected_publishers:
        errors.append(
            "unexpected set of services publishing ports: "
            f"expected {sorted(expected_publishers)}, got {sorted(actual_publishers)}"
        )
    runtime = services.get(expected_embedder)
    init_service = services.get(expected_init)
    if isinstance(runtime, Mapping) and isinstance(init_service, Mapping):
        runtime_mounts = {
            mount.get("target"): mount
            for mount in runtime.get("volumes") or []
            if isinstance(mount, Mapping)
        }
        init_mounts = {
            mount.get("target"): mount
            for mount in init_service.get("volumes") or []
            if isinstance(mount, Mapping)
        }
        for target in ("/app/model_cache", "/app/runtime_state"):
            if runtime_mounts.get(target, {}).get("read_only") is not True:
                errors.append(
                    f"service {expected_embedder} must mount {target} read-only"
                )
            if (
                target not in init_mounts
                or init_mounts[target].get("read_only") is True
            ):
                errors.append(f"service {expected_init} must mount {target} writable")

        runtime_environment = runtime.get("environment") or {}
        init_environment = init_service.get("environment") or {}
        for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            if runtime_environment.get(variable) != "1":
                errors.append(f"service {expected_embedder} must set {variable}=1")
            if init_environment.get(variable) != "0":
                errors.append(f"service {expected_init} must set {variable}=0")
        for service_name, service in (
            (expected_embedder, runtime),
            (expected_init, init_service),
        ):
            if not any(
                str(mount).startswith("/app/runtime_cache:") and ",exec" in str(mount)
                for mount in service.get("tmpfs") or []
            ):
                errors.append(
                    f"service {service_name} must use executable tmpfs for "
                    "/app/runtime_cache"
                )
        for secret_name in ("API_TOKENS", "SERVICE_TOKEN"):
            if init_environment.get(secret_name) != "":
                errors.append(f"service {expected_init} must not receive {secret_name}")
        if init_service.get("user") != "0:0":
            errors.append(
                f"service {expected_init} must start the volume bootstrap as root"
            )
        expected_caps = {"CHOWN", "SETGID", "SETUID"}
        if set(init_service.get("cap_add") or []) != expected_caps:
            errors.append(
                f"service {expected_init} must have only bootstrap capabilities"
            )
        command = init_service.get("command") or []
        if list(command) != ["python3", "model_bootstrap.py"]:
            errors.append(f"service {expected_init} must run model_bootstrap.py")
        dependency = (runtime.get("depends_on") or {}).get(expected_init, {})
        if dependency.get("condition") != "service_completed_successfully":
            errors.append(
                f"service {expected_embedder} must wait for successful preparation"
            )
    return errors


def _gpu_violations(
    service_name: str,
    service: Mapping[str, Any],
    profile: str,
    gpu_device: str | None,
) -> list[str]:
    """Require exactly the intended reservation and exclude alternate GPU access."""
    errors = []
    if "gpus" in service:
        errors.append(f"service {service_name} must not use service-level gpus")
    try:
        devices = (
            service.get("deploy", {})
            .get("resources", {})
            .get("reservations", {})
            .get("devices", [])
        )
    except AttributeError:
        return [*errors, f"service {service_name} has malformed GPU reservations"]
    expected = profile == "gpu" and service_name in {
        "smart-embedder-init-gpu",
        "smart-embedder-gpu",
    }
    if not expected:
        if devices:
            errors.append(f"service {service_name} must not reserve GPU devices")
        return errors
    request = {"driver": "nvidia", "capabilities": ["gpu"]}
    if gpu_device:
        request["device_ids"] = [gpu_device]
        valid = devices == [request]
    else:
        valid = any(devices == [{**request, "count": count}] for count in (-1, "all"))
    if not valid:
        target = "the selected UUID without count" if gpu_device else "all NVIDIA GPUs"
        errors.append(f"service {service_name} must reserve exactly {target}")
    return errors


def resolve_config(
    env_file: Path,
    profile: str,
    *,
    overlays: Sequence[Path] = (),
    environment: Mapping[str, str] | None = None,
    qdrant: bool = False,
) -> Mapping[str, Any]:
    """Resolve one Compose profile as JSON, optionally applying overlay files.

    ``qdrant`` also enables the ``qdrant`` profile, the shape the launchers
    produce for QDRANT_BUNDLED=true.
    """
    command = ["docker", "compose"]
    if overlays:
        # An explicit base is required as soon as one --file is passed.
        command += ["--file", str(REPOSITORY_ROOT / "compose.yml")]
        for overlay in overlays:
            command += ["--file", str(overlay)]
    command += ["--env-file", str(env_file), "--profile", profile]
    if qdrant:
        command += ["--profile", "qdrant"]
    command += ["config", "--format", "json"]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Compose config failed for {profile}: {detail}")
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Compose returned invalid JSON for {profile}") from exc
    if not isinstance(parsed, Mapping):
        raise RuntimeError(f"Compose returned a non-object configuration for {profile}")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the CPU and GPU deployment profiles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=REPOSITORY_ROOT / ".env.example",
        help="environment file used to resolve Compose",
    )
    args = parser.parse_args(argv)
    env_file = args.env_file.resolve()
    if not env_file.is_file():
        parser.error(f"environment file does not exist: {env_file}")

    candidate = os.environ.get("STACK_GPU_DEVICE", "")
    gpu_device = (
        "GPU-" + candidate[4:].lower()
        if GPU_UUID_PATTERN.fullmatch(candidate)
        else GPU_VALIDATION_UUID
    )
    selected_environment = {**os.environ, "STACK_GPU_DEVICE": gpu_device}

    violations: list[str] = []
    for profile in ("cpu", "gpu"):
        # (selected GPU, embedder published, bundled Qdrant). The bundled
        # Qdrant is orthogonal to the other two switches -- different
        # services -- so one shape per hardware profile covers it.
        shapes = [(False, True, False), (False, False, False), (False, True, True)]
        if profile == "gpu":
            shapes += [(True, True, False), (True, False, False)]
        for selected, published, bundled in shapes:
            label = (
                ("selected-gpu" if selected else "default")
                + ("/embedder-private" if not published else "")
                + ("/bundled-qdrant" if bundled else "")
            )
            overlays = [] if published else [EMBEDDER_PRIVATE_COMPOSE]
            if selected:
                overlays.append(GPU_DEVICE_COMPOSE)
            try:
                config = resolve_config(
                    env_file,
                    profile,
                    overlays=overlays,
                    environment=selected_environment if selected else None,
                    qdrant=bundled,
                )
            except RuntimeError as exc:
                violations.append(f"{label}: {exc}")
                continue
            violations.extend(
                f"{profile}/{label}: {error}"
                for error in validate_config(
                    config,
                    profile,
                    embedder_published=published,
                    qdrant_bundled=bundled,
                    gpu_device=gpu_device if selected else None,
                )
            )

    if violations:
        for violation in violations:
            print(f"[ERROR] {violation}", file=sys.stderr)
        return 1
    print(
        "Compose security invariants verified for CPU and GPU profiles, "
        "with and without the published Smart Embedder port, "
        "including single-GPU reservations and the bundled Qdrant (eight shapes)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

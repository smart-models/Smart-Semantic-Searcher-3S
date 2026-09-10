"""Unit tests for the Compose security invariant verifier."""

import unittest
from unittest.mock import patch

import pytest

from scripts import verify_compose_security as verifier
from scripts.verify_compose_security import validate_config


def resolved_config(profile: str, host_ip: str = "127.0.0.1") -> dict:
    runtime_name = f"smart-embedder-{profile}"
    init_name = f"smart-embedder-init-{profile}"
    config = {
        "services": {
            "middleware": {
                "ports": [{"host_ip": host_ip, "target": 6336}],
                "user": "1000:1000",
                "read_only": True,
                "cap_drop": ["ALL"],
            },
            runtime_name: {
                "ports": [{"host_ip": host_ip, "target": 8000}],
                "user": "1000:1000",
                "read_only": True,
                "cap_drop": ["ALL"],
                "environment": {
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                },
                "volumes": [
                    {"target": "/app/model_cache", "read_only": True},
                    {"target": "/app/runtime_state", "read_only": True},
                ],
                "tmpfs": ["/app/runtime_cache:size=2g,exec"],
                "depends_on": {
                    init_name: {"condition": "service_completed_successfully"}
                },
            },
            init_name: {
                "ports": [],
                "user": "0:0",
                "cap_add": ["CHOWN", "SETGID", "SETUID"],
                "command": ["python3", "model_bootstrap.py"],
                "environment": {
                    "API_TOKENS": "",
                    "SERVICE_TOKEN": "",
                    "HF_HUB_OFFLINE": "0",
                    "TRANSFORMERS_OFFLINE": "0",
                },
                "volumes": [
                    {"target": "/app/model_cache"},
                    {"target": "/app/runtime_state"},
                ],
                "tmpfs": ["/app/runtime_cache:size=2g,exec"],
            },
            "smart-qdrant-mcp": {
                "ports": [{"host_ip": "127.0.0.1", "target": 6337}],
                "user": "1000:1000",
                "read_only": True,
                "cap_drop": ["ALL"],
            },
        }
    }
    if profile == "gpu":
        for name in (runtime_name, init_name):
            config["services"][name]["deploy"] = {
                "resources": {
                    "reservations": {
                        "devices": [
                            {"driver": "nvidia", "count": -1, "capabilities": ["gpu"]}
                        ]
                    }
                }
            }
    return config


@pytest.mark.parametrize("published", [False, True])
def test_selected_gpu_both_services_and_private_port(published):
    config = selected_config(published)
    assert (
        validate_config(
            config,
            "gpu",
            embedder_published=published,
            gpu_device=verifier.GPU_VALIDATION_UUID,
        )
        == []
    )


def selected_config(published=True):
    """Return a hardened GPU fixture with a selected UUID on both services."""
    config = resolved_config("gpu")
    for name in ("smart-embedder-init-gpu", "smart-embedder-gpu"):
        request = config["services"][name]["deploy"]["resources"]["reservations"][
            "devices"
        ][0]
        del request["count"]
        request["device_ids"] = [verifier.GPU_VALIDATION_UUID]
    if not published:
        config["services"]["smart-embedder-gpu"]["ports"] = []
    return config


@pytest.mark.parametrize("name", ["smart-embedder-init-gpu", "smart-embedder-gpu"])
@pytest.mark.parametrize(
    "change",
    [
        {"count": -1},
        {"device_ids": ["GPU-wrong"]},
        {"device_ids": []},
        {"device_ids": [verifier.GPU_VALIDATION_UUID, "GPU-extra"]},
        {"driver": "other"},
        {"capabilities": ["gpu", "extra"]},
    ],
)
def test_selected_reservation_rejects_wrong_identity_count_or_shape(name, change):
    config = selected_config()
    config["services"][name]["deploy"]["resources"]["reservations"]["devices"][
        0
    ].update(change)
    assert validate_config(config, "gpu", gpu_device=verifier.GPU_VALIDATION_UUID)


@pytest.mark.parametrize(
    "name",
    ["middleware", "smart-qdrant-mcp", "smart-embedder-cpu", "smart-embedder-init-cpu"],
)
def test_non_gpu_services_cannot_receive_devices(name):
    config = resolved_config("cpu")
    config["services"][name]["deploy"] = {
        "resources": {
            "reservations": {
                "devices": [{"driver": "nvidia", "count": -1, "capabilities": ["gpu"]}]
            }
        }
    }
    assert any(
        "must not reserve GPU" in error for error in validate_config(config, "cpu")
    )


def test_alternative_service_level_gpus_are_rejected():
    config = selected_config()
    config["services"]["smart-embedder-gpu"]["gpus"] = "all"
    assert any(
        "service-level gpus" in error
        for error in validate_config(
            config, "gpu", gpu_device=verifier.GPU_VALIDATION_UUID
        )
    )


@pytest.mark.parametrize("count", [-1, "all"])
def test_default_all_gpu_count_accepts_yaml_and_resolved_json(count):
    config = resolved_config("gpu")
    for name in ("smart-embedder-init-gpu", "smart-embedder-gpu"):
        config["services"][name]["deploy"]["resources"]["reservations"]["devices"][0][
            "count"
        ] = count
    assert validate_config(config, "gpu") == []


@pytest.mark.parametrize("copies", [0, 2])
@pytest.mark.parametrize("name", ["smart-embedder-init-gpu", "smart-embedder-gpu"])
def test_selected_gpu_requires_exactly_one_reservation(name, copies):
    config = selected_config()
    reservation = config["services"][name]["deploy"]["resources"]["reservations"]
    reservation["devices"] *= copies
    assert validate_config(config, "gpu", gpu_device=verifier.GPU_VALIDATION_UUID)


def test_verifier_resolves_six_shapes_with_explicit_selected_environment(monkeypatch):
    monkeypatch.setenv("STACK_GPU_DEVICE", "0")
    with (
        patch.object(verifier, "resolve_config") as resolve,
        patch.object(verifier, "validate_config", return_value=[]),
    ):
        assert (
            verifier.main(
                ["--env-file", str(verifier.REPOSITORY_ROOT / ".env.example")]
            )
            == 0
        )
    assert resolve.call_count == 6
    calls = resolve.call_args_list
    assert [call.args[1] for call in calls] == [
        "cpu",
        "cpu",
        "gpu",
        "gpu",
        "gpu",
        "gpu",
    ]
    for call in calls[-2:]:
        assert call.kwargs["overlays"][-1] == verifier.GPU_DEVICE_COMPOSE
        assert (
            call.kwargs["environment"]["STACK_GPU_DEVICE"]
            == verifier.GPU_VALIDATION_UUID
        )
    assert verifier.os.environ["STACK_GPU_DEVICE"] == "0"


class ComposeSecurityValidationTests(unittest.TestCase):
    """Tests for the compose hardening invariants the verifier enforces."""

    def test_expected_loopback_publishers_are_accepted(self):
        self.assertEqual(validate_config(resolved_config("cpu"), "cpu"), [])

    def test_all_interfaces_is_rejected(self):
        errors = validate_config(resolved_config("gpu", "0.0.0.0"), "gpu")
        self.assertTrue(any("0.0.0.0" in error for error in errors))

    def test_missing_host_ip_is_rejected(self):
        config = resolved_config("cpu")
        del config["services"]["smart-embedder-cpu"]["ports"][0]["host_ip"]
        errors = validate_config(config, "cpu")
        self.assertTrue(any("all interfaces" in error for error in errors))

    def test_host_networking_is_rejected(self):
        config = resolved_config("cpu")
        config["services"]["smart-embedder-cpu"]["network_mode"] = "host"
        errors = validate_config(config, "cpu")
        self.assertTrue(any("host networking" in error for error in errors))

    def test_unexpected_publisher_is_rejected(self):
        config = resolved_config("cpu")
        config["services"]["debug"] = {
            "ports": [{"host_ip": "127.0.0.1", "target": 9000}]
        }
        errors = validate_config(config, "cpu")
        self.assertTrue(any("unexpected set" in error for error in errors))

    def test_writable_runtime_model_cache_is_rejected(self):
        config = resolved_config("cpu")
        config["services"]["smart-embedder-cpu"]["volumes"][0]["read_only"] = False

        errors = validate_config(config, "cpu")

        self.assertTrue(any("model_cache read-only" in error for error in errors))

    def test_writable_middleware_root_filesystem_is_rejected(self):
        config = resolved_config("cpu")
        config["services"]["middleware"]["read_only"] = False

        errors = validate_config(config, "cpu")

        self.assertTrue(
            any("middleware" in error and "read-only" in error for error in errors)
        )

    def test_root_middleware_user_is_rejected(self):
        config = resolved_config("cpu")
        config["services"]["middleware"]["user"] = "0:0"

        errors = validate_config(config, "cpu")

        self.assertTrue(
            any("middleware" in error and "uid:gid" in error for error in errors)
        )

    def test_embedder_without_cap_drop_is_rejected(self):
        config = resolved_config("gpu")
        config["services"]["smart-embedder-gpu"]["cap_drop"] = []

        errors = validate_config(config, "gpu")

        self.assertTrue(
            any(
                "smart-embedder-gpu" in error and "capabilities" in error
                for error in errors
            )
        )

    def test_withdrawn_embedder_port_is_accepted_in_private_mode(self):
        config = resolved_config("cpu")
        config["services"]["smart-embedder-cpu"]["ports"] = []

        errors = validate_config(config, "cpu", embedder_published=False)

        self.assertEqual(errors, [])

    def test_embedder_still_publishing_in_private_mode_is_rejected(self):
        errors = validate_config(
            resolved_config("cpu"), "cpu", embedder_published=False
        )

        self.assertTrue(any("unexpected set" in error for error in errors))

    def test_missing_embedder_port_in_default_mode_is_rejected(self):
        config = resolved_config("gpu")
        config["services"]["smart-embedder-gpu"]["ports"] = []

        errors = validate_config(config, "gpu")

        self.assertTrue(any("unexpected set" in error for error in errors))

    def test_init_api_tokens_is_rejected(self):
        config = resolved_config("gpu")
        config["services"]["smart-embedder-init-gpu"]["environment"]["API_TOKENS"] = (
            "secret:ops"
        )

        errors = validate_config(config, "gpu")

        self.assertTrue(any("must not receive API_TOKENS" in error for error in errors))

    def test_init_service_must_blank_both_token_variables(self):
        """The middleware's credential is a second secret the init must not see."""
        config = resolved_config("gpu")
        environment = config["services"]["smart-embedder-init-gpu"]["environment"]
        environment["API_TOKENS"] = ""
        environment["SERVICE_TOKEN"] = "x"

        errors = validate_config(config, "gpu")

        self.assertTrue(any("SERVICE_TOKEN" in error for error in errors))


if __name__ == "__main__":
    unittest.main()

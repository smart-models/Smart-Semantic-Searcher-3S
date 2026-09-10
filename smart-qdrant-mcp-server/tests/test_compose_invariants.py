"""The compose security invariants must cover the MCP service."""

from __future__ import annotations

import copy

import pytest

from scripts.verify_compose_security import validate_config

BASE = {
    "services": {
        "middleware": {"ports": [{"host_ip": "127.0.0.1", "target": 6336}]},
        "smart-embedder-cpu": {"ports": [{"host_ip": "127.0.0.1", "target": 8000}]},
        "smart-embedder-init-cpu": {},
        "smart-qdrant-mcp": {
            "ports": [{"host_ip": "127.0.0.1", "target": 6337}],
            "user": "1000:1000",
            "read_only": True,
            "cap_drop": ["ALL"],
        },
    }
}


def config(**overrides):
    """Return a minimal resolved config with the MCP service overridden."""
    data = copy.deepcopy(BASE)
    data["services"]["smart-qdrant-mcp"].update(overrides)
    return data


def mcp_errors(cfg) -> list[str]:
    """Return only the violations that mention the MCP service."""
    return [e for e in validate_config(cfg, "cpu") if "smart-qdrant-mcp" in e]


def test_loopback_publish_is_accepted():
    assert mcp_errors(config()) == []


def test_private_address_publish_is_accepted():
    cfg = config(ports=[{"host_ip": "192.168.1.20", "target": 6337}])
    assert mcp_errors(cfg) == []


def test_ipv6_loopback_publish_is_accepted():
    """The only supported IPv6 MCP publish is the loopback address."""
    cfg = config(ports=[{"host_ip": "::1", "target": 6337}])
    assert mcp_errors(cfg) == []


@pytest.mark.parametrize("host_ip", ["0.0.0.0", "::", "203.0.113.7"])
def test_public_or_wildcard_publish_is_refused(host_ip):
    cfg = config(ports=[{"host_ip": host_ip, "target": 6337}])
    assert mcp_errors(cfg)


def test_unexpected_target_port_is_refused():
    cfg = config(ports=[{"host_ip": "127.0.0.1", "target": 9999}])
    assert mcp_errors(cfg)


def test_root_user_is_refused():
    assert mcp_errors(config(user="0:0"))


def test_writable_root_filesystem_is_refused():
    assert mcp_errors(config(read_only=False))


def test_retained_capabilities_are_refused():
    assert mcp_errors(config(cap_drop=[]))


@pytest.mark.parametrize(
    "host_ip",
    [
        "10.0.0.1.evil.com",
        "192.168.1.1:8080",
        "127.0.0.1.evil.com",
        "10.",
        "",
        " 10.0.0.1 ",
    ],
)
def test_malformed_address_strings_are_refused(host_ip):
    """Prefix-matching lookalikes must not be mistaken for private IPv4s."""
    cfg = config(ports=[{"host_ip": host_ip, "target": 6337}])
    assert mcp_errors(cfg)


@pytest.mark.parametrize(
    "host_ip, accepted",
    [
        ("172.15.255.255", False),
        ("172.16.0.0", True),
        ("172.31.255.255", True),
        ("172.32.0.0", False),
    ],
)
def test_172_16_12_boundary_is_pinned(host_ip, accepted):
    cfg = config(ports=[{"host_ip": host_ip, "target": 6337}])
    errors = mcp_errors(cfg)
    if accepted:
        assert errors == []
    else:
        assert errors


def test_ipv4_mapped_ipv6_is_refused():
    # IPv4-mapped IPv6 (::ffff:10.0.0.1) is deliberately out of scope: only a
    # bare RFC 1918 IPv4 literal is accepted for smart-qdrant-mcp, and every IPv6
    # form other than the loopback ::1 stays refused for every service.
    cfg = config(ports=[{"host_ip": "::ffff:10.0.0.1", "target": 6337}])
    assert mcp_errors(cfg)

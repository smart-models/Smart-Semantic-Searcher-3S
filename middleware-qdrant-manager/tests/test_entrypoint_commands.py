"""Tests pinning the Docker/compose entrypoint and environment contracts."""

from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SERVICE_ROOT.parent


def test_runtime_entrypoint_uses_package_app_only():
    expected = "qdrant_middleware.main:app"

    assert expected in (WORKSPACE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert not (SERVICE_ROOT / "main.py").exists()


def test_runtime_image_excludes_frontend_development_dependencies():
    """Keep local Node test dependencies out of every Docker build context."""
    dockerignore = (WORKSPACE_ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert "middleware-qdrant-manager/frontend/node_modules/" in dockerignore


def test_gpu_runtime_contains_torch_compile_toolchain():
    dockerfile = (WORKSPACE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    gpu_runtime = dockerfile.split(" AS smart-embedder-gpu\n", maxsplit=1)[1]

    assert "build-essential" in gpu_runtime
    assert "python3-dev" in gpu_runtime
    assert "CC=/usr/bin/gcc" in gpu_runtime
    assert "CXX=/usr/bin/g++" in gpu_runtime


def test_root_environment_uses_namespaced_middleware_keys():
    compose = (WORKSPACE_ROOT / "compose.yml").read_text(encoding="utf-8")
    env_example = (WORKSPACE_ROOT / ".env.example").read_text(encoding="utf-8")

    expected_mappings = (
        "${MIDDLEWARE_PORT:-6336}",
        "MIDDLEWARE_API_TOKENS: ${MIDDLEWARE_API_TOKENS:-}",
        "SERVICE_TOKEN: ${MIDDLEWARE_CLIENT_TOKEN:-}",
        "MIDDLEWARE_REQUEST_TIMEOUT: ${MIDDLEWARE_REQUEST_TIMEOUT:-30}",
        "MIDDLEWARE_RATE_LIMIT_REQUESTS_PER_MINUTE: "
        "${MIDDLEWARE_RATE_LIMIT_REQUESTS_PER_MINUTE:-300}",
        "MIDDLEWARE_RATE_LIMIT_BURST_SIZE: ${MIDDLEWARE_RATE_LIMIT_BURST_SIZE:-180}",
    )
    assert all(mapping in compose for mapping in expected_mappings)
    assert "QDRANT_API_PORT:" not in compose
    assert "      API_TOKENS: ${MIDDLEWARE_API_TOKENS:-}" not in compose.splitlines()

    expected_keys = (
        "MIDDLEWARE_PORT=",
        "MIDDLEWARE_API_TOKENS=",
        "MIDDLEWARE_CLIENT_TOKEN=",
        "MIDDLEWARE_REQUEST_TIMEOUT=",
        "MIDDLEWARE_RATE_LIMIT_REQUESTS_PER_MINUTE=",
        "MIDDLEWARE_RATE_LIMIT_BURST_SIZE=",
    )
    assert all(key in env_example for key in expected_keys)
    assert "SMART_EMBEDDER_HOST=" not in env_example

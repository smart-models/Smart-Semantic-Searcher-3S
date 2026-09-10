"""Run the MCP server with uvicorn: ``python3 -m smart_qdrant_mcp``."""

from __future__ import annotations

import uvicorn

from smart_qdrant_mcp.config import get_settings


def main() -> None:
    """Serve the streamable HTTP app on the configured port.

    ``settings.listen_host`` (``MCP_LISTEN_HOST``), not ``settings.bind_host``
    (``MCP_BIND_HOST``): the two answer different questions. ``bind_host`` is
    the published/policed address -- what compose publishes on, what
    ``scripts/verify_compose_security.py`` checks, what seeds the Host
    allowlist and issuer URL -- and never reaches a socket. ``listen_host``
    is the address this process actually binds inside its own network
    namespace, defaulting to loopback so running outside Docker is safe by
    default; the compose service sets it explicitly to ``0.0.0.0`` because
    inside a container's own namespace that is exactly what the
    published-port restriction assumes.
    """
    settings = get_settings()
    uvicorn.run(
        "smart_qdrant_mcp.server:app", host=settings.listen_host, port=settings.port
    )


if __name__ == "__main__":
    main()

"""The dependency bundle every tool module receives at registration time."""

from __future__ import annotations

from dataclasses import dataclass

from smart_qdrant_mcp.client import MiddlewareClient
from smart_qdrant_mcp.config import Settings


@dataclass(frozen=True)
class ToolContext:
    """Shared dependencies for tool implementations."""

    client: MiddlewareClient
    settings: Settings

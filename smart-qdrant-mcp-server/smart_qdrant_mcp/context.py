"""The dependency bundle every tool module receives at registration time."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Awaitable, Callable

from smart_qdrant_mcp.client import MiddlewareClient
from smart_qdrant_mcp.config import Settings


@dataclass(frozen=True)
class ToolContext:
    """Shared dependencies for tool implementations."""

    client: MiddlewareClient
    settings: Settings
    close_auth: Callable[[], Awaitable[None]] | None = None

    async def aclose(self) -> None:
        """Release application-owned pools, including after a close failure."""
        try:
            await self.client.aclose()
        finally:
            if self.close_auth is not None:
                await self.close_auth()

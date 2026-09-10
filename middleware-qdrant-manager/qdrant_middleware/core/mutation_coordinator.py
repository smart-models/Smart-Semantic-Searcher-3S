"""Per-application, per-collection exclusion for mutating workflows."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class _LeaseEntry:
    """One collection lock and the number of users waiting for or holding it."""

    lock: asyncio.Lock
    users: int = 0


class MutationCoordinator:
    """Serialize mutations of one collection without blocking other collections."""

    def __init__(self) -> None:
        """Initialize an empty collection-to-lease registry."""
        self._entries: dict[str, _LeaseEntry] = {}
        self._registry_lock = asyncio.Lock()

    @asynccontextmanager
    async def lease(self, collection_name: str) -> AsyncIterator[None]:
        """Hold the collection lease until the guarded mutation finishes."""
        async with self._registry_lock:
            entry = self._entries.get(collection_name)
            if entry is None:
                entry = _LeaseEntry(lock=asyncio.Lock())
                self._entries[collection_name] = entry
            entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            async with self._registry_lock:
                entry.users -= 1
                if entry.users == 0 and self._entries.get(collection_name) is entry:
                    self._entries.pop(collection_name)

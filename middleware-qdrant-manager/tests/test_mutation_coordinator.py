"""Tests for per-collection mutation exclusion."""

import asyncio

import pytest
from fastapi import HTTPException

from qdrant_middleware.core.mutation_coordinator import MutationCoordinator
from qdrant_middleware.services.mutation_guard import mutation_scope


class RunningReindexManager:
    """Minimal manager reporting one process-owned reindex task."""

    def is_running(self, collection_name: str) -> bool:
        """Report the collection as actively regenerated."""
        return collection_name == "docs"


async def test_same_collection_waits_for_the_current_lease():
    """A second mutation cannot enter before the first leaves its collection."""
    coordinator = MutationCoordinator()
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_waiting = asyncio.Event()
    second_entered = asyncio.Event()

    async def first() -> None:
        """Hold the first lease until the test releases it."""
        async with coordinator.lease("docs"):
            first_entered.set()
            await release_first.wait()

    async def second() -> None:
        """Record entry after the first lease becomes available."""
        second_waiting.set()
        async with coordinator.lease("docs"):
            second_entered.set()

    first_task = asyncio.create_task(first())
    await asyncio.wait_for(first_entered.wait(), 2)
    second_task = asyncio.create_task(second())
    await asyncio.wait_for(second_waiting.wait(), 2)
    assert not second_entered.is_set()
    release_first.set()
    await asyncio.wait_for(second_entered.wait(), 2)
    await asyncio.gather(first_task, second_task)


async def test_different_collection_is_not_blocked():
    """A lease on one collection leaves another collection available."""
    coordinator = MutationCoordinator()
    entered = asyncio.Event()

    async def other() -> None:
        """Enter the independent collection lease."""
        async with coordinator.lease("other"):
            entered.set()

    async with coordinator.lease("docs"):
        task = asyncio.create_task(other())
        await asyncio.wait_for(entered.wait(), 2)
    await task


async def test_cancelled_waiter_does_not_leak_a_lease_entry():
    """Cancelling before acquisition releases the registry reference."""
    coordinator = MutationCoordinator()
    entered = asyncio.Event()
    waiting = asyncio.Event()

    async def waiting_mutation() -> None:
        """Wait on the collection held by the outer context."""
        waiting.set()
        async with coordinator.lease("docs"):
            entered.set()

    async with coordinator.lease("docs"):
        task = asyncio.create_task(waiting_mutation())
        await asyncio.wait_for(waiting.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert "docs" in coordinator._entries
        assert not entered.is_set()
    assert coordinator._entries == {}


async def test_mutation_scope_rejects_a_write_after_reindex_admission():
    """The shared coordinator makes a started reindex visible to later writes."""
    coordinator = MutationCoordinator()

    with pytest.raises(HTTPException) as exc_info:
        async with mutation_scope(
            client=None,
            collection_name="docs",
            settings=None,
            coordinator=coordinator,
            manager=RunningReindexManager(),
        ):
            pytest.fail("an active reindex must reject the mutation")

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["status"] == "reindex_in_progress"

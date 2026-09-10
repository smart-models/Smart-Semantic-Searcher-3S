"""In-memory runtime state shared through ``app.state``."""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from qdrant_middleware.core.config import Settings
from qdrant_middleware.core.mutation_coordinator import MutationCoordinator


class ServerStats:
    """Mutable request counters and uptime data exposed by ``/stats``."""

    def __init__(self, collection_name: str) -> None:
        """Initialize counters for the given collection at process start."""
        self.collection_name = collection_name
        self.start_time = time.time()
        self.total_requests = 0
        self.rejected_requests = 0

    def get_stats(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of current counters."""
        uptime_seconds = time.time() - self.start_time
        hours, remainder = divmod(uptime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return {
            "uptime": f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
            "uptime_seconds": int(uptime_seconds),
            "total_requests": self.total_requests,
            "rejected_requests": self.rejected_requests,
            "collection": self.collection_name,
        }


class RateLimiter:
    """Async token-bucket limiter keyed by client identifier.

    Bucket timestamps come from ``time.monotonic()``, not the wall clock: a
    refill is a measure of elapsed process time, so an NTP step or a wall-clock
    change must not grant or withhold tokens.
    """

    def __init__(self, requests_per_minute: int, burst_size: int):
        """Initialize the limiter with a sustained rate and burst capacity."""
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.buckets: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"tokens": float(burst_size), "last_update": time.monotonic()}
        )
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None

    def start_cleanup_task(self) -> None:
        """Start the background cleanup task if it is not already running."""
        if self._cleanup_task and not self._cleanup_task.done():
            return
        self._cleanup_task = asyncio.create_task(self._cleanup_old_buckets())

    async def stop_cleanup_task(self) -> None:
        """Cancel and await the background cleanup task."""
        if not self._cleanup_task:
            return
        self._cleanup_task.cancel()
        try:
            await self._cleanup_task
        except asyncio.CancelledError:
            pass
        finally:
            self._cleanup_task = None

    async def check_rate_limit(self, client_id: str) -> bool:
        """Consume one token for ``client_id`` and return whether it is allowed."""
        async with self._lock:
            now = time.monotonic()
            bucket = self.buckets[client_id]
            elapsed = now - bucket["last_update"]
            bucket["tokens"] = min(
                self.burst_size,
                bucket["tokens"] + elapsed * (self.requests_per_minute / 60),
            )
            bucket["last_update"] = now
            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                return True
            return False

    async def _cleanup_old_buckets(self) -> None:
        """Hourly loop dropping buckets idle for more than one hour."""
        while True:
            await asyncio.sleep(3600)
            cutoff = time.monotonic() - 3600
            async with self._lock:
                for key in [
                    k for k, v in self.buckets.items() if v["last_update"] < cutoff
                ]:
                    del self.buckets[key]


@dataclass
class RuntimeState:
    """Container for per-app mutable runtime objects."""

    settings: Settings
    stats: ServerStats
    rate_limiter: RateLimiter
    mutations: MutationCoordinator
    is_shutting_down: bool = False
    active_exports: int = 0


def create_runtime_state(settings: Settings) -> RuntimeState:
    """Build the runtime state owned by one FastAPI app instance."""
    return RuntimeState(
        settings=settings,
        stats=ServerStats(settings.collection_name),
        rate_limiter=RateLimiter(
            settings.rate_limit_requests_per_minute,
            settings.rate_limit_burst_size,
        ),
        mutations=MutationCoordinator(),
    )

"""Tests for the in-memory runtime state (counters and the rate limiter)."""

import time

import pytest

from qdrant_middleware.core.runtime import RateLimiter


def _monotonic_sequence(*values):
    """Return a callable that yields successive values, then repeats the last."""
    queue = list(values)

    def _next():
        if len(queue) > 1:
            return queue.pop(0)
        return queue[0]

    return _next


@pytest.mark.asyncio
async def test_bucket_refills_after_monotonic_time_passes(monkeypatch):
    limiter = RateLimiter(requests_per_minute=60, burst_size=1)
    clock = _monotonic_sequence(1000.0, 1000.0, 1000.5, 1000.5)
    monkeypatch.setattr(time, "monotonic", clock)

    assert await limiter.check_rate_limit("client") is True
    # No time has passed, so the single-token bucket is empty.
    assert await limiter.check_rate_limit("client") is False
    # Half a second of monotonic time restores 0.5 token, still below one.
    assert await limiter.check_rate_limit("client") is False


@pytest.mark.asyncio
async def test_refill_rate_scales_with_elapsed_monotonic_time(monkeypatch):
    limiter = RateLimiter(requests_per_minute=60, burst_size=1)
    # First read is the bucket creation timestamp, second is the first request
    # (a fresh full bucket), third is one second later (one token refilled).
    clock = _monotonic_sequence(1000.0, 1000.0, 1001.0)
    monkeypatch.setattr(time, "monotonic", clock)

    assert await limiter.check_rate_limit("client") is True
    # One full second refills one token at 60/min, so the request is allowed.
    assert await limiter.check_rate_limit("client") is True


@pytest.mark.asyncio
async def test_a_wall_clock_step_back_does_not_grant_tokens(monkeypatch):
    # The whole point of the monotonic clock: if a negative jump ever reached
    # the refill math, elapsed would be negative and tokens would shrink; if a
    # large forward jump reached it, a bucket would instantly overflow. A
    # monotonic source can only move forward.
    limiter = RateLimiter(requests_per_minute=60, burst_size=2)
    clock = _monotonic_sequence(1000.0, 1000.0)
    monkeypatch.setattr(time, "monotonic", clock)

    assert await limiter.check_rate_limit("client") is True
    assert await limiter.check_rate_limit("client") is True
    assert await limiter.check_rate_limit("client") is False

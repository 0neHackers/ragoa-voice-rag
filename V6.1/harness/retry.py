"""Retry with exponential backoff and full jitter, for every external call.

The task requires the pipeline to survive transient failures rather than crash on
them. Two things here are deliberate:

*Full jitter* (`sleep = random(0, delay)`) rather than fixed backoff, because the
benchmark fires 30–50+ queries in a tight loop. Un-jittered retries from a burst of
concurrent requests re-collide on every attempt and turn one provider hiccup into a
self-inflicted thundering herd.

*A total deadline*, not just an attempt count. This pipeline is latency-graded — three
attempts against a hung socket is a correct retry policy and a failed demo at the same
time. When the deadline passes, retrying stops even with attempts left.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 0.25
    max_delay_s: float = 4.0
    deadline_s: float | None = None
    retry_on: tuple[type[BaseException], ...] = (Exception,)

    def delay_for(self, attempt: int) -> float:
        """Full-jitter exponential backoff. `attempt` is 1-based."""
        ceiling = min(self.max_delay_s, self.base_delay_s * (2 ** (attempt - 1)))
        return random.uniform(0.0, ceiling)


class RetriesExhausted(RuntimeError):
    """Raised after the last attempt. Carries the final cause for classification."""

    def __init__(self, attempts: int, last: BaseException) -> None:
        super().__init__(f"{attempts} attempt(s) failed; last error: {last}")
        self.attempts = attempts
        self.last = last


def retry_sync(fn: Callable[[], T], policy: RetryPolicy | None = None) -> T:
    policy = policy or RetryPolicy()
    started = time.perf_counter()
    last: BaseException | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except policy.retry_on as exc:
            last = exc
            if attempt == policy.max_attempts:
                break
            delay = policy.delay_for(attempt)
            if policy.deadline_s is not None:
                spent = time.perf_counter() - started
                if spent + delay >= policy.deadline_s:
                    break  # no time left to be worth another attempt
            time.sleep(delay)

    raise RetriesExhausted(policy.max_attempts, last or RuntimeError("unknown failure"))


async def retry_async(fn: Callable[[], Awaitable[T]], policy: RetryPolicy | None = None) -> T:
    policy = policy or RetryPolicy()
    started = time.perf_counter()
    last: BaseException | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await fn()
        except policy.retry_on as exc:
            last = exc
            if attempt == policy.max_attempts:
                break
            delay = policy.delay_for(attempt)
            if policy.deadline_s is not None:
                spent = time.perf_counter() - started
                if spent + delay >= policy.deadline_s:
                    break
            await asyncio.sleep(delay)

    raise RetriesExhausted(policy.max_attempts, last or RuntimeError("unknown failure"))


__all__ = ["RetryPolicy", "RetriesExhausted", "retry_sync", "retry_async"]

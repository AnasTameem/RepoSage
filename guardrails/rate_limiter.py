"""
guardrails/rate_limiter.py  – sliding-window per-IP rate limiter.

Uses a simple in-memory deque per client key.  Thread-safe for asyncio
(single-threaded event loop).  Reset-able for testing.
"""
import time
from collections import defaultdict, deque

from config import RATE_LIMIT_RPM

_WINDOW_SECS: int = 60


class RateLimiter:
    """Sliding-window counter.  One instance shared for the whole process."""

    def __init__(self, rpm: int = RATE_LIMIT_RPM, window: int = _WINDOW_SECS) -> None:
        self.rpm = rpm
        self.window = window
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def is_allowed(self, client_key: str) -> bool:
        """Return True if the request is within quota, False otherwise."""
        now = time.monotonic()
        dq: deque[float] = self._hits[client_key]
        # Evict timestamps outside the current window
        cutoff = now - self.window
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= self.rpm:
            return False
        dq.append(now)
        return True

    def reset(self, client_key: str | None = None) -> None:
        """Clear counters (useful in tests)."""
        if client_key is None:
            self._hits.clear()
        else:
            self._hits.pop(client_key, None)


# Singleton reused by main.py
rate_limiter = RateLimiter()

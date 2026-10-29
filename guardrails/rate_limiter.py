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


def get_client_ip(request: object) -> str:
    """
    Extract the real client IP address, properly handling reverse proxies and CDNs.

    Checks in order of specificity:
      1. CF-Connecting-IP (Cloudflare)
      2. X-Real-IP (Nginx / standard reverse proxy)
      3. X-Forwarded-For (Standard multi-hop proxy header: leftmost IP is client)
      4. request.client.host (Direct socket connection fallback)
    """
    headers = getattr(request, "headers", None)
    if headers is not None:
        # 1. Cloudflare
        cf_ip = headers.get("CF-Connecting-IP")
        if cf_ip and cf_ip.strip():
            return cf_ip.strip()

        # 2. Nginx / generic reverse proxy
        x_real_ip = headers.get("X-Real-IP")
        if x_real_ip and x_real_ip.strip():
            return x_real_ip.strip()

        # 3. Standard multi-hop X-Forwarded-For: <client>, <proxy1>, <proxy2>
        xff = headers.get("X-Forwarded-For")
        if xff and xff.strip():
            return xff.split(",")[0].strip()

    # 4. Direct socket connection fallback
    client = getattr(request, "client", None)
    if client and getattr(client, "host", None):
        return client.host

    return "unknown"


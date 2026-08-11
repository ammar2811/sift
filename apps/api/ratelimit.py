"""A per-client ceiling on the endpoint that spends money.

Scope, stated plainly because a rate limiter that is trusted for more than it does is
worse than none: this is an in-process sliding window. Each replica counts its own
requests, so with `maxReplicas: 3` the effective ceiling is three times the configured
one. It exists to stop one client from emptying the Azure OpenAI budget in a loop, not
to enforce a quota, and there is no authentication to hang a real quota on.

A shared counter in Redis would be exact, and Redis is already a dependency. It is not
used here because Redis is optional and fails open: a limiter that stops limiting when
the cache goes down is a worse guarantee than one that never depended on it.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Callable

# Above this many tracked clients, drop the ones whose windows have expired. Bounds
# memory under a spray of one-request-per-address traffic without paying for a sweep on
# every call.
_PRUNE_ABOVE = 1024


class RateLimiter:
    """Sliding window over request timestamps, keyed by client."""

    def __init__(
        self,
        limit: int,
        window_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limit = limit
        self._window = window_s
        self._clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    @property
    def enabled(self) -> bool:
        return self._limit > 0

    def retry_after(self, key: str) -> float | None:
        """Seconds until `key` may retry, or None when the request is allowed.

        Recording the timestamp only on the allowed path means a client that keeps
        hammering while limited does not push its own window forward and lock itself
        out indefinitely.
        """
        if not self.enabled:
            return None

        now = self._clock()
        cutoff = now - self._window
        window = self._hits[key]
        while window and window[0] <= cutoff:
            window.popleft()

        if len(window) >= self._limit:
            return max(0.0, window[0] + self._window - now)

        window.append(now)
        if len(self._hits) > _PRUNE_ABOVE:
            self._prune(cutoff)
        return None

    def _prune(self, cutoff: float) -> None:
        for key in [k for k, v in self._hits.items() if not v or v[-1] <= cutoff]:
            del self._hits[key]


def client_key(client_host: str | None, forwarded_for: str | None) -> str:
    """Identify the caller, preferring the address the proxy reports.

    In the deployed shape every request arrives through nginx and then Container Apps,
    so `client.host` is a proxy and would put every user in one bucket. The left-most
    X-Forwarded-For entry is the original client.

    That header is client-settable, so this is spoofable by anyone talking to the API
    directly rather than through the front end. Given the limiter guards a budget rather
    than an account, being able to spoof it costs an attacker exactly as much as using
    a fresh address would, and the alternative - one shared bucket for all real users -
    is worse.
    """
    if forwarded_for:
        first = forwarded_for.split(",")[0].strip()
        if first:
            return first
    return client_host or "unknown"

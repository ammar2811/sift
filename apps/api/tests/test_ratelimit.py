"""Tests for the ask limiter.

The clock is injected rather than slept through, so the window's behaviour over a minute
is tested in microseconds and without a flaky timing dependency.
"""

from __future__ import annotations

from apps.api.ratelimit import RateLimiter, client_key


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestRateLimiter:
    def test_allows_up_to_the_limit(self) -> None:
        limiter = RateLimiter(3, clock=FakeClock())
        assert [limiter.retry_after("a") for _ in range(3)] == [None, None, None]

    def test_blocks_past_the_limit(self) -> None:
        limiter = RateLimiter(2, clock=FakeClock())
        limiter.retry_after("a")
        limiter.retry_after("a")
        assert limiter.retry_after("a") is not None

    def test_reports_when_to_retry(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter(1, window_s=60.0, clock=clock)
        limiter.retry_after("a")
        clock.advance(20)
        assert limiter.retry_after("a") == 40.0

    def test_the_window_slides(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter(2, window_s=60.0, clock=clock)
        limiter.retry_after("a")
        limiter.retry_after("a")
        assert limiter.retry_after("a") is not None
        clock.advance(61)
        assert limiter.retry_after("a") is None

    def test_a_blocked_client_does_not_extend_its_own_window(self) -> None:
        """Hammering while limited must not push the window forward indefinitely."""
        clock = FakeClock()
        limiter = RateLimiter(1, window_s=60.0, clock=clock)
        limiter.retry_after("a")
        for _ in range(5):
            clock.advance(5)
            assert limiter.retry_after("a") is not None
        clock.advance(40)
        assert limiter.retry_after("a") is None

    def test_clients_are_counted_separately(self) -> None:
        limiter = RateLimiter(1, clock=FakeClock())
        assert limiter.retry_after("a") is None
        assert limiter.retry_after("b") is None
        assert limiter.retry_after("a") is not None

    def test_a_limit_of_zero_disables_it(self) -> None:
        limiter = RateLimiter(0, clock=FakeClock())
        assert not limiter.enabled
        assert all(limiter.retry_after("a") is None for _ in range(100))

    def test_expired_clients_are_forgotten(self) -> None:
        """Otherwise one request each from many addresses grows the map without bound."""
        clock = FakeClock()
        limiter = RateLimiter(5, window_s=60.0, clock=clock)
        for i in range(2000):
            limiter.retry_after(f"client-{i}")
        clock.advance(61)
        for i in range(2000, 3200):
            limiter.retry_after(f"client-{i}")
        assert len(limiter._hits) < 2000


class TestClientKey:
    def test_prefers_the_forwarded_address(self) -> None:
        assert client_key("10.0.0.1", "203.0.113.7") == "203.0.113.7"

    def test_takes_the_original_client_from_a_proxy_chain(self) -> None:
        assert client_key("10.0.0.1", "203.0.113.7, 10.0.0.1, 10.0.0.2") == "203.0.113.7"

    def test_falls_back_to_the_socket_address(self) -> None:
        assert client_key("203.0.113.7", None) == "203.0.113.7"

    def test_an_empty_forwarded_header_does_not_win(self) -> None:
        assert client_key("203.0.113.7", "") == "203.0.113.7"
        assert client_key("203.0.113.7", "  ") == "203.0.113.7"

    def test_an_unknown_client_still_gets_a_key(self) -> None:
        """Falling back to one shared bucket is safer than not limiting at all."""
        assert client_key(None, None) == "unknown"

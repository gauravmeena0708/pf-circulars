"""A small in-process sliding-window rate limiter for a shared credential."""

from __future__ import annotations

import threading
import time
from collections import deque


class SlidingWindowRateLimiter:
    """Thread-safe sliding-window limiter shared across all sessions in one process.

    Streamlit serves every browser session from the same Python process, so a
    single instance of this class is the natural place to protect one shared
    credential (e.g. the server's default Hugging Face token) from being
    exhausted by concurrent users. It coordinates threads within this one
    process only -- it cannot, and is not meant to, coordinate across
    multiple deployed replicas.
    """

    def __init__(self, max_requests: int, window_seconds: float):
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def try_acquire(self, now: float | None = None) -> tuple[bool, float]:
        """Attempts to record one request.

        Returns (allowed, retry_after_seconds). A non-positive max_requests
        disables the limiter entirely (always allowed).
        """
        if self._max_requests <= 0:
            return True, 0.0

        current_time = time.monotonic() if now is None else now
        with self._lock:
            while self._timestamps and current_time - self._timestamps[0] > self._window_seconds:
                self._timestamps.popleft()

            if len(self._timestamps) >= self._max_requests:
                retry_after = self._window_seconds - (current_time - self._timestamps[0])
                return False, max(retry_after, 0.0)

            self._timestamps.append(current_time)
            return True, 0.0

    def reset(self) -> None:
        """Clears all recorded requests. Primarily useful for tests."""
        with self._lock:
            self._timestamps.clear()

import logging
import threading
import time
from collections import deque
from typing import Deque, Dict

logger = logging.getLogger(__name__)


class MinuteRateLimiter:
    def __init__(self, max_requests: int, window_seconds: int = 60) -> None:
        if max_requests <= 0:
            raise ValueError("max_requests must be > 0")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: Deque[float] = deque()
        self._lock = threading.Lock()

    def wait_for_slot(self) -> None:
        # Wait for a slot to become available.
        # The sliding window logic: keep only timestamps of requests in the last 60 seconds.
        # If the window is already full, the thread waits until the nearest slot becomes available.
        while True:
            sleep_for = 0.0
            now = time.monotonic()
            with self._lock:
                threshold = now - self.window_seconds
                while self._timestamps and self._timestamps[0] <= threshold:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.max_requests:
                    self._timestamps.append(now)
                    return

                oldest = self._timestamps[0]
                sleep_for = (oldest + self.window_seconds) - now + 0.01

            if sleep_for > 0:
                logger.info(
                    "Gemini rate limit guard: sleeping %.2f sec to respect RPM=%s",
                    sleep_for,
                    self.max_requests,
                )
                time.sleep(sleep_for)


_LIMITERS: Dict[str, MinuteRateLimiter] = {}
_REGISTRY_LOCK = threading.Lock()


def get_or_create_limiter(
    name: str,
    max_requests: int,
    window_seconds: int = 60,
) -> MinuteRateLimiter:
    with _REGISTRY_LOCK:
        limiter = _LIMITERS.get(name)
        if limiter is None:
            limiter = MinuteRateLimiter(
                max_requests=max_requests,
                window_seconds=window_seconds,
            )
            _LIMITERS[name] = limiter
        return limiter

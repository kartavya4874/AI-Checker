"""
Exponential backoff retry decorator and token bucket rate limiter.
"""

import time
import functools
import threading
from utils.logger import get_logger

logger = get_logger("retry_utils")


class TokenBucketRateLimiter:
    """Thread-safe token bucket rate limiter."""

    def __init__(self, rate_per_minute: int = 50):
        self.rate = rate_per_minute
        self.tokens = float(rate_per_minute)
        self.max_tokens = float(rate_per_minute)
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 120.0) -> bool:
        """Block until a token is available or timeout."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed * (self.rate / 60.0))
        self.last_refill = now


# Global rate limiter instance
_rate_limiter = None
_limiter_lock = threading.Lock()


def get_rate_limiter(rate_per_minute: int = 50) -> TokenBucketRateLimiter:
    """Get or create the global rate limiter."""
    global _rate_limiter
    with _limiter_lock:
        if _rate_limiter is None:
            _rate_limiter = TokenBucketRateLimiter(rate_per_minute)
    return _rate_limiter


def retry_with_backoff(max_retries: int = 3, base_delay: float = 2.0, rate_limit: bool = False):
    """
    Decorator: retry with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds (doubles each retry).
        rate_limit: If True, acquire a token before each attempt.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    if rate_limit:
                        limiter = get_rate_limiter()
                        if not limiter.acquire():
                            raise TimeoutError("Rate limiter timeout")

                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} "
                            f"failed: {e}. Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts: {e}"
                        )
            raise last_exception

        return wrapper

    return decorator

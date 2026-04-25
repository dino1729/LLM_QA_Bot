"""
Per-model token-bucket rate limiter for LLM Council.

Each model registered in LiteLLM has different rate limits.
The limiter blocks until the call is allowed, preventing 429 errors
from overwhelming any single provider during wiki operations.
"""
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass
class ModelRateLimit:
    """Rate limit configuration for a single model."""

    requests_per_minute: int = 60
    tokens_per_minute: int = 100_000


@dataclass
class _TokenBucket:
    """Thread-safe token bucket for rate limiting."""

    capacity: float
    refill_rate: float  # tokens per second
    tokens: float = field(init=False)
    last_refill: float = field(init=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.tokens = self.capacity
        self.last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def acquire(self, cost: float, timeout_seconds: float = 60.0) -> bool:
        """
        Attempt to consume `cost` tokens. Blocks until available or timeout.
        Returns True if acquired, False if timed out.
        """
        deadline = time.monotonic() + timeout_seconds
        while True:
            with self.lock:
                self._refill()
                if self.tokens >= cost:
                    self.tokens -= cost
                    return True

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            # Sleep a fraction of the time needed to refill the deficit
            deficit = cost - self.tokens
            wait = min(deficit / self.refill_rate, remaining, 1.0)
            time.sleep(max(wait, 0.05))


class RateLimiter:
    """
    Per-model rate limiter using dual token buckets (requests + tokens).

    Usage:
        limiter = RateLimiter({"gemini-3.1-pro": ModelRateLimit(60, 100000)})
        limiter.acquire("gemini-3.1-pro", estimated_tokens=2000)
    """

    def __init__(self, limits: Dict[str, Dict]) -> None:
        self._request_buckets: Dict[str, _TokenBucket] = {}
        self._token_buckets: Dict[str, _TokenBucket] = {}

        for model_name, limit_dict in limits.items():
            limit = ModelRateLimit(
                requests_per_minute=limit_dict.get("requests_per_minute", 60),
                tokens_per_minute=limit_dict.get("tokens_per_minute", 100_000),
            )
            # Request bucket: capacity = RPM, refill = RPM/60 per second
            self._request_buckets[model_name] = _TokenBucket(
                capacity=float(limit.requests_per_minute),
                refill_rate=limit.requests_per_minute / 60.0,
            )
            # Token bucket: capacity = TPM, refill = TPM/60 per second
            self._token_buckets[model_name] = _TokenBucket(
                capacity=float(limit.tokens_per_minute),
                refill_rate=limit.tokens_per_minute / 60.0,
            )

    def acquire(self, model_name: str, estimated_tokens: int = 1) -> None:
        """
        Block until rate limit allows this call.
        Raises RateLimitTimeout if blocked for >60 seconds.
        """
        if not self.acquire_with_timeout(model_name, estimated_tokens, timeout_seconds=60.0):
            raise RateLimitTimeout(
                f"Rate limit timeout for model {model_name} "
                f"(requested {estimated_tokens} tokens)"
            )

    def acquire_with_timeout(
        self, model_name: str, estimated_tokens: int = 1, timeout_seconds: float = 60.0
    ) -> bool:
        """
        Attempt to acquire rate limit. Returns False if timeout exceeded.
        For models not in the config, always returns True (no limiting).
        """
        req_bucket = self._request_buckets.get(model_name)
        tok_bucket = self._token_buckets.get(model_name)

        if req_bucket is None or tok_bucket is None:
            # Model not configured for rate limiting - allow through
            logger.debug("No rate limit configured for %s, allowing", model_name)
            return True

        # Acquire request slot first, then token slot
        if not req_bucket.acquire(1.0, timeout_seconds):
            logger.warning("Rate limit timeout (requests) for %s", model_name)
            return False

        if not tok_bucket.acquire(float(estimated_tokens), timeout_seconds):
            logger.warning(
                "Rate limit timeout (tokens) for %s (%d tokens)",
                model_name,
                estimated_tokens,
            )
            return False

        return True


class RateLimitTimeout(Exception):
    """Raised when a rate limit acquire times out."""

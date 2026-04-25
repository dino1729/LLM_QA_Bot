"""Tests for wiki.rate_limiter - RateLimiter and token buckets."""
import threading
import time

import pytest

from wiki.rate_limiter import ModelRateLimit, RateLimiter, RateLimitTimeout, _TokenBucket


class TestTokenBucket:
    def test_initial_capacity(self):
        bucket = _TokenBucket(capacity=10.0, refill_rate=1.0)
        assert bucket.acquire(5.0, timeout_seconds=0.1)

    def test_overdraw_blocks(self):
        bucket = _TokenBucket(capacity=1.0, refill_rate=10.0)
        assert bucket.acquire(1.0, timeout_seconds=0.1)
        # Second acquire should block briefly then succeed (refill rate is high)
        assert bucket.acquire(1.0, timeout_seconds=1.0)

    def test_overdraw_timeout(self):
        bucket = _TokenBucket(capacity=1.0, refill_rate=0.1)  # Very slow refill
        assert bucket.acquire(1.0, timeout_seconds=0.1)
        # Should timeout - not enough time to refill
        assert not bucket.acquire(1.0, timeout_seconds=0.1)


class TestRateLimiter:
    def test_unknown_model_passes(self):
        limiter = RateLimiter({})
        # Unknown models should be allowed through
        assert limiter.acquire_with_timeout("unknown-model", 1000)

    def test_configured_model_basic(self):
        limiter = RateLimiter({
            "test-model": {"requests_per_minute": 60, "tokens_per_minute": 100000}
        })
        assert limiter.acquire_with_timeout("test-model", 100)

    def test_acquire_timeout_returns_false(self):
        limiter = RateLimiter({
            "slow-model": {"requests_per_minute": 1, "tokens_per_minute": 1}
        })
        # First call consumes the single token
        assert limiter.acquire_with_timeout("slow-model", 1, timeout_seconds=0.1)
        # Second should fail - insufficient tokens and short timeout
        assert not limiter.acquire_with_timeout("slow-model", 1, timeout_seconds=0.2)

    def test_multiple_models_independent(self):
        limiter = RateLimiter({
            "model-a": {"requests_per_minute": 60, "tokens_per_minute": 100000},
            "model-b": {"requests_per_minute": 60, "tokens_per_minute": 100000},
        })
        # Both should work independently
        assert limiter.acquire_with_timeout("model-a", 100)
        assert limiter.acquire_with_timeout("model-b", 100)

import threading
import unittest

from rate_limiter import SlidingWindowRateLimiter


class SlidingWindowRateLimiterTests(unittest.TestCase):
    def test_allows_requests_up_to_the_limit(self):
        limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            allowed, retry_after = limiter.try_acquire(now=0.0)
            self.assertTrue(allowed)
            self.assertEqual(retry_after, 0.0)

    def test_rejects_once_the_limit_is_reached_within_the_window(self):
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
        limiter.try_acquire(now=0.0)
        limiter.try_acquire(now=1.0)

        allowed, retry_after = limiter.try_acquire(now=2.0)

        self.assertFalse(allowed)
        # The oldest request (now=0.0) falls out of the 60s window at t=60.
        self.assertAlmostEqual(retry_after, 58.0)

    def test_allows_again_once_the_oldest_request_ages_out_of_the_window(self):
        limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=10)
        limiter.try_acquire(now=0.0)

        still_blocked, _ = limiter.try_acquire(now=9.9)
        self.assertFalse(still_blocked)

        allowed_again, retry_after = limiter.try_acquire(now=10.1)
        self.assertTrue(allowed_again)
        self.assertEqual(retry_after, 0.0)

    def test_zero_max_requests_disables_the_limiter(self):
        limiter = SlidingWindowRateLimiter(max_requests=0, window_seconds=60)
        for _ in range(50):
            allowed, retry_after = limiter.try_acquire(now=0.0)
            self.assertTrue(allowed)
            self.assertEqual(retry_after, 0.0)

    def test_reset_clears_recorded_requests(self):
        limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
        limiter.try_acquire(now=0.0)
        self.assertFalse(limiter.try_acquire(now=1.0)[0])

        limiter.reset()

        self.assertTrue(limiter.try_acquire(now=1.0)[0])

    def test_is_thread_safe_under_concurrent_acquisition(self):
        limiter = SlidingWindowRateLimiter(max_requests=100, window_seconds=60)
        successes = []
        lock = threading.Lock()

        def worker():
            allowed, _ = limiter.try_acquire()
            if allowed:
                with lock:
                    successes.append(1)

        threads = [threading.Thread(target=worker) for _ in range(250)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Exactly max_requests acquisitions should succeed despite the
        # concurrent callers -- no double-counting or lost updates.
        self.assertEqual(len(successes), 100)


if __name__ == "__main__":
    unittest.main()

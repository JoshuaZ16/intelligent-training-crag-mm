import unittest

from agents.download_compat import (
    DEFAULT_IMAGE_USER_AGENT,
    request_with_retries,
    wikimedia_thumbnail_url,
)


class DownloadRetryTest(unittest.TestCase):
    def test_default_user_agent_identifies_the_public_project(self):
        self.assertIn("CRAGMMCourseResearch", DEFAULT_IMAGE_USER_AGENT)
        self.assertIn("github.com/JoshuaZ16/intelligent-training-crag-mm", DEFAULT_IMAGE_USER_AGENT)

    def test_wikimedia_original_is_converted_to_official_thumbnail(self):
        original = (
            "https://upload.wikimedia.org/wikipedia/commons/7/7f/"
            "Example_%28photo%29.jpg"
        )

        self.assertEqual(
            wikimedia_thumbnail_url(original, width=1280),
            "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/"
            "Example_%28photo%29.jpg/1280px-Example_%28photo%29.jpg",
        )

    def test_non_wikimedia_url_is_not_rewritten(self):
        self.assertIsNone(
            wikimedia_thumbnail_url("https://example.com/image.jpg", width=1280)
        )

    def test_transient_failures_use_exponential_backoff_then_succeed(self):
        attempts = []
        sleeps = []
        response = object()

        def flaky_get(url, **kwargs):
            attempts.append((url, kwargs))
            if len(attempts) < 3:
                raise TimeoutError("transient")
            return response

        result = request_with_retries(
            flaky_get,
            "https://example.com/image.jpg",
            attempts=4,
            timeout=30,
            headers={"User-Agent": "test"},
            sleep=sleeps.append,
        )

        self.assertIs(result, response)
        self.assertEqual(len(attempts), 3)
        self.assertEqual([call[1]["timeout"] for call in attempts], [30, 30, 30])
        self.assertEqual(sleeps, [1, 2])

    def test_final_exception_is_preserved_after_retry_budget(self):
        def always_fails(url, **kwargs):
            raise TimeoutError("still unavailable")

        with self.assertRaisesRegex(TimeoutError, "still unavailable"):
            request_with_retries(
                always_fails,
                "https://example.com/image.jpg",
                attempts=2,
                timeout=10,
                sleep=lambda _: None,
            )


if __name__ == "__main__":
    unittest.main()

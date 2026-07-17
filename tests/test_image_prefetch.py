import unittest

from agents.image_prefetch import prefetch_image_urls


class ImagePrefetchTest(unittest.TestCase):
    def test_urls_are_downloaded_and_embedded_images_are_skipped(self):
        rows = {
            4: {"image": None, "image_url": "https://example.com/4.jpg"},
            9: {"image": object(), "image_url": None},
        }
        downloaded = []

        records = prefetch_image_urls(
            rows.__getitem__,
            [4, 9],
            lambda url: downloaded.append(url) or "/cache/4.jpg",
        )

        self.assertEqual(downloaded, ["https://example.com/4.jpg"])
        self.assertEqual(records[0]["status"], "cached")
        self.assertEqual(records[1]["status"], "embedded")

    def test_failure_is_recorded_without_hiding_the_source_index(self):
        def fail(url):
            raise TimeoutError("network unavailable")

        records = prefetch_image_urls(
            lambda _: {"image": None, "image_url": "https://example.com/fail.jpg"},
            [12],
            fail,
        )

        self.assertEqual(records[0]["source_index"], 12)
        self.assertEqual(records[0]["status"], "failed")
        self.assertIn("network unavailable", records[0]["error"])


if __name__ == "__main__":
    unittest.main()

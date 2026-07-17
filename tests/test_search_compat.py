import unittest

from agents.search_compat import (
    LazyCollectionMetadata,
    LazyImageMetadata,
    load_collection_metadata_paged,
)


class FakeCollection:
    def __init__(self):
        self.calls = []
        self.rows = [
            ("0", {"page_name": "zero"}),
            ("1", {"page_name": "one"}),
            ("2", {"page_name": "two"}),
            ("3", {"page_name": "three"}),
            ("4", {"page_name": "four"}),
        ]

    def count(self):
        return len(self.rows)

    def get(self, *, limit=None, offset=None, ids=None, include):
        if ids is not None:
            self.calls.append((ids, include))
            wanted = set(ids)
            page = [row for row in self.rows if row[0] in wanted]
        else:
            self.calls.append((limit, offset, include))
            page = self.rows[offset : offset + limit]
        return {
            "ids": [row_id for row_id, _ in page],
            "metadatas": [metadata for _, metadata in page],
        }


class SearchCompatibilityTest(unittest.TestCase):
    def test_metadata_is_loaded_in_bounded_pages(self):
        collection = FakeCollection()

        metadata = load_collection_metadata_paged(collection, batch_size=2)

        self.assertEqual(metadata["0"]["page_name"], "zero")
        self.assertEqual(metadata["4"]["page_name"], "four")
        self.assertEqual(
            collection.calls,
            [
                (2, 0, ["metadatas"]),
                (2, 2, ["metadatas"]),
                (1, 4, ["metadatas"]),
            ],
        )

    def test_lazy_metadata_only_fetches_requested_ids_and_caches_them(self):
        collection = FakeCollection()
        metadata = LazyCollectionMetadata(collection)

        self.assertEqual(metadata["3"]["page_name"], "three")
        self.assertEqual(metadata["3"]["page_name"], "three")

        self.assertEqual(collection.calls, [(["3"], ["metadatas"])])
        self.assertEqual(len(metadata), 5)

    def test_lazy_metadata_raises_key_error_for_unknown_id(self):
        metadata = LazyCollectionMetadata(FakeCollection())

        with self.assertRaises(KeyError):
            _ = metadata["missing"]

    def test_lazy_image_metadata_populates_entity_cache_on_hit(self):
        collection = FakeCollection()
        collection.rows[2] = (
            "2",
            {
                "image_url": "https://example.com/2.jpg",
                "entities": '["Example Entity"]',
                "info": '{"Example Entity": {"type": "landmark"}}',
            },
        )
        entity_cache = {}
        metadata = LazyImageMetadata(collection, entity_cache)

        self.assertEqual(metadata[2]["image_url"], "https://example.com/2.jpg")
        self.assertEqual(entity_cache["Example Entity"], {"type": "landmark"})
        self.assertEqual(collection.calls, [(["2"], ["metadatas"])])


if __name__ == "__main__":
    unittest.main()

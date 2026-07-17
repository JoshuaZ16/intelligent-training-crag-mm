import unittest
from dataclasses import dataclass

from agents.adaptive_retrieval import (
    merge_web_results,
    rewrite_query,
    should_expand,
)


@dataclass(frozen=True)
class WebItem:
    evidence_id: str
    text: str
    metadata: dict
    score: float = 0.5


class AdaptiveRetrievalTest(unittest.TestCase):
    def test_only_mid_confidence_expands(self):
        self.assertFalse(should_expand(0.2, 0.3, 0.6, False))
        self.assertTrue(should_expand(0.4, 0.3, 0.6, False))
        self.assertFalse(should_expand(0.8, 0.3, 0.6, False))
        self.assertTrue(should_expand(0.8, 0.3, 0.6, True))

    def test_rewrite_begins_with_selected_entity(self):
        self.assertEqual(
            rewrite_query(
                "National Taiwan Museum",
                "Where is this museum located?",
            ),
            "National Taiwan Museum Where is this museum located?",
        )

    def test_merge_deduplicates_url_and_renumbers(self):
        old = [WebItem("WEB1", "old", {"url": "u1"})]
        new = [
            WebItem("WEBX", "duplicate", {"url": "u1"}),
            WebItem("WEBY", "new", {"url": "u2"}),
        ]
        merged = merge_web_results(old, new)
        self.assertEqual(
            [item.evidence_id for item in merged],
            ["WEB1", "WEB2"],
        )
        self.assertEqual(
            [item.metadata["url"] for item in merged],
            ["u1", "u2"],
        )


if __name__ == "__main__":
    unittest.main()

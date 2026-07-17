import unittest
from dataclasses import dataclass, field

from agents.entity_agreement import detect_numeric_conflict, score_and_rerank


@dataclass(frozen=True)
class Item:
    evidence_id: str
    source: str
    text: str
    score: float | None = None
    metadata: dict = field(default_factory=dict)


class EntityAgreementTest(unittest.TestCase):
    def test_web_support_can_promote_second_image_candidate(self):
        image = [
            Item("KG1", "image_kg", "Museum of Texas | location: Austin", 0.82),
            Item(
                "KG2",
                "image_kg",
                "National Taiwan Museum | location: Taipei",
                0.79,
            ),
        ]
        web = [
            Item(
                "WEB1",
                "web",
                "National Taiwan Museum is located in Taipei",
                0.9,
            )
        ]
        result = score_and_rerank(
            image,
            web,
            "Where is this museum located?",
        )
        self.assertEqual(
            [item.evidence_id for item in result.items],
            ["KG2", "KG1"],
        )
        self.assertGreater(result.top_score, result.second_score)

    def test_no_support_preserves_original_tie_order(self):
        image = [
            Item("KG1", "image_kg", "Alpha", 0.8),
            Item("KG2", "image_kg", "Beta", 0.8),
        ]
        result = score_and_rerank(image, [], "Who built it?")
        self.assertEqual(
            [item.evidence_id for item in result.items],
            ["KG1", "KG2"],
        )

    def test_multiple_fields_of_one_entity_count_as_one_candidate(self):
        image = [
            Item("KG1", "image_kg", "Alpha Tower | architect: A", 0.82),
            Item("KG2", "image_kg", "Alpha Tower | owner: B", 0.82),
            Item("KG3", "image_kg", "Beta Tower | location: Taipei", 0.79),
        ]
        web = [
            Item(
                "WEB1",
                "web",
                "Beta Tower is a landmark in Taipei",
                0.9,
            )
        ]
        result = score_and_rerank(image, web, "Where is the tower?")
        self.assertEqual(len(result.candidate_scores), 2)
        self.assertEqual(result.items[0].evidence_id, "KG3")
        self.assertEqual(
            [row["entity_name"] for row in result.candidate_scores],
            ["Beta Tower", "Alpha Tower"],
        )

    def test_numeric_conflict_requires_disjoint_numbers(self):
        self.assertTrue(
            detect_numeric_conflict(
                ["engine: 2.0 L"],
                ["base engine is 2.7 L"],
                "What is the base engine size?",
            )
        )
        self.assertFalse(
            detect_numeric_conflict(
                ["built in 1959"],
                ["award received in 1959"],
                "In what year was the award received?",
            )
        )
        self.assertFalse(
            detect_numeric_conflict(
                ["built in 1959"],
                ["renovated in 2011"],
                "Who designed this building?",
            )
        )


if __name__ == "__main__":
    unittest.main()

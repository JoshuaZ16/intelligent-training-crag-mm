import unittest

from scripts.score_vega_comparison import (
    exact_mcnemar,
    expand_reviews,
    score_labels,
)


class VegaComparisonScoringTest(unittest.TestCase):
    def test_expands_one_review_class_to_all_version_assignments(self):
        review_rows = [
            {
                "response_id": "R001",
                "query_id": "Q001",
                "label": "correct",
                "error_cause": "none",
                "rationale": "matches",
            }
        ]
        mapping = {
            "queries": [
                {"query_id": "Q001", "source_order": 0, "query": "q"}
            ],
            "assignments": [
                {"query_id": "Q001", "version": "b0", "response_id": "R001"},
                {"query_id": "Q001", "version": "a1", "response_id": "R001"},
            ],
        }

        expanded = expand_reviews(review_rows, mapping)

        self.assertEqual(len(expanded), 2)
        self.assertEqual(
            [row["version"] for row in expanded],
            ["b0", "a1"],
        )
        self.assertTrue(all(row["label"] == "correct" for row in expanded))

    def test_score_labels_uses_required_formulas(self):
        metrics = score_labels(
            ["correct", "missing", "partial", "incorrect"]
        )
        self.assertEqual(metrics["correct"], 1)
        self.assertEqual(metrics["missing_count"], 1)
        self.assertEqual(metrics["hallucination_count"], 2)
        self.assertAlmostEqual(metrics["accuracy"], 0.25)
        self.assertAlmostEqual(metrics["missing"], 0.25)
        self.assertAlmostEqual(metrics["hallucination"], 0.5)
        self.assertAlmostEqual(metrics["truthfulness"], -0.25)

    def test_mcnemar_returns_one_when_there_are_no_discordant_pairs(self):
        result = exact_mcnemar(0, 0)
        self.assertEqual(result["discordant"], 0)
        self.assertEqual(result["p_value"], 1.0)


if __name__ == "__main__":
    unittest.main()

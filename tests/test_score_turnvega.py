import unittest

from scripts.score_turnvega import (
    clustered_bootstrap,
    main40_metrics,
    paired_transitions,
    task3_sequence_metrics,
)


class TurnVegaSharedScoringTest(unittest.TestCase):
    def test_main40_formulas_match_hand_calculation(self):
        labels = ["C"] * 20 + ["P"] * 8 + ["I"] * 7 + ["M"] * 5

        result = main40_metrics(labels)

        self.assertEqual(
            {key: result[key] for key in ("N", "C", "P", "I", "M")},
            {"N": 40, "C": 20, "P": 8, "I": 7, "M": 5},
        )
        self.assertAlmostEqual(result["strict_accuracy"], 0.5)
        self.assertAlmostEqual(result["partial_accuracy"], 0.6)
        self.assertAlmostEqual(result["coverage"], 0.7)
        self.assertAlmostEqual(result["missing"], 0.125)
        self.assertAlmostEqual(result["hallucination"], 0.375)
        self.assertAlmostEqual(result["truthfulness"], 0.125)
        self.assertAlmostEqual(
            result["truthfulness"],
            result["strict_accuracy"] - result["hallucination"],
        )

    def test_rejects_unknown_empty_or_non_string_labels(self):
        for labels in ([], ["wrong"], [True], ["C", "correct"]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                main40_metrics(labels)

    def test_paired_transitions_report_absolute_changes_and_net(self):
        baseline = ["C", "P", "I", "M", "C"]
        variant = ["I", "C", "C", "M", "C"]

        result = paired_transitions(baseline, variant)

        self.assertEqual(result["C_to_W"], 1)
        self.assertEqual(result["W_to_C"], 2)
        self.assertEqual(result["net_correct_conversion_count"], 1)
        self.assertAlmostEqual(result["net_correct_conversion"], 0.2)

    def test_clustered_bootstrap_samples_whole_sessions(self):
        sessions = {
            "a": [{"value": 1.0}, {"value": 1.0}],
            "b": [{"value": 0.0}],
        }

        result = clustered_bootstrap(
            sessions,
            samples=200,
            seed=20260720,
            metric=lambda rows: sum(row["value"] for row in rows) / len(rows),
            include_samples=True,
        )

        possible = {0.0, 2.0 / 3.0, 1.0}
        self.assertTrue(result["samples"])
        self.assertTrue(
            all(any(abs(value - target) < 1e-12 for target in possible)
                for value in result["samples"])
        )
        self.assertNotIn(1.0 / 3.0, result["samples"])

    def test_task3_sequence_metrics_match_hand_calculation(self):
        rows = [
            {"session_key": "a", "turn_index": 0, "label": "C"},
            {"session_key": "a", "turn_index": 1, "label": "I"},
            {"session_key": "a", "turn_index": 2, "label": "C"},
            {"session_key": "b", "turn_index": 0, "label": "I"},
            {"session_key": "b", "turn_index": 1, "label": "I"},
            {"session_key": "b", "turn_index": 2, "label": "M"},
        ]

        result = task3_sequence_metrics(rows)

        self.assertAlmostEqual(result["per_turn_accuracy"], 2 / 6)
        self.assertAlmostEqual(result["whole_conversation_accuracy"], 0.0)
        self.assertAlmostEqual(result["average_successful_turns"], 1.0)
        self.assertAlmostEqual(result["recovery_at_1"], 1 / 3)
        self.assertAlmostEqual(result["epc"], -1 / 3)
        self.assertEqual(
            result["per_turn"],
            [
                {"turn_index": 0, "correct": 1, "total": 2, "accuracy": 0.5},
                {"turn_index": 1, "correct": 0, "total": 2, "accuracy": 0.0},
                {"turn_index": 2, "correct": 1, "total": 2, "accuracy": 0.5},
            ],
        )

    def test_history_and_image_harm_use_only_unneeded_paired_rows(self):
        rows = [
            {
                "session_key": "a",
                "turn_index": 0,
                "label": "I",
                "baseline_label": "C",
                "history_needed": False,
                "image_needed": True,
            },
            {
                "session_key": "a",
                "turn_index": 1,
                "label": "C",
                "baseline_label": "I",
                "history_needed": False,
                "image_needed": True,
            },
            {
                "session_key": "b",
                "turn_index": 0,
                "label": "I",
                "baseline_label": "C",
                "history_needed": True,
                "image_needed": False,
            },
        ]

        result = task3_sequence_metrics(rows)

        self.assertAlmostEqual(result["history_harm"], 0.0)
        self.assertAlmostEqual(result["image_harm"], 1.0)
        self.assertEqual(result["history_harm_denominator"], 2)
        self.assertEqual(result["image_harm_denominator"], 1)


if __name__ == "__main__":
    unittest.main()

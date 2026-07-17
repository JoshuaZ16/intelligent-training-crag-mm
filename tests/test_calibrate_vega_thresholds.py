import unittest

from scripts.calibrate_vega_thresholds import score_labels, select_thresholds


class VegaCalibrationTest(unittest.TestCase):
    def test_score_labels_uses_official_compatible_formula(self):
        metrics = score_labels(
            ["correct", "missing", "partial", "incorrect"]
        )
        self.assertEqual(metrics["correct"], 1)
        self.assertEqual(metrics["missing_count"], 1)
        self.assertEqual(metrics["hallucination_count"], 2)
        self.assertAlmostEqual(metrics["accuracy"], 0.25)
        self.assertAlmostEqual(metrics["missing"], 0.25)
        self.assertAlmostEqual(metrics["hallucination"], 0.50)
        self.assertAlmostEqual(metrics["truthfulness"], -0.25)

    def test_select_thresholds_uses_only_real_answer_labels(self):
        rows = [
            {
                "score": 0.25,
                "b0_label": "incorrect",
                "enriched_label": "incorrect",
            },
            {
                "score": 0.55,
                "b0_label": "incorrect",
                "enriched_label": "correct",
            },
            {
                "score": 0.65,
                "b0_label": "correct",
                "enriched_label": "incorrect",
            },
        ]
        chosen = select_thresholds(
            rows,
            grid=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        )
        self.assertEqual(chosen["tau_low"], 0.3)
        self.assertEqual(chosen["tau_high"], 0.6)
        self.assertEqual(
            chosen["selection_order"],
            [
                "truthfulness",
                "accuracy",
                "lower_missing",
                "simpler_threshold",
            ],
        )

    def test_select_thresholds_rejects_non_real_labels(self):
        rows = [
            {
                "score": 0.4,
                "b0_label": "estimated",
                "enriched_label": "correct",
            }
        ]
        with self.assertRaises(ValueError):
            select_thresholds(rows, grid=[0.3, 0.6])

    def test_coverage_constraint_excludes_all_missing_solution(self):
        rows = [
            {
                "score": 0.10,
                "b0_label": "correct",
                "enriched_label": "correct",
            },
            {
                "score": 0.25,
                "b0_label": "incorrect",
                "enriched_label": "incorrect",
            },
            {
                "score": 0.45,
                "b0_label": "incorrect",
                "enriched_label": "incorrect",
            },
        ]
        grid = [0.2, 0.3, 0.5, 0.6]

        unconstrained = select_thresholds(
            rows,
            grid=grid,
            min_coverage=0.0,
        )
        constrained = select_thresholds(
            rows,
            grid=grid,
            min_coverage=0.6,
        )

        self.assertEqual(
            (unconstrained["tau_low"], unconstrained["tau_high"]),
            (0.5, 0.6),
        )
        self.assertEqual(
            (constrained["tau_low"], constrained["tau_high"]),
            (0.2, 0.3),
        )
        self.assertGreaterEqual(constrained["metrics"]["coverage"], 0.6)
        self.assertEqual(constrained["minimum_coverage"], 0.6)
        self.assertGreater(constrained["rejected_candidate_count"], 0)

    def test_coverage_constraint_requires_unit_interval(self):
        rows = [
            {
                "score": 0.4,
                "b0_label": "correct",
                "enriched_label": "correct",
            }
        ]
        with self.assertRaisesRegex(
            ValueError,
            "min_coverage must be in",
        ):
            select_thresholds(
                rows,
                grid=[0.3, 0.6],
                min_coverage=1.1,
            )


if __name__ == "__main__":
    unittest.main()

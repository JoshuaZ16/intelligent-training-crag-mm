import unittest

from agents.turnvega_config import (
    DatasetKind,
    ExperimentBudget,
    TurnVegaVariant,
)
from agents.vega_config import ExperimentVariant, normalize_variant


class TurnVegaConfigTest(unittest.TestCase):
    def test_variants_expose_stable_cli_values(self):
        self.assertEqual(TurnVegaVariant.T2_B0.value, "t2_b0")
        self.assertEqual(
            TurnVegaVariant.T2_CIRCULARITY.value,
            "t2_circularity",
        )
        self.assertEqual(TurnVegaVariant.T3_LAST_TURN.value, "t3_last_turn")
        self.assertEqual(
            TurnVegaVariant.T3_STATE_GATED.value,
            "t3_state_gated",
        )
        self.assertEqual(
            [item.value for item in TurnVegaVariant],
            [
                "t2_b0",
                "t2_budget_b0",
                "t2_candidate_grid",
                "t2_relation_grid",
                "t2_circularity",
                "t2_answerability",
                "t2_evidence_card",
                "t2_typed_repair",
                "t2_core_full",
                "t3_no_history",
                "t3_full_history",
                "t3_last_turn",
                "t3_user_only",
                "t3_structured_state",
                "t3_verified_state",
                "t3_state_gated",
                "t3_core_full",
            ],
        )

    def test_budget_accepts_positive_ordered_limits(self):
        budget = ExperimentBudget(
            candidate_passages=30,
            prompt_evidence=5,
            max_search_calls=4,
        )

        self.assertEqual(budget.candidate_passages, 30)
        self.assertEqual(budget.prompt_evidence, 5)
        self.assertEqual(budget.max_search_calls, 4)

    def test_budget_rejects_non_positive_limits(self):
        for field in (
            "candidate_passages",
            "prompt_evidence",
            "max_search_calls",
        ):
            for invalid_value in (0, -1):
                values = {
                    "candidate_passages": 30,
                    "prompt_evidence": 5,
                    "max_search_calls": 4,
                }
                values[field] = invalid_value
                with self.subTest(field=field, value=invalid_value):
                    with self.assertRaises(ValueError):
                        ExperimentBudget(**values)

    def test_budget_accepts_prompt_evidence_equal_to_candidates(self):
        budget = ExperimentBudget(
            candidate_passages=5,
            prompt_evidence=5,
            max_search_calls=4,
        )

        self.assertEqual(budget.prompt_evidence, budget.candidate_passages)

    def test_budget_rejects_prompt_evidence_above_candidates(self):
        with self.assertRaises(ValueError):
            ExperimentBudget(
                candidate_passages=5,
                prompt_evidence=6,
                max_search_calls=4,
            )

    def test_dataset_kinds_are_stable_cli_values(self):
        self.assertEqual(
            [item.value for item in DatasetKind],
            ["task2", "task3"],
        )

    def test_normalize_variant_accepts_legacy_and_turnvega_values(self):
        for enum_type in (ExperimentVariant, TurnVegaVariant):
            for item in enum_type:
                with self.subTest(enum=enum_type.__name__, value=item.value):
                    self.assertEqual(normalize_variant(item.value), item.value)

    def test_normalize_variant_rejects_unsupported_values(self):
        with self.assertRaisesRegex(
            ValueError,
            "^unsupported experiment variant: unknown$",
        ):
            normalize_variant("unknown")


if __name__ == "__main__":
    unittest.main()

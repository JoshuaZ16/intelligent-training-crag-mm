import unittest

from agents.calibrated_abstention import GateAction, choose_action
from agents.vega_config import ExperimentVariant, VegaThresholds


class CalibratedAbstentionTest(unittest.TestCase):
    def setUp(self):
        self.thresholds = VegaThresholds(0.3, 0.6)

    def test_a3_refuses_only_low_or_conflict(self):
        self.assertEqual(
            choose_action(
                ExperimentVariant.A3,
                0.2,
                False,
                self.thresholds,
            ),
            GateAction.ABSTAIN,
        )
        self.assertEqual(
            choose_action(
                ExperimentVariant.A3,
                0.8,
                True,
                self.thresholds,
            ),
            GateAction.ABSTAIN,
        )
        self.assertEqual(
            choose_action(
                ExperimentVariant.A3,
                0.5,
                False,
                self.thresholds,
            ),
            GateAction.ANSWER,
        )

    def test_full_has_three_branches(self):
        self.assertEqual(
            choose_action(
                ExperimentVariant.FULL,
                0.2,
                False,
                self.thresholds,
            ),
            GateAction.ABSTAIN,
        )
        self.assertEqual(
            choose_action(
                ExperimentVariant.FULL,
                0.4,
                False,
                self.thresholds,
            ),
            GateAction.EXPAND,
        )
        self.assertEqual(
            choose_action(
                ExperimentVariant.FULL,
                0.8,
                False,
                self.thresholds,
            ),
            GateAction.ANSWER,
        )


if __name__ == "__main__":
    unittest.main()

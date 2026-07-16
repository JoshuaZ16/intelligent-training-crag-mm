import json
import tempfile
import unittest
from pathlib import Path

from agents.vega_config import ExperimentVariant, VegaThresholds


class VegaConfigTest(unittest.TestCase):
    def test_variants_are_stable_cli_values(self):
        self.assertEqual(
            [item.value for item in ExperimentVariant],
            ["b0", "a1", "a2", "a3", "full", "a2_enrich_all"],
        )

    def test_thresholds_require_ordered_unit_interval(self):
        with self.assertRaises(ValueError):
            VegaThresholds(tau_low=0.6, tau_high=0.3)
        with self.assertRaises(ValueError):
            VegaThresholds(tau_low=-0.1, tau_high=0.6)

    def test_thresholds_load_frozen_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vega_calibration.json"
            path.write_text(
                json.dumps({"tau_low": 0.3, "tau_high": 0.6}),
                encoding="utf-8",
            )
            value = VegaThresholds.from_json(path)
        self.assertEqual((value.tau_low, value.tau_high), (0.3, 0.6))


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from tools.build_vega_report_assets import (
    parse_gpu_telemetry,
    summarize_trace,
)


class VegaReportAssetsTest(unittest.TestCase):
    def test_gpu_telemetry_is_filtered_by_physical_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gpu.csv"
            path.write_text(
                "2026/07/16 12:00:00, 0, GPU, 40, 50 W, 100 MiB, 40960 MiB, 10 %\n"
                "2026/07/16 12:00:00, 1, GPU, 45, 70 W, 120 MiB, 40960 MiB, 20 %\n"
                "2026/07/16 12:00:05, 1, GPU, 50, 90 W, 200 MiB, 40960 MiB, 80 %\n",
                encoding="utf-8",
            )

            metrics = parse_gpu_telemetry(path, physical_gpu=1)

        self.assertEqual(metrics["telemetry_samples"], 2)
        self.assertEqual(metrics["gpu_peak_memory_mib"], 200.0)
        self.assertEqual(metrics["gpu_peak_utilization_pct"], 80.0)
        self.assertEqual(metrics["gpu_mean_utilization_pct"], 50.0)

    def test_trace_summary_uses_interpolated_percentiles(self):
        traces = [
            {
                "total_ms": value,
                "generation_ms": value / 2,
                "search_call_count": 2,
                "gate_action": "answer",
                "additional_web_evidence": [],
            }
            for value in (1.0, 2.0, 3.0, 4.0)
        ]

        metrics = summarize_trace(traces)

        self.assertEqual(metrics["trace_count"], 4)
        self.assertEqual(metrics["batch_latency_p50_ms"], 2.5)
        self.assertAlmostEqual(metrics["batch_latency_p95_ms"], 3.85)
        self.assertEqual(metrics["answer_count"], 4)
        self.assertEqual(metrics["expand_count"], 0)


if __name__ == "__main__":
    unittest.main()

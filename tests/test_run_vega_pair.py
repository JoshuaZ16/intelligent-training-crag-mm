import json
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.run_vega_pair import run_pair_commands, validate_run_artifacts


class VegaPairRunnerTest(unittest.TestCase):
    def test_pair_assigns_distinct_gpus_and_captures_exit_codes(self):
        code = "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"
        with tempfile.TemporaryDirectory() as tmp:
            pair_dir = Path(tmp) / "pair"
            summary = run_pair_commands(
                [sys.executable, "-c", code],
                [sys.executable, "-c", code],
                pair_dir,
            )
            left_log = (pair_dir / "left" / "stdout_stderr.log").read_text()
            right_log = (pair_dir / "right" / "stdout_stderr.log").read_text()
        self.assertEqual(summary["left_exit_status"], 0)
        self.assertEqual(summary["right_exit_status"], 0)
        self.assertLessEqual(summary["start_delta_seconds"], 30)
        self.assertEqual(left_log.strip(), "0")
        self.assertEqual(right_log.strip(), "1")

    def test_failed_side_is_preserved_and_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            pair_dir = Path(tmp) / "pair"
            summary = run_pair_commands(
                [sys.executable, "-c", "raise SystemExit(3)"],
                [sys.executable, "-c", "print('ok')"],
                pair_dir,
            )
            self.assertTrue((pair_dir / "left" / "exit_status.txt").exists())
            stored = json.loads((pair_dir / "pair_summary.json").read_text())
        self.assertFalse(summary["processes_ok"])
        self.assertEqual(stored["left_exit_status"], 3)

    def test_artifact_validation_requires_30_matching_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left"
            right = Path(tmp) / "right"
            left.mkdir()
            right.mkdir()
            rows = [{"query": f"q{i}", "status": "ok"} for i in range(30)]
            for directory in (left, right):
                (directory / "agent_trace_v3.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
            result = validate_run_artifacts(left, right, expected_count=30)
        self.assertTrue(result["artifacts_ok"])
        self.assertTrue(result["query_sets_match"])

    def test_artifact_validation_rejects_reordered_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left"
            right = Path(tmp) / "right"
            left.mkdir()
            right.mkdir()
            rows = [{"query": f"q{i}", "status": "ok"} for i in range(30)]
            reordered = [rows[1], rows[0], *rows[2:]]
            for directory, values in ((left, rows), (right, reordered)):
                (directory / "agent_trace_v3.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in values),
                    encoding="utf-8",
                )
            result = validate_run_artifacts(left, right, expected_count=30)
        self.assertFalse(result["artifacts_ok"])
        self.assertFalse(result["query_sets_match"])

    def test_artifact_validation_rejects_duplicate_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left"
            right = Path(tmp) / "right"
            left.mkdir()
            right.mkdir()
            rows = [{"query": f"q{i}", "status": "ok"} for i in range(29)]
            rows.append(rows[-1])
            for directory in (left, right):
                (directory / "agent_trace_v3.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
            result = validate_run_artifacts(left, right, expected_count=30)
        self.assertFalse(result["artifacts_ok"])
        self.assertFalse(result["query_sets_match"])


if __name__ == "__main__":
    unittest.main()

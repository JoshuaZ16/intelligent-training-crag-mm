import json
import tempfile
import unittest
from pathlib import Path

from scripts.week2_experiment import (
    file_sha256,
    load_manifest_indices,
    parse_args,
)


class Week2ExperimentCliTest(unittest.TestCase):
    def test_cli_accepts_variant_manifest_and_run_id(self):
        args = parse_args(
            [
                "--mode",
                "task2",
                "--variant",
                "a1",
                "--run-id",
                "r1-a1",
                "--manifest-path",
                "sample_manifest.json",
                "--output-dir",
                "out",
            ]
        )
        self.assertEqual(args.variant, "a1")
        self.assertEqual(args.run_id, "r1-a1")
        self.assertTrue(args.manifest_path.endswith("sample_manifest.json"))

    def test_manifest_indices_and_sha_are_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample_manifest.json"
            path.write_text(
                json.dumps(
                    [
                        {"source_index": 69},
                        {"source_index": 111},
                        {"source_index": 176},
                    ]
                ),
                encoding="utf-8",
            )
            first_hash = file_sha256(path)
            indices = load_manifest_indices(path)
            second_hash = file_sha256(path)
        self.assertEqual(indices, [69, 111, 176])
        self.assertEqual(first_hash, second_hash)
        self.assertEqual(len(first_hash), 64)

    def test_manifest_rejects_duplicate_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample_manifest.json"
            path.write_text(
                json.dumps(
                    [
                        {"source_index": 69},
                        {"source_index": 69},
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_manifest_indices(path)


if __name__ == "__main__":
    unittest.main()

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.build_vega_blind_review import build_review


class VegaBlindReviewTest(unittest.TestCase):
    def _write_run(self, path, answers):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["query", "ground_truth", "agent_response"],
            )
            writer.writeheader()
            for query, ground_truth, answer in answers:
                writer.writerow(
                    {
                        "query": query,
                        "ground_truth": ground_truth,
                        "agent_response": answer,
                    }
                )

    def test_deduplicates_equal_answers_without_leaking_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b0 = root / "b0.csv"
            a1 = root / "a1.csv"
            self._write_run(
                b0,
                [("q1", "g1", "same"), ("q2", "g2", "left")],
            )
            self._write_run(
                a1,
                [("q1", "g1", "same"), ("q2", "g2", "right")],
            )

            review, mapping = build_review(
                {"b0": b0, "a1": a1},
                seed=20260716,
            )
            review_again, mapping_again = build_review(
                {"b0": b0, "a1": a1},
                seed=20260716,
            )

        self.assertEqual(review, review_again)
        self.assertEqual(mapping, mapping_again)
        self.assertEqual(len(review), 3)
        self.assertTrue(all("version" not in row for row in review))
        assignments = sum(
            len(item["versions"])
            for item in mapping["responses"].values()
        )
        self.assertEqual(assignments, 4)
        shared = [
            item
            for item in mapping["responses"].values()
            if len(item["versions"]) == 2
        ]
        self.assertEqual(len(shared), 1)
        self.assertEqual(shared[0]["versions"], ["b0", "a1"])


if __name__ == "__main__":
    unittest.main()

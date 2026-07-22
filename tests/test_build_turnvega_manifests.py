import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import scripts.build_turnvega_manifests as manifest_builder
from scripts.build_turnvega_manifests import build_manifests, write_manifests


MANIFEST_COUNTS = {
    "t2_cal40.json": 40,
    "t2_dev80.json": 80,
    "t2_test120.json": 120,
    "t3_cal20.json": 20,
    "t3_dev40.json": 40,
    "t3_test60.json": 60,
    "t2_main40.json": 40,
    "t3_main40.json": 40,
}
PUBLISHED_NAMES = tuple(MANIFEST_COUNTS) + tuple(
    f"{name}.sha256" for name in MANIFEST_COUNTS
) + ("manifest_summary.json", "MANIFESTS_COMPLETE")


def canonical_bytes(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def make_single_rows(count=270):
    rows = []
    for index in range(count):
        stratum_index = index % 48
        domain = stratum_index % 4
        category = (stratum_index // 4) % 3
        quality = (stratum_index // 12) % 2
        dynamism = (stratum_index // 24) % 2
        turns = [
            {
                "interaction_id": f"single-interaction-{index}-0",
                "query_category": f"category-{category}",
                "domain": f"domain-{domain}",
                "query": f"single query {index} turn 0",
                "image_quality": f"quality-{quality}",
                "dynamism": f"dynamic-{dynamism}",
            },
            {
                "interaction_id": f"single-interaction-{index}-1",
                "query_category": "not-the-first-category",
                "domain": "not-the-first-domain",
                "query": f"single query {index} turn 1",
                "image_quality": "not-the-first-quality",
                "dynamism": "not-the-first-dynamism",
            },
        ]
        if index % 2:
            turn_container = {
                key: [turn[key] for turn in turns]
                for key in turns[0]
            }
        else:
            turn_container = turns
        rows.append(
            {
                "source_index": index,
                "session_id": f"single-{index}",
                "interaction_id": "wrong-top-level-interaction",
                "query_category": "wrong-top-level-category",
                "domain": "wrong-top-level-domain",
                "query": "wrong top-level query",
                "image_quality": "wrong-top-level-quality",
                "dynamism": "wrong-top-level-dynamism",
                "turns": turn_container,
            }
        )
    return rows


def make_multi_rows(count=150):
    rows = []
    for index in range(count):
        turn_count = 2 + index % 3
        turns = [
            {
                "interaction_id": f"multi-interaction-{index}-{turn}",
                "query_category": f"multi-category-{index % 3}",
                "domain": f"multi-domain-{index % 4}",
                "query": f"multi query {index} turn {turn}",
                "image_quality": f"multi-quality-{index % 2}",
            }
            for turn in range(turn_count)
        ]
        if index % 2:
            turn_container = {
                key: [turn[key] for turn in turns]
                for key in turns[0]
            }
        else:
            turn_container = turns
        rows.append(
            {
                "source_index": 1000 + index,
                "session_id": f"multi-{index}",
                "is_ego": bool(index % 2),
                "turns": turn_container,
            }
        )
    return rows


def make_legacy_rows():
    rows = []
    for index in range(15):
        rows.append(
            {
                "source_index": index,
                "session_id": f"multi-{index}",
            }
        )
    for index in range(15, 30):
        rows.append(
            {
                "source_index": 1000 + index,
                "session_id": f"single-{index}",
            }
        )
    return rows


class BuildTurnVegaManifestsTest(unittest.TestCase):
    def setUp(self):
        self.single_rows = make_single_rows()
        self.multi_rows = make_multi_rows()
        self.legacy_rows = make_legacy_rows()

    def test_builds_exact_disjoint_splits_and_derived_suffixes(self):
        manifests = build_manifests(
            self.single_rows,
            self.multi_rows,
            self.legacy_rows,
        )

        self.assertEqual(
            {name: len(rows) for name, rows in manifests.items()},
            MANIFEST_COUNTS,
        )
        self.assertEqual(
            manifests["t2_main40.json"],
            manifests["t2_test120.json"][-40:],
        )
        self.assertEqual(
            manifests["t3_main40.json"],
            manifests["t3_test60.json"][-40:],
        )

        legacy_sources = {row["source_index"] for row in self.legacy_rows}
        legacy_sessions = {row["session_id"] for row in self.legacy_rows}
        required = {
            "source_index",
            "session_id",
            "interaction_id",
            "query_category",
            "domain",
            "query",
        }
        for task, split_names in {
            "t2": ("t2_cal40.json", "t2_dev80.json", "t2_test120.json"),
            "t3": ("t3_cal20.json", "t3_dev40.json", "t3_test60.json"),
        }.items():
            split_rows = [manifests[name] for name in split_names]
            for rows in split_rows:
                for row in rows:
                    self.assertTrue(required.issubset(row))
                    self.assertIs(type(row["source_index"]), int)
                    self.assertNotIn(row["source_index"], legacy_sources)
                    self.assertNotIn(row["session_id"], legacy_sessions)
                    if task == "t3":
                        self.assertIn("turn_count", row)
            for left_index, left in enumerate(split_rows):
                for right in split_rows[left_index + 1 :]:
                    self.assertFalse(
                        {row["source_index"] for row in left}
                        & {row["source_index"] for row in right}
                    )
                    self.assertFalse(
                        {row["session_id"] for row in left}
                        & {row["session_id"] for row in right}
                    )

    def test_keeps_each_task3_session_whole_for_both_turn_shapes(self):
        manifests = build_manifests(
            self.single_rows,
            self.multi_rows,
            self.legacy_rows,
        )
        selected = {
            row["session_id"]: row
            for name in ("t3_cal20.json", "t3_dev40.json", "t3_test60.json")
            for row in manifests[name]
        }

        self.assertEqual(len(selected), 120)
        for source in self.multi_rows[30:]:
            row = selected[source["session_id"]]
            turns = source["turns"]
            expected_count = (
                len(turns["query"]) if isinstance(turns, dict) else len(turns)
            )
            expected_first_interaction = (
                turns["interaction_id"][0]
                if isinstance(turns, dict)
                else turns[0]["interaction_id"]
            )
            self.assertEqual(row["turn_count"], expected_count)
            self.assertEqual(row["interaction_id"], expected_first_interaction)

    def test_task2_uses_first_turn_fields_and_real_strata_for_both_shapes(self):
        manifests = build_manifests(
            self.single_rows,
            self.multi_rows,
            self.legacy_rows,
        )
        selected = [
            row
            for name in ("t2_cal40.json", "t2_dev80.json", "t2_test120.json")
            for row in manifests[name]
        ]

        self.assertEqual(len(selected), 240)
        for row in selected:
            source_index = row["source_index"]
            stratum_index = source_index % 48
            self.assertEqual(
                row["interaction_id"],
                f"single-interaction-{source_index}-0",
            )
            self.assertEqual(
                row["query_category"],
                f"category-{(stratum_index // 4) % 3}",
            )
            self.assertEqual(row["domain"], f"domain-{stratum_index % 4}")
            self.assertEqual(row["query"], f"single query {source_index} turn 0")
            self.assertTrue(row["interaction_id"])
            self.assertTrue(row["query_category"])
            self.assertTrue(row["domain"])
            self.assertTrue(row["query"])

        first_round_strata = {
            (
                row["source_index"] % 48 % 4,
                (row["source_index"] % 48 // 4) % 3,
                (row["source_index"] % 48 // 12) % 2,
                (row["source_index"] % 48 // 24) % 2,
            )
            for row in selected[:48]
        }
        self.assertEqual(len(first_round_strata), 48)

    def test_writes_canonical_reproducible_json_hashes_and_summary(self):
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first_dir = Path(first_tmp)
            second_dir = Path(second_tmp)
            first_summary = write_manifests(
                self.single_rows,
                self.multi_rows,
                self.legacy_rows,
                first_dir,
                seed=20260720,
            )
            second_summary = write_manifests(
                self.single_rows,
                self.multi_rows,
                self.legacy_rows,
                second_dir,
                seed=20260720,
            )

            self.assertEqual(first_summary, second_summary)
            for name, expected_count in MANIFEST_COUNTS.items():
                first_bytes = (first_dir / name).read_bytes()
                second_bytes = (second_dir / name).read_bytes()
                rows = json.loads(first_bytes)
                self.assertEqual(first_bytes, second_bytes)
                self.assertEqual(first_bytes, canonical_bytes(rows))
                self.assertEqual(len(rows), expected_count)
                digest = hashlib.sha256(first_bytes).hexdigest()
                hash_name = f"{name}.sha256"
                self.assertEqual(
                    (first_dir / hash_name).read_text(encoding="utf-8"),
                    f"{digest}\n",
                )
                self.assertEqual(
                    (first_dir / hash_name).read_bytes(),
                    (second_dir / hash_name).read_bytes(),
                )

            summary_path = first_dir / "manifest_summary.json"
            self.assertEqual(
                summary_path.read_bytes(),
                canonical_bytes(first_summary),
            )
            self.assertEqual(first_summary["counts"], MANIFEST_COUNTS)
            self.assertTrue(
                all(
                    overlap["source_overlap_count"] == 0
                    and overlap["session_overlap_count"] == 0
                    for overlap in first_summary[
                        "split_pair_overlap_count"
                    ].values()
                )
            )
            self.assertEqual(
                first_summary["cross_task_session_overlap_count"],
                0,
            )
            self.assertEqual(
                first_summary["source_index_namespace"],
                "dataset-local; cross-task source indices are not compared",
            )
            for task in ("t2", "t3"):
                derived = first_summary["derived_main40"][f"{task}_main40.json"]
                test_name = f"{task}_test{120 if task == 't2' else 60}.json"
                test_rows = json.loads((first_dir / test_name).read_bytes())
                self.assertEqual(derived["source_manifest"], test_name)
                self.assertEqual(
                    derived["source_manifest_sha256"],
                    hashlib.sha256((first_dir / test_name).read_bytes()).hexdigest(),
                )
                self.assertEqual(
                    derived["source_indices"],
                    [row["source_index"] for row in test_rows[-40:]],
                )
                self.assertEqual(
                    derived["session_ids"],
                    [row["session_id"] for row in test_rows[-40:]],
                )
            summary_sha = hashlib.sha256(summary_path.read_bytes()).hexdigest()
            self.assertEqual(
                (first_dir / "MANIFESTS_COMPLETE").read_text(encoding="utf-8"),
                f"{summary_sha}\n",
            )

    def test_rejects_insufficient_task2_or_task3_candidates(self):
        cases = (
            (make_single_rows(239), make_multi_rows(120), "Task 2"),
            (make_single_rows(240), make_multi_rows(119), "Task 3"),
        )
        for single_rows, multi_rows, task_name in cases:
            with self.subTest(task=task_name):
                with self.assertRaisesRegex(ValueError, task_name):
                    build_manifests(single_rows, multi_rows, [])

    def test_input_permutations_produce_identical_bytes_and_hashes(self):
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first_dir = Path(first_tmp)
            second_dir = Path(second_tmp)
            write_manifests(
                self.single_rows,
                self.multi_rows,
                self.legacy_rows,
                first_dir,
            )
            write_manifests(
                list(reversed(self.single_rows)),
                self.multi_rows[::2] + self.multi_rows[1::2],
                list(reversed(self.legacy_rows)),
                second_dir,
            )

            for name in PUBLISHED_NAMES:
                self.assertEqual(
                    (first_dir / name).read_bytes(),
                    (second_dir / name).read_bytes(),
                    name,
                )

    def test_non_manifest_binary_fields_do_not_affect_order_or_hashes(self):
        baseline_single = make_single_rows()
        baseline_multi = make_multi_rows()
        binary_single = make_single_rows()
        binary_multi = make_multi_rows()
        for index, row in enumerate(binary_single):
            row["image"] = bytes((index % 256,))
        for row in binary_multi:
            row["image"] = object()

        baseline = build_manifests(
            baseline_single,
            baseline_multi,
            self.legacy_rows,
        )
        with_binary_fields = build_manifests(
            binary_single,
            binary_multi,
            self.legacy_rows,
        )
        self.assertEqual(with_binary_fields, baseline)

        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first_dir = Path(first_tmp)
            second_dir = Path(second_tmp)
            write_manifests(
                baseline_single,
                baseline_multi,
                self.legacy_rows,
                first_dir,
            )
            write_manifests(
                binary_single,
                binary_multi,
                self.legacy_rows,
                second_dir,
            )
            for name in PUBLISHED_NAMES:
                self.assertEqual(
                    (first_dir / name).read_bytes(),
                    (second_dir / name).read_bytes(),
                    name,
                )

    def test_rejects_invalid_or_duplicate_dataset_identifiers(self):
        cases = []
        for invalid_source in (True, "7"):
            rows = make_single_rows(240)
            rows[0]["source_index"] = invalid_source
            cases.append((rows, make_multi_rows(120), "source_index"))
        for invalid_session in ("", 7):
            rows = make_multi_rows(120)
            rows[0]["session_id"] = invalid_session
            cases.append((make_single_rows(240), rows, "session_id"))
        duplicate_source_rows = make_single_rows(240)
        duplicate_source_rows[1]["source_index"] = duplicate_source_rows[0][
            "source_index"
        ]
        cases.append(
            (duplicate_source_rows, make_multi_rows(120), "duplicate source_index")
        )
        duplicate_session_rows = make_multi_rows(120)
        duplicate_session_rows[1]["session_id"] = duplicate_session_rows[0][
            "session_id"
        ]
        cases.append(
            (make_single_rows(240), duplicate_session_rows, "duplicate session_id")
        )

        for single_rows, multi_rows, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build_manifests(single_rows, multi_rows, [])

    def test_validates_legacy_manifest_shape_and_identifiers(self):
        invalid_legacy_values = (
            {},
            ["not-a-dict"],
            [{}],
            [{"source_index": True}],
            [{"source_index": "1"}],
            [{"session_id": ""}],
            [{"session_id": 1}],
        )
        for legacy_rows in invalid_legacy_values:
            with self.subTest(legacy_rows=legacy_rows):
                with self.assertRaisesRegex(ValueError, "legacy"):
                    build_manifests(
                        make_single_rows(240),
                        make_multi_rows(120),
                        legacy_rows,
                    )

    def test_records_assigned_source_indices_in_summary(self):
        single_rows = make_single_rows(240)
        multi_rows = make_multi_rows(120)
        del single_rows[0]["source_index"]
        del multi_rows[0]["source_index"]
        with tempfile.TemporaryDirectory() as tmp:
            summary = write_manifests(
                single_rows,
                multi_rows,
                [],
                Path(tmp),
            )

        self.assertEqual(summary["assigned_source_index_count"], 2)
        self.assertEqual(
            summary["assigned_source_index_count_by_dataset"],
            {"task2": 1, "task3": 1},
        )

    def test_source_indices_are_local_but_cross_task_sessions_do_not_leak(self):
        single_rows = make_single_rows(240)
        multi_rows = make_multi_rows(121)
        for index, row in enumerate(multi_rows):
            row["source_index"] = index
        shared_session = single_rows[0]["session_id"]
        multi_rows[0]["session_id"] = shared_session

        manifests = build_manifests(single_rows, multi_rows, [])
        task3_rows = [
            row
            for name in ("t3_cal20.json", "t3_dev40.json", "t3_test60.json")
            for row in manifests[name]
        ]
        self.assertNotIn(shared_session, {row["session_id"] for row in task3_rows})
        self.assertTrue(
            {row["source_index"] for row in manifests["t2_cal40.json"]}
            & {row["source_index"] for row in task3_rows}
        )

        insufficient_multi = make_multi_rows(120)
        insufficient_multi[0]["session_id"] = shared_session
        with self.assertRaisesRegex(ValueError, "Task 3"):
            build_manifests(single_rows, insufficient_multi, [])

    def test_staging_failure_does_not_publish_or_change_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            existing = output_dir / "t2_cal40.json"
            existing.write_bytes(b"existing-target\n")
            real_safe_write = manifest_builder._safe_write
            call_count = 0

            def fail_third_write(path, payload):
                nonlocal call_count
                call_count += 1
                if call_count == 3:
                    raise OSError("injected staging failure")
                return real_safe_write(path, payload)

            with mock.patch.object(
                manifest_builder,
                "_safe_write",
                side_effect=fail_third_write,
            ):
                with self.assertRaisesRegex(OSError, "injected staging failure"):
                    write_manifests(
                        self.single_rows,
                        self.multi_rows,
                        self.legacy_rows,
                        output_dir,
                    )

            self.assertEqual(existing.read_bytes(), b"existing-target\n")
            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                ["t2_cal40.json"],
            )

    def test_fsyncs_output_directory_before_and_after_marker_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            events = []
            real_replace = manifest_builder.os.replace
            real_fsync_directory = manifest_builder._fsync_directory

            def tracked_replace(source, target):
                events.append(("replace", Path(target).name))
                return real_replace(source, target)

            def tracked_fsync(path):
                events.append(("fsync", Path(path)))
                return real_fsync_directory(path)

            with mock.patch.object(
                manifest_builder.os,
                "replace",
                side_effect=tracked_replace,
            ), mock.patch.object(
                manifest_builder,
                "_fsync_directory",
                side_effect=tracked_fsync,
            ):
                write_manifests(
                    self.single_rows,
                    self.multi_rows,
                    self.legacy_rows,
                    output_dir,
                )

            summary_replace = events.index(
                ("replace", "manifest_summary.json")
            )
            marker_replace = events.index(("replace", "MANIFESTS_COMPLETE"))
            output_fsyncs = [
                index
                for index, event in enumerate(events)
                if event == ("fsync", output_dir)
            ]
            self.assertTrue(
                any(
                    summary_replace < index < marker_replace
                    for index in output_fsyncs
                )
            )
            self.assertTrue(any(index > marker_replace for index in output_fsyncs))

    def test_rejects_target_symlink_without_touching_victim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            output_dir.mkdir()
            victim = root / "victim.txt"
            victim.write_bytes(b"do-not-touch\n")
            (output_dir / "t2_cal40.json").symlink_to(victim)

            with self.assertRaisesRegex(ValueError, "symlink"):
                write_manifests(
                    self.single_rows,
                    self.multi_rows,
                    self.legacy_rows,
                    output_dir,
                )

            self.assertEqual(victim.read_bytes(), b"do-not-touch\n")


if __name__ == "__main__":
    unittest.main()

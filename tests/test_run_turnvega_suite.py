import copy
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import scripts.run_turnvega_suite as suite


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = PROJECT_ROOT / "configs" / "turnvega_core_experiments.json"
EXPECTED_IDS = [
    "T2-R1", "T2-R2", "T2-R3", "T2-R4", "T2-R5", "T2-R6",
    "T2-C0", "T2-C1", "T2-C2",
    "T3-R1", "T3-R2", "T3-R3", "T3-R4", "T3-R5", "T3-R6",
    "T3-C0", "T3-C1", "T3-C2",
]
COMMON = {
    "seed": 20260720,
    "temperature": 0,
    "candidate_passages": 30,
    "prompt_evidence": 5,
    "max_search_calls": 4,
    "max_evidence_chars": 5500,
    "trace_schema": "v5",
    "execution_mode": "auto",
}


class TurnVegaSuiteTest(unittest.TestCase):
    def _write_matrix(self, root, payload=None):
        path = Path(root) / "matrix.json"
        payload = suite.load_matrix(MATRIX_PATH) if payload is None else payload
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _run(self, root, fake, **kwargs):
        root = Path(root)
        kwargs.setdefault("stdout", io.StringIO())
        kwargs.setdefault("single_runner", fake)
        return suite.run_suite(
            self._write_matrix(root),
            project_root=root,
            status_path=root / "suite_status.json",
            pair_runner=fake,
            execution_mode_resolver=lambda requested: "sequential_triplet",
            **kwargs,
        )

    @staticmethod
    def _write_result(entry, output_path, accepted=True):
        output_path.mkdir(parents=True, exist_ok=True)
        if accepted:
            if entry["phase"] == "dev":
                (output_path / "triplet_summary.json").write_text(
                    json.dumps({"status": "completed", "runner_valid": True}),
                    encoding="utf-8",
                )
            else:
                (output_path / "run_config.json").write_text(
                    json.dumps({
                        "status": "completed",
                        "variant": entry["variant"],
                        "dataset_kind": entry["dataset_kind"],
                    }),
                    encoding="utf-8",
                )
        (output_path / "audit.json").write_text(
            json.dumps({"accepted": accepted}), encoding="utf-8"
        )

    @classmethod
    def _accepted_runner(cls, calls):
        def fake(entry, output_path, execution_mode):
            calls.append((entry["pair_id"], execution_mode))
            cls._write_result(entry, output_path)
            return 0

        return fake

    def test_runner_module_and_frozen_matrix_exist(self):
        self.assertIsNotNone(importlib.util.find_spec("scripts.run_turnvega_suite"))
        self.assertTrue(MATRIX_PATH.is_file())

    def test_script_entrypoint_can_show_help_from_project_root(self):
        result = subprocess.run(
            [sys.executable, "scripts/run_turnvega_suite.py", "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--from-pair", result.stdout)

    def test_matrix_has_exact_order_and_repeats_every_frozen_field(self):
        matrix = suite.load_matrix(MATRIX_PATH)
        self.assertEqual(matrix["protocol_version"], "v1.2-dual-gpu-crossover")
        self.assertEqual([entry["pair_id"] for entry in matrix["entries"]], EXPECTED_IDS)
        self.assertEqual(len(matrix["entries"]), 18)
        for key, value in COMMON.items():
            self.assertEqual(matrix[key], value)
            for entry in matrix["entries"]:
                self.assertEqual(entry[key], value)
        self.assertEqual(matrix["entries"][16]["history_mode"], "state_gated")
        c2 = matrix["entries"][-1]
        self.assertEqual(c2["config_id"], "t3_equal_token_history_summary_control")
        self.assertEqual(c2["variant"], "t3_core_full")
        self.assertEqual(c2["history_mode"], "equal_token_summary_control")
        self.assertEqual(c2["control_of"], "t3_core_full")
        self.assertNotEqual(c2["variant"], "t3_last_turn")

    def test_load_is_deterministic_and_returns_fresh_values(self):
        first = suite.load_matrix(MATRIX_PATH)
        second = suite.load_matrix(MATRIX_PATH)
        self.assertEqual(first, second)
        first["entries"][0]["seed"] = -1
        self.assertEqual(second["entries"][0]["seed"], 20260720)

    def test_first_invalid_audit_stops_and_marks_later_not_started(self):
        calls = []

        def fake(entry, output_path, execution_mode):
            calls.append(entry["pair_id"])
            accepted = entry["pair_id"] != "T2-R2"
            self._write_result(entry, output_path, accepted)
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            result = self._run(tmp, fake, stdout=stdout)
            state = json.loads((Path(tmp) / "suite_status.json").read_text())
        self.assertEqual(result, 1)
        self.assertEqual(calls, ["T2-R1", "T2-R2"])
        self.assertIn("T2-R2", stdout.getvalue())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["entries"][1]["status"], "failed")
        self.assertTrue(all(item["status"] == "not_started" for item in state["entries"][2:]))

    def test_nonzero_and_exception_stop_immediately(self):
        for kind in ("nonzero", "exception"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                calls = []

                def fake(entry, output_path, execution_mode):
                    calls.append(entry["pair_id"])
                    if entry["pair_id"] == "T2-R2":
                        if kind == "exception":
                            raise RuntimeError("planned")
                        return 9
                    self._write_result(entry, output_path)
                    return 0

                stdout = io.StringIO()
                result = self._run(tmp, fake, stdout=stdout)
                state = json.loads((Path(tmp) / "suite_status.json").read_text())
                self.assertEqual(result, 1)
                self.assertEqual(calls, ["T2-R1", "T2-R2"])
                self.assertEqual(state["entries"][1]["status"], "failed")
                self.assertEqual(state["entries"][1]["exit_status"], 9 if kind == "nonzero" else 1)
                self.assertIn("T2-R2", stdout.getvalue())

    def test_resume_requires_matching_resolution_and_restarts_failed_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial_calls = []

            def fail_second(entry, output_path, execution_mode):
                initial_calls.append(entry["pair_id"])
                self._write_result(
                    entry, output_path, entry["pair_id"] != "T2-R2"
                )
                return 0

            self.assertEqual(self._run(root, fail_second), 1)
            matrix_path = root / "matrix.json"
            state_path = root / "suite_status.json"
            accepted_calls = []
            accepted = self._accepted_runner(accepted_calls)
            base_kwargs = dict(
                matrix_path=matrix_path,
                project_root=root,
                status_path=state_path,
                pair_runner=accepted,
                single_runner=accepted,
                execution_mode_resolver=lambda requested: "sequential_triplet",
                from_pair="T2-R2",
                stdout=io.StringIO(),
            )
            self.assertEqual(suite.run_suite(**base_kwargs), 1)
            resolution = root / "resolution.jsonl"
            for record in (
                {"pair_id": "T2-R2", "status": "open", "resolution": "x", "recorded_by": "tester", "recorded_at_utc": "now"},
                {"pair_id": "T2-R3", "status": "resolved", "resolution": "x", "recorded_by": "tester", "recorded_at_utc": "now"},
            ):
                resolution.write_text(json.dumps(record) + "\n", encoding="utf-8")
                self.assertEqual(suite.run_suite(**base_kwargs, resolution_log=resolution), 1)
            resolution.write_text(
                json.dumps({
                    "pair_id": "T2-R2",
                    "status": "resolved",
                    "resolution": "fixed runner input",
                    "recorded_by": "tester",
                    "recorded_at_utc": "2026-07-21T00:00:00Z",
                }) + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                suite, "audit_triplet", return_value={"accepted": True}
            ) as fresh_audit:
                result = suite.run_suite(**base_kwargs, resolution_log=resolution)
                self.assertEqual(result, 0, base_kwargs["stdout"].getvalue())
            fresh_audit.assert_called_once()
            state = json.loads(state_path.read_text())
            self.assertEqual(state["status"], "completed")
            self.assertEqual(
                state["matrix_sha256"],
                hashlib.sha256((root / "matrix.json").read_bytes()).hexdigest(),
            )
            self.assertRegex(state["suite_run_id"], r"^[0-9a-f]{32}$")
            self.assertEqual([item[0] for item in accepted_calls], EXPECTED_IDS[1:])
            self.assertTrue(list((root / "artifacts").rglob("*.failed-attempt-1")))

    def test_resume_rejects_nonfailed_state_unknown_id_and_skipping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []
            self.assertEqual(self._run(root, self._accepted_runner(calls)), 0)
            resolution = root / "resolution.jsonl"
            resolution.write_text(json.dumps({
                "pair_id": "T2-R1", "status": "resolved", "resolution": "x",
                "recorded_by": "tester", "recorded_at_utc": "now",
            }) + "\n", encoding="utf-8")
            kwargs = dict(
                matrix_path=root / "matrix.json", project_root=root,
                status_path=root / "suite_status.json", pair_runner=self._accepted_runner([]),
                execution_mode_resolver=lambda requested: "sequential_triplet",
                resolution_log=resolution,
                stdout=io.StringIO(),
            )
            self.assertEqual(suite.run_suite(**kwargs, from_pair="NO-SUCH-ID"), 1)
            self.assertEqual(suite.run_suite(**kwargs, from_pair="T2-R1"), 1)

    def test_matrix_rejects_bool_paths_duplicates_unknown_missing_extra_and_order(self):
        mutations = []
        base = suite.load_matrix(MATRIX_PATH)
        value = copy.deepcopy(base); value["entries"][0]["seed"] = True; mutations.append(value)
        value = copy.deepcopy(base); value["entries"][0]["output_path"] = "/tmp/escape"; mutations.append(value)
        value = copy.deepcopy(base); value["entries"][0]["output_path"] = "../escape"; mutations.append(value)
        value = copy.deepcopy(base); value["entries"][1]["pair_id"] = "T2-R1"; mutations.append(value)
        value = copy.deepcopy(base); value["entries"][1]["output_path"] = value["entries"][0]["output_path"]; mutations.append(value)
        value = copy.deepcopy(base); value["entries"][0]["variant"] = "unknown"; mutations.append(value)
        value = copy.deepcopy(base); del value["entries"][0]["seed"]; mutations.append(value)
        value = copy.deepcopy(base); value["entries"][0]["surprise"] = 1; mutations.append(value)
        value = copy.deepcopy(base); value["entries"][0], value["entries"][1] = value["entries"][1], value["entries"][0]; mutations.append(value)
        value = copy.deepcopy(base); value["entries"][0]["candidate_passages"] = 29; mutations.append(value)
        with tempfile.TemporaryDirectory() as tmp:
            for index, payload in enumerate(mutations):
                with self.subTest(index=index):
                    path = Path(tmp) / f"bad-{index}.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        suite.load_matrix(path)

    def test_rejects_symlink_bad_json_nonempty_output_and_symlink_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = root / "bad.json"
            bad.write_text("{", encoding="utf-8")
            with self.assertRaises(ValueError):
                suite.load_matrix(bad)
            link = root / "matrix-link.json"
            link.symlink_to(MATRIX_PATH)
            with self.assertRaises(ValueError):
                suite.load_matrix(link)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix = suite.load_matrix(MATRIX_PATH)
            unexpected = root / matrix["entries"][0]["output_path"]
            unexpected.mkdir(parents=True)
            (unexpected / "foreign.txt").write_text("keep", encoding="utf-8")
            calls = []
            self.assertEqual(self._run(root, self._accepted_runner(calls)), 1)
            self.assertEqual(calls, [])
            self.assertEqual((unexpected / "foreign.txt").read_text(), "keep")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.json"
            target.write_text(json.dumps({"accepted": True}), encoding="utf-8")

            def symlink_audit(entry, output_path, execution_mode):
                output_path.mkdir(parents=True, exist_ok=True)
                (output_path / "audit.json").symlink_to(target)
                return 0

            self.assertEqual(self._run(root, symlink_audit), 1)

    def test_resume_rejects_bad_or_symlinked_status_and_resolution_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix_path = self._write_matrix(root)
            status = root / "suite_status.json"
            status.write_text("{", encoding="utf-8")
            resolution = root / "resolution.jsonl"
            resolution.write_text("{}\n", encoding="utf-8")
            kwargs = dict(
                matrix_path=matrix_path,
                project_root=root,
                status_path=status,
                pair_runner=self._accepted_runner([]),
                execution_mode_resolver=lambda requested: "sequential_triplet",
                from_pair="T2-R1",
                resolution_log=resolution,
                stdout=io.StringIO(),
            )
            self.assertEqual(suite.run_suite(**kwargs), 1)
            status.unlink()
            status_target = root / "status-target.json"
            status_target.write_text("{}", encoding="utf-8")
            status.symlink_to(status_target)
            self.assertEqual(suite.run_suite(**kwargs), 1)
            status.unlink()

            def fail_first(entry, output_path, execution_mode):
                self._write_result(entry, output_path, False)
                return 0

            self.assertEqual(self._run(root, fail_first), 1)
            resolution_target = root / "resolution-target.jsonl"
            resolution_target.write_text(json.dumps({
                "pair_id": "T2-R1", "status": "resolved", "resolution": "x",
                "recorded_by": "tester", "recorded_at_utc": "now",
            }) + "\n", encoding="utf-8")
            resolution.unlink()
            resolution.symlink_to(resolution_target)
            self.assertEqual(suite.run_suite(**kwargs), 1)

    def test_status_is_atomic_has_hashes_and_fake_runner_never_starts_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []
            self.assertEqual(self._run(root, self._accepted_runner(calls)), 0)
            state_path = root / "suite_status.json"
            state = json.loads(state_path.read_text())
            self.assertEqual(state["status"], "completed")
            self.assertEqual(len(calls), 18)
            self.assertTrue(all(mode == "sequential_triplet" for _, mode in calls))
            self.assertFalse(list(root.glob(".suite_status.json.tmp-*")))
            for item in state["entries"]:
                self.assertTrue(item["started_at_utc"])
                self.assertTrue(item["completed_at_utc"])
                self.assertEqual(item["exit_status"], 0)
                audit = root / item["output_path"] / "audit.json"
                self.assertEqual(item["audit_sha256"], hashlib.sha256(audit.read_bytes()).hexdigest())

    def test_confirmatory_dispatches_once_to_single_runner_never_triplet_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            dev_calls = []
            single_calls = []
            dev = self._accepted_runner(dev_calls)
            single = self._accepted_runner(single_calls)
            self.assertEqual(self._run(tmp, dev, single_runner=single), 0)
        self.assertEqual(
            [pair_id for pair_id, _ in dev_calls],
            [pair_id for pair_id in EXPECTED_IDS if "-R" in pair_id],
        )
        self.assertEqual(
            [pair_id for pair_id, _ in single_calls],
            [pair_id for pair_id in EXPECTED_IDS if "-C" in pair_id],
        )
        self.assertEqual([pair_id for pair_id, _ in single_calls].count("T2-C0"), 1)
        self.assertEqual([pair_id for pair_id, _ in single_calls].count("T3-C2"), 1)

    def test_default_single_runner_uses_one_command_and_never_triplet_api(self):
        entry = suite.load_matrix(MATRIX_PATH)["entries"][6]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / entry["output_path"]
            completed = subprocess.CompletedProcess(args=["single"], returncode=0)
            with mock.patch.object(
                suite.subprocess, "run", return_value=completed
            ) as command, mock.patch.object(
                suite, "audit_single_config", return_value={"accepted": True}
            ) as audit, mock.patch.object(
                suite, "run_triplet_commands"
            ) as triplet:
                result = suite.run_single_config(
                    entry,
                    output,
                    "sequential_triplet",
                    project_root=root,
                )
            self.assertEqual(result, 0)
            command.assert_called_once()
            audit.assert_called_once_with(entry, output, root)
            triplet.assert_not_called()

    def test_single_config_audit_checks_manifest_config_and_trace(self):
        entry = suite.load_matrix(MATRIX_PATH)["entries"][6]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / entry["manifest_path"]
            manifest_path.parent.mkdir(parents=True)
            manifest = [{
                "source_index": 0,
                "interaction_id": "i-0",
                "query": "question",
            }]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output = root / entry["output_path"]
            output.mkdir(parents=True)
            run_id = "t2-c0-single"
            config = {
                "status": "completed",
                "experiment_family": "turnvega",
                "variant": entry["variant"],
                "dataset_kind": entry["dataset_kind"],
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "manifest_row_count": 1,
                "seed": 20260720,
                "temperature": 0,
                "candidate_budget": 30,
                "prompt_evidence": 5,
                "max_search_calls": 4,
                "max_evidence_chars": 5500,
                "history_mode": "none",
                "trace_schema": "v5",
                "trace_filename": "agent_trace_v5.jsonl",
                "run_id": run_id,
            }
            (output / "run_config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            packet = {}
            packet_sha = hashlib.sha256(
                json.dumps(packet, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            trace = {
                "trace_schema": "v5", "experiment_variant": entry["variant"],
                "run_id": run_id, "interaction_id": "i-0", "dataset_kind": "task2",
                "session_key": "", "turn_index": 0, "query": "question",
                "answer": "answer", "status": "ok", "search_call_count": 2,
                "history_mode": "none", "question_frame": {}, "image_needed": False,
                "history_needed": False, "web_needed": True,
                "entity_candidates_before": [], "entity_candidates_after": [],
                "candidate_queries": [], "query_family": "entity", "candidate_budget": 30,
                "atomic_evidence": [], "source_clusters": [], "relation_coverage": {},
                "circularity_flags": [], "answerability_scores": {}, "typed_conflicts": [],
                "evidence_packet": packet, "evidence_packet_sha256": packet_sha,
                "evidence_token_count": 0, "memory_state_before": {},
                "memory_state_after": {}, "provisional_claims": [], "verified_claims": [],
                "quarantined_claims": [], "state_version": 0,
            }
            (output / "agent_trace_v5.jsonl").write_text(
                json.dumps(trace) + "\n", encoding="utf-8"
            )
            accepted = suite.audit_single_config(entry, output, root)
            self.assertTrue(accepted["accepted"], accepted["reasons"])
            config["variant"] = "t2_budget_b0"
            (output / "run_config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            rejected = suite.audit_single_config(entry, output, root)
            self.assertFalse(rejected["accepted"])

    def test_resume_rejects_forged_completed_prefix_without_real_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fail_first(entry, output_path, execution_mode):
                self._write_result(entry, output_path, False)
                return 0

            self.assertEqual(self._run(root, fail_first), 1)
            state_path = root / "suite_status.json"
            state = json.loads(state_path.read_text())
            first_output = root / state["entries"][0]["output_path"]
            (first_output / "audit.json").write_text(
                json.dumps({"accepted": True}), encoding="utf-8"
            )
            (first_output / "triplet_summary.json").write_text(
                json.dumps({"status": "completed", "runner_valid": True}),
                encoding="utf-8",
            )
            first = state["entries"][0]
            first.update(
                status="completed",
                exit_status=0,
                completed_at_utc="2026-07-21T00:00:00Z",
                audit_sha256=hashlib.sha256(
                    (first_output / "audit.json").read_bytes()
                ).hexdigest(),
                error=None,
            )
            second = state["entries"][1]
            second.update(
                status="failed",
                started_at_utc="2026-07-21T00:00:01Z",
                completed_at_utc="2026-07-21T00:00:02Z",
                exit_status=1,
                audit_sha256=None,
                error="forged skip",
            )
            state["status"] = "failed"
            state["matrix_sha256"] = hashlib.sha256(
                (root / "matrix.json").read_bytes()
            ).hexdigest()
            state["suite_run_id"] = "forged-run"
            first["proof_sha256"] = suite._write_suite_proof(
                suite.load_matrix(root / "matrix.json")["entries"][0],
                root,
                state,
                first["audit_sha256"],
            )
            state_path.write_text(json.dumps(state), encoding="utf-8")
            resolution = root / "resolution.jsonl"
            resolution.write_text(json.dumps({
                "pair_id": "T2-R2", "status": "resolved", "resolution": "x",
                "recorded_by": "tester", "recorded_at_utc": "now",
            }) + "\n", encoding="utf-8")
            calls = []
            result = suite.run_suite(
                root / "matrix.json",
                project_root=root,
                status_path=state_path,
                pair_runner=self._accepted_runner(calls),
                single_runner=self._accepted_runner(calls),
                execution_mode_resolver=lambda requested: "sequential_triplet",
                from_pair="T2-R2",
                resolution_log=resolution,
                stdout=io.StringIO(),
            )
            self.assertEqual(result, 1)
            self.assertEqual(calls, [])

    def test_resume_rejects_suite_run_id_not_bound_to_completed_artifact_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fail_second(entry, output_path, execution_mode):
                self._write_result(
                    entry, output_path, entry["pair_id"] != "T2-R2"
                )
                return 0

            self.assertEqual(self._run(root, fail_second), 1)
            state_path = root / "suite_status.json"
            state = json.loads(state_path.read_text())
            original_run_id = state["suite_run_id"]
            state["suite_run_id"] = "forged-" + original_run_id
            state_path.write_text(json.dumps(state), encoding="utf-8")
            resolution = root / "resolution.jsonl"
            resolution.write_text(json.dumps({
                "pair_id": "T2-R2", "status": "resolved", "resolution": "x",
                "recorded_by": "tester", "recorded_at_utc": "now",
            }) + "\n", encoding="utf-8")
            calls = []
            accepted = self._accepted_runner(calls)
            result = suite.run_suite(
                root / "matrix.json",
                project_root=root,
                status_path=state_path,
                pair_runner=accepted,
                single_runner=accepted,
                execution_mode_resolver=lambda requested: "sequential_triplet",
                from_pair="T2-R2",
                resolution_log=resolution,
                stdout=io.StringIO(),
            )
            self.assertEqual(result, 1)
            self.assertEqual(calls, [])

    def test_post_runner_output_directory_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root.parent / (root.name + "-outside")
            outside.mkdir()
            try:
                (outside / "audit.json").write_text(
                    json.dumps({"accepted": True}), encoding="utf-8"
                )

                def replace_with_symlink(entry, output_path, execution_mode):
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.symlink_to(outside, target_is_directory=True)
                    return 0

                self.assertEqual(self._run(root, replace_with_symlink), 1)
                state = json.loads((root / "suite_status.json").read_text())
                self.assertEqual(state["status"], "failed")
                self.assertEqual(state["entries"][0]["status"], "failed")
            finally:
                for child in outside.iterdir():
                    child.unlink()
                outside.rmdir()

    def test_resolution_log_parses_all_lines_and_rejects_duplicate_or_tail_list(self):
        valid = {
            "pair_id": "T2-R1", "status": "resolved", "resolution": "x",
            "recorded_by": "tester", "recorded_at_utc": "now",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, lines in (
                ("tail", [valid, []]),
                ("duplicate", [valid, valid]),
                ("extra", [{**valid, "extra": 1}]),
            ):
                with self.subTest(name=name):
                    path = root / (name + ".jsonl")
                    path.write_text(
                        "".join(json.dumps(line) + "\n" for line in lines),
                        encoding="utf-8",
                    )
                    with self.assertRaises(ValueError):
                        suite._resolution_allows(path, "T2-R1")

    def test_runner_polluting_next_output_marks_that_item_and_suite_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix = suite.load_matrix(MATRIX_PATH)

            def pollute_next(entry, output_path, execution_mode):
                self._write_result(entry, output_path)
                if entry["pair_id"] == "T2-R1":
                    next_output = root / matrix["entries"][1]["output_path"]
                    next_output.mkdir(parents=True)
                    (next_output / "pollution.txt").write_text("x", encoding="utf-8")
                return 0

            self.assertEqual(self._run(root, pollute_next), 1)
            state = json.loads((root / "suite_status.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["entries"][0]["status"], "completed")
            self.assertEqual(state["entries"][1]["status"], "failed")
            self.assertTrue(all(
                item["status"] == "not_started" for item in state["entries"][2:]
            ))


if __name__ == "__main__":
    unittest.main()

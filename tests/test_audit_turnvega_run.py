import importlib.util
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import scripts.audit_turnvega_run as auditor


REQUIRED_V5_FIELDS = {
    "trace_schema",
    "experiment_variant",
    "run_id",
    "interaction_id",
    "dataset_kind",
    "session_key",
    "turn_index",
    "query",
    "answer",
    "status",
    "search_call_count",
    "history_mode",
    "question_frame",
    "image_needed",
    "history_needed",
    "web_needed",
    "entity_candidates_before",
    "entity_candidates_after",
    "candidate_queries",
    "query_family",
    "candidate_budget",
    "atomic_evidence",
    "source_clusters",
    "relation_coverage",
    "circularity_flags",
    "answerability_scores",
    "typed_conflicts",
    "evidence_packet",
    "evidence_packet_sha256",
    "evidence_token_count",
    "memory_state_before",
    "memory_state_after",
    "provisional_claims",
    "verified_claims",
    "quarantined_claims",
    "state_version",
}


class TurnVegaAuditTest(unittest.TestCase):
    def setUp(self):
        self._lock_tmp = tempfile.TemporaryDirectory()
        self._lock_patch = mock.patch.object(
            auditor,
            "CANONICAL_FAMILY_LOCK_ROOT",
            Path(self._lock_tmp.name).resolve() / "canonical-lock-root",
            create=True,
        )
        self._lock_patch.start()
        self.lock_path = auditor.CANONICAL_FAMILY_LOCK_ROOT / (
            hashlib.sha256(b"turnvega").hexdigest() + ".json"
        )
        self.lock_path.parent.mkdir(parents=True)
        self.lock_path.write_text(
            json.dumps(
                {
                    "experiment_family": "turnvega",
                    "family_id": "turnvega",
                    "execution_mode": "sequential_triplet",
                }
            ),
            encoding="utf-8",
        )
        self.lock_hash = hashlib.sha256(
            str(self.lock_path.resolve()).encode("utf-8")
        ).hexdigest()

    def tearDown(self):
        self._lock_patch.stop()
        self._lock_tmp.cleanup()

    def test_audit_module_exists(self):
        self.assertIsNotNone(
            importlib.util.find_spec("scripts.audit_turnvega_run")
        )

    def _require(self, name):
        self.assertTrue(hasattr(auditor, name), f"missing audit API: {name}")
        return getattr(auditor, name)

    def _trace(self, index, run_id, answer=None, experiment_variant="t3_last_turn"):
        packet = {}
        packet_hash = hashlib.sha256(
            json.dumps(
                packet,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        row = {
            "trace_schema": "v5",
            "experiment_variant": experiment_variant,
            "run_id": run_id,
            "interaction_id": f"i-{index}",
            "dataset_kind": "task3",
            "session_key": "session-a",
            "turn_index": index,
            "query": f"question {index}",
            "answer": answer or f"answer {index}",
            "status": "ok",
            "search_call_count": 2,
            "history_mode": "last_turn",
            "question_frame": {},
            "image_needed": True,
            "history_needed": True,
            "web_needed": True,
            "entity_candidates_before": [],
            "entity_candidates_after": [],
            "candidate_queries": [],
            "query_family": "entity",
            "candidate_budget": 12,
            "atomic_evidence": [],
            "source_clusters": [],
            "relation_coverage": {},
            "circularity_flags": [],
            "answerability_scores": {},
            "typed_conflicts": [],
            "evidence_packet": packet,
            "evidence_packet_sha256": packet_hash,
            "evidence_token_count": 0,
            "memory_state_before": {},
            "memory_state_after": {},
            "provisional_claims": [],
            "verified_claims": [],
            "quarantined_claims": [],
            "state_version": index,
        }
        self.assertTrue(REQUIRED_V5_FIELDS.issubset(row))
        return row

    def _fixture(self, root, count=3):
        manifest = [
            {
                "source_index": index,
                "session_key": "session-a",
                "turn_index": index,
                "interaction_id": f"i-{index}",
                "query": f"question {index}",
            }
            for index in range(count)
        ]
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        triplet = root / "triplet"
        triplet.mkdir()
        run_names = ("anchor_before", "variant", "anchor_after")
        runs = []
        for name in run_names:
            run_dir = triplet / name
            run_dir.mkdir()
            run_id = "run-" + name
            run_variant = (
                "t3_core_full" if name == "variant" else "t3_last_turn"
            )
            rows = [
                self._trace(index, run_id, experiment_variant=run_variant)
                for index in range(count)
            ]
            (run_dir / "agent_trace_v5.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            (run_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "run_id": run_id,
                        "experiment_family": "turnvega",
                        "execution_mode": "sequential_triplet",
                        "triplet_role": name,
                        "assigned_gpu": 0,
                        "family_lock_identity": "turnvega",
                        "family_lock_path_sha256": self.lock_hash,
                        "trace_schema": "v5",
                        "dataset_kind": "task3",
                        "variant": run_variant,
                        "manifest_sha256": manifest_sha,
                        "manifest_row_count": count,
                        "candidate_budget": 12,
                        "max_search_calls": 4,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "scores_dictionary.json").write_text(
                json.dumps({"all": {"accuracy": 0.75}, "ego": {}}),
                encoding="utf-8",
            )
            runs.append(
                {
                    "logical_run": name,
                    "path": name,
                    "gpu": 0,
                    "exit_status": 0,
                }
            )
        (triplet / "triplet_summary.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "runner_valid": True,
                    "oom_detected": False,
                    "experiment_family": "turnvega",
                    "family_lock_identity": "turnvega",
                    "family_lock_path_sha256": self.lock_hash,
                    "anchor_variant": "t3_last_turn",
                    "variant": "t3_core_full",
                    "execution_mode": "sequential_triplet",
                    "runs": runs,
                    "reasons": [],
                }
            ),
            encoding="utf-8",
        )
        return triplet, manifest_path

    def _rewrite_json(self, path, mutate):
        payload = json.loads(path.read_text(encoding="utf-8"))
        mutate(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _rewrite_trace(self, path, mutate):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        mutate(rows)
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_valid_triplet_has_exact_required_audit_fields_and_is_accepted(self):
        audit_triplet = self._require("audit_triplet")
        with tempfile.TemporaryDirectory() as tmp:
            triplet, manifest = self._fixture(Path(tmp))
            result = audit_triplet(triplet, manifest)
            stored = json.loads((triplet / "audit.json").read_text())
        required = {
            "runner_valid",
            "scope_isolation_valid",
            "schema_valid",
            "budget_valid",
            "anchor_equivalence_rate",
            "anchor_accuracy_delta",
            "anchor_drift",
            "accepted",
            "reasons",
        }
        self.assertTrue(required.issubset(result))
        self.assertEqual(stored, result)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["anchor_equivalence_rate"], 1.0)
        self.assertEqual(result["anchor_accuracy_delta"], 0.0)
        self.assertFalse(result["anchor_drift"])

    def test_strict_v5_count_hash_required_fields_and_order(self):
        audit_triplet = self._require("audit_triplet")
        mutations = (
            (
                "count",
                lambda triplet, manifest: self._rewrite_trace(
                    triplet / "variant" / "agent_trace_v5.jsonl",
                    lambda rows: rows.pop(),
                ),
            ),
            (
                "manifest hash",
                lambda triplet, manifest: self._rewrite_json(
                    triplet / "variant" / "run_config.json",
                    lambda config: config.update(manifest_sha256="0" * 64),
                ),
            ),
            (
                "required fields",
                lambda triplet, manifest: self._rewrite_trace(
                    triplet / "variant" / "agent_trace_v5.jsonl",
                    lambda rows: rows[0].pop("evidence_packet_sha256"),
                ),
            ),
            (
                "ordered identity",
                lambda triplet, manifest: self._rewrite_trace(
                    triplet / "variant" / "agent_trace_v5.jsonl",
                    lambda rows: rows.__setitem__(slice(None), [rows[1], rows[0], *rows[2:]]),
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                triplet, manifest = self._fixture(Path(tmp))
                mutate(triplet, manifest)
                result = audit_triplet(triplet, manifest)
                self.assertFalse(result["schema_valid"])
                self.assertFalse(result["accepted"])

    def test_task3_duplicate_key_ignores_query_text(self):
        audit_triplet = self._require("audit_triplet")
        with tempfile.TemporaryDirectory() as tmp:
            triplet, manifest = self._fixture(Path(tmp))

            def duplicate_identity_with_different_query(rows):
                rows[1]["session_key"] = rows[0]["session_key"]
                rows[1]["turn_index"] = rows[0]["turn_index"]
                rows[1]["interaction_id"] = rows[0]["interaction_id"]
                rows[1]["query"] = "different query must not make key unique"

            for name in ("anchor_before", "variant", "anchor_after"):
                self._rewrite_trace(
                    triplet / name / "agent_trace_v5.jsonl",
                    duplicate_identity_with_different_query,
                )
            result = audit_triplet(triplet, manifest)
        self.assertFalse(result["schema_valid"])
        self.assertTrue(any("duplicate" in reason.lower() for reason in result["reasons"]))

    def test_budget_totals_must_match(self):
        audit_triplet = self._require("audit_triplet")
        with tempfile.TemporaryDirectory() as tmp:
            triplet, manifest = self._fixture(Path(tmp))
            self._rewrite_trace(
                triplet / "variant" / "agent_trace_v5.jsonl",
                lambda rows: rows[0].update(search_call_count=3),
            )
            result = audit_triplet(triplet, manifest)
        self.assertFalse(result["budget_valid"])
        self.assertFalse(result["accepted"])

    def test_failed_run_and_oom_are_excluded_and_rejected(self):
        audit_triplet = self._require("audit_triplet")
        with tempfile.TemporaryDirectory() as tmp:
            triplet, manifest = self._fixture(Path(tmp))
            self._rewrite_json(
                triplet / "anchor_after" / "run_config.json",
                lambda config: config.update(status="failed"),
            )
            self._rewrite_json(
                triplet / "triplet_summary.json",
                lambda summary: summary.update(
                    status="failed", runner_valid=False, oom_detected=True
                ),
            )
            result = audit_triplet(triplet, manifest)
        self.assertFalse(result["runner_valid"])
        self.assertFalse(result["accepted"])
        self.assertTrue(any("OOM" in reason for reason in result["reasons"]))

    def test_scope_isolation_rejects_escape_symlink_and_mixed_mode(self):
        audit_triplet = self._require("audit_triplet")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            triplet, manifest = self._fixture(root)
            outside = root / "outside"
            outside.mkdir()
            summary_path = triplet / "triplet_summary.json"
            self._rewrite_json(
                summary_path,
                lambda summary: summary["runs"][1].update(path="../outside"),
            )
            result = audit_triplet(triplet, manifest)
            self.assertFalse(result["scope_isolation_valid"])

        with tempfile.TemporaryDirectory() as tmp:
            triplet, manifest = self._fixture(Path(tmp))
            self._rewrite_json(
                triplet / "variant" / "run_config.json",
                lambda config: config.update(execution_mode="dual_gpu_crossover"),
            )
            result = audit_triplet(triplet, manifest)
        self.assertFalse(result["scope_isolation_valid"])
        self.assertFalse(result["accepted"])

    def test_exact_sequential_inventory_rejects_missing_extra_and_wrong_family(self):
        audit_triplet = self._require("audit_triplet")
        mutations = (
            (
                "missing variant",
                lambda summary: summary["runs"].__setitem__(
                    slice(None),
                    [run for run in summary["runs"] if run["logical_run"] != "variant"],
                ),
            ),
            (
                "extra role",
                lambda summary: summary["runs"].append(
                    {
                        "logical_run": "extra",
                        "path": "anchor_before",
                        "gpu": 0,
                        "exit_status": 0,
                    }
                ),
            ),
            (
                "wrong family",
                lambda summary: summary.update(experiment_family="other"),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                triplet, manifest = self._fixture(Path(tmp))
                self._rewrite_json(triplet / "triplet_summary.json", mutate)
                result = audit_triplet(triplet, manifest)
                self.assertFalse(result["runner_valid"])
                self.assertFalse(result["scope_isolation_valid"])
                self.assertFalse(result["accepted"])

    def test_dual_inventory_requires_reciprocal_pair_summaries(self):
        audit_triplet = self._require("audit_triplet")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            triplet, manifest = self._fixture(root)
            summary_path = triplet / "triplet_summary.json"
            summary = json.loads(summary_path.read_text())
            summary["execution_mode"] = "dual_gpu_crossover"
            self._rewrite_json(
                self.lock_path,
                lambda lock: lock.update(execution_mode="dual_gpu_crossover"),
            )
            dual_specs = (
                ("anchor_before", "round_a/anchor", 0, "anchor_before"),
                ("variant_round_a", "round_a/variant", 1, "variant"),
                ("variant_round_b", "round_b/variant", 0, "variant"),
                ("anchor_after", "round_b/anchor", 1, "anchor_after"),
            )
            for logical, relative, gpu, source_name in dual_specs:
                target = triplet / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                source = triplet / source_name
                if target != source:
                    target.mkdir(exist_ok=True)
                    for filename in (
                        "run_config.json",
                        "agent_trace_v5.jsonl",
                        "scores_dictionary.json",
                    ):
                        target.joinpath(filename).write_bytes(source.joinpath(filename).read_bytes())
                    self._rewrite_json(
                        target / "run_config.json",
                        lambda config, logical=logical, gpu=gpu: config.update(
                            run_id="run-" + logical,
                            execution_mode="dual_gpu_crossover",
                            triplet_role=logical,
                            assigned_gpu=gpu,
                        ),
                    )
                    self._rewrite_trace(
                        target / "agent_trace_v5.jsonl",
                        lambda rows, logical=logical: [
                            row.update(run_id="run-" + logical) for row in rows
                        ],
                    )
            summary["runs"] = [
                {
                    "logical_run": logical,
                    "path": relative,
                    "gpu": gpu,
                    "exit_status": 0,
                }
                for logical, relative, gpu, _ in dual_specs
            ]
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            (triplet / "round_a" / "pair_summary.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "execution_mode": "dual_gpu_crossover",
                        "anchor_gpu": 0,
                        "variant_gpu": 1,
                        "anchor_exit_status": 0,
                        "variant_exit_status": 0,
                        "pair_valid": True,
                        "oom_detected": False,
                    }
                ),
                encoding="utf-8",
            )
            (triplet / "round_b" / "pair_summary.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "execution_mode": "dual_gpu_crossover",
                        "variant_gpu": 0,
                        "anchor_gpu": 1,
                        "variant_exit_status": 0,
                        "anchor_exit_status": 0,
                        "pair_valid": True,
                        "oom_detected": False,
                    }
                ),
                encoding="utf-8",
            )
            valid = audit_triplet(triplet, manifest)
            self.assertTrue(valid["accepted"])
            real_round_a = triplet / "_round_a"
            (triplet / "round_a").rename(real_round_a)
            (triplet / "round_a").symlink_to(
                real_round_a, target_is_directory=True
            )
            symlinked = audit_triplet(triplet, manifest)
            self.assertFalse(symlinked["scope_isolation_valid"])
            self.assertFalse(symlinked["accepted"])
            (triplet / "round_a").unlink()
            real_round_a.rename(triplet / "round_a")
            self._rewrite_json(
                triplet / "round_a" / "pair_summary.json",
                lambda pair: pair.update(anchor_exit_status=False),
            )
            wrong_type = audit_triplet(triplet, manifest)
            self.assertFalse(wrong_type["accepted"])
            self._rewrite_json(
                triplet / "round_a" / "pair_summary.json",
                lambda pair: pair.update(anchor_exit_status=0),
            )
            (triplet / "round_b" / "pair_summary.json").unlink()
            result = audit_triplet(triplet, manifest)
            self.assertFalse(result["runner_valid"])
            self.assertFalse(result["accepted"])
            self.assertTrue(any("pair" in reason.lower() for reason in result["reasons"]))

    def test_anchor_drift_thresholds_are_inclusive(self):
        audit_triplet = self._require("audit_triplet")
        with tempfile.TemporaryDirectory() as tmp:
            triplet, manifest = self._fixture(Path(tmp), count=50)
            self._rewrite_trace(
                triplet / "anchor_after" / "agent_trace_v5.jsonl",
                lambda rows: rows[0].update(answer="one changed answer"),
            )
            self._rewrite_json(
                triplet / "anchor_after" / "scores_dictionary.json",
                lambda scores: scores["all"].update(accuracy=0.775),
            )
            at_limit = audit_triplet(triplet, manifest)
            self.assertFalse(at_limit["anchor_drift"])
            self.assertAlmostEqual(at_limit["anchor_equivalence_rate"], 0.98)
            self.assertAlmostEqual(at_limit["anchor_accuracy_delta"], 0.025)

            self._rewrite_json(
                triplet / "anchor_after" / "scores_dictionary.json",
                lambda scores: scores["all"].update(accuracy=0.776),
            )
            over_limit = audit_triplet(triplet, manifest)
        self.assertTrue(over_limit["anchor_drift"])
        self.assertFalse(over_limit["accepted"])

    def test_reads_accuracy_from_real_evaluator_scores_dictionary_shape(self):
        audit_triplet = self._require("audit_triplet")
        with tempfile.TemporaryDirectory() as tmp:
            triplet, manifest = self._fixture(Path(tmp))
            for name, accuracy in (
                ("anchor_before", 0.8),
                ("anchor_after", 0.81),
            ):
                (triplet / name / "scores_dictionary.json").write_text(
                    json.dumps(
                        {
                            "all": {"accuracy": accuracy},
                            "ego": {"accuracy": accuracy},
                        }
                    ),
                    encoding="utf-8",
                )
            result = audit_triplet(triplet, manifest)
        self.assertIsNotNone(result["anchor_accuracy_delta"])
        self.assertAlmostEqual(result["anchor_accuracy_delta"], 0.01)
        self.assertFalse(result["anchor_drift"])
        self.assertTrue(result["accepted"])

    def test_accuracy_rejects_missing_nonfinite_bool_out_of_range_and_ego_only(self):
        audit_triplet = self._require("audit_triplet")
        cases = (
            ("missing", {}),
            ("nan", {"all": {"accuracy": float("nan")}}),
            ("inf", {"all": {"accuracy": float("inf")}}),
            ("bool", {"all": {"accuracy": True}}),
            ("negative", {"all": {"accuracy": -0.1}}),
            ("above one", {"all": {"accuracy": 1.1}}),
            ("ego only", {"ego": {"accuracy": 0.8}}),
        )
        for label, scores in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                triplet, manifest = self._fixture(Path(tmp))
                for name in ("anchor_before", "anchor_after"):
                    (triplet / name / "scores_dictionary.json").write_text(
                        json.dumps(scores), encoding="utf-8"
                    )
                result = audit_triplet(triplet, manifest)
                self.assertTrue(result["anchor_drift"])
                self.assertFalse(result["accepted"])
                self.assertTrue(any("accuracy" in reason.lower() for reason in result["reasons"]))

    def test_budget_fields_reject_bool_and_float_even_when_numerically_integral(self):
        audit_triplet = self._require("audit_triplet")
        for field, value in (
            ("search_call_count", True),
            ("search_call_count", 2.0),
            ("candidate_budget", False),
            ("candidate_budget", 12.0),
        ):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as tmp:
                triplet, manifest = self._fixture(Path(tmp))
                self._rewrite_trace(
                    triplet / "variant" / "agent_trace_v5.jsonl",
                    lambda rows, field=field, value=value: rows[0].update({field: value}),
                )
                result = audit_triplet(triplet, manifest)
                self.assertFalse(result["budget_valid"])
                self.assertFalse(result["schema_valid"])
                self.assertFalse(result["accepted"])

    def test_manifest_row_count_is_required_exact_int_not_bool(self):
        audit_triplet = self._require("audit_triplet")
        cases = (
            ("missing", lambda config: config.pop("manifest_row_count")),
            ("none", lambda config: config.update(manifest_row_count=None)),
            ("bool", lambda config: config.update(manifest_row_count=True)),
            ("wrong", lambda config: config.update(manifest_row_count=999)),
        )
        for label, mutate in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                triplet, manifest = self._fixture(Path(tmp))
                self._rewrite_json(triplet / "variant" / "run_config.json", mutate)
                result = audit_triplet(triplet, manifest)
                self.assertFalse(result["schema_valid"])
                self.assertFalse(result["accepted"])

    def test_trace_dataset_variant_and_role_semantics_must_match_config(self):
        audit_triplet = self._require("audit_triplet")
        mutations = (
            (
                "config dataset",
                lambda triplet: self._rewrite_json(
                    triplet / "variant" / "run_config.json",
                    lambda config: config.update(dataset_kind="task2"),
                ),
            ),
            (
                "trace dataset",
                lambda triplet: self._rewrite_trace(
                    triplet / "variant" / "agent_trace_v5.jsonl",
                    lambda rows: rows[0].update(dataset_kind="task2"),
                ),
            ),
            (
                "trace variant",
                lambda triplet: self._rewrite_trace(
                    triplet / "anchor_after" / "agent_trace_v5.jsonl",
                    lambda rows: rows[0].update(experiment_variant="wrong"),
                ),
            ),
            (
                "role variant",
                lambda triplet: self._rewrite_json(
                    triplet / "variant" / "run_config.json",
                    lambda config: config.update(variant="wrong"),
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                triplet, manifest = self._fixture(Path(tmp))
                mutate(triplet)
                result = audit_triplet(triplet, manifest)
                self.assertFalse(result["schema_valid"])
                self.assertFalse(result["accepted"])

    def test_family_lock_identity_and_path_hash_are_cross_checked(self):
        audit_triplet = self._require("audit_triplet")
        mutations = (
            lambda triplet: self._rewrite_json(
                triplet / "triplet_summary.json",
                lambda summary: summary.update(family_lock_identity="other"),
            ),
            lambda triplet: self._rewrite_json(
                triplet / "variant" / "run_config.json",
                lambda config: config.update(family_lock_path_sha256="0" * 64),
            ),
        )
        for mutate in mutations:
            with tempfile.TemporaryDirectory() as tmp:
                triplet, manifest = self._fixture(Path(tmp))
                mutate(triplet)
                result = audit_triplet(triplet, manifest)
                self.assertFalse(result["scope_isolation_valid"])
                self.assertFalse(result["accepted"])

    def test_uniform_counterfeit_lock_hash_is_rejected_against_canonical_path(self):
        audit_triplet = self._require("audit_triplet")
        with tempfile.TemporaryDirectory() as tmp:
            triplet, manifest = self._fixture(Path(tmp))
            counterfeit = "a" * 64
            self._rewrite_json(
                triplet / "triplet_summary.json",
                lambda summary: summary.update(
                    family_lock_path_sha256=counterfeit
                ),
            )
            for name in ("anchor_before", "variant", "anchor_after"):
                self._rewrite_json(
                    triplet / name / "run_config.json",
                    lambda config: config.update(
                        family_lock_path_sha256=counterfeit
                    ),
                )
            result = audit_triplet(triplet, manifest)
        self.assertFalse(result["scope_isolation_valid"])
        self.assertFalse(result["accepted"])
        self.assertTrue(any("canonical" in reason.lower() for reason in result["reasons"]))

    def test_evidence_packet_hash_is_canonical_and_tamper_evident(self):
        audit_triplet = self._require("audit_triplet")
        mutations = (
            lambda rows: rows[0].update(evidence_packet={"tampered": True}),
            lambda rows: rows[0].update(evidence_packet_sha256="not-a-sha"),
        )
        for mutate in mutations:
            with tempfile.TemporaryDirectory() as tmp:
                triplet, manifest = self._fixture(Path(tmp))
                self._rewrite_trace(
                    triplet / "variant" / "agent_trace_v5.jsonl", mutate
                )
                result = audit_triplet(triplet, manifest)
                self.assertFalse(result["schema_valid"])
                self.assertFalse(result["accepted"])

    def test_bool_gpu_inventory_is_not_accepted_as_integer_zero(self):
        audit_triplet = self._require("audit_triplet")
        with tempfile.TemporaryDirectory() as tmp:
            triplet, manifest = self._fixture(Path(tmp))
            self._rewrite_json(
                triplet / "triplet_summary.json",
                lambda summary: summary["runs"][0].update(gpu=False),
            )
            result = audit_triplet(triplet, manifest)
        self.assertFalse(result["scope_isolation_valid"])
        self.assertFalse(result["accepted"])

    def test_malformed_or_symlink_security_inputs_write_rejected_audit(self):
        audit_triplet = self._require("audit_triplet")
        for label in ("bad-summary", "list-summary", "symlink-manifest"):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                triplet, manifest = self._fixture(root)
                if label == "bad-summary":
                    (triplet / "triplet_summary.json").write_text("{")
                elif label == "list-summary":
                    (triplet / "triplet_summary.json").write_text("[]")
                else:
                    target = root / "manifest-target.json"
                    target.write_bytes(manifest.read_bytes())
                    manifest.unlink()
                    manifest.symlink_to(target)
                result = audit_triplet(triplet, manifest)
                self.assertFalse(result["accepted"])
                self.assertTrue((triplet / "audit.json").is_file())
                self.assertTrue(result["reasons"])

    def test_held_dirfd_survives_parent_swap_without_reading_outside(self):
        open_chain = self._require("_open_dirfd_chain")
        read_at = self._require("_read_text_at")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "triplet"
            original = root / "round_a"
            original.mkdir(parents=True)
            (original / "pair_summary.json").write_text("REAL")
            outside = Path(tmp) / "outside"
            outside.mkdir()
            (outside / "pair_summary.json").write_text("OUTSIDE")
            with open_chain(root, Path("round_a")) as directory_fd:
                original.rename(root / "_round_a")
                (root / "round_a").symlink_to(
                    outside, target_is_directory=True
                )
                self.assertEqual(
                    read_at(directory_fd, "pair_summary.json"), "REAL"
                )

    def test_dirfd_chain_context_does_not_leak_descriptors(self):
        open_chain = self._require("_open_dirfd_chain")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "triplet"
            (root / "round_a").mkdir(parents=True)
            before = len(os.listdir("/dev/fd"))
            for _ in range(200):
                with open_chain(root, Path("round_a")) as directory_fd:
                    self.assertGreaterEqual(directory_fd, 0)
            after = len(os.listdir("/dev/fd"))
        self.assertLessEqual(after, before + 1)

    def test_main_returns_one_without_deleting_artifacts(self):
        main = self._require("main")
        with tempfile.TemporaryDirectory() as tmp:
            triplet, manifest = self._fixture(Path(tmp))
            self._rewrite_json(
                triplet / "triplet_summary.json",
                lambda summary: summary.update(runner_valid=False, status="failed"),
            )
            status = main(
                ["--triplet-dir", str(triplet), "--manifest-path", str(manifest)]
            )
            self.assertTrue((triplet / "anchor_before").exists())
            self.assertTrue((triplet / "audit.json").exists())
        self.assertEqual(status, 1)


if __name__ == "__main__":
    unittest.main()

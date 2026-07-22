import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agents.turnvega_config import ExperimentBudget
import scripts.turnvega_experiment as experiment
from scripts.turnvega_experiment import (
    DATASET_IDS,
    DATASET_REVISION,
    TRACE_FILENAME,
    OFFICIAL_SESSIONS_TO_SKIP,
    build_agent_config,
    build_run_config,
    build_trace_identity_provider,
    collect_runtime_versions,
    current_git_diff_sha256,
    parse_args,
    validate_selected_rows,
    validate_manifest_rows,
)


def valid_argv(output_dir, manifest_path):
    return [
        "--dataset-kind",
        "task2",
        "--variant",
        "t2_b0",
        "--manifest-path",
        str(manifest_path),
        "--candidate-passages",
        "12",
        "--prompt-evidence",
        "5",
        "--max-search-calls",
        "4",
        "--trace-schema",
        "v5",
        "--run-id",
        "task4-unit",
        "--output-dir",
        str(output_dir),
    ]


class TurnVegaExperimentCliTest(unittest.TestCase):
    def test_accepts_required_controls_and_old_or_turnvega_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            args = parse_args(valid_argv(path, path / "manifest.json"))
            self.assertEqual(args.dataset_kind, "task2")
            self.assertEqual(args.variant, "t2_b0")
            self.assertEqual(args.candidate_passages, 12)
            self.assertEqual(args.prompt_evidence, 5)
            self.assertEqual(args.max_search_calls, 4)
            self.assertEqual(args.trace_schema, "v5")
            self.assertEqual(args.backend, "vllm")

            old_args = valid_argv(path, path / "manifest.json")
            old_args[old_args.index("t2_b0")] = "b0"
            self.assertEqual(parse_args(old_args).variant, "b0")

    def test_rejects_non_positive_or_inconsistent_budgets(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = valid_argv(tmp, Path(tmp) / "manifest.json")
            for flag, value in (
                ("--candidate-passages", "0"),
                ("--prompt-evidence", "0"),
                ("--max-search-calls", "0"),
                ("--prompt-evidence", "13"),
            ):
                argv = list(base)
                argv[argv.index(flag) + 1] = value
                with self.subTest(flag=flag, value=value):
                    with self.assertRaises(SystemExit):
                        parse_args(argv)

    def test_rejects_trace_schema_other_than_v5(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = valid_argv(tmp, Path(tmp) / "manifest.json")
            argv[argv.index("--trace-schema") + 1] = "v3"
            with self.assertRaises(SystemExit):
                parse_args(argv)

    def test_rejects_nonzero_temperature(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = valid_argv(tmp, Path(tmp) / "manifest.json")
            argv.extend(["--temperature", "0.01"])
            with self.assertRaises(SystemExit):
                parse_args(argv)


class TurnVegaManifestTest(unittest.TestCase):
    def test_preserves_order_and_rejects_duplicate_source_indices(self):
        rows = [
            {"source_index": 9, "session_id": "s9"},
            {"source_index": 2, "session_id": "s2"},
        ]
        validated = validate_manifest_rows(rows, "task2")
        self.assertEqual([row["source_index"] for row in validated], [9, 2])

        with self.assertRaisesRegex(ValueError, "source_index"):
            validate_manifest_rows(
                [rows[0], {"source_index": 9, "session_id": "other"}],
                "task2",
            )

    def test_rejects_non_list_or_empty_manifest(self):
        for rows in ({}, []):
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    validate_manifest_rows(rows, "task2")

    def test_task3_rejects_duplicate_session_turn_interaction_key(self):
        rows = [
            {
                "source_index": 1,
                "session_key": "session-a",
                "turn_index": 2,
                "interaction_id": "i-2",
            },
            {
                "source_index": 2,
                "session_id": "session-a",
                "turn_index": 2,
                "interaction_id": "i-2",
            },
        ]
        with self.assertRaisesRegex(ValueError, "Task3 identity"):
            validate_manifest_rows(rows, "task3")

    def test_task3_rejects_each_official_skipped_session(self):
        self.assertIsInstance(OFFICIAL_SESSIONS_TO_SKIP, frozenset)
        self.assertEqual(len(OFFICIAL_SESSIONS_TO_SKIP), 2)
        for index, session_id in enumerate(sorted(OFFICIAL_SESSIONS_TO_SKIP)):
            with self.subTest(session_id=session_id):
                with self.assertRaisesRegex(
                    ValueError,
                    "official skipped session",
                ):
                    validate_manifest_rows(
                        [
                            {
                                "source_index": index,
                                "session_id": session_id,
                                "turn_index": 0,
                                "interaction_id": f"skip-{index}",
                            }
                        ],
                        "task3",
                    )

    def test_selected_rows_must_match_full_manifest_order(self):
        manifest = [{"source_index": 7}, {"source_index": 3}]
        validate_selected_rows(
            [{"source_index": 7}, {"source_index": 3}],
            manifest,
        )
        with self.assertRaisesRegex(ValueError, "manifest order"):
            validate_selected_rows(
                [{"source_index": 3}, {"source_index": 7}],
                manifest,
            )
        with self.assertRaisesRegex(ValueError, "row count"):
            validate_selected_rows([{"source_index": 7}], manifest)

    def test_task3_selected_turns_must_match_each_manifest_identity(self):
        manifest = [
            {
                "source_index": 10,
                "session_id": "same-session",
                "turn_index": 0,
                "interaction_id": "turn-0",
            },
            {
                "source_index": 11,
                "session_id": "same-session",
                "turn_index": 1,
                "interaction_id": "turn-1",
            },
        ]
        selected_without_source_indices = [
            {
                "session_key": "same-session",
                "turn_index": 0,
                "interaction_id": "turn-0",
            },
            {
                "session_key": "same-session",
                "turn_index": 1,
                "interaction_id": "turn-1",
            },
        ]
        validate_selected_rows(selected_without_source_indices, manifest)
        with self.assertRaisesRegex(ValueError, "manifest order"):
            validate_selected_rows(
                list(reversed(selected_without_source_indices)),
                manifest,
            )

    def test_conversation_row_identity_uses_first_turn_and_rejects_missing(self):
        manifest = [
            {
                "source_index": 4,
                "session_id": "conversation",
                "interaction_id": "first-interaction",
            }
        ]
        validate_selected_rows(
            [
                {
                    "session_id": "conversation",
                    "turns": {
                        "interaction_id": [
                            "first-interaction",
                            "second-interaction",
                        ]
                    },
                }
            ],
            manifest,
        )
        with self.assertRaisesRegex(ValueError, "cannot extract"):
            validate_selected_rows(
                [{"session_id": "conversation", "turns": {}}],
                manifest,
            )


class TurnVegaProvenanceTest(unittest.TestCase):
    def test_missing_package_versions_are_explicit_null(self):
        with mock.patch(
            "scripts.turnvega_experiment.metadata.version",
            side_effect=PackageNotFoundError,
        ):
            versions = collect_runtime_versions()
        self.assertIsInstance(versions["python"], str)
        self.assertIsNone(versions["torch"])
        self.assertIsNone(versions["transformers"])
        self.assertIsNone(versions["vllm"])

    def test_builds_v5_agent_config_without_loading_optional_packages(self):
        args = SimpleNamespace(
            run_id="task4-config",
            dataset_kind="task3",
            trace_schema="v5",
            variant="t3_last_turn",
            candidate_passages=12,
            prompt_evidence=5,
            max_search_calls=4,
            max_evidence_chars=4096,
            history_mode="last_turn",
        )
        config = build_agent_config(args, "/tmp/agent_trace_v5.jsonl")
        self.assertEqual(config.dataset_kind, "task3")
        self.assertEqual(config.trace_schema, "v5")
        self.assertEqual(config.variant.value, "t3_last_turn")
        self.assertEqual(config.trace_path, "/tmp/agent_trace_v5.jsonl")
        self.assertEqual(config.run_id, "task4-config")

    def test_run_config_contains_core_fixed_provenance(self):
        args = SimpleNamespace(
            run_id="run-123",
            dataset_kind="task2",
            variant="t2_b0",
            manifest_path="/tmp/t2.json",
            model="org/model",
            model_weight_sha256=None,
            model_weight_lfs_oid=None,
            seed=17,
            temperature=0.0,
            max_evidence_chars=4096,
            backend="vllm",
            dataset_path=None,
            trace_schema="v5",
            history_mode="none",
        )
        budget = ExperimentBudget(12, 5, 4)
        run = build_run_config(
            args,
            budget,
            manifest_sha256="abc123",
            versions={
                "python": "3.test",
                "torch": None,
                "transformers": None,
                "vllm": None,
            },
            git_commit="deadbeef",
            git_diff_sha256_value="diffhash",
            started_at_utc="2026-07-20T00:00:00+00:00",
            completed_at_utc=None,
        )
        self.assertEqual(run["experiment_family"], "turnvega")
        self.assertEqual(run["dataset_id"], DATASET_IDS["task2"])
        self.assertEqual(run["dataset_revision"], DATASET_REVISION)
        self.assertEqual(run["manifest_sha256"], "abc123")
        self.assertEqual(run["candidate_budget"], 12)
        self.assertEqual(run["prompt_evidence"], 5)
        self.assertEqual(run["max_search_calls"], 4)
        self.assertEqual(run["python_version"], "3.test")
        self.assertIsNone(run["vllm_version"])
        self.assertIsNone(run["model_weight_sha256"])
        self.assertEqual(run["trace_filename"], TRACE_FILENAME)
        self.assertEqual(run["backend"], "vllm")
        self.assertEqual(run["model_name"], "org/model")
        self.assertEqual(run["history_mode"], "none")
        self.assertIsNone(run["completed_at_utc"])

    def test_source_state_hash_covers_head_diff_and_untracked_but_not_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)

            def git(*args):
                subprocess.run(
                    ["git", *args],
                    cwd=repo,
                    check=True,
                    capture_output=True,
                )

            git("init")
            git("config", "user.email", "task4@example.test")
            git("config", "user.name", "Task 4")
            (repo / ".gitignore").write_text("ignored.bin\n", encoding="utf-8")
            tracked = repo / "tracked.txt"
            tracked.write_text("base\n", encoding="utf-8")
            git("add", ".gitignore", "tracked.txt")
            git("commit", "-m", "base")
            clean_hash = current_git_diff_sha256(repo)

            tracked.write_text("unstaged\n", encoding="utf-8")
            self.assertNotEqual(current_git_diff_sha256(repo), clean_hash)

            tracked.write_text("staged\n", encoding="utf-8")
            git("add", "tracked.txt")
            staged_hash = current_git_diff_sha256(repo)
            self.assertNotEqual(staged_hash, clean_hash)

            git("restore", "--staged", "tracked.txt")
            tracked.write_text("base\n", encoding="utf-8")
            untracked = repo / "untracked.txt"
            untracked.write_text("first\n", encoding="utf-8")
            untracked_hash = current_git_diff_sha256(repo)
            self.assertNotEqual(untracked_hash, clean_hash)
            untracked.write_text("second\n", encoding="utf-8")
            self.assertNotEqual(
                current_git_diff_sha256(repo),
                untracked_hash,
            )

            untracked.unlink()
            ignored = repo / "ignored.bin"
            ignored.write_bytes(os.urandom(32))
            self.assertEqual(current_git_diff_sha256(repo), clean_hash)

    def test_source_state_hash_has_no_one_file_two_file_record_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)

            def git(*args):
                subprocess.run(
                    ["git", *args], cwd=repo, check=True, capture_output=True
                )

            git("init")
            git("config", "user.email", "task4@example.test")
            git("config", "user.name", "Task 4")
            (repo / "base").write_text("base", encoding="utf-8")
            git("add", "base")
            git("commit", "-m", "base")

            first = repo / "a"
            second = repo / "b"
            first.write_bytes(b"first")
            mode = f"{first.lstat().st_mode:o}".encode("ascii")
            first.write_bytes(
                b"first\0PATH\0b\0" + mode + b"\0FILE\0second"
            )
            one_file_hash = current_git_diff_sha256(repo)

            first.write_bytes(b"first")
            second.write_bytes(b"second")
            two_file_hash = current_git_diff_sha256(repo)
            self.assertNotEqual(one_file_hash, two_file_hash)


class TurnVegaTraceIdentityTest(unittest.TestCase):
    def test_provider_matches_official_deque_iterator_for_batch_size_two(self):
        rows = [
            {
                "session_id": "a",
                "turns": [
                    {"query": "a0", "interaction_id": "ai0"},
                    {"query": "a1", "interaction_id": "ai1"},
                    {"query": "a2", "interaction_id": "ai2"},
                ],
            },
            {
                "session_id": "b",
                "turns": [
                    {"query": "b0", "interaction_id": "bi0"},
                    {"query": "b1", "interaction_id": "bi1"},
                ],
            },
        ]
        provider = build_trace_identity_provider(rows, batch_size=2)
        observed = [
            provider.take("a0", []),
            provider.take("b0", []),
            provider.take("b1", [{}]),
            provider.take("a1", [{}]),
            provider.take("a2", [{}, {}]),
        ]
        self.assertEqual(
            [(row["session_key"], row["turn_index"]) for row in observed],
            [("a", 0), ("b", 0), ("b", 1), ("a", 1), ("a", 2)],
        )
        self.assertEqual([row["query"] for row in observed], ["a0", "b0", "b1", "a1", "a2"])

    def test_provider_keeps_each_session_contiguous_for_batch_size_one(self):
        rows = [
            {
                "session_id": "a",
                "turns": [
                    {"query": "a0"},
                    {"query": "a1"},
                ],
            },
            {"session_id": "b", "turns": [{"query": "b0"}]},
        ]
        provider = build_trace_identity_provider(rows, batch_size=1)
        observed = [
            provider.take("a0", []),
            provider.take("a1", [{}]),
            provider.take("b0", []),
        ]
        self.assertEqual(
            [(row["session_key"], row["turn_index"]) for row in observed],
            [("a", 0), ("a", 1), ("b", 0)],
        )

    def test_provider_rejects_query_or_history_misordering(self):
        provider = build_trace_identity_provider(
            [{"session_id": "a", "turns": [{"query": "q0"}, {"query": "q1"}]}],
            batch_size=1,
        )
        with self.assertRaisesRegex(ValueError, "query"):
            provider.take("wrong", [])
        provider.take("q0", [])
        with self.assertRaisesRegex(ValueError, "history"):
            provider.take("q1", [])

    def test_scalar_turn_mapping_produces_one_identity(self):
        provider = build_trace_identity_provider(
            [
                {
                    "session_id": "scalar",
                    "turns": {
                        "query": "only query",
                        "interaction_id": "only-id",
                    },
                }
            ],
            batch_size=4,
        )
        identity = provider.take("only query", [])
        self.assertEqual(identity["interaction_id"], "only-id")
        with self.assertRaises(RuntimeError):
            provider.take("only query", [])


class TurnVegaRunLifecycleTest(unittest.TestCase):
    def _manifest_and_args(self, root, *, dry_run=False, variant="t2_b0", with_weight=True):
        root = Path(root)
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                [
                    {
                        "source_index": 0,
                        "session_id": "s0",
                        "interaction_id": "i0",
                    }
                ]
            ),
            encoding="utf-8",
        )
        argv = valid_argv(root / "output", manifest)
        argv[argv.index("t2_b0")] = variant
        argv.extend(["--model", "org/remote-model"])
        if with_weight:
            argv.extend(["--model-weight-sha256", "a" * 64])
        if dry_run:
            argv.append("--dry-run")
        return parse_args(argv)

    def test_existing_products_raise_without_modification(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._manifest_and_args(tmp, dry_run=True)
            output = Path(args.output_dir)
            output.mkdir()
            run_path = output / "run_config.json"
            trace_path = output / TRACE_FILENAME
            run_path.write_bytes(b"old-run")
            trace_path.write_bytes(b"old-trace")
            with self.assertRaises(FileExistsError):
                experiment.run_experiment(args)
            self.assertEqual(run_path.read_bytes(), b"old-run")
            self.assertEqual(trace_path.read_bytes(), b"old-trace")

    def test_official_skipped_session_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._manifest_and_args(tmp, dry_run=True)
            args.dataset_kind = "task3"
            Path(args.manifest_path).write_text(
                json.dumps(
                    [
                        {
                            "source_index": 0,
                            "session_id": next(iter(OFFICIAL_SESSIONS_TO_SKIP)),
                            "turn_index": 0,
                            "interaction_id": "skipped",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "official skipped session",
            ):
                experiment.run_experiment(args)
            self.assertFalse(Path(args.output_dir).exists())

    def test_any_existing_output_entry_or_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._manifest_and_args(tmp, dry_run=True)
            output = Path(args.output_dir)
            output.mkdir()
            unknown = output / "scores_dictionary.json"
            unknown.write_bytes(b"old-score")
            with self.assertRaises(FileExistsError):
                experiment.run_experiment(args)
            self.assertEqual(unknown.read_bytes(), b"old-score")

        with tempfile.TemporaryDirectory() as tmp:
            args = self._manifest_and_args(tmp, dry_run=True)
            output = Path(args.output_dir)
            output.mkdir()
            target = Path(tmp) / "victim"
            target.write_bytes(b"victim")
            symlink = output / "unknown-link"
            symlink.symlink_to(target)
            with self.assertRaises(FileExistsError):
                experiment.run_experiment(args)
            self.assertEqual(target.read_bytes(), b"victim")
            self.assertTrue(symlink.is_symlink())

    def test_invalid_remote_model_identity_preserves_old_products(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._manifest_and_args(tmp, with_weight=False)
            output = Path(args.output_dir)
            output.mkdir()
            run_path = output / "run_config.json"
            trace_path = output / TRACE_FILENAME
            run_path.write_bytes(b"old-run")
            trace_path.write_bytes(b"old-trace")
            with self.assertRaisesRegex(ValueError, "local model"):
                experiment.run_experiment(args)
            self.assertEqual(run_path.read_bytes(), b"old-run")
            self.assertEqual(trace_path.read_bytes(), b"old-trace")

    def test_atomic_run_config_replace_and_fsync(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with mock.patch(
                "scripts.turnvega_experiment.os.replace",
                wraps=os.replace,
            ) as replace, mock.patch(
                "scripts.turnvega_experiment.os.fsync",
                wraps=os.fsync,
            ) as fsync:
                experiment._write_run_config(output, {"status": "prepared"})
            self.assertTrue(replace.called)
            self.assertGreaterEqual(fsync.call_count, 2)
            self.assertEqual(
                json.loads((output / "run_config.json").read_text()),
                {"status": "prepared"},
            )

    def test_run_failure_atomically_records_failed_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._manifest_and_args(tmp, with_weight=False)
            local_model = Path(tmp) / "local-model"
            local_model.mkdir()
            (local_model / "weights.bin").write_bytes(b"real weights")
            args.model = str(local_model)

            class FakeDataset(list):
                def select(self, indices):
                    return FakeDataset([self[index] for index in indices])

            dataset_module = types.ModuleType("datasets")
            dataset_module.load_dataset = lambda *a, **k: FakeDataset(
                [
                    {
                        "session_id": "s0",
                        "turns": {
                            "query": "q0",
                            "interaction_id": "i0",
                        },
                    }
                ]
            )

            class FailingEvaluator:
                def __init__(self, **kwargs):
                    pass

                def evaluate_agent(self):
                    raise RuntimeError("planned evaluator failure")

            evaluation_module = types.ModuleType("local_evaluation")
            evaluation_module.CRAGEvaluator = FailingEvaluator
            with mock.patch.dict(
                sys.modules,
                {
                    "datasets": dataset_module,
                    "local_evaluation": evaluation_module,
                },
            ), mock.patch.object(
                experiment, "build_search_pipeline", return_value=None
            ), mock.patch.object(
                experiment, "create_backend", return_value=object()
            ):
                with self.assertRaisesRegex(RuntimeError, "planned"):
                    experiment.run_experiment(args)

            run = json.loads(
                (Path(args.output_dir) / "run_config.json").read_text()
            )
            self.assertEqual(run["status"], "failed")
            self.assertEqual(run["error_type"], "RuntimeError")
            self.assertIn("planned evaluator failure", run["error_message"])
            self.assertTrue(run["completed_at_utc"])

    def test_unsupported_variant_is_not_reported_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._manifest_and_args(
                tmp,
                dry_run=True,
                variant="t2_core_full",
            )
            run = experiment.run_experiment(args)
            self.assertEqual(run["status"], "unsupported")
            self.assertNotEqual(run["status"], "completed")

    def test_task3_legacy_variants_are_unsupported(self):
        for variant in ("b0", "a1"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                args = self._manifest_and_args(
                    tmp,
                    dry_run=True,
                    variant=variant,
                )
                args.dataset_kind = "task3"
                manifest_path = Path(args.manifest_path)
                manifest_path.write_text(
                    json.dumps(
                        [
                            {
                                "source_index": 0,
                                "session_id": "s0",
                                "turn_index": 0,
                                "interaction_id": "i0",
                            }
                        ]
                    ),
                    encoding="utf-8",
                )
                run = experiment.run_experiment(args)
                self.assertEqual(run["status"], "unsupported")

    def test_non_dry_remote_model_is_rejected_even_with_declared_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._manifest_and_args(tmp, with_weight=True)
            with self.assertRaisesRegex(ValueError, "local model"):
                experiment.run_experiment(args)
            self.assertFalse(Path(args.output_dir).exists())

    def test_local_model_rejects_incorrect_declared_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._manifest_and_args(tmp, with_weight=True)
            model = Path(tmp) / "model"
            model.mkdir()
            (model / "weights.bin").write_bytes(b"actual")
            args.model = str(model)
            with self.assertRaisesRegex(ValueError, "does not match"):
                experiment.validate_input_identity(args)

    def test_dry_run_remote_model_is_never_marked_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._manifest_and_args(tmp, dry_run=True)
            run = experiment.run_experiment(args)
            self.assertIn(run["status"], {"unsupported", "prepared"})
            self.assertNotIn("completed", run["status"])

    def test_local_inputs_are_hashed_and_remote_weights_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            (model / "weights.bin").write_bytes(b"weights")
            dataset = root / "data.parquet"
            dataset.write_bytes(b"dataset")
            args = self._manifest_and_args(tmp, dry_run=True, with_weight=False)
            args.model = str(model)
            args.dataset_path = str(dataset)
            identity = experiment.validate_input_identity(args)
            self.assertEqual(len(identity["model_weight_sha256"]), 64)
            self.assertEqual(len(identity["dataset_path_sha256"]), 64)

            args.model = "org/remote"
            args.dry_run = False
            with self.assertRaisesRegex(ValueError, "local model"):
                experiment.validate_input_identity(args)

    def test_task2_selection_injects_manifest_source_indices(self):
        class FakeDataset(list):
            def select(self, indices):
                return FakeDataset([self[index] for index in indices])

        dataset = FakeDataset(
            [
                {"session_id": "zero", "interaction_id": "i0"},
                {"session_id": "one", "interaction_id": "i1"},
            ]
        )
        manifest = [
            {"source_index": 1, "session_id": "one", "interaction_id": "i1"},
            {"source_index": 0, "session_id": "zero", "interaction_id": "i0"},
        ]
        selected = experiment.select_manifest_rows(dataset, manifest)
        self.assertEqual([row["source_index"] for row in selected], [1, 0])


if __name__ == "__main__":
    unittest.main()

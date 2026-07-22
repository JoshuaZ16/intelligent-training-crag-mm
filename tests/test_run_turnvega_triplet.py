import importlib.util
import inspect
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import scripts.run_turnvega_triplet as runner


class TurnVegaTripletRunnerTest(unittest.TestCase):
    def setUp(self):
        self._lock_tmp = tempfile.TemporaryDirectory()
        self._lock_patch = mock.patch.object(
            runner,
            "CANONICAL_FAMILY_LOCK_ROOT",
            Path(self._lock_tmp.name).resolve() / "canonical-lock-root",
            create=True,
        )
        self._lock_patch.start()

    def tearDown(self):
        self._lock_patch.stop()
        self._lock_tmp.cleanup()

    def test_runner_module_exists(self):
        self.assertIsNotNone(
            importlib.util.find_spec("scripts.run_turnvega_triplet")
        )

    def _require(self, name):
        self.assertTrue(hasattr(runner, name), f"missing runner API: {name}")
        return getattr(runner, name)

    def _config_command(self, body="", *, status="completed", family="turnvega"):
        code = (
            "import json,os,pathlib,sys,time;"
            "p=pathlib.Path(sys.argv[1]);"
            "(p/'run_config.json').write_text(json.dumps({"
            f"'status':{status!r},'experiment_family':{family!r},"
            "'variant':'test-variant'"
            "}));"
            + body
        )
        return [sys.executable, "-c", code, "{output_dir}"]

    def _assert_pid_dead(self, pid):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.02)
        self.fail(f"process {pid} is still alive")

    def test_auto_is_smoke_only_and_freezes_deterministic_fallback(self):
        resolve = self._require("resolve_execution_mode")
        self.assertEqual(
            resolve("auto", pre_formal_smoke=True, dual_gpu_smoke_ok=True),
            "dual_gpu_crossover",
        )
        self.assertEqual(
            resolve("auto", pre_formal_smoke=True, dual_gpu_smoke_ok=False),
            "sequential_triplet",
        )
        with self.assertRaisesRegex(ValueError, "pre-formal smoke"):
            resolve("auto", pre_formal_smoke=False, dual_gpu_smoke_ok=True)

    def test_dual_gpu_smoke_runs_both_sides_and_any_failure_falls_back(self):
        smoke = self._require("run_dual_gpu_smoke")
        probe = lambda: {
            "ram_available_bytes": 8 * 1024**3,
            "swap_thrashing": False,
            "gpu_memory_mib": {0: 0, 1: 0},
        }
        ok = smoke(
            [
                sys.executable,
                "-c",
                "import os;assert os.environ['CUDA_VISIBLE_DEVICES'] in {'0','1'}",
            ],
            resource_probe=probe,
        )
        failed = smoke(
            [sys.executable, "-c", "raise SystemExit(2)"],
            resource_probe=probe,
        )
        self.assertTrue(ok)
        self.assertFalse(failed)

    def test_dual_gpu_rounds_are_reciprocal_and_each_round_is_concurrent(self):
        run_triplet = self._require("run_triplet_commands")
        body = (
            "ready=p/'ready';ready.write_text('1');"
            "peer=p.parent/('variant' if p.name=='anchor' else 'anchor')/'ready';"
            "deadline=time.time()+2;"
            "exec(\"while not peer.exists() and time.time()<deadline: time.sleep(0.01)\");"
            "assert peer.exists();"
            "print(json.dumps({'gpu':os.environ['CUDA_VISIBLE_DEVICES'],"
            "'started':time.time()}))"
        )
        commands = {
            name: self._config_command(body)
            for name in ("anchor_before", "variant", "anchor_after")
        }
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_triplet(
                commands,
                Path(tmp) / "triplet",
                execution_mode="dual_gpu_crossover",
                resource_probe=lambda: {
                    "ram_available_bytes": 8 * 1024**3,
                    "swap_thrashing": False,
                    "gpu_memory_mib": {0: 0, 1: 0},
                },
            )
            root = Path(tmp) / "triplet"
            a_anchor = json.loads(
                (root / "round_a" / "anchor" / "stdout_stderr.log")
                .read_text()
                .strip()
            )
            a_variant = json.loads(
                (root / "round_a" / "variant" / "stdout_stderr.log")
                .read_text()
                .strip()
            )
            b_variant = json.loads(
                (root / "round_b" / "variant" / "stdout_stderr.log")
                .read_text()
                .strip()
            )
            b_anchor = json.loads(
                (root / "round_b" / "anchor" / "stdout_stderr.log")
                .read_text()
                .strip()
            )
        self.assertEqual((a_anchor["gpu"], a_variant["gpu"]), ("0", "1"))
        self.assertEqual((b_variant["gpu"], b_anchor["gpu"]), ("0", "1"))
        self.assertTrue(summary["runner_valid"])

    def test_mlx_sequential_does_not_require_nvidia_telemetry(self):
        run_triplet = self._require("run_triplet_commands")
        self.assertIn("require_gpu_telemetry", inspect.signature(run_triplet).parameters)
        commands = {
            name: self._config_command()
            for name in ("anchor_before", "variant", "anchor_after")
        }
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_triplet(
                commands,
                Path(tmp) / "triplet",
                execution_mode="sequential_triplet",
                require_gpu_telemetry=False,
                resource_probe=lambda: {
                    "ram_available_bytes": 8 * 1024**3,
                    "swap_thrashing": False,
                    "gpu_memory_mib": {},
                },
            )
        self.assertTrue(summary["runner_valid"])

    def test_macos_vm_stat_fallback_parses_available_ram_and_swap_counters(self):
        parse = self._require("_parse_macos_vm_stat")
        snapshot = parse(
            "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n"
            "Pages free: 1000.\n"
            "Pages inactive: 2000.\n"
            "Pages speculative: 100.\n"
            "Pageins: 500.\n"
            "Pageouts: 600.\n"
        )
        self.assertEqual(snapshot["ram_available_bytes"], 3100 * 4096)
        self.assertEqual(snapshot["swap_pages"], (500, 600))

    def test_canonical_lock_rejects_symlink_in_root_chain(self):
        run_triplet = self._require("run_triplet_commands")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            real = base / "real"
            real.mkdir()
            link = base / "linked"
            link.symlink_to(real, target_is_directory=True)
            runner.CANONICAL_FAMILY_LOCK_ROOT = link / "locks"
            commands = {
                name: self._config_command()
                for name in ("anchor_before", "variant", "anchor_after")
            }
            with self.assertRaisesRegex(FileExistsError, "symlink"):
                run_triplet(
                    commands,
                    base / "triplet",
                    execution_mode="sequential_triplet",
                    resource_probe=lambda: {
                        "ram_available_bytes": 8 * 1024**3,
                        "swap_thrashing": False,
                        "gpu_memory_mib": {0: 0},
                    },
                )

    def test_incremental_oom_scanner_handles_large_log_and_cross_chunk_marker(self):
        scan = self._require("_scan_log_incremental")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "large.log"
            path.write_bytes(b"x" * (2 * 1024 * 1024) + b"CUDA out of mem")
            state = {"log_offset": 0, "log_tail": b""}
            self.assertFalse(scan(path, state))
            first_offset = state["log_offset"]
            with path.open("ab") as handle:
                handle.write(b"ory\n")
            self.assertTrue(scan(path, state))
            self.assertGreater(state["log_offset"], first_offset)
            self.assertLessEqual(len(state["log_tail"]), 128)

    def test_sequential_fallback_order_is_anchor_variant_anchor(self):
        run_triplet = self._require("run_triplet_commands")
        commands = {
            name: self._config_command("print(os.environ['CUDA_VISIBLE_DEVICES'])")
            for name in ("anchor_before", "variant", "anchor_after")
        }
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_triplet(
                commands,
                Path(tmp) / "triplet",
                execution_mode="sequential_triplet",
                resource_probe=lambda: {
                    "ram_available_bytes": 8 * 1024**3,
                    "swap_thrashing": False,
                    "gpu_memory_mib": {0: 0},
                },
            )
        self.assertEqual(
            [item["logical_run"] for item in summary["runs"]],
            ["anchor_before", "variant", "anchor_after"],
        )
        self.assertEqual([item["gpu"] for item in summary["runs"]], [0, 0, 0])

    def test_nonzero_side_invalidates_pair_and_preserves_both_artifacts(self):
        run_triplet = self._require("run_triplet_commands")
        commands = {
            "anchor_before": self._config_command("raise SystemExit(7)"),
            "variant": self._config_command("print('variant artifact')"),
            "anchor_after": self._config_command("print('unused')"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "triplet"
            summary = run_triplet(
                commands,
                root,
                execution_mode="dual_gpu_crossover",
                resource_probe=lambda: {
                    "ram_available_bytes": 8 * 1024**3,
                    "swap_thrashing": False,
                    "gpu_memory_mib": {0: 0, 1: 0},
                },
            )
            pair = json.loads((root / "round_a" / "pair_summary.json").read_text())
            self.assertFalse(pair["pair_valid"])
            self.assertFalse(summary["runner_valid"])
            self.assertEqual(pair["anchor_exit_status"], 7)
            self.assertTrue((root / "round_a" / "anchor" / "exit_status.txt").exists())
            self.assertTrue((root / "round_a" / "variant" / "exit_status.txt").exists())
            self.assertEqual(json.loads((root / "triplet_summary.json").read_text())["status"], "failed")

    def test_low_ram_swap_or_oom_stops_and_preserves_failed_summary(self):
        run_triplet = self._require("run_triplet_commands")
        ok_commands = {
            name: self._config_command("print('artifact')")
            for name in ("anchor_before", "variant", "anchor_after")
        }
        bad_snapshots = (
            {"ram_available_bytes": 5 * 1024**3, "swap_thrashing": False, "gpu_memory_mib": {0: 0}},
            {"ram_available_bytes": 8 * 1024**3, "swap_thrashing": True, "gpu_memory_mib": {0: 0}},
        )
        for snapshot in bad_snapshots:
            with self.subTest(snapshot=snapshot), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "triplet"
                summary = run_triplet(
                    ok_commands,
                    root,
                    execution_mode="sequential_triplet",
                    resource_probe=lambda snapshot=snapshot: snapshot,
                )
                self.assertFalse(summary["runner_valid"])
                self.assertEqual(summary["status"], "failed")
                self.assertTrue((root / "triplet_summary.json").exists())

        oom_commands = dict(ok_commands)
        oom_commands["anchor_before"] = self._config_command(
            "print('CUDA out of memory',flush=True);raise SystemExit(1)"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "triplet"
            summary = run_triplet(
                oom_commands,
                root,
                execution_mode="sequential_triplet",
                resource_probe=lambda: {
                    "ram_available_bytes": 8 * 1024**3,
                    "swap_thrashing": False,
                    "gpu_memory_mib": {0: 0},
                },
            )
            self.assertTrue((root / "anchor_before" / "stdout_stderr.log").exists())
        self.assertTrue(summary["oom_detected"])
        self.assertFalse(summary["runner_valid"])

    def test_gpu_memory_must_clear_below_500_mib_between_rounds(self):
        wait_for_clear = self._require("_wait_for_gpu_clear")
        reason = wait_for_clear(
            lambda: {
                "ram_available_bytes": 8 * 1024**3,
                "swap_thrashing": False,
                "gpu_memory_mib": {0: 700},
            },
            (0,),
            0,
        )
        self.assertIn("500 MiB", reason)

    def test_missing_gpu_telemetry_fails_closed(self):
        wait_for_clear = self._require("_wait_for_gpu_clear")
        reason = wait_for_clear(
            lambda: {
                "ram_available_bytes": 8 * 1024**3,
                "swap_thrashing": False,
                "gpu_memory_mib": {},
            },
            (0,),
            0,
        )
        self.assertIn("telemetry", reason)

    def test_swap_thrashing_uses_bidirectional_page_rate(self):
        detect = self._require("detect_swap_thrashing")
        self.assertFalse(detect((100, 200, 10.0), (110, 220, 11.0)))
        self.assertTrue(detect((100, 200, 10.0), (1500, 1700, 11.0)))
        self.assertFalse(detect((100, 200, 10.0), (2500, 200, 11.0)))

    def test_output_protection_and_family_mode_lock(self):
        run_triplet = self._require("run_triplet_commands")
        commands = {
            name: self._config_command("print('ok')")
            for name in ("anchor_before", "variant", "anchor_after")
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nonempty = root / "nonempty"
            nonempty.mkdir()
            (nonempty / "owned.txt").write_text("keep")
            with self.assertRaisesRegex(FileExistsError, "empty"):
                run_triplet(commands, nonempty, execution_mode="sequential_triplet")
            target = root / "target"
            target.mkdir()
            link = root / "link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(FileExistsError, "symlink"):
                run_triplet(commands, link, execution_mode="sequential_triplet")

            family_root = runner.CANONICAL_FAMILY_LOCK_ROOT
            runner._freeze_family_mode("dual_gpu_crossover")
            with self.assertRaisesRegex(ValueError, "mix"):
                run_triplet(
                    commands,
                    root / "new-triplet",
                    execution_mode="sequential_triplet",
                    family_root=family_root,
                )

    def test_family_mode_lock_is_enforced_by_default_across_sibling_runs(self):
        run_triplet = self._require("run_triplet_commands")
        commands = {
            name: self._config_command("print('ok')")
            for name in ("anchor_before", "variant", "anchor_after")
        }
        probe = lambda: {
            "ram_available_bytes": 8 * 1024**3,
            "swap_thrashing": False,
            "gpu_memory_mib": {0: 0, 1: 0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = run_triplet(
                commands,
                root / "first",
                execution_mode="sequential_triplet",
                resource_probe=probe,
            )
            self.assertTrue(first["runner_valid"])
            with self.assertRaisesRegex(ValueError, "mix"):
                run_triplet(
                    commands,
                    root / "second",
                    execution_mode="dual_gpu_crossover",
                    resource_probe=probe,
                )

    def test_canonical_family_lock_path_cannot_escape_or_be_redirected(self):
        canonical = self._require("canonical_family_lock_path")
        try:
            lock = canonical()
        except TypeError as exc:
            self.fail("canonical lock still accepts caller path/id: " + str(exc))
        self.assertEqual(lock.parent, runner.CANONICAL_FAMILY_LOCK_ROOT.resolve())
        self.assertEqual(lock.suffix, ".json")
        self.assertNotIn("..", lock.parts)

    def test_family_root_or_id_cannot_redirect_canonical_lock(self):
        run_triplet = self._require("run_triplet_commands")
        commands = {
            name: self._config_command()
            for name in ("anchor_before", "variant", "anchor_after")
        }
        probe = lambda: {
            "ram_available_bytes": 8 * 1024**3,
            "swap_thrashing": False,
            "gpu_memory_mib": {0: 0, 1: 0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for kwargs in (
                {"family_root": root / "redirect"},
                {"family_id": "other-family"},
            ):
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                    ValueError, "canonical"
                ):
                    run_triplet(
                        commands,
                        root / ("run-" + str(len(kwargs))),
                        execution_mode="sequential_triplet",
                        resource_probe=probe,
                        **kwargs,
                    )

    def test_running_oom_kills_peer_without_waiting_for_natural_exit(self):
        run_triplet = self._require("run_triplet_commands")
        commands = {
            "anchor_before": self._config_command(
                "print('CUDA out of memory',flush=True);time.sleep(5)"
            ),
            "variant": self._config_command("time.sleep(5)"),
            "anchor_after": self._config_command(),
        }
        probe = lambda: {
            "ram_available_bytes": 8 * 1024**3,
            "swap_thrashing": False,
            "gpu_memory_mib": {0: 0, 1: 0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "triplet"
            started = time.monotonic()
            summary = run_triplet(
                commands,
                root,
                execution_mode="dual_gpu_crossover",
                resource_probe=probe,
            )
            elapsed = time.monotonic() - started
            peer_status = int(
                (root / "round_a" / "variant" / "exit_status.txt").read_text()
            )
        self.assertLess(elapsed, 2.0)
        self.assertTrue(summary["oom_detected"])
        self.assertNotEqual(peer_status, 0)

    def test_oom_kills_forked_grandchild_process_group(self):
        run_triplet = self._require("run_triplet_commands")
        malicious = self._config_command(
            "(p/'parent.pid').write_text(str(os.getpid()));"
            "child=os.fork();"
            "(time.sleep(10) if child==0 else None);"
            "((p/'grandchild.pid').write_text(str(child)) if child else None);"
            "(print('CUDA out of memory',flush=True) if child else None);"
            "(time.sleep(10) if child else None)"
        )
        commands = {
            "anchor_before": malicious,
            "variant": self._config_command("time.sleep(10)"),
            "anchor_after": self._config_command(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "triplet"
            summary = run_triplet(
                commands,
                root,
                execution_mode="dual_gpu_crossover",
                resource_probe=lambda: {
                    "ram_available_bytes": 8 * 1024**3,
                    "swap_thrashing": False,
                    "gpu_memory_mib": {0: 0, 1: 0},
                },
            )
            run_dir = root / "round_a" / "anchor"
            parent_pid = int((run_dir / "parent.pid").read_text())
            grandchild_pid = int((run_dir / "grandchild.pid").read_text())
            self._assert_pid_dead(parent_pid)
            self._assert_pid_dead(grandchild_pid)
        self.assertFalse(summary["runner_valid"])

    def test_sequential_probe_exception_reaps_child_and_grandchild(self):
        run_triplet = self._require("run_triplet_commands")
        forking = self._config_command(
            "(p/'parent.pid').write_text(str(os.getpid()));"
            "child=os.fork();"
            "(time.sleep(10) if child==0 else None);"
            "((p/'grandchild.pid').write_text(str(child)) if child else None);"
            "(time.sleep(10) if child else None)"
        )
        calls = 0

        def probe():
            nonlocal calls
            calls += 1
            if calls > 1:
                time.sleep(0.2)
                raise RuntimeError("probe exploded")
            return {
                "ram_available_bytes": 8 * 1024**3,
                "swap_thrashing": False,
                "gpu_memory_mib": {0: 0},
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "triplet"
            summary = run_triplet(
                {
                    "anchor_before": forking,
                    "variant": self._config_command(),
                    "anchor_after": self._config_command(),
                },
                root,
                execution_mode="sequential_triplet",
                resource_probe=probe,
            )
            run_dir = root / "anchor_before"
            parent_pid = int((run_dir / "parent.pid").read_text())
            grandchild_pid = int((run_dir / "grandchild.pid").read_text())
            self._assert_pid_dead(parent_pid)
            self._assert_pid_dead(grandchild_pid)
        self.assertEqual(summary["status"], "failed")
        self.assertTrue(
            any("probe exploded" in reason for reason in summary["reasons"]),
            summary,
        )

    def test_running_low_ram_kills_both_sides_and_preserves_logs(self):
        run_triplet = self._require("run_triplet_commands")
        commands = {
            name: self._config_command("time.sleep(5)")
            for name in ("anchor_before", "variant", "anchor_after")
        }
        snapshots = iter(
            [
                {"ram_available_bytes": 8 * 1024**3, "swap_thrashing": False, "gpu_memory_mib": {0: 0, 1: 0}},
                {"ram_available_bytes": 5 * 1024**3, "swap_thrashing": False, "gpu_memory_mib": {0: 0, 1: 0}},
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "triplet"
            started = time.monotonic()
            summary = run_triplet(
                commands,
                root,
                execution_mode="dual_gpu_crossover",
                resource_probe=lambda: next(snapshots),
            )
            elapsed = time.monotonic() - started
            artifacts_exist = all(
                (root / "round_a" / side / "stdout_stderr.log").exists()
                for side in ("anchor", "variant")
            )
        self.assertLess(elapsed, 2.0)
        self.assertFalse(summary["runner_valid"])
        self.assertTrue(artifacts_exist)
        self.assertTrue(any("RAM" in reason for reason in summary["reasons"]))

    def test_missing_invalid_or_prepared_run_config_fails_runner(self):
        run_triplet = self._require("run_triplet_commands")
        cases = (
            ("missing", [sys.executable, "-c", "print('no config')"]),
            (
                "invalid",
                [
                    sys.executable,
                    "-c",
                    "import pathlib,sys;p=pathlib.Path(sys.argv[1]);(p/'run_config.json').write_text('{')",
                    "{output_dir}",
                ],
            ),
            ("prepared", self._config_command(status="prepared")),
        )
        probe = lambda: {
            "ram_available_bytes": 8 * 1024**3,
            "swap_thrashing": False,
            "gpu_memory_mib": {0: 0},
        }
        for label, first_command in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                commands = {
                    "anchor_before": first_command,
                    "variant": self._config_command(),
                    "anchor_after": self._config_command(),
                }
                root = Path(tmp) / "triplet"
                try:
                    summary = run_triplet(
                        commands,
                        root,
                        execution_mode="sequential_triplet",
                        resource_probe=probe,
                    )
                except Exception:
                    summary = json.loads((root / "triplet_summary.json").read_text())
                self.assertEqual(summary["status"], "failed")
                self.assertFalse(summary["runner_valid"])

    def test_launch_exception_still_kills_started_peer_and_writes_terminal_summary(self):
        run_triplet = self._require("run_triplet_commands")
        commands = {
            "anchor_before": self._config_command("time.sleep(5)"),
            "variant": ["/definitely/not/a/real/executable"],
            "anchor_after": self._config_command(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "triplet"
            started = time.monotonic()
            returned = run_triplet(
                commands,
                root,
                execution_mode="dual_gpu_crossover",
                resource_probe=lambda: {
                    "ram_available_bytes": 8 * 1024**3,
                    "swap_thrashing": False,
                    "gpu_memory_mib": {0: 0, 1: 0},
                },
            )
            elapsed = time.monotonic() - started
            stored = json.loads((root / "triplet_summary.json").read_text())
        self.assertLess(elapsed, 2.0)
        self.assertEqual(stored["status"], "failed")
        self.assertEqual(returned, stored)
        self.assertTrue(stored["completed_at_utc"])
        self.assertTrue(stored["reasons"])

    def test_execution_mode_is_atomically_injected_into_each_run_config(self):
        run_triplet = self._require("run_triplet_commands")
        code = (
            "import json,pathlib,sys;"
            "p=pathlib.Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=True);"
            "(p/'run_config.json').write_text(json.dumps({'status':'completed','experiment_family':'turnvega','variant':'test-variant'}))"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "triplet"
            commands = {
                "anchor_before": [sys.executable, "-c", code, str(root / "anchor_before")],
                "variant": [sys.executable, "-c", code, str(root / "variant")],
                "anchor_after": [sys.executable, "-c", code, str(root / "anchor_after")],
            }
            summary = run_triplet(
                commands,
                root,
                execution_mode="sequential_triplet",
                resource_probe=lambda: {
                    "ram_available_bytes": 8 * 1024**3,
                    "swap_thrashing": False,
                    "gpu_memory_mib": {0: 0},
                },
            )
            configs = [
                json.loads((root / name / "run_config.json").read_text())
                for name in ("anchor_before", "variant", "anchor_after")
            ]
        self.assertTrue(summary["runner_valid"])
        self.assertEqual(
            [config["execution_mode"] for config in configs],
            ["sequential_triplet"] * 3,
        )

    def test_output_dir_placeholder_is_expanded_for_all_four_dual_runs(self):
        run_triplet = self._require("run_triplet_commands")
        command = self._config_command("print(sys.argv[1])")
        commands = {
            name: command
            for name in ("anchor_before", "variant", "anchor_after")
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "triplet"
            summary = run_triplet(
                commands,
                root,
                execution_mode="dual_gpu_crossover",
                resource_probe=lambda: {
                    "ram_available_bytes": 8 * 1024**3,
                    "swap_thrashing": False,
                    "gpu_memory_mib": {0: 0, 1: 0},
                },
            )
            actual = {
                str(path.relative_to(root)): path.joinpath("stdout_stderr.log")
                .read_text()
                .strip()
                for path in (
                    root / "round_a" / "anchor",
                    root / "round_a" / "variant",
                    root / "round_b" / "variant",
                    root / "round_b" / "anchor",
                )
            }
        self.assertTrue(summary["runner_valid"])
        self.assertEqual(
            actual,
            {
                relative: str(root / relative)
                for relative in (
                    "round_a/anchor",
                    "round_a/variant",
                    "round_b/variant",
                    "round_b/anchor",
                )
            },
        )


if __name__ == "__main__":
    unittest.main()

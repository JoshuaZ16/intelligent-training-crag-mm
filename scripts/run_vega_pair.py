#!/usr/bin/env python3
"""Launch a reproducible VEGA-RAG pair on two isolated GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _trace_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_run_artifacts(
    left_dir: Path,
    right_dir: Path,
    expected_count: int,
) -> dict:
    left_rows = _trace_rows(left_dir / "agent_trace_v3.jsonl")
    right_rows = _trace_rows(right_dir / "agent_trace_v3.jsonl")
    left_queries = [row.get("query") for row in left_rows]
    right_queries = [row.get("query") for row in right_rows]
    counts_ok = (
        len(left_rows) == expected_count
        and len(right_rows) == expected_count
    )
    query_sets_match = (
        left_queries == right_queries
        and len(set(left_queries)) == expected_count
    )
    statuses_ok = all(row.get("status") == "ok" for row in left_rows + right_rows)
    return {
        "expected_count": expected_count,
        "left_trace_count": len(left_rows),
        "right_trace_count": len(right_rows),
        "query_sets_match": query_sets_match,
        "trace_statuses_ok": statuses_ok,
        "artifacts_ok": counts_ok and query_sets_match and statuses_ok,
    }


def run_pair_commands(
    left_command: Sequence[str],
    right_command: Sequence[str],
    pair_dir: Path,
    telemetry_command: Sequence[str] | None = None,
) -> dict:
    left_dir = pair_dir / "left"
    right_dir = pair_dir / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    telemetry_handle = None
    telemetry_process = None
    if telemetry_command:
        telemetry_handle = (pair_dir / "gpu_telemetry.csv").open(
            "w",
            encoding="utf-8",
        )
        telemetry_process = subprocess.Popen(
            list(telemetry_command),
            stdout=telemetry_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    left_env = os.environ.copy()
    left_env["CUDA_VISIBLE_DEVICES"] = "0"
    right_env = os.environ.copy()
    right_env["CUDA_VISIBLE_DEVICES"] = "1"

    left_started = time.time()
    left_started_utc = datetime.now(timezone.utc).isoformat()
    with (left_dir / "stdout_stderr.log").open("w", encoding="utf-8") as left_log:
        left_process = subprocess.Popen(
            list(left_command),
            env=left_env,
            stdout=left_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        right_started = time.time()
        right_started_utc = datetime.now(timezone.utc).isoformat()
        with (right_dir / "stdout_stderr.log").open("w", encoding="utf-8") as right_log:
            right_process = subprocess.Popen(
                list(right_command),
                env=right_env,
                stdout=right_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            left_status = left_process.wait()
            right_status = right_process.wait()

    if telemetry_process is not None:
        telemetry_process.terminate()
        try:
            telemetry_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            telemetry_process.kill()
            telemetry_process.wait()
    if telemetry_handle is not None:
        telemetry_handle.close()

    (left_dir / "exit_status.txt").write_text(
        f"{left_status}\n",
        encoding="utf-8",
    )
    (right_dir / "exit_status.txt").write_text(
        f"{right_status}\n",
        encoding="utf-8",
    )
    summary = {
        "left_command": list(left_command),
        "right_command": list(right_command),
        "left_started_at_utc": left_started_utc,
        "right_started_at_utc": right_started_utc,
        "start_delta_seconds": abs(right_started - left_started),
        "left_exit_status": left_status,
        "right_exit_status": right_status,
        "processes_ok": left_status == 0 and right_status == 0,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(pair_dir / "pair_summary.json", summary)
    return summary


def build_experiment_command(
    python: str,
    variant: str,
    run_id: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        python,
        "scripts/week2_experiment.py",
        "--mode",
        "task2",
        "--backend",
        "vllm",
        "--variant",
        variant,
        "--run-id",
        run_id,
        "--num-samples",
        str(args.num_samples),
        "--seed",
        str(args.seed),
        "--manifest-path",
        args.manifest_path,
        "--dataset-path",
        args.dataset_path,
        "--model",
        args.model,
        "--output-dir",
        str(output_dir),
    ]
    if args.thresholds_path:
        command.extend(["--thresholds-path", args.thresholds_path])
    command.extend(["--adaptive-k", str(args.adaptive_k)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-dir", required=True)
    parser.add_argument("--left-variant", default="b0")
    parser.add_argument("--right-variant", required=True)
    parser.add_argument("--left-run-id", required=True)
    parser.add_argument("--right-run-id", required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--thresholds-path")
    parser.add_argument("--adaptive-k", type=int, default=5)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    pair_dir = Path(args.pair_dir)
    left_dir = pair_dir / "left"
    right_dir = pair_dir / "right"
    left_command = build_experiment_command(
        args.python,
        args.left_variant,
        args.left_run_id,
        left_dir,
        args,
    )
    right_command = build_experiment_command(
        args.python,
        args.right_variant,
        args.right_run_id,
        right_dir,
        args,
    )
    telemetry_command = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader",
        "--loop-ms=5000",
    ]
    summary = run_pair_commands(
        left_command,
        right_command,
        pair_dir,
        telemetry_command,
    )
    artifact_status = validate_run_artifacts(
        left_dir,
        right_dir,
        args.num_samples,
    )
    summary.update(artifact_status)
    summary["pair_valid"] = (
        summary["processes_ok"]
        and summary["start_delta_seconds"] <= 30
        and artifact_status["artifacts_ok"]
    )
    _atomic_json(pair_dir / "pair_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    raise SystemExit(0 if summary["pair_valid"] else 1)


if __name__ == "__main__":
    main()

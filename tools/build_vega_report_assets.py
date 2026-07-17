#!/usr/bin/env python3
"""Validate accepted VEGA-RAG runs and build report-ready assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path


def _number(value: str) -> float:
    return float(value.strip().split()[0])


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def parse_gpu_telemetry(path: Path, physical_gpu: int) -> dict[str, float | int]:
    selected = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle, skipinitialspace=True):
            if len(row) < 8 or int(row[1].strip()) != physical_gpu:
                continue
            selected.append(
                {
                    "temperature": _number(row[3]),
                    "power": _number(row[4]),
                    "memory": _number(row[5]),
                    "utilization": _number(row[7]),
                }
            )
    if not selected:
        raise ValueError(f"no telemetry rows for physical GPU {physical_gpu}")
    return {
        "telemetry_samples": len(selected),
        "gpu_peak_memory_mib": max(row["memory"] for row in selected),
        "gpu_peak_utilization_pct": max(row["utilization"] for row in selected),
        "gpu_mean_utilization_pct": sum(row["utilization"] for row in selected) / len(selected),
        "gpu_peak_temperature_c": max(row["temperature"] for row in selected),
        "gpu_peak_power_w": max(row["power"] for row in selected),
    }


def summarize_trace(traces: list[dict]) -> dict[str, float | int]:
    if not traces:
        raise ValueError("trace rows must not be empty")
    total_ms = [float(row["total_ms"]) for row in traces]
    generation_ms = [float(row["generation_ms"]) for row in traces]
    actions = [row["gate_action"] for row in traces]
    return {
        "trace_count": len(traces),
        "batch_latency_p50_ms": _percentile(total_ms, 0.50),
        "batch_latency_p95_ms": _percentile(total_ms, 0.95),
        "generation_p50_ms": _percentile(generation_ms, 0.50),
        "generation_p95_ms": _percentile(generation_ms, 0.95),
        "mean_search_calls": sum(float(row["search_call_count"]) for row in traces) / len(traces),
        "additional_evidence_count": sum(len(row.get("additional_web_evidence") or []) for row in traces),
        "answer_count": sum(action == "answer" for action in actions),
        "expand_count": sum(action == "expand" for action in actions),
        "abstain_count": sum(action == "abstain" for action in actions),
    }


def _read_traces(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_assets(
    ablation_root: Path,
    accepted_runs_path: Path,
    comparison_dir: Path,
    output_dir: Path,
) -> dict:
    accepted_config = json.loads(
        accepted_runs_path.read_text(encoding="utf-8")
    )
    expected_threshold_sha = accepted_config["thresholds_sha256"]
    threshold_path = ablation_root / "calibration/vega_calibration.json"
    actual_threshold_sha = hashlib.sha256(threshold_path.read_bytes()).hexdigest()
    if actual_threshold_sha != expected_threshold_sha:
        raise ValueError("frozen threshold SHA-256 does not match accepted manifest")

    system_rows = []
    accepted_validation = []
    for item in accepted_config["accepted"]:
        pair_dir = ablation_root / item["pair_dir"]
        run_dir = pair_dir / item["side"]
        pair_summary = json.loads(
            (pair_dir / "pair_summary.json").read_text(encoding="utf-8")
        )
        run_config = json.loads(
            (run_dir / "run_config.json").read_text(encoding="utf-8")
        )
        traces = _read_traces(run_dir / "agent_trace_v3.jsonl")
        with (run_dir / "turn_evaluation_results_all.csv").open(
            newline="",
            encoding="utf-8",
        ) as handle:
            results = list(csv.DictReader(handle))
        valid = (
            pair_summary.get("pair_valid") is True
            and run_config.get("status") == "completed"
            and run_config.get("thresholds_sha256") == expected_threshold_sha
            and len(traces) == len(results) == 30
            and all(row.get("status") == "ok" for row in traces)
            and {row["query"] for row in traces}
            == {row["query"] for row in results}
        )
        if not valid:
            raise ValueError(f"accepted run failed validation: {item['run_key']}")
        trace_metrics = summarize_trace(traces)
        gpu_metrics = parse_gpu_telemetry(
            pair_dir / "gpu_telemetry.csv",
            physical_gpu=int(item["physical_gpu"]),
        )
        system_rows.append(
            {
                "round": item["round"],
                "run_key": item["run_key"],
                "version": item["version"],
                "run_id": run_config["run_id"],
                "physical_gpu": item["physical_gpu"],
                "wall_time_seconds_including_load": run_config["elapsed_seconds"],
                **trace_metrics,
                **gpu_metrics,
                "latency_note": "Per-turn trace values share batch generation time; model/index load is excluded from P50/P95 and included in wall time.",
            }
        )
        accepted_validation.append(
            {
                "run_key": item["run_key"],
                "valid": True,
                "trace_count": len(traces),
                "result_count": len(results),
            }
        )

    comparison_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(comparison_dir / "system_metrics.csv", system_rows)
    _write_csv(output_dir / "system_metrics.csv", system_rows)
    failed_rows = [
        {
            "pair_dir": item["pair_dir"],
            "runner_pair_valid": item["runner_pair_valid"],
            "strict_acceptance": item["strict_acceptance"],
            "reason": item["reason"],
        }
        for item in accepted_config.get("excluded", [])
    ]
    if failed_rows:
        _write_csv(comparison_dir / "failed_runs.csv", failed_rows)
        _write_csv(output_dir / "failed_runs.csv", failed_rows)

    asset_names = [
        "metrics_primary_20.csv",
        "metrics_all_30.csv",
        "paired_deltas_primary_20.csv",
        "bootstrap_ci.csv",
        "mcnemar.csv",
        "transition_matrix.csv",
        "error_causes.csv",
        "refusal_quality.csv",
        "review_final.csv",
    ]
    for name in asset_names:
        source = comparison_dir / name
        if not source.exists():
            raise ValueError(f"missing comparison asset: {name}")
        shutil.copy2(source, output_dir / name)
    validation = {
        "thresholds_sha256": expected_threshold_sha,
        "accepted_runs": accepted_validation,
        "excluded_runs": accepted_config.get("excluded", []),
        "system_metric_rows": len(system_rows),
    }
    (output_dir / "asset_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-root", required=True)
    parser.add_argument("--accepted-runs", required=True)
    parser.add_argument("--comparison-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    validation = build_assets(
        Path(args.ablation_root),
        Path(args.accepted_runs),
        Path(args.comparison_dir),
        Path(args.output_dir),
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

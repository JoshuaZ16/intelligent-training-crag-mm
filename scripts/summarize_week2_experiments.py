#!/usr/bin/env python3
"""Combine experiment outputs without presenting exact match as official scoring."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def percentile(values, percentile_value):
    if not values:
        return 0.0
    values = sorted(values)
    index = min(round((percentile_value / 100) * (len(values) - 1)), len(values) - 1)
    return values[index]


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    for raw_dir in args.run_dirs:
        run_dir = Path(raw_dir)
        run = load_json(run_dir / "run_config.json")
        scores = load_json(run_dir / "scores_dictionary.json")["all"]
        traces = [json.loads(line) for line in (run_dir / "agent_trace.jsonl").read_text(encoding="utf-8").splitlines() if line]
        totals = [float(trace["total_ms"]) for trace in traces]
        rows.append({
            "mode": run["mode"],
            "backend": run["backend"],
            "model": run.get("model") or "backend default",
            "samples": run["sample_count"],
            "exact_match_accuracy": scores["accuracy"],
            "exact_match_missing": scores["missing"],
            "exact_match_hallucination": scores["hallucination_rate"],
            "exact_match_truthfulness": scores["truthfulness_score"],
            "mean_total_ms": statistics.fmean(totals) if totals else 0.0,
            "p95_total_ms": percentile(totals, 95),
            "run_failures": sum(trace["status"] != "ok" for trace in traces),
            "empty_retrieval": sum(not trace["image_evidence"] and not trace["web_evidence"] for trace in traces),
            "refusals": sum(trace["answer"] == "I don't know" for trace in traces),
            "over_75_tokens": sum(int(trace["answer_token_count"]) > 75 for trace in traces),
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

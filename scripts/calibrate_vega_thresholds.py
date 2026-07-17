#!/usr/bin/env python3
"""Freeze VEGA-RAG thresholds from reviewed real calibration answers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


VALID_LABELS = {"correct", "partial", "incorrect", "missing"}
DEFAULT_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
DEFAULT_MIN_COVERAGE = 0.7
SELECTION_ORDER = [
    "truthfulness",
    "accuracy",
    "lower_missing",
    "simpler_threshold",
]


def score_labels(labels: Iterable[str]) -> dict[str, float | int]:
    values = list(labels)
    if not values:
        raise ValueError("at least one reviewed label is required")
    invalid = sorted(set(values) - VALID_LABELS)
    if invalid:
        raise ValueError(f"unsupported review labels: {invalid}")
    total = len(values)
    correct = sum(value == "correct" for value in values)
    missing_count = sum(value == "missing" for value in values)
    hallucination_count = total - correct - missing_count
    return {
        "total": total,
        "correct": correct,
        "missing_count": missing_count,
        "hallucination_count": hallucination_count,
        "accuracy": correct / total,
        "missing": missing_count / total,
        "hallucination": hallucination_count / total,
        "truthfulness": ((2 * correct + missing_count) / total) - 1,
        "coverage": 1 - (missing_count / total),
    }


def select_thresholds(
    rows: list[dict],
    grid: list[float] | None = None,
    min_coverage: float = 0.0,
) -> dict:
    if not rows:
        raise ValueError("calibration rows must not be empty")
    if not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage must be in [0, 1]")
    candidate_grid = list(grid or DEFAULT_GRID)
    for row in rows:
        if row.get("b0_label") not in VALID_LABELS:
            raise ValueError("every row needs a reviewed b0_label")
        if row.get("enriched_label") not in VALID_LABELS:
            raise ValueError("every row needs a reviewed enriched_label")
        score = row.get("score")
        if not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise ValueError("every row score must be in [0, 1]")

    candidates = []
    best = None
    best_key = None
    for tau_low in candidate_grid:
        for tau_high in candidate_grid:
            if tau_low >= tau_high:
                continue
            labels = []
            for row in rows:
                score = float(row["score"])
                if score <= tau_low:
                    labels.append("missing")
                elif score < tau_high:
                    labels.append(row["enriched_label"])
                else:
                    labels.append(row["b0_label"])
            metrics = score_labels(labels)
            feasible = metrics["coverage"] + 1e-12 >= min_coverage
            candidate = {
                "tau_low": tau_low,
                "tau_high": tau_high,
                "metrics": metrics,
                "labels": labels,
                "feasible": feasible,
            }
            candidates.append(candidate)
            if not feasible:
                continue
            key = (
                metrics["truthfulness"],
                metrics["accuracy"],
                -metrics["missing"],
                -tau_low,
                -tau_high,
            )
            if best_key is None or key > best_key:
                best_key = key
                best = candidate

    if best is None:
        raise ValueError(
            "grid must contain at least one ordered threshold pair "
            "that satisfies min_coverage"
        )
    feasible_count = sum(candidate["feasible"] for candidate in candidates)
    return {
        "tau_low": best["tau_low"],
        "tau_high": best["tau_high"],
        "metrics": best["metrics"],
        "minimum_coverage": min_coverage,
        "selection_order": SELECTION_ORDER,
        "tie_breaker": [
            "lower_tau_low",
            "lower_tau_high",
        ],
        "grid": candidate_grid,
        "candidate_count": len(candidates),
        "feasible_candidate_count": feasible_count,
        "rejected_candidate_count": len(candidates) - feasible_count,
        "candidates": candidates,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=DEFAULT_MIN_COVERAGE,
    )
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_json)
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    result = select_thresholds(rows, min_coverage=args.min_coverage)
    result["input_sha256"] = _sha256(input_path)
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["calibration_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

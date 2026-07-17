#!/usr/bin/env python3
"""Score reviewed VEGA-RAG responses with paired statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


VALID_LABELS = {"correct", "partial", "incorrect", "missing"}
METRIC_NAMES = ["accuracy", "missing", "hallucination", "truthfulness"]


def score_labels(labels: Iterable[str]) -> dict[str, float | int]:
    values = list(labels)
    if not values:
        raise ValueError("at least one label is required")
    invalid = set(values) - VALID_LABELS
    if invalid:
        raise ValueError(f"unsupported labels: {sorted(invalid)}")
    total = len(values)
    correct = sum(label == "correct" for label in values)
    missing_count = sum(label == "missing" for label in values)
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


def expand_reviews(review_rows: list[dict], mapping: dict) -> list[dict]:
    reviews = {row["response_id"]: row for row in review_rows}
    if len(reviews) != len(review_rows):
        raise ValueError("review response_id values must be unique")
    query_meta = {
        row["query_id"]: row for row in mapping.get("queries", [])
    }
    expanded = []
    seen = set()
    for assignment in mapping.get("assignments", []):
        response_id = assignment["response_id"]
        query_id = assignment["query_id"]
        key = (query_id, assignment["version"])
        if key in seen:
            raise ValueError(f"duplicate assignment: {key}")
        seen.add(key)
        if response_id not in reviews or query_id not in query_meta:
            raise ValueError("mapping references an unknown review or query")
        review = reviews[response_id]
        if review["query_id"] != query_id:
            raise ValueError("review and mapping query_id differ")
        if review["label"] not in VALID_LABELS:
            raise ValueError(f"unreviewed response: {response_id}")
        expanded.append(
            {
                "query_id": query_id,
                "source_order": int(query_meta[query_id]["source_order"]),
                "query": query_meta[query_id]["query"],
                "version": assignment["version"],
                "response_id": response_id,
                "label": review["label"],
                "error_cause": review["error_cause"],
                "rationale": review["rationale"],
            }
        )
    return expanded


def exact_mcnemar(b: int, c: int) -> dict[str, float | int]:
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = b + c
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(b, c) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    return {
        "b_baseline_correct_variant_wrong": b,
        "c_baseline_wrong_variant_correct": c,
        "discordant": discordant,
        "p_value": p_value,
    }


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


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metric_rows(
    expanded: list[dict],
    query_ids: set[str],
    versions: list[str],
    scope: str,
) -> list[dict]:
    rows = []
    for version in versions:
        labels = [
            row["label"]
            for row in expanded
            if row["version"] == version and row["query_id"] in query_ids
        ]
        values = score_labels(labels)
        rows.append({"scope": scope, "version": version, **values})
    return rows


def score_comparison(
    review_rows: list[dict],
    mapping: dict,
    bootstrap_samples: int = 10_000,
    seed: int = 20260716,
) -> dict[str, list[dict]]:
    expanded = expand_reviews(review_rows, mapping)
    versions = list(mapping.get("version_order") or [])
    if not versions:
        versions = list(dict.fromkeys(row["version"] for row in expanded))
    if "b0" not in versions:
        raise ValueError("version_order must contain b0")
    all_ids = {
        row["query_id"]
        for row in expanded
    }
    primary_ids = {
        row["query_id"]
        for row in expanded
        if int(row["source_order"]) >= 10
    }
    if not primary_ids:
        raise ValueError("primary scope must contain post-calibration rows")
    metrics_all = _metric_rows(expanded, all_ids, versions, "all_30")
    metrics_primary = _metric_rows(
        expanded,
        primary_ids,
        versions,
        "primary_20",
    )
    metrics_primary_by_version = {
        row["version"]: row for row in metrics_primary
    }
    baseline_metrics = metrics_primary_by_version["b0"]
    paired_deltas = []
    for version in versions:
        values = metrics_primary_by_version[version]
        paired_deltas.append(
            {
                "scope": "primary_20",
                "version": version,
                **{
                    f"delta_{metric}": values[metric] - baseline_metrics[metric]
                    for metric in METRIC_NAMES
                },
            }
        )

    labels_by_version = {
        version: {
            row["query_id"]: row["label"]
            for row in expanded
            if row["version"] == version
        }
        for version in versions
    }
    ordered_primary = sorted(
        primary_ids,
        key=lambda query_id: next(
            row["source_order"]
            for row in expanded
            if row["query_id"] == query_id
        ),
    )
    rng = random.Random(seed)
    bootstrap_values = {
        (version, metric): []
        for version in versions
        if version != "b0"
        for metric in METRIC_NAMES
    }
    for _ in range(bootstrap_samples):
        sample_ids = [
            rng.choice(ordered_primary) for _ in ordered_primary
        ]
        baseline = score_labels(
            labels_by_version["b0"][query_id]
            for query_id in sample_ids
        )
        for version in versions:
            if version == "b0":
                continue
            variant = score_labels(
                labels_by_version[version][query_id]
                for query_id in sample_ids
            )
            for metric in METRIC_NAMES:
                bootstrap_values[(version, metric)].append(
                    variant[metric] - baseline[metric]
                )
    bootstrap = []
    for (version, metric), values in bootstrap_values.items():
        bootstrap.append(
            {
                "scope": "primary_20",
                "version": version,
                "metric": metric,
                "observed_delta": metrics_primary_by_version[version][metric]
                - baseline_metrics[metric],
                "ci_low": _percentile(values, 0.025),
                "ci_high": _percentile(values, 0.975),
                "bootstrap_samples": bootstrap_samples,
                "seed": seed,
            }
        )

    mcnemar = []
    for version in versions:
        if version == "b0":
            continue
        b = sum(
            labels_by_version["b0"][query_id] == "correct"
            and labels_by_version[version][query_id] != "correct"
            for query_id in primary_ids
        )
        c = sum(
            labels_by_version["b0"][query_id] != "correct"
            and labels_by_version[version][query_id] == "correct"
            for query_id in primary_ids
        )
        mcnemar.append(
            {
                "scope": "primary_20",
                "version": version,
                **exact_mcnemar(b, c),
            }
        )

    transitions = []
    for scope, ids in (("primary_20", primary_ids), ("all_30", all_ids)):
        for version in versions:
            if version == "b0":
                continue
            counts = Counter(
                (
                    labels_by_version["b0"][query_id],
                    labels_by_version[version][query_id],
                )
                for query_id in ids
            )
            for (baseline_label, variant_label), count in sorted(counts.items()):
                transitions.append(
                    {
                        "scope": scope,
                        "version": version,
                        "baseline_label": baseline_label,
                        "variant_label": variant_label,
                        "count": count,
                    }
                )

    error_causes = []
    for scope, ids in (("primary_20", primary_ids), ("all_30", all_ids)):
        for version in versions:
            counts = Counter(
                row["error_cause"]
                for row in expanded
                if row["version"] == version
                and row["query_id"] in ids
                and row["label"] != "correct"
            )
            for cause, count in sorted(counts.items()):
                error_causes.append(
                    {
                        "scope": scope,
                        "version": version,
                        "error_cause": cause,
                        "count": count,
                    }
                )

    refusal_quality = []
    for scope, ids in (("primary_20", primary_ids), ("all_30", all_ids)):
        for version in versions:
            missing_ids = [
                query_id
                for query_id in ids
                if labels_by_version[version][query_id] == "missing"
            ]
            categories = Counter()
            for query_id in missing_ids:
                baseline_label = labels_by_version["b0"][query_id]
                if baseline_label == "correct":
                    categories["harmful_refusal"] += 1
                elif baseline_label == "missing":
                    categories["baseline_also_missing"] += 1
                else:
                    categories["avoided_hallucination"] += 1
            refusal_quality.append(
                {
                    "scope": scope,
                    "version": version,
                    "refusal_count": len(missing_ids),
                    "avoided_hallucination": categories["avoided_hallucination"],
                    "harmful_refusal": categories["harmful_refusal"],
                    "baseline_also_missing": categories["baseline_also_missing"],
                }
            )

    return {
        "review_expanded": expanded,
        "metrics_primary_20": metrics_primary,
        "metrics_all_30": metrics_all,
        "paired_deltas_primary_20": paired_deltas,
        "bootstrap_ci": bootstrap,
        "mcnemar": mcnemar,
        "transition_matrix": transitions,
        "error_causes": error_causes,
        "refusal_quality": refusal_quality,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--mapping-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    with Path(args.review_csv).open(newline="", encoding="utf-8") as handle:
        review_rows = list(csv.DictReader(handle))
    mapping = json.loads(Path(args.mapping_json).read_text(encoding="utf-8"))
    outputs = score_comparison(
        review_rows,
        mapping,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output_dir = Path(args.output_dir)
    for name, rows in outputs.items():
        _write_csv(output_dir / f"{name}.csv", rows)
    print(
        json.dumps(
            {name: len(rows) for name, rows in outputs.items()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

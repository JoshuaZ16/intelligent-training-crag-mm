#!/usr/bin/env python3
"""Build a deterministic version-blind review sheet for VEGA-RAG."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


REQUIRED_COLUMNS = {"query", "ground_truth", "agent_response"}
REVIEW_COLUMNS = [
    "response_id",
    "query_id",
    "query",
    "ground_truth",
    "answer",
    "label",
    "error_cause",
    "rationale",
]


def _read_run(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    queries = [row["query"] for row in rows]
    if not rows or len(queries) != len(set(queries)):
        raise ValueError(f"{path} must contain unique, non-empty queries")
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_review(
    run_files: dict[str, str | Path],
    seed: int = 20260716,
) -> tuple[list[dict[str, str]], dict]:
    if not run_files:
        raise ValueError("at least one version run is required")
    version_order = list(run_files)
    loaded = {
        version: _read_run(Path(path))
        for version, path in run_files.items()
    }
    base_rows = loaded[version_order[0]]
    query_order = [row["query"] for row in base_rows]
    ground_truth = {
        row["query"]: row["ground_truth"] for row in base_rows
    }
    by_version = {
        version: {row["query"]: row for row in rows}
        for version, rows in loaded.items()
    }
    expected_queries = set(query_order)
    for version in version_order:
        if set(by_version[version]) != expected_queries:
            raise ValueError(f"query set differs for version {version}")
        for query in query_order:
            if by_version[version][query]["ground_truth"] != ground_truth[query]:
                raise ValueError(
                    f"ground truth differs for version {version}: {query}"
                )

    classes: list[dict] = []
    queries = []
    for query_index, query in enumerate(query_order, start=1):
        query_id = f"Q{query_index:03d}"
        queries.append(
            {
                "query_id": query_id,
                "source_order": query_index - 1,
                "query": query,
            }
        )
        answer_versions: dict[str, list[str]] = {}
        for version in version_order:
            answer = by_version[version][query]["agent_response"].strip()
            answer_versions.setdefault(answer, []).append(version)
        for answer, versions in answer_versions.items():
            classes.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "ground_truth": ground_truth[query],
                    "answer": answer,
                    "versions": versions,
                }
            )

    random.Random(seed).shuffle(classes)
    review = []
    responses = {}
    assignments = []
    for response_index, item in enumerate(classes, start=1):
        response_id = f"R{response_index:03d}"
        review.append(
            {
                "response_id": response_id,
                "query_id": item["query_id"],
                "query": item["query"],
                "ground_truth": item["ground_truth"],
                "answer": item["answer"],
                "label": "",
                "error_cause": "",
                "rationale": "",
            }
        )
        responses[response_id] = {
            "query_id": item["query_id"],
            "versions": item["versions"],
        }
        assignments.extend(
            {
                "query_id": item["query_id"],
                "version": version,
                "response_id": response_id,
            }
            for version in item["versions"]
        )

    mapping = {
        "seed": seed,
        "version_order": version_order,
        "input_sha256": {
            version: _sha256(Path(run_files[version]))
            for version in version_order
        },
        "queries": queries,
        "responses": responses,
        "assignments": assignments,
    }
    return review, mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="VERSION=CSV",
    )
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--mapping-json", required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    run_files = {}
    for value in args.run:
        version, separator, path = value.partition("=")
        if not separator or not version or not path:
            parser.error("--run must use VERSION=CSV")
        run_files[version] = Path(path)
    review, mapping = build_review(run_files, seed=args.seed)

    review_path = Path(args.review_csv)
    mapping_path = Path(args.mapping_json)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with review_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(review)
    mapping_path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "review_rows": len(review),
                "assignments": len(mapping["assignments"]),
                "review_csv": str(review_path),
                "mapping_json": str(mapping_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

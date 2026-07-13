#!/usr/bin/env python3
"""Deterministic 30+30 contract acceptance; this is not a CRAG benchmark."""

from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.course_agent_v2 import AgentConfig, CourseRAGAgentV2, IDK_RESPONSE, TaskMode


class ContractSearch:
    def __call__(self, value, k):
        if isinstance(value, Image.Image):
            case_id = int(value.info["case_id"])
            case_type = value.info["case_type"]
            attrs = {"object_type": "building"} if case_type == "task1" else {"product_category": "test product"}
            if case_type == "task1":
                attrs["architect"] = f"Architect {case_id}"
            return [{
                "score": 0.95,
                "entities": [{"entity_name": f"Entity {case_id}", "entity_attributes": attrs}],
            }]
        match = re.search(r"Product (\d+)", str(value))
        case_id = int(match.group(1)) if match else 0
        return [{
            "score": 0.9,
            "page_name": f"Product {case_id} official listing",
            "page_url": f"https://example.test/products/{case_id}",
            "page_snippet": f"Product {case_id} costs ${case_id + 10}.",
        }]


class ContractBackend:
    def answer_batch(self, prompts, images):
        answers = []
        for prompt in prompts:
            architect = re.search(r"architect: (Architect \d+)", prompt)
            price = re.search(r"costs (\$\d+)", prompt)
            answers.append(architect.group(1) if architect else (price.group(1) if price else IDK_RESPONSE))
        return answers

    def truncate(self, text, max_tokens):
        return " ".join(text.split()[:max_tokens])

    def count_tokens(self, text):
        return len(text.split())


def image_for(case_id, case_type):
    image = Image.new("RGB", (16, 16), (240, 240, 240))
    image.info["case_id"] = case_id
    image.info["case_type"] = case_type
    return image


def score(rows):
    correct = sum(row["prediction"] == row["ground_truth"] for row in rows)
    missing = sum(row["prediction"] == IDK_RESPONSE for row in rows)
    hallucination = len(rows) - correct - missing
    return {
        "samples": len(rows),
        "correct": correct,
        "missing": missing,
        "hallucination": hallucination,
        "accuracy": correct / len(rows),
        "missing_rate": missing / len(rows),
        "hallucination_rate": hallucination / len(rows),
        "truthfulness": (correct - hallucination) / len(rows),
    }


def run(mode, cases, trace_path):
    config = AgentConfig(task_mode=mode, trace_path=str(trace_path), batch_size=6)
    agent = CourseRAGAgentV2(ContractSearch(), ContractBackend(), config)
    rows = []
    for offset in range(0, len(cases), config.batch_size):
        batch = cases[offset:offset + config.batch_size]
        predictions = agent.batch_generate_response(
            [case["query"] for case in batch],
            [case["image"] for case in batch],
            [[] for _ in batch],
        )
        for case, prediction in zip(batch, predictions):
            rows.append({
                "suite": case["suite"],
                "case_id": case["case_id"],
                "mode": mode.value,
                "query": case["query"],
                "ground_truth": case["ground_truth"],
                "prediction": prediction,
                "is_correct": prediction == case["ground_truth"],
            })
    return rows


def main():
    output_dir = PROJECT_ROOT / "artifacts" / "week2" / "offline_acceptance"
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_trace in output_dir.glob("*_trace.jsonl"):
        old_trace.unlink()

    task1_cases = [{
        "suite": "task1_contract",
        "case_id": index,
        "query": f"Who is the architect of Entity {index}?",
        "ground_truth": f"Architect {index}",
        "image": image_for(index, "task1"),
    } for index in range(1, 31)]
    task2_cases = [{
        "suite": "task2_contract",
        "case_id": index,
        "query": f"What is the price of Product {index}?",
        "ground_truth": f"${index + 10}",
        "image": image_for(index, "task2"),
    } for index in range(1, 31)]

    runs = [
        (TaskMode.VISION, task1_cases, "vision_on_task1"),
        (TaskMode.TASK1, task1_cases, "task1_on_task1"),
        (TaskMode.TASK1, task2_cases, "task1_on_task2"),
        (TaskMode.TASK2, task2_cases, "task2_on_task2"),
    ]
    all_rows = []
    summaries = []
    for mode, cases, label in runs:
        rows = run(mode, cases, output_dir / f"{label}_trace.jsonl")
        all_rows.extend(rows)
        summaries.append({"run": label, "mode": mode.value, **score(rows)})

    with (output_dir / "raw_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    metadata = {
        "scope": "synthetic_contract_acceptance",
        "not_official_benchmark": True,
        "purpose": "Verify branching, retrieval calls, batching, evidence flow, refusals, traces, and metrics only.",
        "runs": summaries,
    }
    (output_dir / "README.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

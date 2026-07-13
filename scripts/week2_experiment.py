#!/usr/bin/env python3
"""Run a fixed, traceable CRAG-MM week-2 experiment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.course_agent_v2 import AgentConfig, CourseRAGAgentV2, TaskMode, create_backend
from local_evaluation import CRAGEvaluator


DATASET_ID = "crag-mm-2025/crag-mm-single-turn-public"
DATASET_REVISION = "v0.1.2"


def first_value(value, key):
    if isinstance(value, dict):
        column = value.get(key, [])
        return column[0] if isinstance(column, list) and column else column
    if isinstance(value, list) and value:
        return value[0].get(key)
    return None


def stratified_indices(dataset: Dataset, count: int, seed: int) -> list[int]:
    groups: dict[tuple, list[int]] = {}
    for index, row in enumerate(dataset):
        key = (
            first_value(row["turns"], "query_category"),
            first_value(row["turns"], "domain"),
        )
        groups.setdefault(key, []).append(index)
    rng = random.Random(seed)
    for indices in groups.values():
        rng.shuffle(indices)
    keys = list(groups)
    rng.shuffle(keys)
    selected = []
    while len(selected) < min(count, len(dataset)):
        progressed = False
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop())
                progressed = True
                if len(selected) >= count:
                    break
        if not progressed:
            break
    return selected


def build_search_pipeline(mode: TaskMode):
    if mode is TaskMode.VISION:
        return None
    from cragmm_search.search import UnifiedSearchPipeline

    kwargs = {
        "image_model_name": "openai/clip-vit-large-patch14-336",
        "image_hf_dataset_id": "crag-mm-2025/image-search-index-validation",
    }
    if mode is TaskMode.TASK2:
        kwargs.update({
            "text_model_name": "BAAI/bge-large-en-v1.5",
            "web_hf_dataset_id": "crag-mm-2025/web-search-index-validation",
        })
    return UnifiedSearchPipeline(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=[mode.value for mode in TaskMode], required=True)
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", choices=["mlx", "vllm"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset-path", default=None, help="Optional local parquet shard for constrained local runs")
    parser.add_argument("--prepare-only", action="store_true", help="Write the fixed sample manifest without loading models")
    args = parser.parse_args()

    if args.backend:
        os.environ["CRAG_BACKEND"] = args.backend
    if args.model:
        os.environ["CRAG_MODEL"] = args.model

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "agent_trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()

    started = time.perf_counter()
    if args.dataset_path:
        dataset = load_dataset("parquet", data_files=args.dataset_path, split="train")
        dataset_source = str(Path(args.dataset_path).resolve())
        dataset_revision = "local parquet shard from " + DATASET_REVISION
    else:
        dataset = load_dataset(DATASET_ID, split="validation", revision=DATASET_REVISION)
        dataset_source = DATASET_ID
        dataset_revision = DATASET_REVISION
    indices = stratified_indices(dataset, args.num_samples, args.seed)
    selected = dataset.select(indices)
    manifest = []
    for source_index, row in zip(indices, selected):
        manifest.append({
            "source_index": source_index,
            "session_id": row["session_id"],
            "interaction_id": first_value(row["turns"], "interaction_id"),
            "query_category": first_value(row["turns"], "query_category"),
            "domain": first_value(row["turns"], "domain"),
            "query": first_value(row["turns"], "query"),
            "ground_truth": first_value(row["answers"], "ans_full"),
        })
    (output_dir / "sample_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if args.prepare_only:
        run = {
            "status": "sample_manifest_prepared",
            "dataset_source": dataset_source,
            "dataset_revision": dataset_revision,
            "seed": args.seed,
            "sample_count": len(selected),
            "source_indices": indices,
        }
        (output_dir / "run_config.json").write_text(
            json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(run, indent=2, ensure_ascii=False))
        return

    mode = TaskMode(args.mode)
    config = AgentConfig(task_mode=mode, trace_path=str(trace_path))
    pipeline = build_search_pipeline(mode)
    agent = CourseRAGAgentV2(pipeline, backend=create_backend(), config=config)
    evaluator = CRAGEvaluator(
        dataset=selected,
        agent=agent,
        eval_model_name=None,
        num_conversations=None,
        show_progress=True,
        num_workers=1,
    )
    turn_results, scores = evaluator.evaluate_agent()
    evaluator.save_results(turn_results, scores, str(output_dir))

    run = {
        "status": "completed",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode.value,
        "backend": os.getenv("CRAG_BACKEND") or ("mlx" if platform.system() == "Darwin" else "vllm"),
        "model": os.getenv("CRAG_MODEL"),
        "dataset_source": dataset_source,
        "dataset_revision": dataset_revision,
        "seed": args.seed,
        "sample_count": len(selected),
        "source_indices": indices,
        "elapsed_seconds": time.perf_counter() - started,
        "metric_note": "Semantic judging disabled; exact-match metrics are not official leaderboard scores.",
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"run": run, "scores": scores}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

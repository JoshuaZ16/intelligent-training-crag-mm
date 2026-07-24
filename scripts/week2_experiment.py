#!/usr/bin/env python3
"""Run a fixed, traceable CRAG-MM week-2 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.course_agent_v2 import AgentConfig, CourseRAGAgentV2, TaskMode, create_backend
from agents.search_compat import (
    install_cragmm_lazy_image_metadata,
    install_cragmm_lazy_web_metadata,
)
from agents.vega_config import ExperimentVariant, VegaThresholds
DATASET_ID = "crag-mm-2025/crag-mm-single-turn-public"
DATASET_REVISION = "v0.1.2"
DEFAULT_IMAGE_MODEL = "openai/clip-vit-large-patch14-336"
DEFAULT_TEXT_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_IMAGE_INDEX = "crag-mm-2025/image-search-index-validation"
DEFAULT_WEB_INDEX = "crag-mm-2025/web-search-index-validation"
IMAGE_INDEX_REVISION = "19b5f4dca7218b0231b59e2c3da74da73b6acad7"
WEB_INDEX_REVISION = "ad1614b964d62575637babb7469f8c3086adb402"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest_indices(path: str | Path) -> list[int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("manifest must be a non-empty JSON list")
    indices = []
    for row in payload:
        if not isinstance(row, dict) or not isinstance(row.get("source_index"), int):
            raise ValueError("every manifest row must have an integer source_index")
        indices.append(row["source_index"])
    if len(indices) != len(set(indices)):
        raise ValueError("manifest source_index values must be unique")
    return indices


def git_diff_sha256() -> str:
    result = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=[mode.value for mode in TaskMode], required=True)
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", choices=["mlx", "vllm"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional local parquet shard for constrained local runs",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write the fixed sample manifest without loading models",
    )
    parser.add_argument(
        "--variant",
        choices=[variant.value for variant in ExperimentVariant],
        default=ExperimentVariant.B0.value,
    )
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--thresholds-path", default=None)
    parser.add_argument("--adaptive-k", type=int, default=5)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args(argv)


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
    install_cragmm_lazy_image_metadata()
    if mode is TaskMode.TASK2:
        install_cragmm_lazy_web_metadata()
    from cragmm_search.search import UnifiedSearchPipeline

    kwargs = {
        "image_model_name": os.getenv(
            "CRAG_IMAGE_MODEL",
            DEFAULT_IMAGE_MODEL,
        ),
        "image_hf_dataset_id": os.getenv(
            "CRAG_IMAGE_INDEX",
            DEFAULT_IMAGE_INDEX,
        ),
        "image_hf_dataset_tag": os.getenv(
            "CRAG_IMAGE_INDEX_REVISION",
            IMAGE_INDEX_REVISION,
        ),
    }
    if mode is TaskMode.TASK2:
        kwargs.update({
            "text_model_name": os.getenv(
                "CRAG_TEXT_MODEL",
                DEFAULT_TEXT_MODEL,
            ),
            "web_hf_dataset_id": os.getenv(
                "CRAG_WEB_INDEX",
                DEFAULT_WEB_INDEX,
            ),
            "web_hf_dataset_tag": os.getenv(
                "CRAG_WEB_INDEX_REVISION",
                WEB_INDEX_REVISION,
            ),
        })
    return UnifiedSearchPipeline(**kwargs)


def main() -> None:
    from datasets import load_dataset
    from local_evaluation import CRAGEvaluator

    args = parse_args()

    if args.backend:
        os.environ["CRAG_BACKEND"] = args.backend
    if args.model:
        os.environ["CRAG_MODEL"] = args.model

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "agent_trace_v3.jsonl"
    if trace_path.exists():
        trace_path.unlink()

    started = time.perf_counter()
    started_at_utc = datetime.now(timezone.utc).isoformat()
    if args.dataset_path:
        dataset = load_dataset("parquet", data_files=args.dataset_path, split="train")
        dataset_source = str(Path(args.dataset_path).resolve())
        dataset_revision = "local parquet shard from " + DATASET_REVISION
    else:
        dataset = load_dataset(DATASET_ID, split="validation", revision=DATASET_REVISION)
        dataset_source = DATASET_ID
        dataset_revision = DATASET_REVISION
    if args.manifest_path:
        manifest_indices = load_manifest_indices(args.manifest_path)
        if args.num_samples > len(manifest_indices):
            raise ValueError(
                f"requested {args.num_samples} samples but manifest has "
                f"only {len(manifest_indices)}"
            )
        indices = manifest_indices[: args.num_samples]
    else:
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
    output_manifest_path = output_dir / "sample_manifest.json"
    output_manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    variant = ExperimentVariant(args.variant)
    thresholds = (
        VegaThresholds.from_json(args.thresholds_path)
        if args.thresholds_path
        else VegaThresholds()
    )
    provenance = {
        "run_id": args.run_id,
        "variant": variant.value,
        "manifest_input_sha256": (
            file_sha256(args.manifest_path) if args.manifest_path else None
        ),
        "manifest_output_sha256": file_sha256(output_manifest_path),
        "thresholds_sha256": (
            file_sha256(args.thresholds_path) if args.thresholds_path else None
        ),
        "tau_low": thresholds.tau_low,
        "tau_high": thresholds.tau_high,
        "adaptive_k": args.adaptive_k,
        "git_diff_sha256": git_diff_sha256(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "started_at_utc": started_at_utc,
    }

    if args.prepare_only:
        run = {
            "status": "sample_manifest_prepared",
            **provenance,
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
    config = AgentConfig(
        task_mode=mode,
        trace_path=str(trace_path),
        variant=variant,
        thresholds=thresholds,
        adaptive_k=args.adaptive_k,
    )
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
        **provenance,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
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

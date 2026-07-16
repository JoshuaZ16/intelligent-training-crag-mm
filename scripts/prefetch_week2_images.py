#!/usr/bin/env python3
"""Prefetch images for a fixed week-2 sample manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.image_prefetch import prefetch_image_urls
from utils import download_image_url


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset = load_dataset(
        "parquet",
        data_files=args.dataset_path,
        split="train",
    )
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    source_indices = [item["source_index"] for item in manifest]
    records = prefetch_image_urls(dataset.__getitem__, source_indices, download_image_url)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
        print(json.dumps(record, ensure_ascii=False))
    print(json.dumps({"counts": counts, "total": len(records)}, ensure_ascii=False))
    if counts.get("failed") or counts.get("missing"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

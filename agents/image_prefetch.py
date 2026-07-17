"""Prefetch fixed-sample images before allocating the generation model."""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable


def prefetch_image_urls(
    row_getter: Callable[[int], dict[str, Any]],
    source_indices: Iterable[int],
    downloader: Callable[[str], str],
) -> list[dict[str, Any]]:
    records = []
    for source_index in source_indices:
        row = row_getter(source_index)
        image_url = row.get("image_url")
        started = time.perf_counter()
        if row.get("image") is not None:
            records.append({
                "source_index": source_index,
                "image_url": image_url,
                "status": "embedded",
                "elapsed_seconds": time.perf_counter() - started,
            })
            continue
        if not image_url:
            records.append({
                "source_index": source_index,
                "image_url": None,
                "status": "missing",
                "elapsed_seconds": time.perf_counter() - started,
            })
            continue
        try:
            cache_path = downloader(image_url)
            records.append({
                "source_index": source_index,
                "image_url": image_url,
                "status": "cached",
                "cache_path": cache_path,
                "elapsed_seconds": time.perf_counter() - started,
            })
        except Exception as exc:
            records.append({
                "source_index": source_index,
                "image_url": image_url,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.perf_counter() - started,
            })
    return records

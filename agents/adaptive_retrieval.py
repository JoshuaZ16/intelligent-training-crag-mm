"""Adaptive Web-retrieval helpers for VEGA-RAG."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from agents.entity_agreement import tokens


def should_expand(
    score: float,
    tau_low: float,
    tau_high: float,
    force_all: bool = False,
) -> bool:
    return force_all or (tau_low < score < tau_high)


def rewrite_query(
    entity_name: str,
    query: str,
    max_chars: int = 240,
) -> str:
    value = " ".join([entity_name.strip(), query.strip()]).strip()
    return value[:max_chars]


def merge_web_results(
    existing: Sequence[Any],
    additional: Sequence[Any],
) -> list[Any]:
    seen: set[tuple[str, str]] = set()
    merged: list[Any] = []
    for item in [*existing, *additional]:
        url = item.metadata.get("url") or ""
        key = (
            ("url", url)
            if url
            else ("text", " ".join(tokens(item.text)))
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(
            replace(item, evidence_id=f"WEB{len(merged) + 1}")
        )
    return merged

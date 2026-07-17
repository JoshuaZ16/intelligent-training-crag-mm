"""Deterministic entity-agreement scoring for VEGA-RAG."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence


TOKEN_RE = re.compile(r"[a-z0-9]+")
NUMBER_RE = re.compile(r"(?<![a-z])\d+(?:\.\d+)?")
NUMERIC_QUERY_RE = re.compile(
    r"\b(how many|how much|what year|when|age|number|size|length|width|"
    r"height|engine|horsepower|capacity|century|date|score|population|"
    r"distance|weight|speed|duration)\b",
    re.IGNORECASE,
)


def tokens(value: str) -> list[str]:
    return TOKEN_RE.findall((value or "").lower())


def token_f1(left: str, right: str) -> float:
    left_tokens = set(tokens(left))
    right_tokens = set(tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class AgreementResult:
    items: list[Any]
    candidate_scores: list[dict[str, Any]]
    top_score: float
    second_score: float
    margin: float


def score_and_rerank(
    image_items: Sequence[Any],
    web_items: Sequence[Any],
    query: str,
) -> AgreementResult:
    web_text = " ".join(item.text for item in web_items)
    grouped: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(image_items):
        name, _, attributes = item.text.partition(" | ")
        key = name.strip().lower() or f"__unnamed_{index}"
        group = grouped.setdefault(
            key,
            {
                "index": index,
                "name": name.strip(),
                "items": [],
                "attributes": [],
                "image_score": 0.0,
            },
        )
        group["items"].append(item)
        if attributes:
            group["attributes"].append(attributes)
        group["image_score"] = max(
            group["image_score"],
            float(item.score or 0.0),
        )

    scored: list[tuple[int, dict[str, Any], float, float, float, float]] = []
    for group in grouped.values():
        image_norm = min(1.0, max(0.0, group["image_score"]))
        attributes = " | ".join(group["attributes"])
        name = group["name"]
        web_support = token_f1(name, web_text)
        attribute_support = token_f1(query, attributes)
        total = (
            0.50 * image_norm
            + 0.35 * web_support
            + 0.15 * attribute_support
        )
        scored.append(
            (
                group["index"],
                group,
                total,
                image_norm,
                web_support,
                attribute_support,
            )
        )

    scored.sort(key=lambda row: (-row[2], row[0]))
    values = [row[2] for row in scored]
    top_score = values[0] if values else 0.0
    second_score = values[1] if len(values) > 1 else 0.0
    return AgreementResult(
        items=[item for row in scored for item in row[1]["items"]],
        candidate_scores=[
            {
                "entity_name": row[1]["name"],
                "evidence_ids": [item.evidence_id for item in row[1]["items"]],
                "entity_score": row[2],
                "image_score": row[3],
                "web_support": row[4],
                "attribute_support": row[5],
            }
            for row in scored
        ],
        top_score=top_score,
        second_score=second_score,
        margin=max(0.0, top_score - second_score),
    )


def detect_numeric_conflict(
    image_texts: Sequence[str],
    web_texts: Sequence[str],
    query: str,
) -> bool:
    if not NUMERIC_QUERY_RE.search(query or "") and not NUMBER_RE.search(
        query or ""
    ):
        return False
    image_numbers = set(
        NUMBER_RE.findall(" ".join(image_texts).lower())
    )
    web_numbers = set(NUMBER_RE.findall(" ".join(web_texts).lower()))
    return bool(
        image_numbers
        and web_numbers
        and image_numbers.isdisjoint(web_numbers)
    )

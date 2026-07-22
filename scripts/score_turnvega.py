#!/usr/bin/env python3
"""Shared frozen metrics for TurnVEGA Task 2 and Task 3 experiments."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


LABELS = ("C", "P", "I", "M")


def _validated_labels(labels: Iterable[str]) -> list[str]:
    values = list(labels)
    if not values:
        raise ValueError("at least one C/P/I/M label is required")
    if any(type(label) is not str or label not in LABELS for label in values):
        raise ValueError("labels must use the exact C/P/I/M codes")
    return values


def main40_metrics(labels: Iterable[str]) -> dict[str, float | int]:
    """Return the protocol's exact C/P/I/M display metrics."""
    values = _validated_labels(labels)
    total = len(values)
    counts = {label: values.count(label) for label in LABELS}
    correct = counts["C"]
    partial = counts["P"]
    incorrect = counts["I"]
    missing_count = counts["M"]
    strict_accuracy = correct / total
    hallucination = (partial + incorrect) / total
    return {
        "N": total,
        **counts,
        "strict_accuracy": strict_accuracy,
        "partial_accuracy": (correct + 0.5 * partial) / total,
        "coverage": (correct + partial) / total,
        "missing": missing_count / total,
        "hallucination": hallucination,
        "truthfulness": ((2 * correct + missing_count) / total) - 1,
    }


def paired_transitions(
    baseline_labels: Iterable[str],
    variant_labels: Iterable[str],
) -> dict[str, float | int]:
    """Count correct-to-wrong and wrong-to-correct paired transitions."""
    baseline = _validated_labels(baseline_labels)
    variant = _validated_labels(variant_labels)
    if len(baseline) != len(variant):
        raise ValueError("paired label sequences must have equal length")
    correct_to_wrong = sum(
        left == "C" and right != "C"
        for left, right in zip(baseline, variant)
    )
    wrong_to_correct = sum(
        left != "C" and right == "C"
        for left, right in zip(baseline, variant)
    )
    net = wrong_to_correct - correct_to_wrong
    return {
        "N": len(baseline),
        "C_to_W": correct_to_wrong,
        "W_to_C": wrong_to_correct,
        "net_correct_conversion_count": net,
        "net_correct_conversion": net / len(baseline),
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def clustered_bootstrap(
    session_to_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    samples: int = 10_000,
    seed: int = 20260720,
    metric: Callable[[list[Mapping[str, Any]]], float] | None = None,
    include_samples: bool = False,
) -> dict[str, Any]:
    """Bootstrap whole sessions with replacement, never individual turns."""
    if type(samples) is not int or samples <= 0:
        raise ValueError("samples must be a positive integer")
    if not session_to_rows:
        raise ValueError("at least one session is required")
    ordered_keys = sorted(session_to_rows)
    groups = []
    for key in ordered_keys:
        rows = list(session_to_rows[key])
        if not rows:
            raise ValueError("every session must contain at least one row")
        groups.append(rows)
    if metric is None:
        metric = lambda rows: sum(row.get("label") == "C" for row in rows) / len(rows)

    observed_rows = [row for group in groups for row in group]
    observed = float(metric(observed_rows))
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        sampled_rows = []
        for _ in ordered_keys:
            sampled_rows.extend(rng.choice(groups))
        values.append(float(metric(sampled_rows)))
    result = {
        "observed": observed,
        "ci_low": _percentile(values, 0.025),
        "ci_high": _percentile(values, 0.975),
        "bootstrap_samples": samples,
        "seed": seed,
    }
    if include_samples:
        result["samples"] = values
    return result


def _net_harm(
    rows: Sequence[Mapping[str, Any]],
    needed_field: str,
) -> tuple[float | None, int]:
    eligible = [row for row in rows if row.get(needed_field) is False]
    if not eligible:
        return None, 0
    if any(row.get("baseline_label") not in LABELS for row in eligible):
        raise ValueError(
            f"{needed_field} harm requires paired baseline_label values"
        )
    transitions = paired_transitions(
        (row["baseline_label"] for row in eligible),
        (row["label"] for row in eligible),
    )
    harm = (
        transitions["C_to_W"] - transitions["W_to_C"]
    ) / len(eligible)
    return harm, len(eligible)


def task3_sequence_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score ordered Task 3 turns, including error propagation and recovery."""
    if not rows:
        raise ValueError("Task 3 rows must not be empty")
    by_session: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen = set()
    for row in rows:
        session_key = row.get("session_key")
        turn_index = row.get("turn_index")
        label = row.get("label")
        if not isinstance(session_key, str) or not session_key:
            raise ValueError("every Task 3 row needs a session_key")
        if type(turn_index) is not int or turn_index < 0:
            raise ValueError("every Task 3 row needs a non-negative turn_index")
        if type(label) is not str or label not in LABELS:
            raise ValueError("Task 3 labels must use exact C/P/I/M codes")
        identity = (session_key, turn_index)
        if identity in seen:
            raise ValueError("duplicate Task 3 session/turn identity")
        seen.add(identity)
        by_session[session_key].append(row)

    for session_key, session_rows in by_session.items():
        session_rows.sort(key=lambda row: row["turn_index"])
        if [row["turn_index"] for row in session_rows] != list(
            range(len(session_rows))
        ):
            raise ValueError(f"Task 3 turns are not contiguous: {session_key}")

    total = len(rows)
    correct = sum(row["label"] == "C" for row in rows)
    whole_correct = sum(
        all(row["label"] == "C" for row in session_rows)
        for session_rows in by_session.values()
    )
    per_turn = []
    for turn_index in sorted({row["turn_index"] for row in rows}):
        turn_rows = [row for row in rows if row["turn_index"] == turn_index]
        turn_correct = sum(row["label"] == "C" for row in turn_rows)
        per_turn.append(
            {
                "turn_index": turn_index,
                "correct": turn_correct,
                "total": len(turn_rows),
                "accuracy": turn_correct / len(turn_rows),
            }
        )

    after_error = []
    after_correct = []
    for session_rows in by_session.values():
        for previous, current in zip(session_rows, session_rows[1:]):
            current_is_error = current["label"] != "C"
            if previous["label"] == "C":
                after_correct.append(current_is_error)
            else:
                after_error.append(current_is_error)
    error_after_error = (
        sum(after_error) / len(after_error) if after_error else None
    )
    error_after_correct = (
        sum(after_correct) / len(after_correct) if after_correct else None
    )
    epc = (
        error_after_error - error_after_correct
        if error_after_error is not None and error_after_correct is not None
        else None
    )
    recovery_at_1 = (
        sum(not value for value in after_error) / len(after_error)
        if after_error
        else None
    )
    history_harm, history_denominator = _net_harm(rows, "history_needed")
    image_harm, image_denominator = _net_harm(rows, "image_needed")
    return {
        "turn_count": total,
        "conversation_count": len(by_session),
        "per_turn_accuracy": correct / total,
        "per_turn": per_turn,
        "average_successful_turns": correct / len(by_session),
        "whole_conversation_accuracy": whole_correct / len(by_session),
        "error_after_error": error_after_error,
        "error_after_correct": error_after_correct,
        "epc": epc,
        "recovery_at_1": recovery_at_1,
        "history_harm": history_harm,
        "history_harm_denominator": history_denominator,
        "image_harm": image_harm,
        "image_harm_denominator": image_denominator,
    }

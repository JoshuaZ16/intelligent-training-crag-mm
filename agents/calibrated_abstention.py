"""Deterministic answer/expand/abstain policy for VEGA-RAG."""

from __future__ import annotations

from enum import Enum

from agents.vega_config import ExperimentVariant, VegaThresholds


class GateAction(str, Enum):
    ANSWER = "answer"
    EXPAND = "expand"
    ABSTAIN = "abstain"


def choose_action(
    variant: ExperimentVariant,
    score: float,
    conflict: bool,
    thresholds: VegaThresholds,
) -> GateAction:
    if variant is ExperimentVariant.A3:
        if conflict or score <= thresholds.tau_low:
            return GateAction.ABSTAIN
        return GateAction.ANSWER

    if variant is ExperimentVariant.FULL:
        if conflict or score <= thresholds.tau_low:
            return GateAction.ABSTAIN
        if score < thresholds.tau_high:
            return GateAction.EXPAND

    if variant is ExperimentVariant.A2:
        if thresholds.tau_low < score < thresholds.tau_high:
            return GateAction.EXPAND
        return GateAction.ANSWER

    if variant is ExperimentVariant.A2_ENRICH_ALL:
        if score <= thresholds.tau_low:
            return GateAction.ABSTAIN
        return GateAction.EXPAND

    return GateAction.ANSWER

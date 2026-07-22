"""Shared experiment configuration for TurnVEGA Task 2 and Task 3 runs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TurnVegaVariant(str, Enum):
    T2_B0 = "t2_b0"
    T2_BUDGET_B0 = "t2_budget_b0"
    T2_CANDIDATE_GRID = "t2_candidate_grid"
    T2_RELATION_GRID = "t2_relation_grid"
    T2_CIRCULARITY = "t2_circularity"
    T2_ANSWERABILITY = "t2_answerability"
    T2_EVIDENCE_CARD = "t2_evidence_card"
    T2_TYPED_REPAIR = "t2_typed_repair"
    T2_CORE_FULL = "t2_core_full"
    T3_NO_HISTORY = "t3_no_history"
    T3_FULL_HISTORY = "t3_full_history"
    T3_LAST_TURN = "t3_last_turn"
    T3_USER_ONLY = "t3_user_only"
    T3_STRUCTURED_STATE = "t3_structured_state"
    T3_VERIFIED_STATE = "t3_verified_state"
    T3_STATE_GATED = "t3_state_gated"
    T3_CORE_FULL = "t3_core_full"


class DatasetKind(str, Enum):
    TASK2 = "task2"
    TASK3 = "task3"


@dataclass(frozen=True)
class ExperimentBudget:
    candidate_passages: int
    prompt_evidence: int
    max_search_calls: int

    def __post_init__(self) -> None:
        if self.candidate_passages <= 0:
            raise ValueError("candidate_passages must be positive")
        if self.prompt_evidence <= 0:
            raise ValueError("prompt_evidence must be positive")
        if self.max_search_calls <= 0:
            raise ValueError("max_search_calls must be positive")
        if self.prompt_evidence > self.candidate_passages:
            raise ValueError(
                "prompt_evidence must not exceed candidate_passages"
            )

"""Configuration primitives for reproducible VEGA-RAG ablations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ExperimentVariant(str, Enum):
    B0 = "b0"
    A1 = "a1"
    A2 = "a2"
    A3 = "a3"
    FULL = "full"
    A2_ENRICH_ALL = "a2_enrich_all"


@dataclass(frozen=True)
class VegaThresholds:
    tau_low: float = 0.3
    tau_high: float = 0.6

    def __post_init__(self) -> None:
        if not (0.0 <= self.tau_low < self.tau_high <= 1.0):
            raise ValueError(
                "thresholds must satisfy 0 <= tau_low < tau_high <= 1"
            )

    @classmethod
    def from_json(cls, path: str | Path) -> "VegaThresholds":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            tau_low=float(payload["tau_low"]),
            tau_high=float(payload["tau_high"]),
        )

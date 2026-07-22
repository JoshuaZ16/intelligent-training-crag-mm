import dataclasses
import hashlib
import json
import math


@dataclasses.dataclass(frozen=True)
class QuestionFrame:
    target_relation: str
    answer_type: str
    temporal_intent: bool
    relation_terms: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation_terms", tuple(self.relation_terms))


@dataclasses.dataclass(frozen=True)
class EntityCandidate:
    candidate_id: str
    entity_name: str
    aliases: tuple[str, ...] = ()
    image_score: float = 0.0
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        image_score = float(self.image_score)
        if not math.isfinite(image_score):
            raise ValueError("image_score must be finite")
        if image_score == 0.0:
            image_score = 0.0
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(self, "image_score", image_score)
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))


@dataclasses.dataclass(frozen=True)
class AtomicEvidence:
    evidence_id: str
    source: str
    source_cluster: str
    entity: str
    predicate: str
    value: str
    unit: str
    valid_time: str
    text: str
    relevance: float

    def __post_init__(self) -> None:
        relevance = float(self.relevance)
        if not math.isfinite(relevance) or not 0.0 <= relevance <= 1.0:
            raise ValueError("relevance must be in the closed interval [0, 1]")
        if relevance == 0.0:
            relevance = 0.0
        object.__setattr__(self, "relevance", relevance)


@dataclasses.dataclass(frozen=True)
class EvidencePacket:
    question: str
    question_frame: QuestionFrame
    entity_candidates: tuple[EntityCandidate, ...]
    evidence: tuple[AtomicEvidence, ...]
    conflicts: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "entity_candidates",
            tuple(self.entity_candidates),
        )
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "conflicts", tuple(self.conflicts))

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)

    def sha256(self) -> str:
        canonical_json = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

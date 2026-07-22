"""Evidence-traceable CRAG-MM agents for the first two course weeks.

The retrieval and prompt-building code is backend agnostic.  The official
submission can use vLLM while Apple Silicon development can use MLX-VLM.
"""

from __future__ import annotations

import html
import json
import logging
import os
import platform
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Protocol, Sequence

from PIL import Image

from agents.adaptive_retrieval import merge_web_results, rewrite_query
from agents.calibrated_abstention import GateAction, choose_action
from agents.entity_agreement import detect_numeric_conflict, score_and_rerank
from agents.turnvega_config import TurnVegaVariant
from agents.vega_config import ExperimentVariant, VegaThresholds

try:
    from agents.base_agent import BaseAgent
except ModuleNotFoundError:
    class BaseAgent:  # pragma: no cover - permits dependency-light unit tests.
        def __init__(self, search_pipeline: Any):
            self.search_pipeline = search_pipeline


LOGGER = logging.getLogger(__name__)
IDK_RESPONSE = "I don't know"
DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
DEFAULT_MLX_MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
MAX_ANSWER_TOKENS = 75


class TaskMode(str, Enum):
    VISION = "vision"
    TASK1 = "task1"
    TASK2 = "task2"


class RefusalReason(str, Enum):
    NO_RETRIEVAL = "no_retrieval"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    SEARCH_ERROR = "search_error"
    EMPTY_GENERATION = "empty_generation"


@dataclass(frozen=True)
class AgentConfig:
    task_mode: TaskMode = TaskMode.TASK1
    batch_size: int = 4
    image_results: int = 3
    web_results: int = 3
    max_image_fields: int = 10
    max_evidence_chars: int = 5500
    max_answer_tokens: int = MAX_ANSWER_TOKENS
    web_score_threshold: float = 0.20
    trace_path: str | None = None
    variant: ExperimentVariant | TurnVegaVariant = ExperimentVariant.B0
    thresholds: VegaThresholds = field(default_factory=VegaThresholds)
    adaptive_k: int = 5
    dataset_kind: str = ""
    trace_schema: str = "v3"
    history_mode: str = ""
    candidate_passages: int | None = None
    prompt_evidence: int | None = None
    max_search_calls: int | None = None
    run_id: str = ""
    seed: int = 20260720
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "AgentConfig":
        raw_mode = os.getenv("CRAG_TASK_MODE", TaskMode.TASK1.value).lower()
        try:
            mode = TaskMode(raw_mode)
        except ValueError as exc:
            raise ValueError(f"Unsupported CRAG_TASK_MODE={raw_mode!r}") from exc
        raw_variant = os.getenv(
            "CRAG_EXPERIMENT_VARIANT",
            ExperimentVariant.B0.value,
        ).lower()
        try:
            variant: ExperimentVariant | TurnVegaVariant = ExperimentVariant(
                raw_variant
            )
        except ValueError:
            variant = TurnVegaVariant(raw_variant)
        return cls(
            task_mode=mode,
            batch_size=int(os.getenv("CRAG_BATCH_SIZE", "4")),
            trace_path=os.getenv("CRAG_TRACE_PATH") or None,
            variant=variant,
            thresholds=VegaThresholds(
                tau_low=float(os.getenv("CRAG_TAU_LOW", "0.3")),
                tau_high=float(os.getenv("CRAG_TAU_HIGH", "0.6")),
            ),
            adaptive_k=int(os.getenv("CRAG_ADAPTIVE_K", "5")),
            dataset_kind=os.getenv("CRAG_DATASET_KIND", ""),
            trace_schema=os.getenv("CRAG_TRACE_SCHEMA", "v3"),
            history_mode=os.getenv("CRAG_HISTORY_MODE", ""),
            run_id=os.getenv("CRAG_RUN_ID", ""),
            seed=int(os.getenv("CRAG_SEED", "20260720")),
            temperature=float(os.getenv("CRAG_TEMPERATURE", "0")),
        )


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    source: str
    text: str
    score: float | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnTrace:
    agent_version: str
    task_mode: str
    query: str
    search_query: str = ""
    image_evidence: List[Dict[str, Any]] = field(default_factory=list)
    web_evidence: List[Dict[str, Any]] = field(default_factory=list)
    selected_evidence_ids: List[str] = field(default_factory=list)
    answer: str = ""
    refusal_reason: str | None = None
    image_search_ms: float = 0.0
    web_search_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    answer_token_count: int = 0
    status: str = "ok"
    errors: List[str] = field(default_factory=list)
    trace_schema: str = "v3"
    experiment_variant: str = ExperimentVariant.B0.value
    entity_candidates: List[Dict[str, Any]] = field(default_factory=list)
    entity_agreement_before: float = 0.0
    entity_agreement: float = 0.0
    entity_margin: float = 0.0
    evidence_conflict: bool = False
    gate_action: str = GateAction.ANSWER.value
    additional_search_query: str = ""
    additional_web_evidence: List[Dict[str, Any]] = field(default_factory=list)
    additional_web_search_ms: float = 0.0
    search_call_count: int = 0
    run_id: str = ""
    interaction_id: str = ""
    dataset_kind: str = ""
    session_key: str = ""
    turn_index: int = 0
    history_mode: str = ""
    question_frame: Dict[str, Any] = field(default_factory=dict)
    image_needed: bool = False
    history_needed: bool = False
    web_needed: bool = False
    entity_candidates_before: List[Dict[str, Any]] = field(default_factory=list)
    entity_candidates_after: List[Dict[str, Any]] = field(default_factory=list)
    candidate_queries: List[str] = field(default_factory=list)
    query_family: str = ""
    candidate_budget: int = 0
    atomic_evidence: List[Dict[str, Any]] = field(default_factory=list)
    source_clusters: List[Dict[str, Any]] = field(default_factory=list)
    relation_coverage: Dict[str, Any] = field(default_factory=dict)
    circularity_flags: List[str] = field(default_factory=list)
    answerability_scores: Dict[str, Any] = field(default_factory=dict)
    typed_conflicts: List[Dict[str, Any]] = field(default_factory=list)
    evidence_packet: Dict[str, Any] = field(default_factory=dict)
    evidence_packet_sha256: str = ""
    evidence_token_count: int = 0
    memory_state_before: Dict[str, Any] = field(default_factory=dict)
    memory_state_after: Dict[str, Any] = field(default_factory=dict)
    provisional_claims: List[Dict[str, Any]] = field(default_factory=list)
    verified_claims: List[Dict[str, Any]] = field(default_factory=list)
    quarantined_claims: List[Dict[str, Any]] = field(default_factory=list)
    state_version: int = 0


@dataclass
class PreparedTurn:
    prompt: str
    trace: TurnTrace
    forced_answer: str | None = None


class GenerationBackend(Protocol):
    def answer_batch(
        self,
        prompts: Sequence[str],
        images: Sequence[Image.Image],
    ) -> List[str]: ...

    def truncate(self, text: str, max_tokens: int) -> str: ...

    def count_tokens(self, text: str) -> int: ...


def clean_markup(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[([^|\]]*\|)?([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\{\{URL\|([^{}]+)\}\}", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\{\{convert\|([^|{}]+)\|([^|{}]+)[^{}]*\}\}", r"\1 \2", text, flags=re.IGNORECASE)
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,;|")


def terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", clean_markup(text).lower())
        if len(token) > 1 and token not in {"the", "is", "a", "an", "of", "in", "on", "to", "what", "which", "who", "how"}
    }


def _relevance(query_terms: set[str], *parts: str) -> float:
    candidate_terms = terms(" ".join(parts))
    if not candidate_terms:
        return 0.0
    overlap = len(query_terms & candidate_terms)
    return overlap + overlap / max(len(query_terms), 1)


def parse_image_evidence(
    results: Sequence[Dict[str, Any]] | None,
    query: str,
    max_items: int = 3,
    max_fields: int = 10,
) -> List[EvidenceItem]:
    query_terms = terms(query)
    candidates: List[tuple[float, float, str, str, Dict[str, Any]]] = []
    for result_index, result in enumerate(results or []):
        retrieval_score = float(result.get("score") or 0.0)
        for entity in result.get("entities") or []:
            entity_name = clean_markup(entity.get("entity_name"))
            attrs = entity.get("entity_attributes") or {}
            for field_name, value in attrs.items():
                field = clean_markup(field_name)
                value_text = clean_markup(value)
                if not field or not value_text:
                    continue
                relevance = _relevance(query_terms, entity_name, field, value_text)
                candidates.append((relevance, retrieval_score, entity_name, f"{field}: {value_text}", {"result_index": result_index}))

    candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
    selected: List[EvidenceItem] = []
    seen: set[tuple[str, str]] = set()
    for relevance, retrieval_score, entity_name, attribute, metadata in candidates:
        key = (entity_name.lower(), attribute.lower())
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            EvidenceItem(
                evidence_id=f"KG{len(selected) + 1}",
                source="image_kg",
                text=f"{entity_name} | {attribute}" if entity_name else attribute,
                score=retrieval_score,
                metadata={**metadata, "query_relevance": relevance},
            )
        )
        if len(selected) >= min(max_fields, max_items * max_fields):
            break
    return selected


def parse_web_evidence(
    results: Sequence[Dict[str, Any]] | None,
    query: str,
    max_items: int = 3,
    score_threshold: float = 0.20,
) -> List[EvidenceItem]:
    query_terms = terms(query)
    ranked: List[tuple[float, float, str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for result in results or []:
        title = clean_markup(result.get("page_name")) or "Untitled page"
        snippet = clean_markup(result.get("page_snippet"))
        url = clean_markup(result.get("page_url"))
        score = float(result.get("score") or 0.0)
        dedupe_key = (url.lower(), snippet.lower())
        if not snippet or dedupe_key in seen or score < score_threshold:
            continue
        seen.add(dedupe_key)
        relevance = _relevance(query_terms, title, snippet)
        ranked.append((relevance, score, title, snippet, url))
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [
        EvidenceItem(
            evidence_id=f"WEB{index}",
            source="web",
            text=f"{title}: {snippet}",
            score=score,
            metadata={"url": url, "query_relevance": relevance},
        )
        for index, (relevance, score, title, snippet, url) in enumerate(ranked[:max_items], start=1)
    ]


def build_search_query(query: str, image_evidence: Sequence[EvidenceItem], max_chars: int = 240) -> str:
    entity_names: List[str] = []
    for item in image_evidence:
        name = item.text.split(" | ", 1)[0].strip()
        if name and name.lower() not in {value.lower() for value in entity_names}:
            entity_names.append(name)
        if len(entity_names) >= 2:
            break
    return clean_markup(" ".join([*entity_names, query]))[:max_chars]


def format_evidence(items: Sequence[EvidenceItem], max_chars: int) -> str:
    blocks: List[str] = []
    used = 0
    for item in items:
        score = f" score={item.score:.3f}" if item.score is not None else ""
        line = f"[{item.evidence_id}{score}] {item.text}"
        if used + len(line) > max_chars:
            break
        blocks.append(line)
        used += len(line)
    return "\n".join(blocks)


def build_prompt(
    query: str,
    mode: TaskMode,
    image_evidence: Sequence[EvidenceItem],
    web_evidence: Sequence[EvidenceItem],
    max_evidence_chars: int,
) -> str:
    if mode is TaskMode.VISION:
        evidence_section = "No external retrieval evidence is available. Answer only from the current image."
    else:
        kg = format_evidence(image_evidence, max_evidence_chars)
        web = format_evidence(web_evidence, max_evidence_chars - len(kg))
        evidence_section = f"IMAGE KG EVIDENCE\n{kg or '(none)'}"
        if mode is TaskMode.TASK2:
            evidence_section += f"\n\nWEB EVIDENCE\n{web or '(none)'}"

    return (
        "You are answering a factual CRAG-MM visual question.\n"
        "Rules:\n"
        "1. The current image is direct visual evidence. Similar-image KG records are external evidence, not proof that every visible detail is identical.\n"
        "2. Use only evidence that directly supports the question. Ignore irrelevant fields and web noise.\n"
        "3. If required evidence is absent or sources materially conflict, answer exactly: I don't know\n"
        "4. Return only the concise final answer, with no explanation and no citations.\n\n"
        f"{evidence_section}\n\nQUESTION\n{clean_markup(query)}"
    )


def normalize_answer(text: str) -> tuple[str, RefusalReason | None]:
    answer = clean_markup(text)
    if not answer:
        return IDK_RESPONSE, RefusalReason.EMPTY_GENERATION
    if re.search(r"\bi\s+(?:do\s+not|don't)\s+know\b", answer, re.IGNORECASE):
        return IDK_RESPONSE, RefusalReason.INSUFFICIENT_EVIDENCE
    return answer, None


class VllmBackend:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        seed: int = 20260720,
        temperature: float = 0.0,
    ):
        import vllm

        self.vllm = vllm
        self.seed = seed
        self.temperature = temperature
        self.llm = vllm.LLM(
            model_name,
            seed=seed,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            max_model_len=8192,
            max_num_seqs=4,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=True,
            limit_mm_per_prompt={"image": 1},
        )
        self.tokenizer = self.llm.get_tokenizer()

    def answer_batch(self, prompts: Sequence[str], images: Sequence[Image.Image]) -> List[str]:
        inputs = []
        for prompt, image in zip(prompts, images):
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
            ]
            formatted = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs.append({"prompt": formatted, "multi_modal_data": {"image": image}})
        outputs = self.llm.generate(
            inputs,
            sampling_params=self.vllm.SamplingParams(
                temperature=self.temperature,
                max_tokens=MAX_ANSWER_TOKENS,
                seed=self.seed,
            ),
        )
        return [output.outputs[0].text for output in outputs]

    def truncate(self, text: str, max_tokens: int) -> str:
        token_ids = self.tokenizer.encode(text)[:max_tokens]
        return self.tokenizer.decode(token_ids).strip()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))


class MlxVlmBackend:
    """Local Apple Silicon backend.  Import is lazy to keep official images clean."""

    def __init__(self, model_name: str = DEFAULT_MLX_MODEL):
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template

        self._generate = generate
        self._apply_chat_template = apply_chat_template
        self.model, self.processor = load(model_name)
        self.config = self.model.config
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

    def answer_batch(self, prompts: Sequence[str], images: Sequence[Image.Image]) -> List[str]:
        answers = []
        for prompt, image in zip(prompts, images):
            formatted = self._apply_chat_template(self.processor, self.config, prompt, num_images=1)
            result = self._generate(
                self.model,
                self.processor,
                formatted,
                [image.convert("RGB")],
                max_tokens=MAX_ANSWER_TOKENS,
                temperature=0.0,
                verbose=False,
            )
            answers.append(getattr(result, "text", str(result)))
        return answers

    def truncate(self, text: str, max_tokens: int) -> str:
        token_ids = self.tokenizer.encode(text)[:max_tokens]
        return self.tokenizer.decode(token_ids).strip()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))


def create_backend(config: AgentConfig | None = None) -> GenerationBackend:
    backend_name = os.getenv("CRAG_BACKEND", "mlx" if platform.system() == "Darwin" else "vllm").lower()
    if backend_name == "mlx":
        return MlxVlmBackend(os.getenv("CRAG_MODEL", DEFAULT_MLX_MODEL))
    if backend_name == "vllm":
        return VllmBackend(
            os.getenv("CRAG_MODEL", DEFAULT_MODEL),
            seed=config.seed if config else 20260720,
            temperature=config.temperature if config else 0.0,
        )
    raise ValueError(f"Unsupported CRAG_BACKEND={backend_name!r}")


class CourseRAGAgentV2(BaseAgent):
    VERSION = "week2-v2"
    _trace_lock = threading.Lock()

    def __init__(
        self,
        search_pipeline: Any,
        backend: GenerationBackend | None = None,
        config: AgentConfig | None = None,
        load_backend: bool = True,
        trace_identity_provider: Callable[
            [str, Image.Image, Sequence[Dict[str, Any]]],
            Mapping[str, Any],
        ] | None = None,
    ):
        super().__init__(search_pipeline)
        self.config = config or AgentConfig.from_env()
        self.backend = backend if backend is not None else (create_backend() if load_backend else None)
        self.trace_identity_provider = trace_identity_provider

    def get_batch_size(self) -> int:
        return self.config.batch_size

    def _search(self, value: Any, k: int, source: str, trace: TurnTrace) -> List[Dict[str, Any]]:
        started = time.perf_counter()
        if (
            self.config.max_search_calls is not None
            and trace.search_call_count >= self.config.max_search_calls
        ):
            trace.errors.append(
                "search call budget exhausted before " + source
            )
            trace.status = "search_budget_exhausted"
            return []
        trace.search_call_count += 1
        try:
            if self.search_pipeline is None:
                raise RuntimeError("search pipeline is not configured")
            return list(self.search_pipeline(value, k=k) or [])
        except Exception as exc:
            message = f"{source} search failed: {type(exc).__name__}: {exc}"
            LOGGER.exception(message)
            trace.errors.append(message)
            trace.status = "search_error"
            return []
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            if source == "image":
                trace.image_search_ms = elapsed
            elif source == "web_extra":
                trace.additional_web_search_ms += elapsed
            else:
                trace.web_search_ms = elapsed

    def _write_trace(self, trace: TurnTrace) -> None:
        if not self.config.trace_path:
            return
        path = Path(self.config.trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._trace_lock, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(trace), ensure_ascii=False) + "\n")

    def _prepare_turn(
        self,
        query: str,
        image: Image.Image,
        message_history: Sequence[Dict[str, Any]] = (),
    ) -> PreparedTurn:
        trace_identity: Mapping[str, Any] = {}
        if self.trace_identity_provider is not None:
            take = getattr(self.trace_identity_provider, "take", None)
            if callable(take):
                trace_identity = take(query, message_history)
            else:
                trace_identity = self.trace_identity_provider(
                    query,
                    image,
                    message_history,
                )
            if not isinstance(trace_identity, Mapping):
                raise TypeError("trace_identity_provider must return a mapping")
        turn_index = trace_identity.get("turn_index", 0)
        if type(turn_index) is not int:
            raise ValueError("trace turn_index must be an integer")
        trace = TurnTrace(
            agent_version=self.VERSION,
            task_mode=self.config.task_mode.value,
            query=query,
            trace_schema=self.config.trace_schema,
            experiment_variant=self.config.variant.value,
            run_id=self.config.run_id,
            interaction_id=str(trace_identity.get("interaction_id") or ""),
            dataset_kind=self.config.dataset_kind,
            session_key=str(
                trace_identity.get("session_key")
                or trace_identity.get("session_id")
                or ""
            ),
            turn_index=turn_index,
            history_mode=self.config.history_mode,
            candidate_budget=self.config.candidate_passages or 0,
        )
        if self.config.task_mode is TaskMode.VISION:
            prompt = build_prompt(query, self.config.task_mode, [], [], self.config.max_evidence_chars)
            return PreparedTurn(prompt=prompt, trace=trace)

        raw_image = self._search(image, self.config.image_results, "image", trace)
        image_evidence = parse_image_evidence(
            raw_image,
            query,
            max_items=self.config.image_results,
            max_fields=self.config.max_image_fields,
        )
        web_evidence: List[EvidenceItem] = []
        if self.config.task_mode is TaskMode.TASK2:
            trace.search_query = build_search_query(query, image_evidence)
            candidate_passages = (
                self.config.candidate_passages or self.config.web_results
            )
            prompt_evidence = (
                self.config.prompt_evidence or self.config.web_results
            )
            raw_web = self._search(
                trace.search_query,
                candidate_passages,
                "web",
                trace,
            )
            web_evidence = parse_web_evidence(
                raw_web,
                query,
                max_items=prompt_evidence,
                score_threshold=self.config.web_score_threshold,
            )

        prompt_image_evidence = image_evidence
        forced_answer = None
        if (
            self.config.task_mode is TaskMode.TASK2
            and isinstance(self.config.variant, ExperimentVariant)
        ):
            agreement = score_and_rerank(image_evidence, web_evidence, query)
            conflict = detect_numeric_conflict(
                [item.text for item in image_evidence],
                [item.text for item in web_evidence],
                query,
            )
            trace.entity_candidates = agreement.candidate_scores
            trace.entity_agreement_before = agreement.top_score
            trace.entity_agreement = agreement.top_score
            trace.entity_margin = agreement.margin
            trace.evidence_conflict = conflict
            action = choose_action(
                self.config.variant,
                agreement.top_score,
                conflict,
                self.config.thresholds,
            )
            trace.gate_action = action.value

            if self.config.variant is not ExperimentVariant.B0:
                prompt_image_evidence = agreement.items

            if action is GateAction.EXPAND and agreement.candidate_scores:
                entity_name = str(
                    agreement.candidate_scores[0]["entity_name"]
                )
                trace.additional_search_query = rewrite_query(
                    entity_name,
                    query,
                )
                raw_additional_web = self._search(
                    trace.additional_search_query,
                    self.config.adaptive_k,
                    "web_extra",
                    trace,
                )
                additional_web = parse_web_evidence(
                    raw_additional_web,
                    query,
                    max_items=self.config.adaptive_k,
                    score_threshold=self.config.web_score_threshold,
                )
                trace.additional_web_evidence = [
                    asdict(item) for item in additional_web
                ]
                web_evidence = merge_web_results(
                    web_evidence,
                    additional_web,
                )
                agreement = score_and_rerank(
                    image_evidence,
                    web_evidence,
                    query,
                )
                prompt_image_evidence = agreement.items
                trace.entity_candidates = agreement.candidate_scores
                trace.entity_agreement = agreement.top_score
                trace.entity_margin = agreement.margin

            if action is GateAction.ABSTAIN:
                forced_answer = IDK_RESPONSE
                trace.refusal_reason = (
                    RefusalReason.CONFLICTING_EVIDENCE.value
                    if conflict
                    else RefusalReason.INSUFFICIENT_EVIDENCE.value
                )

        trace.image_evidence = [
            asdict(item) for item in prompt_image_evidence
        ]
        trace.web_evidence = [asdict(item) for item in web_evidence]
        trace.selected_evidence_ids = [
            item.evidence_id
            for item in [*prompt_image_evidence, *web_evidence]
        ]
        prompt = build_prompt(
            query,
            self.config.task_mode,
            prompt_image_evidence,
            web_evidence,
            self.config.max_evidence_chars,
        )
        return PreparedTurn(
            prompt=prompt,
            trace=trace,
            forced_answer=forced_answer,
        )

    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        if not (len(queries) == len(images) == len(message_histories)):
            raise ValueError("queries, images, and message_histories must have identical lengths")
        if self.backend is None:
            raise RuntimeError("generation backend is not loaded")

        started = time.perf_counter()
        prepared = [
            self._prepare_turn(query, image, message_history)
            for query, image, message_history in zip(
                queries,
                images,
                message_histories,
            )
        ]
        traces = [item.trace for item in prepared]
        active_indices = [
            index
            for index, item in enumerate(prepared)
            if item.forced_answer is None
        ]
        raw_answers = [
            item.forced_answer or IDK_RESPONSE for item in prepared
        ]
        generation_started = time.perf_counter()
        if active_indices:
            active_prompts = [prepared[index].prompt for index in active_indices]
            active_images = [images[index] for index in active_indices]
            try:
                generated = self.backend.answer_batch(
                    active_prompts,
                    active_images,
                )
            except Exception as exc:
                generated = [IDK_RESPONSE] * len(active_indices)
                for index in active_indices:
                    traces[index].status = "generation_error"
                    traces[index].errors.append(
                        f"generation failed: {type(exc).__name__}: {exc}"
                    )
            if len(generated) != len(active_indices):
                raise RuntimeError(
                    f"backend returned {len(generated)} answers for "
                    f"{len(active_indices)} active queries"
                )
            for index, answer in zip(active_indices, generated):
                raw_answers[index] = answer
        generation_ms = (time.perf_counter() - generation_started) * 1000

        answers: List[str] = []
        per_turn_generation = generation_ms / max(len(active_indices), 1)
        per_turn_total = (time.perf_counter() - started) * 1000 / max(len(queries), 1)
        for index, (raw_answer, trace) in enumerate(zip(raw_answers, traces)):
            answer, refusal = normalize_answer(raw_answer)
            truncated = (
                self.backend.truncate(
                    answer,
                    self.config.max_answer_tokens,
                )
                or IDK_RESPONSE
            )
            answer, truncated_refusal = normalize_answer(truncated)
            refusal = refusal or truncated_refusal
            trace.answer = answer
            if answer == IDK_RESPONSE:
                if trace.status == "generation_error":
                    trace.refusal_reason = "generation_error"
                elif trace.errors:
                    trace.refusal_reason = RefusalReason.SEARCH_ERROR.value
                elif trace.refusal_reason:
                    pass
                elif not trace.image_evidence and not trace.web_evidence and self.config.task_mode is not TaskMode.VISION:
                    trace.refusal_reason = RefusalReason.NO_RETRIEVAL.value
                else:
                    trace.refusal_reason = (refusal or RefusalReason.INSUFFICIENT_EVIDENCE).value
            trace.generation_ms = (
                per_turn_generation if index in active_indices else 0.0
            )
            trace.total_ms = per_turn_total
            trace.answer_token_count = self.backend.count_tokens(answer)
            self._write_trace(trace)
            answers.append(answer)
        return answers


class VisionBaselineAgent(CourseRAGAgentV2):
    def __init__(self, search_pipeline: Any, **kwargs: Any):
        config = kwargs.pop("config", AgentConfig(task_mode=TaskMode.VISION))
        super().__init__(search_pipeline, config=config, **kwargs)


class Task1Agent(CourseRAGAgentV2):
    def __init__(self, search_pipeline: Any, **kwargs: Any):
        config = kwargs.pop("config", AgentConfig(task_mode=TaskMode.TASK1))
        super().__init__(search_pipeline, config=config, **kwargs)


class Task2Agent(CourseRAGAgentV2):
    def __init__(self, search_pipeline: Any, **kwargs: Any):
        config = kwargs.pop("config", AgentConfig(task_mode=TaskMode.TASK2))
        super().__init__(search_pipeline, config=config, **kwargs)

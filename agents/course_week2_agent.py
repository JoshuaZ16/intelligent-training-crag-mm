import re
from typing import Any, Dict, Iterable, List

from PIL import Image

try:
    from agents.base_agent import BaseAgent
except ModuleNotFoundError:
    class BaseAgent:  # pragma: no cover - only used when optional CRAG deps are absent locally.
        def __init__(self, search_pipeline: Any):
            self.search_pipeline = search_pipeline


AICROWD_SUBMISSION_BATCH_SIZE = 4
MAX_GENERATION_TOKENS = 75
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 2
NUM_IMAGE_RESULTS = 3
NUM_WEB_RESULTS = 3
VLLM_GPU_MEMORY_UTILIZATION = 0.85
VLLM_TENSOR_PARALLEL_SIZE = 1


def clean_markup(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"\[\[([^|\]]*\|)?([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\{\{[^{}]*\|([^{}|]+)\}\}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,;")


def build_search_query(
    query: str,
    image_summary: str = "",
    message_history: List[Dict[str, Any]] | None = None,
    max_chars: int = 280,
) -> str:
    history_terms: List[str] = []
    for message in reversed(message_history or []):
        if message.get("role") in {"user", "assistant"} and message.get("content"):
            history_terms.append(str(message["content"]))
        if len(history_terms) >= 2:
            break

    parts = [query, image_summary, *reversed(history_terms)]
    search_query = " ".join(clean_markup(part) for part in parts if clean_markup(part))
    return search_query[:max_chars]


def _iter_entity_lines(result: Dict[str, Any], max_fields: int) -> Iterable[str]:
    for entity in result.get("entities") or []:
        entity_name = clean_markup(entity.get("entity_name"))
        attrs = entity.get("entity_attributes") or {}
        fields = []
        for key, value in attrs.items():
            value_text = clean_markup(value)
            if value_text:
                fields.append(f"{clean_markup(key)}: {value_text}")
            if len(fields) >= max_fields:
                break
        if entity_name or fields:
            yield "; ".join([part for part in [entity_name, *fields] if part])


def format_image_evidence(
    image_results: List[Dict[str, Any]] | None,
    max_items: int = NUM_IMAGE_RESULTS,
    max_fields: int = 8,
) -> str:
    blocks = []
    for idx, result in enumerate((image_results or [])[:max_items], start=1):
        lines = list(_iter_entity_lines(result, max_fields=max_fields))
        if lines:
            score = result.get("score")
            score_text = f", score={score:.3f}" if isinstance(score, float) else ""
            blocks.append(f"[Image KG {idx}{score_text}] " + " | ".join(lines))
    return "\n".join(blocks)


def format_web_evidence(
    web_results: List[Dict[str, Any]] | None,
    max_items: int = NUM_WEB_RESULTS,
) -> str:
    blocks = []
    for idx, result in enumerate((web_results or [])[:max_items], start=1):
        snippet = clean_markup(result.get("page_snippet"))
        if not snippet:
            continue
        title = clean_markup(result.get("page_name")) or "Untitled page"
        url = clean_markup(result.get("page_url"))
        score = result.get("score")
        score_text = f", score={score:.3f}" if isinstance(score, float) else ""
        source = f"{title} ({url})" if url else title
        blocks.append(f"[Web {idx}{score_text}] {source}: {snippet}")
    return "\n".join(blocks)


class CourseWeek2RAGAgent(BaseAgent):
    """
    Week-2 course agent: image KG for Task 1, image KG + web snippets for Task 2.

    The implementation keeps model usage close to the starter-kit vLLM agents, while
    making the retrieval evidence explicit and separable for report/debugging needs.
    """

    def __init__(
        self,
        search_pipeline: Any,
        model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
        load_model: bool = True,
    ):
        super().__init__(search_pipeline)
        self.model_name = model_name
        self.llm = None
        self.tokenizer = None
        if load_model:
            self.initialize_models()

    def initialize_models(self) -> None:
        import vllm

        self.vllm = vllm
        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=True,
            limit_mm_per_prompt={"image": 1},
        )
        self.tokenizer = self.llm.get_tokenizer()

    def get_batch_size(self) -> int:
        return AICROWD_SUBMISSION_BATCH_SIZE

    def batch_summarize_images(self, images: List[Image.Image]) -> List[str]:
        if not self.llm or not self.tokenizer:
            return [""] * len(images)

        messages = [
            {
                "role": "system",
                "content": "Describe the image for retrieval. Mention visible objects, text, brands, places, and entities.",
            },
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": "Summarize the image in one concise sentence."}],
            },
        ]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = [{"prompt": prompt, "multi_modal_data": {"image": image}} for image in images]
        outputs = self.llm.generate(
            inputs,
            sampling_params=self.vllm.SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=32,
                skip_special_tokens=True,
            ),
        )
        return [output.outputs[0].text.strip() for output in outputs]

    def _safe_search(self, query_or_image: Any, k: int) -> List[Dict[str, Any]]:
        if not self.search_pipeline:
            return []
        try:
            return self.search_pipeline(query_or_image, k=k) or []
        except Exception:
            return []

    def build_prompt_messages(
        self,
        query: str,
        image_evidence: str,
        web_evidence: str,
        message_history: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        system_prompt = (
            "You are a truthful CRAG-MM visual question answering agent. "
            "Answer using the image and the provided retrieval evidence. "
            "Image KG evidence and Web evidence are separate sources; prefer facts directly supported by them. "
            "If the evidence is insufficient or conflicting, answer exactly: I don't know. "
            "Keep the final answer short and factual, within 75 BPE tokens."
        )
        evidence_parts = []
        if image_evidence:
            evidence_parts.append("Image KG evidence:\n" + image_evidence)
        if web_evidence:
            evidence_parts.append("Web evidence:\n" + web_evidence)
        evidence_text = "\n\n".join(evidence_parts) or "No retrieval evidence was found."

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "image"}]},
        ]
        if message_history:
            messages.extend(message_history)
        messages.append({"role": "user", "content": evidence_text})
        messages.append({"role": "user", "content": query})
        return messages

    def prepare_inputs(
        self,
        queries: List[str],
        images: List[Image.Image],
        image_summaries: List[str],
        message_histories: List[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        inputs = []
        for query, image, image_summary, history in zip(queries, images, image_summaries, message_histories):
            image_results = self._safe_search(image, k=NUM_IMAGE_RESULTS)
            search_query = build_search_query(query, image_summary, history)
            web_results = self._safe_search(search_query, k=NUM_WEB_RESULTS)
            messages = self.build_prompt_messages(
                query=query,
                image_evidence=format_image_evidence(image_results),
                web_evidence=format_web_evidence(web_results),
                message_history=history,
            )
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs.append({"prompt": formatted_prompt, "multi_modal_data": {"image": image}})
        return inputs

    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        if not self.llm:
            return ["I don't know" for _ in queries]

        image_summaries = self.batch_summarize_images(images)
        inputs = self.prepare_inputs(queries, images, image_summaries, message_histories)
        outputs = self.llm.generate(
            inputs,
            sampling_params=self.vllm.SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=MAX_GENERATION_TOKENS,
                skip_special_tokens=True,
            ),
        )
        return [output.outputs[0].text.strip() or "I don't know" for output in outputs]

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from agents.course_agent_v2 import (
    AgentConfig,
    CourseRAGAgentV2,
    IDK_RESPONSE,
    RefusalReason,
    TaskMode,
)
from agents.vega_config import ExperimentVariant, VegaThresholds


class RecordingBackend:
    def __init__(self, answer="Taipei"):
        self.answer = answer
        self.prompts = []

    def answer_batch(self, prompts, images):
        self.prompts.extend(prompts)
        return [self.answer] * len(prompts)

    def truncate(self, text, max_tokens):
        return " ".join(text.split()[:max_tokens])

    def count_tokens(self, text):
        return len(text.split())


class SpecialTokenBackend(RecordingBackend):
    def truncate(self, text, max_tokens):
        return f"<|begin_of_text|>{text}"


class TwoEntitySearch:
    def __init__(self):
        self.calls = []

    def __call__(self, value, k):
        kind = "image" if isinstance(value, Image.Image) else "web"
        self.calls.append((kind, value, k))
        if kind == "image":
            return [
                {
                    "score": 0.82,
                    "entities": [
                        {
                            "entity_name": "Museum of Texas",
                            "entity_attributes": {"location": "Austin"},
                        }
                    ],
                },
                {
                    "score": 0.79,
                    "entities": [
                        {
                            "entity_name": "National Taiwan Museum",
                            "entity_attributes": {"location": "Taipei"},
                        }
                    ],
                },
            ]
        if len([call for call in self.calls if call[0] == "web"]) == 1:
            return [
                {
                    "score": 0.9,
                    "page_name": "National Taiwan Museum",
                    "page_url": "https://example.test/taiwan-museum",
                    "page_snippet": "National Taiwan Museum is located in Taipei.",
                }
            ]
        return [
            {
                "score": 0.88,
                "page_name": "Taipei museum location",
                "page_url": "https://example.test/taipei-location",
                "page_snippet": "The museum stands in Taipei, Taiwan.",
            }
        ]


class NumericConflictSearch:
    def __call__(self, value, k):
        if isinstance(value, Image.Image):
            return [
                {
                    "score": 0.9,
                    "entities": [
                        {
                            "entity_name": "Example Vehicle",
                            "entity_attributes": {
                                "engine": "2.0 L",
                            },
                        }
                    ],
                }
            ]
        return [
            {
                "score": 0.9,
                "page_name": "Example Vehicle engine",
                "page_url": "https://example.test/engine",
                "page_snippet": "The base engine is 2.7 L.",
            }
        ]


class VegaAgentIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (4, 4), "white")
        self.query = "Where is this museum located?"

    def build_agent(
        self,
        variant,
        thresholds=VegaThresholds(0.3, 0.7),
        trace_path=None,
    ):
        search = TwoEntitySearch()
        backend = RecordingBackend()
        config = AgentConfig(
            task_mode=TaskMode.TASK2,
            variant=variant,
            thresholds=thresholds,
            trace_path=trace_path,
        )
        return CourseRAGAgentV2(search, backend, config), search, backend

    def test_b0_keeps_original_prompt_and_one_image_one_web_call(self):
        agent, search, backend = self.build_agent(ExperimentVariant.B0)
        agent.batch_generate_response([self.query], [self.image], [[]])
        self.assertEqual(
            [call[0] for call in search.calls],
            ["image", "web"],
        )
        self.assertLess(
            backend.prompts[0].index("KG1"),
            backend.prompts[0].index("KG2"),
        )

    def test_a1_reorders_without_extra_search_or_refusal(self):
        agent, search, backend = self.build_agent(ExperimentVariant.A1)
        answer = agent.batch_generate_response(
            [self.query],
            [self.image],
            [[]],
        )
        self.assertEqual(answer, ["Taipei"])
        self.assertEqual(
            [call[0] for call in search.calls],
            ["image", "web"],
        )
        self.assertLess(
            backend.prompts[0].index("KG2"),
            backend.prompts[0].index("KG1"),
        )

    def test_a2_adds_exactly_one_web_call_in_mid_band(self):
        agent, search, _ = self.build_agent(ExperimentVariant.A2)
        agent.batch_generate_response([self.query], [self.image], [[]])
        self.assertEqual(
            [call[0] for call in search.calls],
            ["image", "web", "web"],
        )

    def test_a3_low_score_returns_idk_without_generation(self):
        agent, _, backend = self.build_agent(
            ExperimentVariant.A3,
            thresholds=VegaThresholds(0.7, 0.9),
        )
        answer = agent.batch_generate_response(
            [self.query],
            [self.image],
            [[]],
        )
        self.assertEqual(answer, [IDK_RESPONSE])
        self.assertEqual(backend.prompts, [])

    def test_a3_refusal_stays_exact_after_special_token_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = str(Path(tmp) / "trace.jsonl")
            backend = SpecialTokenBackend()
            agent = CourseRAGAgentV2(
                TwoEntitySearch(),
                backend,
                AgentConfig(
                    task_mode=TaskMode.TASK2,
                    variant=ExperimentVariant.A3,
                    thresholds=VegaThresholds(0.7, 0.9),
                    trace_path=trace_path,
                ),
            )
            answer = agent.batch_generate_response(
                [self.query],
                [self.image],
                [[]],
            )
            trace = json.loads(
                Path(trace_path).read_text(encoding="utf-8")
            )

        self.assertEqual(answer, [IDK_RESPONSE])
        self.assertEqual(trace["answer"], IDK_RESPONSE)
        self.assertEqual(trace["generation_ms"], 0.0)
        self.assertTrue(trace["refusal_reason"])
        self.assertEqual(backend.prompts, [])

    def test_a3_conflict_reason_survives_refusal_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = str(Path(tmp) / "trace.jsonl")
            agent = CourseRAGAgentV2(
                NumericConflictSearch(),
                SpecialTokenBackend(),
                AgentConfig(
                    task_mode=TaskMode.TASK2,
                    variant=ExperimentVariant.A3,
                    thresholds=VegaThresholds(0.2, 0.3),
                    trace_path=trace_path,
                ),
            )
            answer = agent.batch_generate_response(
                ["What engine size does this vehicle have?"],
                [self.image],
                [[]],
            )
            trace = json.loads(
                Path(trace_path).read_text(encoding="utf-8")
            )

        self.assertEqual(answer, [IDK_RESPONSE])
        self.assertTrue(trace["evidence_conflict"])
        self.assertEqual(
            trace["refusal_reason"],
            RefusalReason.CONFLICTING_EVIDENCE.value,
        )
        self.assertEqual(trace["generation_ms"], 0.0)

    def test_full_writes_trace_v3_action_and_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = str(Path(tmp) / "trace.jsonl")
            agent, _, _ = self.build_agent(
                ExperimentVariant.FULL,
                trace_path=trace_path,
            )
            agent.batch_generate_response(
                [self.query],
                [self.image],
                [[]],
            )
            trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
        self.assertEqual(trace["trace_schema"], "v3")
        self.assertEqual(trace["experiment_variant"], "full")
        self.assertEqual(trace["gate_action"], "expand")
        self.assertEqual(len(trace["entity_candidates"]), 2)
        self.assertEqual(trace["search_call_count"], 3)


if __name__ == "__main__":
    unittest.main()

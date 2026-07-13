import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from agents.course_agent_v2 import (
    AgentConfig,
    CourseRAGAgentV2,
    EvidenceItem,
    IDK_RESPONSE,
    TaskMode,
    build_prompt,
    build_search_query,
    clean_markup,
    normalize_answer,
    parse_image_evidence,
    parse_web_evidence,
)


class FakeBackend:
    def __init__(self, answers=None):
        self.answers = answers
        self.prompts = []

    def answer_batch(self, prompts, images):
        self.prompts.extend(prompts)
        return list(self.answers or ["Frank Gehry"] * len(prompts))

    def truncate(self, text, max_tokens):
        return " ".join(text.split()[:max_tokens])

    def count_tokens(self, text):
        return len(text.split())


class FakeSearch:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def __call__(self, value, k):
        kind = "image" if isinstance(value, Image.Image) else "web"
        self.calls.append((kind, value, k))
        if self.fail_on == kind:
            raise RuntimeError("planned failure")
        if kind == "image":
            return [{
                "score": 0.91,
                "entities": [{
                    "entity_name": "8 Spruce Street",
                    "entity_attributes": {
                        "architect": "[[Frank Gehry]]",
                        "floor_count": "76",
                        "empty": "",
                    },
                }],
            }]
        return [{
            "score": 0.72,
            "page_name": "8 Spruce Street",
            "page_url": "https://example.test/building",
            "page_snippet": "The tower was designed by Frank Gehry.",
        }]


class CourseAgentCoreTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (4, 4), "white")

    def test_clean_markup_handles_html_wikilinks_convert_and_entities(self):
        raw = "[[Frank Gehry|Gehry]]<br />{{convert|870|ft|m|0}} &amp; Co."
        self.assertEqual(clean_markup(raw), "Gehry 870 ft & Co.")

    def test_image_fields_are_ranked_for_the_question(self):
        evidence = parse_image_evidence(
            [{"score": 0.9, "entities": [{"entity_name": "Tower", "entity_attributes": {
                "opening": "2011", "architect": "Frank Gehry", "owner": "Example LLC"
            }}]}],
            "Who is the architect of this tower?",
            max_fields=3,
        )
        self.assertIn("architect: Frank Gehry", evidence[0].text)
        self.assertEqual([item.evidence_id for item in evidence], ["KG1", "KG2", "KG3"])

    def test_image_evidence_deduplicates_same_entity_field(self):
        result = {"score": 0.8, "entities": [{"entity_name": "Tower", "entity_attributes": {"architect": "Gehry"}}]}
        evidence = parse_image_evidence([result, result], "architect", max_fields=10)
        self.assertEqual(len(evidence), 1)

    def test_web_evidence_filters_empty_low_score_and_duplicates(self):
        rows = [
            {"score": 0.1, "page_name": "Low", "page_url": "u1", "page_snippet": "architect Gehry"},
            {"score": 0.8, "page_name": "Good", "page_url": "u2", "page_snippet": "architect Gehry"},
            {"score": 0.8, "page_name": "Good", "page_url": "u2", "page_snippet": "architect Gehry"},
            {"score": 0.9, "page_name": "Empty", "page_url": "u3", "page_snippet": ""},
        ]
        evidence = parse_web_evidence(rows, "Who is the architect?", score_threshold=0.2)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].evidence_id, "WEB1")

    def test_search_query_uses_entities_without_history_dump(self):
        query = build_search_query(
            "Who designed it?",
            [EvidenceItem("KG1", "image_kg", "8 Spruce Street | architect: Frank Gehry")],
        )
        self.assertEqual(query, "8 Spruce Street Who designed it?")

    def test_task1_prompt_has_no_web_section(self):
        prompt = build_prompt("Who?", TaskMode.TASK1, [], [], 1000)
        self.assertIn("IMAGE KG EVIDENCE", prompt)
        self.assertNotIn("WEB EVIDENCE", prompt)

    def test_task2_prompt_keeps_sources_separate(self):
        prompt = build_prompt("Who?", TaskMode.TASK2, [], [], 1000)
        self.assertIn("IMAGE KG EVIDENCE", prompt)
        self.assertIn("WEB EVIDENCE", prompt)

    def test_normalize_answer_standardizes_refusal_and_empty(self):
        self.assertEqual(normalize_answer("I do not know.")[0], IDK_RESPONSE)
        self.assertEqual(normalize_answer("  ")[0], IDK_RESPONSE)

    def test_batch_lengths_must_match(self):
        agent = CourseRAGAgentV2(FakeSearch(), FakeBackend(), AgentConfig(task_mode=TaskMode.TASK1))
        with self.assertRaises(ValueError):
            agent.batch_generate_response(["q"], [], [[]])

    def test_task1_never_calls_web_search(self):
        search = FakeSearch()
        backend = FakeBackend()
        agent = CourseRAGAgentV2(search, backend, AgentConfig(task_mode=TaskMode.TASK1))
        answer = agent.batch_generate_response(["Who designed it?"], [self.image], [[]])
        self.assertEqual(answer, ["Frank Gehry"])
        self.assertEqual([call[0] for call in search.calls], ["image"])

    def test_task2_calls_image_then_web_and_numbers_evidence(self):
        search = FakeSearch()
        backend = FakeBackend()
        agent = CourseRAGAgentV2(search, backend, AgentConfig(task_mode=TaskMode.TASK2))
        agent.batch_generate_response(["Who designed it?"], [self.image], [[]])
        self.assertEqual([call[0] for call in search.calls], ["image", "web"])
        self.assertIn("[KG1", backend.prompts[0])
        self.assertIn("[WEB1", backend.prompts[0])

    def test_vision_mode_does_not_search(self):
        search = FakeSearch()
        agent = CourseRAGAgentV2(search, FakeBackend(), AgentConfig(task_mode=TaskMode.VISION))
        agent.batch_generate_response(["What is shown?"], [self.image], [[]])
        self.assertEqual(search.calls, [])

    def test_search_error_is_logged_in_trace_and_refuses_safely(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = str(Path(tmp) / "trace.jsonl")
            config = AgentConfig(task_mode=TaskMode.TASK1, trace_path=trace_path)
            agent = CourseRAGAgentV2(FakeSearch(fail_on="image"), FakeBackend([IDK_RESPONSE]), config)
            self.assertEqual(agent.batch_generate_response(["q"], [self.image], [[]]), [IDK_RESPONSE])
            trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
            self.assertEqual(trace["status"], "search_error")
            self.assertEqual(trace["refusal_reason"], "search_error")
            self.assertTrue(trace["errors"])

    def test_answer_is_truncated_to_configured_limit(self):
        backend = FakeBackend(["one two three four"])
        config = AgentConfig(task_mode=TaskMode.VISION, max_answer_tokens=3)
        agent = CourseRAGAgentV2(FakeSearch(), backend, config)
        self.assertEqual(agent.batch_generate_response(["q"], [self.image], [[]]), ["one two three"])

    def test_backend_answer_count_must_match_batch(self):
        agent = CourseRAGAgentV2(FakeSearch(), FakeBackend(["only one"]), AgentConfig(task_mode=TaskMode.VISION))
        with self.assertRaises(RuntimeError):
            agent.batch_generate_response(["q1", "q2"], [self.image, self.image], [[], []])


if __name__ == "__main__":
    unittest.main()

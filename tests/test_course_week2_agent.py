import unittest

from agents.course_week2_agent import (
    build_search_query,
    format_image_evidence,
    format_web_evidence,
)
from agents.course_agent_v2 import CourseRAGAgentV2
from agents.user_config import UserAgent


class CourseWeek2AgentHelpersTest(unittest.TestCase):
    def test_user_config_selects_verified_v2_agent(self):
        self.assertIs(UserAgent, CourseRAGAgentV2)

    def test_build_search_query_combines_question_summary_and_latest_history(self):
        query = build_search_query(
            "How much does it cost?",
            "a Dyson Airwrap hair styler on a table",
            [
                {"role": "user", "content": "What product is shown?"},
                {"role": "assistant", "content": "It is a Dyson Airwrap."},
            ],
        )

        self.assertIn("How much does it cost?", query)
        self.assertIn("Dyson Airwrap", query)
        self.assertIn("a Dyson Airwrap hair styler", query)

    def test_format_image_evidence_extracts_structured_kg_fields(self):
        evidence = format_image_evidence(
            [
                {
                    "score": 0.91,
                    "entities": [
                        {
                            "entity_name": "8 Spruce Street",
                            "entity_attributes": {
                                "architect": "[[Frank Gehry]]",
                                "floor_count": "76",
                                "irrelevant": "",
                            },
                        }
                    ],
                }
            ],
            max_items=1,
        )

        self.assertIn("Image KG 1", evidence)
        self.assertIn("8 Spruce Street", evidence)
        self.assertIn("architect: Frank Gehry", evidence)
        self.assertIn("floor_count: 76", evidence)
        self.assertNotIn("irrelevant", evidence)

    def test_format_web_evidence_uses_snippets_without_fetching_full_pages(self):
        evidence = format_web_evidence(
            [
                {
                    "page_name": "Example Product",
                    "page_url": "https://example.com/product",
                    "page_snippet": "The product costs $49 and ships worldwide.",
                    "score": 0.72,
                }
            ],
            max_items=1,
        )

        self.assertIn("Web 1", evidence)
        self.assertIn("Example Product", evidence)
        self.assertIn("https://example.com/product", evidence)
        self.assertIn("The product costs $49", evidence)


if __name__ == "__main__":
    unittest.main()

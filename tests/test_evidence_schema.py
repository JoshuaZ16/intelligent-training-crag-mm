import math
import unittest
from dataclasses import FrozenInstanceError, fields
from typing import get_type_hints

from agents.evidence_schema import (
    AtomicEvidence,
    EntityCandidate,
    EvidencePacket,
    QuestionFrame,
)


class EvidenceSchemaTest(unittest.TestCase):
    def setUp(self):
        self.frame = QuestionFrame(
            "location",
            "place",
            False,
            ("located", "where"),
        )
        self.web_evidence = AtomicEvidence(
            "WEB1",
            "web",
            "example.test",
            "Museum",
            "location",
            "Taipei",
            "",
            "",
            "The museum is in Taipei.",
            0.9,
        )
        self.image_evidence = AtomicEvidence(
            "IMG1",
            "image",
            "photo-1",
            "Museum",
            "sign_text",
            "Taipei Museum",
            "",
            "",
            "The sign reads Taipei Museum.",
            0.8,
        )

    def make_packet(self, evidence=None):
        if evidence is None:
            evidence = (self.web_evidence,)
        return EvidencePacket(
            "Where is it?",
            self.frame,
            (),
            evidence,
            (),
        )

    def test_field_order_matches_positional_api(self):
        self.assertEqual(
            [field.name for field in fields(QuestionFrame)],
            [
                "target_relation",
                "answer_type",
                "temporal_intent",
                "relation_terms",
            ],
        )
        self.assertEqual(
            [field.name for field in fields(EntityCandidate)],
            [
                "candidate_id",
                "entity_name",
                "aliases",
                "image_score",
                "evidence_ids",
            ],
        )
        self.assertEqual(
            [field.name for field in fields(AtomicEvidence)],
            [
                "evidence_id",
                "source",
                "source_cluster",
                "entity",
                "predicate",
                "value",
                "unit",
                "valid_time",
                "text",
                "relevance",
            ],
        )
        self.assertEqual(
            [field.name for field in fields(EvidencePacket)],
            [
                "question",
                "question_frame",
                "entity_candidates",
                "evidence",
                "conflicts",
            ],
        )

    def test_plan_core_example_runs_directly(self):
        packet = self.make_packet()

        self.assertEqual(packet.question_frame, self.frame)
        self.assertEqual(packet.evidence, (self.web_evidence,))

    def test_entity_candidate_defaults_and_fields_serialize(self):
        minimal = EntityCandidate("entity-1", "Museum")
        candidate = EntityCandidate(
            "entity-2",
            "Taipei Museum",
            ("Museum of Taipei",),
            0.75,
            ("WEB1", "IMG1"),
        )
        packet = EvidencePacket(
            "Where is it?",
            self.frame,
            (minimal, candidate),
            (self.web_evidence,),
            (),
        )

        serialized = packet.to_dict()["entity_candidates"]
        self.assertEqual(
            serialized[0],
            {
                "candidate_id": "entity-1",
                "entity_name": "Museum",
                "aliases": (),
                "image_score": 0.0,
                "evidence_ids": (),
            },
        )
        self.assertEqual(serialized[1]["candidate_id"], "entity-2")
        self.assertEqual(serialized[1]["aliases"], ("Museum of Taipei",))
        self.assertEqual(serialized[1]["image_score"], 0.75)
        self.assertEqual(serialized[1]["evidence_ids"], ("WEB1", "IMG1"))

    def test_to_dict_serializes_nested_evidence_predicate(self):
        packet = self.make_packet()

        self.assertEqual(packet.to_dict()["evidence"][0]["predicate"], "location")

    def test_equal_packets_have_identical_hashes(self):
        first = self.make_packet()
        second = self.make_packet()

        self.assertEqual(first.sha256(), second.sha256())

    def test_negative_zero_normalizes_for_packet_equality_and_hash(self):
        positive_evidence = AtomicEvidence(
            "WEB1",
            "web",
            "example.test",
            "Museum",
            "location",
            "Taipei",
            "",
            "",
            "The museum is in Taipei.",
            0.0,
        )
        negative_evidence = AtomicEvidence(
            "WEB1",
            "web",
            "example.test",
            "Museum",
            "location",
            "Taipei",
            "",
            "",
            "The museum is in Taipei.",
            -0.0,
        )
        positive_candidate = EntityCandidate("entity-1", "Museum", (), 0.0, ())
        negative_candidate = EntityCandidate("entity-1", "Museum", (), -0.0, ())
        positive_packet = EvidencePacket(
            "Where is it?",
            self.frame,
            (positive_candidate,),
            (positive_evidence,),
            (),
        )
        negative_packet = EvidencePacket(
            "Where is it?",
            self.frame,
            (negative_candidate,),
            (negative_evidence,),
            (),
        )

        self.assertEqual(math.copysign(1.0, negative_evidence.relevance), 1.0)
        self.assertEqual(math.copysign(1.0, negative_candidate.image_score), 1.0)
        self.assertEqual(positive_packet, negative_packet)
        self.assertEqual(positive_packet.sha256(), negative_packet.sha256())

    def test_integer_and_float_zero_have_equal_hashes(self):
        integer_evidence = AtomicEvidence(
            "WEB1",
            "web",
            "example.test",
            "Museum",
            "location",
            "Taipei",
            "",
            "",
            "The museum is in Taipei.",
            0,
        )
        float_evidence = AtomicEvidence(
            "WEB1",
            "web",
            "example.test",
            "Museum",
            "location",
            "Taipei",
            "",
            "",
            "The museum is in Taipei.",
            0.0,
        )
        integer_candidate = EntityCandidate("entity-1", "Museum", (), 0, ())
        float_candidate = EntityCandidate("entity-1", "Museum", (), 0.0, ())
        integer_packet = EvidencePacket(
            "Where is it?",
            self.frame,
            (integer_candidate,),
            (integer_evidence,),
            (),
        )
        float_packet = EvidencePacket(
            "Where is it?",
            self.frame,
            (float_candidate,),
            (float_evidence,),
            (),
        )

        self.assertIsInstance(integer_evidence.relevance, float)
        self.assertIsInstance(integer_candidate.image_score, float)
        self.assertEqual(integer_packet, float_packet)
        self.assertEqual(integer_packet.sha256(), float_packet.sha256())

    def test_evidence_order_is_preserved_and_changes_hash(self):
        first = self.make_packet((self.web_evidence, self.image_evidence))
        second = self.make_packet((self.image_evidence, self.web_evidence))

        self.assertEqual(
            [item["evidence_id"] for item in first.to_dict()["evidence"]],
            ["WEB1", "IMG1"],
        )
        self.assertEqual(
            [item["evidence_id"] for item in second.to_dict()["evidence"]],
            ["IMG1", "WEB1"],
        )
        self.assertNotEqual(first.sha256(), second.sha256())

    def test_unicode_hash_matches_fixed_canonical_utf8_digest(self):
        frame = QuestionFrame("位置", "地点", False, ("位于", "哪里"))
        evidence = AtomicEvidence(
            "网页1",
            "网页",
            "示例.测试",
            "博物馆",
            "位置",
            "台北",
            "",
            "",
            "博物馆位于台北。",
            1.0,
        )
        packet = EvidencePacket("它在哪里？", frame, (), (evidence,), ())

        self.assertEqual(
            packet.sha256(),
            "c0c73618584924ff60a2fa0366fe10b8de99f66f7e542ac3ef03d3392670c14c",
        )

    def test_input_lists_are_defensively_converted_to_tuples(self):
        relation_terms = ["located", "where"]
        aliases = ["Taipei Museum"]
        evidence_ids = ["WEB1"]
        candidates = []
        evidence_items = [self.web_evidence]
        conflicts = ["source disagreement"]
        frame = QuestionFrame("location", "place", False, relation_terms)
        candidate = EntityCandidate(
            "entity-1",
            "Museum",
            aliases,
            0.75,
            evidence_ids,
        )
        candidates.append(candidate)
        packet = EvidencePacket(
            "Where is it?",
            frame,
            candidates,
            evidence_items,
            conflicts,
        )
        original_hash = packet.sha256()

        relation_terms.append("site")
        aliases.append("Changed alias")
        evidence_ids.append("IMG1")
        candidates.append(EntityCandidate("entity-2", "Other"))
        evidence_items.append(self.image_evidence)
        conflicts.append("changed")

        self.assertEqual(frame.relation_terms, ("located", "where"))
        self.assertEqual(candidate.aliases, ("Taipei Museum",))
        self.assertEqual(candidate.evidence_ids, ("WEB1",))
        self.assertEqual(packet.entity_candidates, (candidate,))
        self.assertEqual(packet.evidence, (self.web_evidence,))
        self.assertEqual(packet.conflicts, ("source disagreement",))
        self.assertEqual(packet.sha256(), original_hash)

    def test_all_schema_dataclasses_are_frozen(self):
        instances_and_fields = (
            (self.frame, "target_relation"),
            (EntityCandidate("entity-1", "Museum"), "entity_name"),
            (self.web_evidence, "predicate"),
            (self.make_packet(), "question"),
        )

        for instance, field_name in instances_and_fields:
            with self.subTest(type=type(instance).__name__):
                with self.assertRaises(FrozenInstanceError):
                    setattr(instance, field_name, "changed")

    def test_relevance_rejects_values_outside_closed_unit_interval(self):
        for relevance in (-0.1, 1.1, float("nan"), float("inf"), float("-inf")):
            with self.subTest(relevance=relevance):
                with self.assertRaises(ValueError):
                    AtomicEvidence(
                        "WEB1",
                        "web",
                        "example.test",
                        "Museum",
                        "location",
                        "Taipei",
                        "",
                        "",
                        "The museum is in Taipei.",
                        relevance,
                    )

    def test_relevance_accepts_closed_interval_boundaries(self):
        for relevance in (0.0, 1.0):
            with self.subTest(relevance=relevance):
                evidence = AtomicEvidence(
                    "WEB1",
                    "web",
                    "example.test",
                    "Museum",
                    "location",
                    "Taipei",
                    "",
                    "",
                    "The museum is in Taipei.",
                    relevance,
                )
                self.assertEqual(evidence.relevance, relevance)

    def test_image_score_must_be_finite_without_unit_interval_restriction(self):
        for image_score in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(image_score=image_score):
                with self.assertRaises(ValueError):
                    EntityCandidate("entity-1", "Museum", (), image_score, ())

        for image_score in (-1.25, 12.0):
            with self.subTest(image_score=image_score):
                candidate = EntityCandidate(
                    "entity-1",
                    "Museum",
                    (),
                    image_score,
                    (),
                )
                self.assertEqual(candidate.image_score, image_score)

    def test_public_method_return_annotations_are_stable(self):
        for schema_type in (
            QuestionFrame,
            EntityCandidate,
            AtomicEvidence,
            EvidencePacket,
        ):
            with self.subTest(type=schema_type.__name__):
                self.assertIs(
                    get_type_hints(schema_type.__post_init__)["return"],
                    type(None),
                )

        self.assertEqual(
            get_type_hints(EvidencePacket.to_dict)["return"],
            dict[str, object],
        )
        self.assertIs(get_type_hints(EvidencePacket.sha256)["return"], str)


if __name__ == "__main__":
    unittest.main()

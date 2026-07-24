import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import scripts.week2_experiment as week2_experiment
from agents.course_agent_v2 import TaskMode
from scripts.week2_experiment import (
    IMAGE_INDEX_REVISION,
    WEB_INDEX_REVISION,
    build_search_pipeline,
    file_sha256,
    load_manifest_indices,
    parse_args,
)


class Week2ExperimentCliTest(unittest.TestCase):
    def test_cli_accepts_variant_manifest_and_run_id(self):
        args = parse_args(
            [
                "--mode",
                "task2",
                "--variant",
                "a1",
                "--run-id",
                "r1-a1",
                "--manifest-path",
                "sample_manifest.json",
                "--output-dir",
                "out",
            ]
        )
        self.assertEqual(args.variant, "a1")
        self.assertEqual(args.run_id, "r1-a1")
        self.assertTrue(args.manifest_path.endswith("sample_manifest.json"))

    def test_manifest_indices_and_sha_are_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample_manifest.json"
            path.write_text(
                json.dumps(
                    [
                        {"source_index": 69},
                        {"source_index": 111},
                        {"source_index": 176},
                    ]
                ),
                encoding="utf-8",
            )
            first_hash = file_sha256(path)
            indices = load_manifest_indices(path)
            second_hash = file_sha256(path)
        self.assertEqual(indices, [69, 111, 176])
        self.assertEqual(first_hash, second_hash)
        self.assertEqual(len(first_hash), 64)

    def test_manifest_rejects_duplicate_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample_manifest.json"
            path.write_text(
                json.dumps(
                    [
                        {"source_index": 69},
                        {"source_index": 69},
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_manifest_indices(path)

    def test_search_pipeline_uses_audited_local_resources_and_revisions(self):
        captured = {}
        fake_search = types.ModuleType("cragmm_search.search")

        class FakePipeline:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        fake_search.UnifiedSearchPipeline = FakePipeline
        environment = {
            "CRAG_IMAGE_MODEL": "/frozen/clip",
            "CRAG_TEXT_MODEL": "/frozen/bge",
            "CRAG_IMAGE_INDEX": "/frozen/image-index",
            "CRAG_WEB_INDEX": "/frozen/web-index",
        }
        with mock.patch.dict(
            sys.modules,
            {"cragmm_search.search": fake_search},
        ), mock.patch.dict(
            os.environ,
            environment,
            clear=True,
        ), mock.patch.object(
            week2_experiment,
            "install_cragmm_lazy_image_metadata",
        ), mock.patch.object(
            week2_experiment,
            "install_cragmm_lazy_web_metadata",
        ):
            pipeline = build_search_pipeline(TaskMode.TASK2)

        self.assertIsInstance(pipeline, FakePipeline)
        self.assertEqual(captured["image_model_name"], "/frozen/clip")
        self.assertEqual(captured["text_model_name"], "/frozen/bge")
        self.assertEqual(
            captured["image_hf_dataset_id"],
            "/frozen/image-index",
        )
        self.assertEqual(
            captured["web_hf_dataset_id"],
            "/frozen/web-index",
        )
        self.assertEqual(
            captured["image_hf_dataset_tag"],
            IMAGE_INDEX_REVISION,
        )
        self.assertEqual(
            captured["web_hf_dataset_tag"],
            WEB_INDEX_REVISION,
        )


if __name__ == "__main__":
    unittest.main()

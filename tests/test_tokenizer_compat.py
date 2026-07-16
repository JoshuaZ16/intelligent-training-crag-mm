import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.tokenizer_compat import load_evaluator_tokenizer


class FakeTokenizer:
    calls = []

    @classmethod
    def from_file(cls, path):
        cls.calls.append(("file", path))
        return ("file", path)

    @classmethod
    def from_pretrained(cls, name):
        cls.calls.append(("pretrained", name))
        return ("pretrained", name)


class EvaluatorTokenizerCompatibilityTest(unittest.TestCase):
    def setUp(self):
        FakeTokenizer.calls = []

    def test_local_crag_model_tokenizer_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer_file = Path(tmp_dir) / "tokenizer.json"
            tokenizer_file.write_text("{}", encoding="utf-8")
            with patch.dict(os.environ, {"CRAG_MODEL": tmp_dir}, clear=False):
                tokenizer = load_evaluator_tokenizer(FakeTokenizer)

        self.assertEqual(tokenizer, ("file", str(tokenizer_file)))
        self.assertEqual(FakeTokenizer.calls, [("file", str(tokenizer_file))])

    def test_default_remote_tokenizer_remains_the_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            tokenizer = load_evaluator_tokenizer(FakeTokenizer)

        self.assertEqual(
            tokenizer,
            ("pretrained", "meta-llama/Llama-3.2-1B-Instruct"),
        )


if __name__ == "__main__":
    unittest.main()

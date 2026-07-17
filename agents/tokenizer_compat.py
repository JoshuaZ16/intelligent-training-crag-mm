"""Resolve the evaluator tokenizer without requiring a second gated model."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_EVALUATOR_TOKENIZER = "meta-llama/Llama-3.2-1B-Instruct"


def load_evaluator_tokenizer(tokenizer_class: Any):
    """Prefer a local CRAG model tokenizer, then preserve the upstream fallback."""
    source = os.getenv("CRAG_EVAL_TOKENIZER") or os.getenv("CRAG_MODEL")
    if source:
        path = Path(source).expanduser()
        tokenizer_file = path / "tokenizer.json" if path.is_dir() else path
        if tokenizer_file.is_file():
            return tokenizer_class.from_file(str(tokenizer_file))
        return tokenizer_class.from_pretrained(source)
    return tokenizer_class.from_pretrained(DEFAULT_EVALUATOR_TOKENIZER)

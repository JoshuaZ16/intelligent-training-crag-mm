"""Run one real-image MLX-VLM smoke test and persist auditable evidence."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.course_agent_v2 import MlxVlmBackend


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", default="What dish is shown in this image? Answer briefly.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "model": args.model,
        "image": args.image,
        "question": args.question,
        "status": "error",
    }
    started = time.perf_counter()
    try:
        image = Image.open(args.image).convert("RGB")
        backend = MlxVlmBackend(args.model)
        record["answer"] = backend.answer_batch([args.question], [image])[0]
        record["status"] = "ok"
    except Exception as exc:  # Evidence must preserve the full failure context.
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)
        record["traceback"] = traceback.format_exc()
    finally:
        record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0 if record["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Write a machine-readable week-2 environment readiness snapshot."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


PACKAGES = [
    "datasets",
    "torch",
    "torchvision",
    "transformers",
    "cragmm-search-pipeline",
    "mlx",
    "mlx-vlm",
    "pandas",
    "Pillow",
]


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    imports = {}
    for module_name in ["datasets", "torch", "torchvision", "transformers", "cragmm_search", "mlx", "mlx_vlm"]:
        try:
            importlib.import_module(module_name)
            imports[module_name] = "ok"
        except Exception as exc:
            imports[module_name] = f"{type(exc).__name__}: {exc}"

    torch_info = {}
    if imports.get("torch") == "ok":
        import torch

        torch_info = {
            "mps_available": bool(torch.backends.mps.is_available()),
            "cuda_available": bool(torch.cuda.is_available()),
        }

    snapshot = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {name: package_version(name) for name in PACKAGES},
        "imports": imports,
        "torch": torch_info,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(snapshot, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

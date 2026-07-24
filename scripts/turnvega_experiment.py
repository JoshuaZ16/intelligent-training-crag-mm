#!/usr/bin/env python3
"""Run a fixed-manifest TurnVEGA Task 2 or Task 3 experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import random
import re
import secrets
import stat
import subprocess
import sys
import threading
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.course_agent_v2 import (
    AgentConfig,
    CourseRAGAgentV2,
    TaskMode,
    create_backend,
)
from agents.turnvega_config import (
    DatasetKind,
    ExperimentBudget,
    TurnVegaVariant,
)
from agents.vega_config import ExperimentVariant
from scripts.week2_experiment import (
    build_search_pipeline,
    file_sha256,
    load_manifest_indices,
)


DATASET_IDS = {
    DatasetKind.TASK2.value: "crag-mm-2025/crag-mm-single-turn-public",
    DatasetKind.TASK3.value: "crag-mm-2025/crag-mm-multi-turn-public",
}
DATASET_REVISION = "v0.1.2"
TRACE_FILENAME = "agent_trace_v5.jsonl"
EXPERIMENT_FAMILY = "turnvega"
OFFICIAL_SESSIONS_TO_SKIP = frozenset(
    {
        "04d98259-27af-41b1-a7be-5798fd1b8e95",
        "695b4b5c-7c65-4f7b-8968-50fe10482a16",
    }
)
PACKAGE_DISTRIBUTIONS = {
    "torch": "torch",
    "transformers": "transformers",
    "vllm": "vllm",
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _zero_float(value: str) -> float:
    parsed = float(value)
    if parsed != 0.0:
        raise argparse.ArgumentTypeError(
            "TurnVEGA Task 4 only supports temperature=0"
        )
    return 0.0


def _variant_values() -> List[str]:
    return [
        variant.value
        for enum_type in (ExperimentVariant, TurnVegaVariant)
        for variant in enum_type
    ]


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-kind",
        choices=[kind.value for kind in DatasetKind],
        required=True,
    )
    parser.add_argument("--variant", choices=_variant_values(), required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument(
        "--candidate-passages",
        type=_positive_int,
        required=True,
    )
    parser.add_argument(
        "--prompt-evidence",
        type=_positive_int,
        required=True,
    )
    parser.add_argument(
        "--max-search-calls",
        type=_positive_int,
        required=True,
    )
    parser.add_argument("--trace-schema", choices=["v5"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--backend",
        choices=["mlx", "vllm"],
        default="vllm",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-weight-sha256", default=None)
    parser.add_argument("--model-weight-lfs-oid", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--temperature", type=_zero_float, default=0.0)
    parser.add_argument(
        "--max-evidence-chars",
        type=_positive_int,
        default=5500,
    )
    parser.add_argument("--history-mode", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        ExperimentBudget(
            candidate_passages=args.candidate_passages,
            prompt_evidence=args.prompt_evidence,
            max_search_calls=args.max_search_calls,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def validate_manifest_rows(
    rows: Any,
    dataset_kind: str | DatasetKind,
) -> List[Dict[str, Any]]:
    """Validate manifest identities while preserving caller-provided order."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest must be a non-empty ordered JSON list")
    kind = DatasetKind(dataset_kind)
    validated: List[Dict[str, Any]] = []
    source_indices = set()
    task3_identities = set()
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"manifest row {position} must be an object")
        source_index = row.get("source_index")
        if type(source_index) is not int:
            raise ValueError(
                f"manifest row {position} source_index must be an integer"
            )
        if source_index in source_indices:
            raise ValueError("manifest source_index values must be unique")
        source_indices.add(source_index)

        if kind is DatasetKind.TASK3:
            session_key = row.get("session_key") or row.get("session_id")
            turn_index = row.get("turn_index", 0)
            interaction_id = row.get("interaction_id")
            if not isinstance(session_key, str) or not session_key.strip():
                raise ValueError(
                    f"Task3 manifest row {position} needs session_key/session_id"
                )
            if session_key in OFFICIAL_SESSIONS_TO_SKIP:
                raise ValueError(
                    "Task3 manifest contains official skipped session: "
                    + session_key
                )
            if type(turn_index) is not int:
                raise ValueError(
                    f"Task3 manifest row {position} turn_index must be an integer"
                )
            if not isinstance(interaction_id, str) or not interaction_id.strip():
                raise ValueError(
                    f"Task3 manifest row {position} needs interaction_id"
                )
            identity = (session_key, turn_index, interaction_id)
            if identity in task3_identities:
                raise ValueError("Task3 identity values must be unique")
            task3_identities.add(identity)
        validated.append(dict(row))
    return validated


def load_validated_manifest(
    path: str | Path,
    dataset_kind: str | DatasetKind,
) -> List[Dict[str, Any]]:
    """Load the complete manifest and cross-check the shared index loader."""
    manifest_path = Path(path)
    rows = validate_manifest_rows(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        dataset_kind,
    )
    shared_indices = load_manifest_indices(manifest_path)
    indices = [row["source_index"] for row in rows]
    if shared_indices != indices:
        raise ValueError("manifest index order changed during validation")
    return rows


def validate_selected_rows(
    selected_rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Reject incomplete, truncated, or reordered dataset selections."""
    if len(selected_rows) != len(manifest_rows):
        raise ValueError(
            "selected dataset row count must exactly match manifest row count"
        )
    for position, (selected, manifest) in enumerate(
        zip(selected_rows, manifest_rows)
    ):
        actual_source_index = selected.get("source_index")
        if (
            actual_source_index is not None
            and actual_source_index != manifest["source_index"]
        ):
            raise ValueError(
                f"selected row {position} does not match manifest order"
            )
        if "session_key" in manifest or "session_id" in manifest:
            expected_session = manifest.get("session_key") or manifest.get(
                "session_id"
            )
            actual_session = _selected_identity_value(
                selected,
                ("session_key", "session_id"),
            )
            if actual_session is _MISSING:
                raise ValueError(
                    f"selected row {position} cannot extract session identity"
                )
            if actual_session != expected_session:
                raise ValueError(
                    f"selected row {position} does not match manifest order"
                )
        for identity_field in ("turn_index", "interaction_id"):
            if identity_field not in manifest:
                continue
            actual_value = _selected_identity_value(
                selected,
                (identity_field,),
            )
            if actual_value is _MISSING:
                raise ValueError(
                    f"selected row {position} cannot extract {identity_field}"
                )
            if actual_value != manifest[identity_field]:
                raise ValueError(
                    f"selected row {position} does not match manifest order"
                )


_MISSING = object()


def _first_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else _MISSING
    return value


def _selected_identity_value(
    row: Mapping[str, Any],
    field_names: Sequence[str],
) -> Any:
    for field_name in field_names:
        if field_name in row:
            return _first_value(row[field_name])
    turns = row.get("turns")
    if isinstance(turns, Mapping):
        for field_name in field_names:
            if field_name in turns:
                return _first_value(turns[field_name])
    elif isinstance(turns, (list, tuple)) and turns:
        first_turn = turns[0]
        if isinstance(first_turn, Mapping):
            for field_name in field_names:
                if field_name in first_turn:
                    return _first_value(first_turn[field_name])
    return _MISSING


def collect_runtime_versions() -> Dict[str, str | None]:
    versions: Dict[str, str | None] = {
        "python": platform.python_version(),
    }
    for key, distribution in PACKAGE_DISTRIBUTIONS.items():
        try:
            versions[key] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[key] = None
    return versions


def current_git_commit(project_root: str | Path = PROJECT_ROOT) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def current_git_diff_sha256(
    project_root: str | Path = PROJECT_ROOT,
) -> str:
    """Hash HEAD, all tracked changes, and every non-ignored untracked file."""
    root = Path(project_root)
    commit = current_git_commit(root)
    if commit is None:
        raise RuntimeError("cannot determine git HEAD for source-state hash")
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    untracked_output = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    untracked_paths = sorted(
        path for path in untracked_output.split(b"\0") if path
    )

    digest = hashlib.sha256()
    _update_length_prefixed(digest, b"HEAD")
    _update_length_prefixed(digest, commit.encode("ascii"))
    _update_length_prefixed(digest, b"DIFF")
    _update_length_prefixed(digest, diff)
    for relative_bytes in untracked_paths:
        relative_path = os.fsdecode(relative_bytes)
        path = root / relative_path
        file_stat = path.lstat()
        record = hashlib.sha256()
        _update_length_prefixed(record, relative_bytes)
        _update_length_prefixed(
            record,
            f"{file_stat.st_mode:o}".encode("ascii"),
        )
        if stat.S_ISLNK(file_stat.st_mode):
            _update_length_prefixed(record, b"SYMLINK")
            _update_length_prefixed(
                record,
                hashlib.sha256(
                    os.fsencode(os.readlink(path))
                ).digest(),
            )
        elif stat.S_ISREG(file_stat.st_mode):
            _update_length_prefixed(record, b"FILE")
            content_digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    content_digest.update(chunk)
            _update_length_prefixed(record, content_digest.digest())
        else:
            _update_length_prefixed(record, b"OTHER")
        _update_length_prefixed(digest, record.digest())
    return digest.hexdigest()


def _update_length_prefixed(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def directory_tree_sha256(path: str | Path) -> str:
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"local model path is not a directory: {root}")
    digest = hashlib.sha256()
    for item in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if item.is_dir() and not item.is_symlink():
            continue
        relative = os.fsencode(item.relative_to(root).as_posix())
        item_stat = item.lstat()
        record = hashlib.sha256()
        _update_length_prefixed(record, relative)
        _update_length_prefixed(
            record,
            f"{item_stat.st_mode:o}".encode("ascii"),
        )
        if item.is_symlink():
            _update_length_prefixed(record, b"SYMLINK")
            content = os.fsencode(os.readlink(item))
            _update_length_prefixed(
                record,
                hashlib.sha256(content).digest(),
            )
        elif item.is_file():
            _update_length_prefixed(record, b"FILE")
            _update_length_prefixed(
                record,
                bytes.fromhex(file_sha256(item)),
            )
        else:
            _update_length_prefixed(record, b"OTHER")
        _update_length_prefixed(digest, record.digest())
    return digest.hexdigest()


def _is_sha256(value: str | None) -> bool:
    return bool(value and re.fullmatch(r"[0-9a-fA-F]{64}", value))


def _is_lfs_oid(value: str | None) -> bool:
    return bool(
        value and re.fullmatch(r"sha256:[0-9a-fA-F]{64}", value)
    )


def validate_input_identity(args: argparse.Namespace) -> Dict[str, Any]:
    """Validate model/dataset inputs without importing runtime packages."""
    model_sha = args.model_weight_sha256
    model_lfs_oid = args.model_weight_lfs_oid
    dataset_sha = None
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
        if not dataset_path.is_file():
            raise ValueError(
                f"local dataset path is not a readable file: {dataset_path}"
            )
        dataset_sha = file_sha256(dataset_path)

    if args.model:
        model_path = Path(args.model)
        if model_path.exists():
            actual_sha = (
                directory_tree_sha256(model_path)
                if model_path.is_dir()
                else file_sha256(model_path)
            )
            if model_sha and model_sha.lower() != actual_sha:
                raise ValueError("declared model weight sha256 does not match local model")
            model_sha = actual_sha
        elif not args.dry_run:
            raise ValueError(
                "non-dry runs require an existing local model file or directory"
            )
    elif not args.dry_run:
        raise ValueError("--model is required for non-dry runs")

    if model_sha and not _is_sha256(model_sha):
        raise ValueError("model weight sha256 must contain 64 hex characters")
    if model_lfs_oid and not _is_lfs_oid(model_lfs_oid):
        raise ValueError("model weight LFS OID must use sha256:<64 hex>")
    return {
        "model_weight_sha256": model_sha.lower() if model_sha else None,
        "model_weight_lfs_oid": (
            model_lfs_oid.lower() if model_lfs_oid else None
        ),
        "dataset_path_sha256": dataset_sha,
        "model_is_local": bool(args.model and Path(args.model).exists()),
    }


def _coerce_variant(value: str) -> ExperimentVariant | TurnVegaVariant:
    try:
        return ExperimentVariant(value)
    except ValueError:
        return TurnVegaVariant(value)


def build_agent_config(
    args: argparse.Namespace,
    trace_path: str | Path,
) -> AgentConfig:
    return AgentConfig(
        task_mode=TaskMode.TASK2,
        max_evidence_chars=args.max_evidence_chars,
        trace_path=str(trace_path),
        variant=_coerce_variant(args.variant),
        dataset_kind=args.dataset_kind,
        trace_schema=args.trace_schema,
        history_mode=args.history_mode,
        candidate_passages=args.candidate_passages,
        prompt_evidence=args.prompt_evidence,
        max_search_calls=args.max_search_calls,
        run_id=args.run_id,
        seed=getattr(args, "seed", 20260720),
        temperature=getattr(args, "temperature", 0.0),
    )


def build_run_config(
    args: argparse.Namespace,
    budget: ExperimentBudget,
    *,
    manifest_sha256: str,
    versions: Mapping[str, str | None] | None = None,
    git_commit: str | None = None,
    git_diff_sha256_value: str | None = None,
    started_at_utc: str | None = None,
    completed_at_utc: str | None = None,
) -> Dict[str, Any]:
    versions = dict(versions or collect_runtime_versions())
    started = started_at_utc or datetime.now(timezone.utc).isoformat()
    return {
        "run_id": args.run_id,
        "experiment_family": EXPERIMENT_FAMILY,
        "variant": args.variant,
        "dataset_kind": args.dataset_kind,
        "dataset_id": DATASET_IDS[args.dataset_kind],
        "dataset_revision": DATASET_REVISION,
        "dataset_path": args.dataset_path,
        "dataset_path_sha256": getattr(
            args,
            "dataset_path_sha256",
            None,
        ),
        "manifest_path": str(Path(args.manifest_path).resolve()),
        "manifest_sha256": manifest_sha256,
        "model_name": args.model,
        "model_weight_sha256": args.model_weight_sha256,
        "model_weight_lfs_oid": args.model_weight_lfs_oid,
        "model_is_local": getattr(args, "model_is_local", None),
        "python_version": versions.get("python"),
        "torch_version": versions.get("torch"),
        "transformers_version": versions.get("transformers"),
        "vllm_version": versions.get("vllm"),
        "runtime_versions": versions,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "seed": args.seed,
        "temperature": args.temperature,
        "candidate_budget": budget.candidate_passages,
        "prompt_evidence": budget.prompt_evidence,
        "max_search_calls": budget.max_search_calls,
        "experiment_budget": asdict(budget),
        "max_evidence_chars": args.max_evidence_chars,
        "backend": args.backend,
        "history_mode": args.history_mode,
        "trace_schema": args.trace_schema,
        "trace_filename": TRACE_FILENAME,
        "git_commit": git_commit,
        "git_diff_sha256": git_diff_sha256_value,
        "started_at_utc": started,
        "completed_at_utc": completed_at_utc,
    }


def _write_run_config(output_dir: Path, run_config: Mapping[str, Any]) -> None:
    target = output_dir / "run_config.json"
    temporary = output_dir / (
        ".run_config.json.tmp-"
        + str(os.getpid())
        + "-"
        + secrets.token_hex(8)
    )
    payload = (
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(output_dir, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _turn_dicts(row: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    turns = row.get("turns")
    if isinstance(turns, (list, tuple)):
        return [turn for turn in turns if isinstance(turn, Mapping)]
    if isinstance(turns, Mapping):
        lengths = [
            len(value)
            for value in turns.values()
            if isinstance(value, (list, tuple))
        ]
        if not lengths:
            return [dict(turns)]
        result: List[Mapping[str, Any]] = []
        for index in range(max(lengths, default=0)):
            result.append(
                {
                    key: (
                        value[index]
                        if isinstance(value, (list, tuple))
                        and index < len(value)
                        else value
                    )
                    for key, value in turns.items()
                }
            )
        return result
    return [row]


def _history_turn_count(message_history: Sequence[Mapping[str, Any]]) -> int:
    user_messages = sum(
        1
        for message in message_history
        if isinstance(message, Mapping) and message.get("role") == "user"
    )
    return user_messages if user_messages else len(message_history)


class TraceIdentityProvider:
    def __init__(self, identities: Sequence[Mapping[str, Any]]):
        self._identities = [dict(identity) for identity in identities]
        self._position = 0
        self._lock = threading.Lock()

    def take(
        self,
        query: str,
        message_history: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        with self._lock:
            if self._position >= len(self._identities):
                raise RuntimeError("trace identity provider exhausted")
            expected = self._identities[self._position]
            if query != expected["query"]:
                raise ValueError(
                    "trace query order mismatch: expected "
                    + repr(expected["query"])
                    + ", received "
                    + repr(query)
                )
            history_turn_count = _history_turn_count(message_history)
            if history_turn_count != expected["turn_index"]:
                raise ValueError(
                    "trace history order mismatch: expected turn_index "
                    + str(expected["turn_index"])
                    + ", received history for "
                    + str(history_turn_count)
                    + " turns"
                )
            self._position += 1
            return dict(expected)

    def __call__(self, query, image, message_history):
        del image
        return self.take(query, message_history)


def build_trace_identity_provider(
    selected_rows: Sequence[Mapping[str, Any]],
    batch_size: int,
) -> TraceIdentityProvider:
    """Return an ordered, thread-safe identity hook for evaluator turns."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows_with_turns = [
        (
            row,
            _turn_dicts(row),
        )
        for row in selected_rows
    ]
    for row, _ in rows_with_turns:
        session_key = row.get("session_key") or row.get("session_id")
        if session_key in OFFICIAL_SESSIONS_TO_SKIP:
            raise ValueError(
                "trace identities contain official skipped session: "
                + str(session_key)
            )
    identities: List[Dict[str, Any]] = []
    positions = [0] * len(rows_with_turns)
    pending = deque(
        index
        for index, (_, turns) in enumerate(rows_with_turns)
        if turns
    )
    while pending:
        current_convs = [
            pending.popleft()
            for _ in range(min(batch_size, len(pending)))
        ]
        for row_index in current_convs:
            row, turns = rows_with_turns[row_index]
            index = positions[row_index]
            turn = turns[index]
            session_key = (
                row.get("session_key") or row.get("session_id") or ""
            )
            identities.append(
                {
                    "session_key": session_key,
                    "turn_index": turn.get("turn_index", index),
                    "interaction_id": turn.get("interaction_id", ""),
                    "query": turn.get("query", row.get("query", "")),
                }
            )
            positions[row_index] += 1
            if positions[row_index] < len(turns):
                pending.appendleft(row_index)
    return TraceIdentityProvider(identities)


def select_manifest_rows(dataset: Any, manifest_rows: Sequence[Mapping[str, Any]]):
    indices = [row["source_index"] for row in manifest_rows]
    selected = dataset.select(indices)
    column_names = getattr(selected, "column_names", None)
    if column_names is not None:
        if "source_index" not in column_names:
            selected = selected.add_column("source_index", indices)
    else:
        selected = [
            {**dict(row), "source_index": source_index}
            for row, source_index in zip(selected, indices)
        ]
    validate_selected_rows(selected, manifest_rows)
    return selected


def _variant_support(args: argparse.Namespace) -> tuple[bool, str | None]:
    variant = _coerce_variant(args.variant)
    if isinstance(variant, ExperimentVariant):
        if args.dataset_kind == DatasetKind.TASK2.value:
            return True, None
        return (
            False,
            "legacy ExperimentVariant values are only implemented for task2",
        )
    if (
        variant is TurnVegaVariant.T2_B0
        and args.dataset_kind == DatasetKind.TASK2.value
    ):
        return True, None
    if (
        variant is TurnVegaVariant.T3_NO_HISTORY
        and args.dataset_kind == DatasetKind.TASK3.value
    ):
        return True, None
    return (
        False,
        f"variant {variant.value} is not implemented by the Task 4 runner",
    )


def _ensure_output_products_absent(output_dir: Path) -> None:
    if output_dir.is_symlink():
        raise FileExistsError(
            f"refusing symlink output directory: {output_dir}"
        )
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise FileExistsError(f"output path is not a directory: {output_dir}")
    first_entry = next(output_dir.iterdir(), None)
    if first_entry is not None:
        raise FileExistsError(
            f"output directory must be empty; found: {first_entry}"
        )


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if args.temperature != 0.0:
        raise ValueError("TurnVEGA Task 4 only supports temperature=0")
    budget = ExperimentBudget(
        candidate_passages=args.candidate_passages,
        prompt_evidence=args.prompt_evidence,
        max_search_calls=args.max_search_calls,
    )
    manifest_rows = load_validated_manifest(
        args.manifest_path,
        args.dataset_kind,
    )
    input_identity = validate_input_identity(args)
    for key, value in input_identity.items():
        setattr(args, key, value)
    supported, unsupported_reason = _variant_support(args)
    if args.dry_run and args.model and not args.model_is_local:
        supported = False
        unsupported_reason = unsupported_reason or (
            "remote model IDs are unsupported by the Task 4 runner"
        )
    if not supported and not args.dry_run:
        raise NotImplementedError(unsupported_reason)

    started_at = datetime.now(timezone.utc).isoformat()
    provenance_args = {
        "manifest_sha256": file_sha256(args.manifest_path),
        "versions": collect_runtime_versions(),
        "git_commit": current_git_commit(),
        "git_diff_sha256_value": current_git_diff_sha256(),
        "started_at_utc": started_at,
    }
    run_config = build_run_config(args, budget, **provenance_args)
    run_config.update(
        status="prepared" if supported else "unsupported",
        supported=supported,
        unsupported_reason=unsupported_reason,
        manifest_row_count=len(manifest_rows),
        source_indices=[row["source_index"] for row in manifest_rows],
    )

    output_dir = Path(args.output_dir)
    _ensure_output_products_absent(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / TRACE_FILENAME
    _write_run_config(output_dir, run_config)

    if not supported:
        run_config["completed_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        _write_run_config(output_dir, run_config)
        return run_config

    if args.dry_run:
        run_config["status"] = "dry_run_completed"
        run_config["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_run_config(output_dir, run_config)
        return run_config
    agent = None
    cleanup_errors: List[str] = []
    try:
        from datasets import load_dataset
        from local_evaluation import CRAGEvaluator

        if args.dataset_path:
            dataset = load_dataset(
                "parquet",
                data_files=args.dataset_path,
                split="train",
            )
        else:
            dataset = load_dataset(
                DATASET_IDS[args.dataset_kind],
                split="validation",
                revision=DATASET_REVISION,
            )
        selected = select_manifest_rows(dataset, manifest_rows)

        random.seed(args.seed)
        os.environ["CRAG_BACKEND"] = args.backend
        os.environ["CRAG_MODEL"] = args.model
        config = build_agent_config(args, trace_path)
        pipeline = build_search_pipeline(TaskMode.TASK2)
        agent = CourseRAGAgentV2(
            pipeline,
            backend=create_backend(config),
            config=config,
            trace_identity_provider=build_trace_identity_provider(
                selected,
                batch_size=config.batch_size,
            ),
        )
        evaluator = CRAGEvaluator(
            dataset=selected,
            agent=agent,
            eval_model_name=None,
            num_conversations=None,
            show_progress=True,
            num_workers=1,
        )
        turn_results, scores = evaluator.evaluate_agent()
        evaluator.save_results(turn_results, scores, str(output_dir))
        run_config["status"] = "results_saved"
        run_config["results_saved_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        _write_run_config(output_dir, run_config)
    except Exception as exc:
        run_config.update(
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        _write_run_config(output_dir, run_config)
        raise
    finally:
        if agent is not None:
            run_config["cleanup_started_at_utc"] = datetime.now(
                timezone.utc
            ).isoformat()
            cleanup_errors = agent.close()
            run_config["cleanup_errors"] = cleanup_errors
            run_config["cleanup_completed_at_utc"] = datetime.now(
                timezone.utc
            ).isoformat()
            _write_run_config(output_dir, run_config)

    run_config["status"] = "completed"
    run_config["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_run_config(output_dir, run_config)
    return {"run": run_config, "scores": scores}


def main() -> None:
    print(
        json.dumps(
            run_experiment(parse_args()),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

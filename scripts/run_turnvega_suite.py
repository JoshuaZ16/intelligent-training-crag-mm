#!/usr/bin/env python3
"""Run the frozen TurnVEGA core experiment suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, List, Mapping, Optional, Sequence, TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.turnvega_config import TurnVegaVariant
import scripts.audit_turnvega_run as turnvega_audit
from scripts.audit_turnvega_run import audit_triplet
from scripts.run_turnvega_triplet import (
    resolve_execution_mode,
    run_dual_gpu_smoke,
    run_triplet_commands,
)


DEFAULT_MATRIX_PATH = PROJECT_ROOT / "configs" / "turnvega_core_experiments.json"
PROTOCOL_VERSION = "v1.2-dual-gpu-crossover"
FROZEN_COMMON = {
    "protocol_version": PROTOCOL_VERSION,
    "seed": 20260720,
    "temperature": 0,
    "candidate_passages": 30,
    "prompt_evidence": 5,
    "max_search_calls": 4,
    "max_evidence_chars": 5500,
    "trace_schema": "v5",
    "execution_mode": "auto",
}
TOP_LEVEL_KEYS = frozenset({*FROZEN_COMMON, "entries"})
ENTRY_COMMON_KEYS = frozenset(
    {
        "pair_id",
        "dataset_kind",
        "phase",
        "history_mode",
        "manifest_path",
        "output_path",
        *FROZEN_COMMON,
    }
)
DEV_ENTRY_KEYS = ENTRY_COMMON_KEYS | {"anchor_variant", "variant"}
CONFIG_ENTRY_KEYS = ENTRY_COMMON_KEYS | {"config_id", "variant"}
CONTROL_ENTRY_KEYS = CONFIG_ENTRY_KEYS | {"control_of"}
INT_FIELDS = frozenset(
    {
        "seed",
        "temperature",
        "candidate_passages",
        "prompt_evidence",
        "max_search_calls",
        "max_evidence_chars",
    }
)
ALLOWED_VARIANTS = frozenset(item.value for item in TurnVegaVariant)
STATE_KEYS = frozenset(
    {
        "protocol_version",
        "matrix_sha256",
        "suite_run_id",
        "status",
        "execution_mode",
        "started_at_utc",
        "completed_at_utc",
        "entries",
    }
)
STATE_ENTRY_KEYS = frozenset(
    {
        "pair_id",
        "config_id",
        "output_path",
        "status",
        "started_at_utc",
        "completed_at_utc",
        "exit_status",
        "audit_sha256",
        "proof_sha256",
        "error",
    }
)
RESOLUTION_KEYS = frozenset(
    {"pair_id", "status", "resolution", "recorded_by", "recorded_at_utc"}
)
PROOF_KEYS = frozenset(
    {
        "protocol_version",
        "matrix_sha256",
        "suite_run_id",
        "pair_id",
        "config_id",
        "phase",
        "output_path",
        "execution_mode",
        "audit_sha256",
        "terminal_filename",
        "terminal_sha256",
    }
)


def _spec(
    pair_id: str,
    dataset_kind: str,
    phase: str,
    variant: str,
    history_mode: str,
    manifest_name: str,
    output_suffix: str,
    *,
    anchor_variant: Optional[str] = None,
    config_id: Optional[str] = None,
    control_of: Optional[str] = None,
) -> Dict[str, str]:
    value = {
        "pair_id": pair_id,
        "dataset_kind": dataset_kind,
        "phase": phase,
        "variant": variant,
        "history_mode": history_mode,
        "manifest_path": (
            "artifacts/turnvega/20260720/manifests/" + manifest_name
        ),
        "output_path": "artifacts/turnvega/20260720/" + output_suffix,
    }
    if anchor_variant is not None:
        value["anchor_variant"] = anchor_variant
    if config_id is not None:
        value["config_id"] = config_id
    if control_of is not None:
        value["control_of"] = control_of
    return value


EXPECTED_ENTRIES = (
    _spec("T2-R1", "task2", "dev", "t2_candidate_grid", "none", "t2_dev80.json", "task2/dev/r1_candidate_grid", anchor_variant="t2_budget_b0"),
    _spec("T2-R2", "task2", "dev", "t2_relation_grid", "none", "t2_dev80.json", "task2/dev/r2_relation_grid", anchor_variant="t2_candidate_grid"),
    _spec("T2-R3", "task2", "dev", "t2_circularity", "none", "t2_dev80.json", "task2/dev/r3_circularity", anchor_variant="t2_relation_grid"),
    _spec("T2-R4", "task2", "dev", "t2_answerability", "none", "t2_dev80.json", "task2/dev/r4_answerability", anchor_variant="t2_circularity"),
    _spec("T2-R5", "task2", "dev", "t2_evidence_card", "none", "t2_dev80.json", "task2/dev/r5_evidence_card", anchor_variant="t2_answerability"),
    _spec("T2-R6", "task2", "dev", "t2_typed_repair", "none", "t2_dev80.json", "task2/dev/r6_typed_repair", anchor_variant="t2_evidence_card"),
    _spec("T2-C0", "task2", "confirmatory", "t2_b0", "none", "t2_test120.json", "task2/test/confirmatory/c0_b0", config_id="t2_b0"),
    _spec("T2-C1", "task2", "confirmatory", "t2_core_full", "none", "t2_test120.json", "task2/test/confirmatory/c1_core_full", config_id="t2_core_full"),
    _spec("T2-C2", "task2", "confirmatory", "t2_budget_b0", "none", "t2_test120.json", "task2/test/confirmatory/c2_budget_b0", config_id="t2_budget_b0"),
    _spec("T3-R1", "task3", "dev", "t3_full_history", "full_history", "t3_dev40.json", "task3/dev/r1_full_history", anchor_variant="t3_no_history"),
    _spec("T3-R2", "task3", "dev", "t3_last_turn", "last_turn", "t3_dev40.json", "task3/dev/r2_last_turn", anchor_variant="t3_full_history"),
    _spec("T3-R3", "task3", "dev", "t3_user_only", "user_only", "t3_dev40.json", "task3/dev/r3_user_only", anchor_variant="t3_last_turn"),
    _spec("T3-R4", "task3", "dev", "t3_structured_state", "structured_state", "t3_dev40.json", "task3/dev/r4_structured_state", anchor_variant="t3_last_turn"),
    _spec("T3-R5", "task3", "dev", "t3_verified_state", "verified_state", "t3_dev40.json", "task3/dev/r5_verified_state", anchor_variant="t3_structured_state"),
    _spec("T3-R6", "task3", "dev", "t3_state_gated", "state_gated", "t3_dev40.json", "task3/dev/r6_state_gated", anchor_variant="t3_verified_state"),
    _spec("T3-C0", "task3", "confirmatory", "t3_last_turn", "last_turn", "t3_test60.json", "task3/test/confirmatory/c0_last_turn", config_id="t3_last_turn"),
    _spec("T3-C1", "task3", "confirmatory", "t3_core_full", "state_gated", "t3_test60.json", "task3/test/confirmatory/c1_core_full", config_id="t3_core_full"),
    _spec("T3-C2", "task3", "confirmatory", "t3_core_full", "equal_token_summary_control", "t3_test60.json", "task3/test/confirmatory/c2_equal_token_summary", config_id="t3_equal_token_history_summary_control", control_of="t3_core_full"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_regular_bytes(path: Path, description: str) -> bytes:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(description + " must not be a symlink")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(description + " is not a readable regular file") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(description + " must be a regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _load_regular_json(path: Path, description: str) -> object:
    try:
        return json.loads(_read_regular_bytes(path, description).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(description + " contains invalid JSON") from exc


def _validate_relative_path(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(field + " must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(field + " must be a safe relative path")
    if "\\" in value:
        raise ValueError(field + " must use POSIX path separators")
    return value


def _validate_matrix(payload: object) -> Dict[str, object]:
    if type(payload) is not dict:
        raise ValueError("matrix must be a JSON object")
    if set(payload) != TOP_LEVEL_KEYS:
        raise ValueError("matrix top-level keys do not match the frozen schema")
    for key, expected in FROZEN_COMMON.items():
        value = payload.get(key)
        required_type = int if key in INT_FIELDS else str
        if type(value) is not required_type or value != expected:
            raise ValueError("matrix frozen field drift: " + key)
    entries = payload.get("entries")
    if type(entries) is not list or len(entries) != len(EXPECTED_ENTRIES):
        raise ValueError("matrix must contain exactly 18 ordered entries")
    ids = []
    outputs = []
    validated_entries: List[Dict[str, object]] = []
    for index, (entry, expected) in enumerate(zip(entries, EXPECTED_ENTRIES)):
        if type(entry) is not dict:
            raise ValueError("entry " + str(index) + " must be an object")
        expected_keys = (
            DEV_ENTRY_KEYS
            if "anchor_variant" in expected
            else CONTROL_ENTRY_KEYS
            if "control_of" in expected
            else CONFIG_ENTRY_KEYS
        )
        if set(entry) != expected_keys:
            raise ValueError("entry " + str(index) + " keys do not match schema")
        for key in expected_keys:
            required_type = int if key in INT_FIELDS else str
            if type(entry.get(key)) is not required_type:
                raise ValueError("entry " + str(index) + " field type: " + key)
        for key, frozen in FROZEN_COMMON.items():
            if entry[key] != frozen:
                raise ValueError("entry budget/protocol drift: " + key)
        for key, expected_value in expected.items():
            if entry[key] != expected_value:
                raise ValueError(
                    "entry order or frozen value mismatch at " + expected["pair_id"]
                )
        _validate_relative_path(entry["manifest_path"], "manifest_path")
        _validate_relative_path(entry["output_path"], "output_path")
        variants = [entry["variant"]]
        if "anchor_variant" in entry:
            variants.append(entry["anchor_variant"])
        if any(value not in ALLOWED_VARIANTS for value in variants):
            raise ValueError("unknown TurnVEGA variant")
        ids.append(entry["pair_id"])
        outputs.append(entry["output_path"])
        validated_entries.append(dict(entry))
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate pair ID")
    if len(outputs) != len(set(outputs)):
        raise ValueError("duplicate output path")
    result = dict(payload)
    result["entries"] = validated_entries
    return result


def _load_matrix_and_hash(path: Path) -> tuple[Dict[str, object], str]:
    data = _read_regular_bytes(Path(path), "matrix")
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("matrix contains invalid JSON") from exc
    return _validate_matrix(payload), hashlib.sha256(data).hexdigest()


def load_matrix(path: Path = DEFAULT_MATRIX_PATH) -> Dict[str, object]:
    """Load a matrix without defaults, inheritance, or type coercion."""
    return _load_matrix_and_hash(Path(path))[0]


def _reject_symlink_chain(path: Path, root: Path) -> None:
    root = Path(root).absolute()
    candidate = Path(path).absolute()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("path escapes project root") from exc
    while True:
        try:
            mode = candidate.lstat().st_mode
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISLNK(mode):
                raise ValueError("refusing symlink path: " + str(candidate))
        if candidate == root:
            return
        candidate = candidate.parent


def _project_path(root: Path, relative: str) -> Path:
    _validate_relative_path(relative, "project path")
    root = Path(root).absolute()
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    _reject_symlink_chain(candidate, root)
    try:
        candidate.resolve(strict=False).relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("project path escapes through a symlink") from exc
    return candidate


@contextmanager
def _open_project_directory(root: Path, relative: str):
    """Open a relative directory one component at a time without symlinks."""
    _validate_relative_path(relative, "output path")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptors: List[int] = []
    try:
        current = os.open(Path(root).absolute(), flags)
        descriptors.append(current)
        for component in PurePosixPath(relative).parts:
            current = os.open(component, flags, dir_fd=current)
            descriptors.append(current)
        yield current
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _read_regular_at(directory_fd: int, filename: str, description: str) -> bytes:
    if not filename or filename in (".", "..") or "/" in filename:
        raise ValueError(description + " filename is unsafe")
    descriptor = os.open(
        filename,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(description + " must be a regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _secure_output_files(
    root: Path, relative: str, filenames: Sequence[str]
) -> Dict[str, bytes]:
    path = _project_path(root, relative)
    _reject_symlink_chain(path, root)
    with _open_project_directory(root, relative) as directory_fd:
        return {
            name: _read_regular_at(directory_fd, name, name)
            for name in filenames
        }


def _secure_project_file(root: Path, relative: str, description: str) -> bytes:
    path = PurePosixPath(_validate_relative_path(relative, description))
    parent = path.parent.as_posix()
    if parent in ("", "."):
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        directory_fd = os.open(Path(root).absolute(), flags)
        try:
            return _read_regular_at(directory_fd, path.name, description)
        finally:
            os.close(directory_fd)
    with _open_project_directory(root, parent) as directory_fd:
        return _read_regular_at(directory_fd, path.name, description)


def _ensure_empty_output(path: Path, root: Path) -> None:
    _reject_symlink_chain(path, root)
    if not path.exists():
        return
    if not path.is_dir():
        raise ValueError("output path is not a directory: " + str(path))
    if next(path.iterdir(), None) is not None:
        raise ValueError("unexpected non-empty output: " + str(path))


def _atomic_json(path: Path, payload: Mapping[str, object], root: Path) -> None:
    path = Path(path)
    _reject_symlink_chain(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_chain(path.parent, root)
    temporary = path.parent / (
        "." + path.name + ".tmp-" + str(os.getpid()) + "-" + str(time.time_ns())
    )
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    try:
        data = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode(
            "utf-8"
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def _state_entry(entry: Mapping[str, object]) -> Dict[str, object]:
    return {
        "pair_id": entry["pair_id"],
        "config_id": entry.get("config_id"),
        "output_path": entry["output_path"],
        "status": "not_started",
        "started_at_utc": None,
        "completed_at_utc": None,
        "exit_status": None,
        "audit_sha256": None,
        "proof_sha256": None,
        "error": None,
    }


def _terminal_filename(entry: Mapping[str, object]) -> str:
    return "triplet_summary.json" if entry["phase"] == "dev" else "run_config.json"


def _write_suite_proof(
    entry: Mapping[str, object],
    root: Path,
    state: Mapping[str, object],
    audit_sha256: str,
) -> str:
    relative = str(entry["output_path"])
    terminal_filename = _terminal_filename(entry)
    terminal = _secure_output_files(root, relative, (terminal_filename,))[
        terminal_filename
    ]
    proof = {
        "protocol_version": state["protocol_version"],
        "matrix_sha256": state["matrix_sha256"],
        "suite_run_id": state["suite_run_id"],
        "pair_id": entry["pair_id"],
        "config_id": entry.get("config_id"),
        "phase": entry["phase"],
        "output_path": entry["output_path"],
        "execution_mode": state["execution_mode"],
        "audit_sha256": audit_sha256,
        "terminal_filename": terminal_filename,
        "terminal_sha256": hashlib.sha256(terminal).hexdigest(),
    }
    output_path = _project_path(root, relative)
    _atomic_json(output_path / "suite_proof.json", proof, root)
    stored = _secure_output_files(root, relative, ("suite_proof.json",))[
        "suite_proof.json"
    ]
    return hashlib.sha256(stored).hexdigest()


def _new_state(
    matrix: Mapping[str, object],
    mode: str,
    matrix_sha256: str,
    suite_run_id: str,
) -> Dict[str, object]:
    entries = matrix["entries"]
    assert isinstance(entries, list)
    return {
        "protocol_version": matrix["protocol_version"],
        "matrix_sha256": matrix_sha256,
        "suite_run_id": suite_run_id,
        "status": "prepared",
        "execution_mode": mode,
        "started_at_utc": _utc_now(),
        "completed_at_utc": None,
        "entries": [_state_entry(entry) for entry in entries],
    }


def _parse_audit(data: bytes) -> tuple[bool, str]:
    digest = hashlib.sha256(data).hexdigest()
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("audit contains invalid JSON") from exc
    if type(payload) is not dict or type(payload.get("accepted")) is not bool:
        raise ValueError("audit accepted must be an explicit boolean")
    return payload["accepted"], digest


def _read_audit_at(root: Path, output_path: str) -> tuple[bool, str]:
    data = _secure_output_files(root, output_path, ("audit.json",))["audit.json"]
    return _parse_audit(data)


def _normalize_exit_status(result: object) -> int:
    if type(result) is int:
        return result
    returncode = getattr(result, "returncode", None)
    if type(returncode) is int:
        return returncode
    if type(result) is dict and type(result.get("runner_valid")) is bool:
        return 0 if result["runner_valid"] else 1
    raise TypeError("pair runner must return an integer exit status")


def _history_mode_for_variant(variant: str) -> str:
    values = {
        "t3_no_history": "none",
        "t3_full_history": "full_history",
        "t3_last_turn": "last_turn",
        "t3_user_only": "user_only",
        "t3_structured_state": "structured_state",
        "t3_verified_state": "verified_state",
        "t3_state_gated": "state_gated",
        "t3_core_full": "state_gated",
    }
    return values.get(variant, "none")


def _experiment_command(
    entry: Mapping[str, object],
    variant: str,
    history_mode: str,
    run_id: str,
    project_root: Path,
    *,
    python: str,
    backend: str,
    model: Optional[str],
    dataset_path: Optional[str],
) -> List[str]:
    command = [
        python,
        str(project_root / "scripts" / "turnvega_experiment.py"),
        "--dataset-kind", str(entry["dataset_kind"]),
        "--variant", variant,
        "--manifest-path", str(_project_path(project_root, str(entry["manifest_path"]))),
        "--candidate-passages", str(entry["candidate_passages"]),
        "--prompt-evidence", str(entry["prompt_evidence"]),
        "--max-search-calls", str(entry["max_search_calls"]),
        "--max-evidence-chars", str(entry["max_evidence_chars"]),
        "--trace-schema", str(entry["trace_schema"]),
        "--seed", str(entry["seed"]),
        "--temperature", str(entry["temperature"]),
        "--history-mode", history_mode,
        "--run-id", run_id,
        "--output-dir", "{output_dir}",
        "--backend", backend,
    ]
    if model:
        command.extend(["--model", model])
    if dataset_path:
        command.extend(["--dataset-path", dataset_path])
    return command


def run_with_task5(
    entry: Mapping[str, object],
    output_path: Path,
    execution_mode: str,
    *,
    project_root: Path = PROJECT_ROOT,
    python: str = sys.executable,
    backend: str = "vllm",
    model: Optional[str] = None,
    dataset_paths: Optional[Mapping[str, str]] = None,
) -> int:
    """Run one frozen item through Task5's public triplet and audit APIs."""
    target = str(entry["variant"])
    anchor = str(entry.get("anchor_variant", target))
    pair_id = str(entry["pair_id"]).lower()
    dataset_path = (dataset_paths or {}).get(str(entry["dataset_kind"]))
    target_history = str(entry["history_mode"])
    anchor_history = (
        target_history
        if "config_id" in entry
        else _history_mode_for_variant(anchor)
    )
    commands = {
        "anchor_before": _experiment_command(
            entry, anchor, anchor_history, pair_id + "-anchor-before", project_root,
            python=python, backend=backend, model=model, dataset_path=dataset_path,
        ),
        "variant": _experiment_command(
            entry, target, target_history, pair_id + "-variant-{run_suffix}", project_root,
            python=python, backend=backend, model=model, dataset_path=dataset_path,
        ),
        "anchor_after": _experiment_command(
            entry, anchor, anchor_history, pair_id + "-anchor-after", project_root,
            python=python, backend=backend, model=model, dataset_path=dataset_path,
        ),
    }
    summary = run_triplet_commands(
        commands,
        output_path,
        execution_mode=execution_mode,
        require_gpu_telemetry=(backend == "vllm"),
    )
    if not summary.get("runner_valid"):
        return 1
    audit = audit_triplet(
        output_path, _project_path(project_root, str(entry["manifest_path"]))
    )
    return 0 if audit.get("accepted") is True else 1


def audit_single_config(
    entry: Mapping[str, object], output_path: Path, project_root: Path
) -> Dict[str, object]:
    """Strictly audit one confirmatory configuration without a triplet."""
    reasons: List[str] = []
    runner_valid = True
    schema_valid = True
    budget_valid = True
    scope_valid = True
    try:
        relative = str(entry["output_path"])
        files = _secure_output_files(
            project_root, relative, ("run_config.json", "agent_trace_v5.jsonl")
        )
        manifest_data = _secure_project_file(
            project_root, str(entry["manifest_path"]), "manifest"
        )
        config = json.loads(files["run_config.json"].decode("utf-8"))
        manifest = json.loads(manifest_data.decode("utf-8"))
        trace_text = files["agent_trace_v5.jsonl"].decode("utf-8")
        rows = turnvega_audit._trace_rows_text(trace_text)
        if type(config) is not dict:
            raise ValueError("run_config must be an object")
        if type(manifest) is not list or not manifest or not all(
            type(row) is dict for row in manifest
        ):
            raise ValueError("manifest must be a non-empty object list")
        expected_count = len(manifest)
        manifest_sha = hashlib.sha256(manifest_data).hexdigest()
        manifest_path = _project_path(
            project_root, str(entry["manifest_path"])
        ).resolve()
        config_path = config.get("manifest_path")
        config_contract = {
            "status": "completed",
            "experiment_family": "turnvega",
            "variant": entry["variant"],
            "dataset_kind": entry["dataset_kind"],
            "manifest_sha256": manifest_sha,
            "manifest_row_count": expected_count,
            "seed": entry["seed"],
            "candidate_budget": entry["candidate_passages"],
            "prompt_evidence": entry["prompt_evidence"],
            "max_search_calls": entry["max_search_calls"],
            "max_evidence_chars": entry["max_evidence_chars"],
            "history_mode": entry["history_mode"],
            "trace_schema": "v5",
            "trace_filename": "agent_trace_v5.jsonl",
        }
        for key, expected in config_contract.items():
            value = config.get(key)
            if key in (
                "manifest_row_count",
                "seed",
                "candidate_budget",
                "prompt_evidence",
                "max_search_calls",
                "max_evidence_chars",
            ):
                valid = type(value) is int and value == expected
            else:
                valid = type(value) is str and value == expected
            if not valid:
                schema_valid = False
                reasons.append("run_config mismatch: " + key)
        temperature = config.get("temperature")
        if (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or temperature != 0
        ):
            schema_valid = False
            reasons.append("run_config mismatch: temperature")
        if (
            type(config_path) is not str
            or Path(config_path).resolve() != manifest_path
        ):
            scope_valid = False
            reasons.append("run_config manifest path mismatch")
        run_id = config.get("run_id")
        if type(run_id) is not str or not run_id:
            schema_valid = False
            reasons.append("run_config run_id is missing")
        if len(rows) != expected_count:
            schema_valid = False
            reasons.append("trace count differs from manifest")
        expected_order = [
            turnvega_audit._ordered_identity(row) for row in manifest
        ]
        actual_order = []
        keys = []
        for row_index, row in enumerate(rows):
            missing = turnvega_audit.V5_REQUIRED_FIELDS.difference(row)
            if (
                missing
                or not turnvega_audit._trace_types_valid(row)
                or row.get("trace_schema") != "v5"
                or row.get("status") != "ok"
                or row.get("run_id") != run_id
                or row.get("dataset_kind") != entry["dataset_kind"]
                or row.get("experiment_variant") != entry["variant"]
                or row.get("history_mode") != entry["history_mode"]
            ):
                schema_valid = False
                reasons.append("trace row contract mismatch: " + str(row_index))
            search_calls = row.get("search_call_count")
            candidates = row.get("candidate_budget")
            if (
                type(search_calls) is not int
                or search_calls < 0
                or search_calls > entry["max_search_calls"]
                or type(candidates) is not int
                or candidates != entry["candidate_passages"]
            ):
                budget_valid = False
                reasons.append("trace row budget mismatch: " + str(row_index))
            try:
                packet = json.dumps(
                    row.get("evidence_packet"),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
                packet_sha = hashlib.sha256(packet).hexdigest()
            except (TypeError, ValueError):
                packet_sha = ""
            if row.get("evidence_packet_sha256") != packet_sha:
                schema_valid = False
                reasons.append("trace evidence hash mismatch: " + str(row_index))
            actual_order.append(turnvega_audit._ordered_identity(row))
            keys.append(
                turnvega_audit._turn_key(row, str(entry["dataset_kind"]))
            )
        if actual_order != expected_order or len(set(keys)) != len(keys):
            schema_valid = False
            reasons.append("trace identity/order differs from manifest")
        if config.get("status") != "completed":
            runner_valid = False
            reasons.append("single run is not terminal completed")
    except BaseException as exc:
        runner_valid = False
        schema_valid = False
        budget_valid = False
        scope_valid = False
        reasons.append(type(exc).__name__ + ": " + str(exc))
    accepted = runner_valid and schema_valid and budget_valid and scope_valid
    result: Dict[str, object] = {
        "pair_id": entry["pair_id"],
        "config_id": entry["config_id"],
        "runner_valid": runner_valid,
        "scope_isolation_valid": scope_valid,
        "schema_valid": schema_valid,
        "budget_valid": budget_valid,
        "accepted": accepted,
        "reasons": reasons,
    }
    if output_path.is_dir() and not output_path.is_symlink():
        _atomic_json(output_path / "audit.json", result, project_root)
    return result


def run_single_config(
    entry: Mapping[str, object],
    output_path: Path,
    execution_mode: str,
    *,
    project_root: Path = PROJECT_ROOT,
    python: str = sys.executable,
    backend: str = "vllm",
    model: Optional[str] = None,
    dataset_paths: Optional[Mapping[str, str]] = None,
) -> int:
    """Execute one confirmatory config exactly once, then audit it in-process."""
    dataset_path = (dataset_paths or {}).get(str(entry["dataset_kind"]))
    command = _experiment_command(
        entry,
        str(entry["variant"]),
        str(entry["history_mode"]),
        str(entry["pair_id"]).lower() + "-single",
        project_root,
        python=python,
        backend=backend,
        model=model,
        dataset_path=dataset_path,
    )
    command = [str(output_path) if value == "{output_dir}" else value for value in command]
    completed = subprocess.run(command, cwd=project_root, check=False)
    if completed.returncode != 0:
        return int(completed.returncode)
    audit_single_config(entry, output_path, project_root)
    return 0


def resolve_suite_execution_mode(requested: str, python: str = sys.executable) -> str:
    if requested != "auto":
        return resolve_execution_mode(
            requested, pre_formal_smoke=False, dual_gpu_smoke_ok=None
        )
    smoke_code = (
        "import torch;assert torch.cuda.is_available();"
        "x=torch.empty(1,device='cuda');assert x.is_cuda"
    )
    dual_ok = run_dual_gpu_smoke([python, "-c", smoke_code])
    return resolve_execution_mode(
        requested, pre_formal_smoke=True, dual_gpu_smoke_ok=dual_ok
    )


def _resolution_allows(path: Path, pair_id: str) -> bool:
    data = _read_regular_bytes(path, "resolution log")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("resolution log must be UTF-8 JSONL") from exc
    records: List[Dict[str, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "resolution log line " + str(line_number) + " is invalid JSON"
            ) from exc
        if type(record) is not dict or set(record) != RESOLUTION_KEYS:
            raise ValueError("resolution records must match the exact schema")
        if not all(
            type(record.get(field)) is str and bool(record[field].strip())
            for field in RESOLUTION_KEYS
        ):
            raise ValueError("resolution record fields must be non-empty strings")
        records.append(dict(record))
    ids = [record["pair_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("resolution log contains duplicate pair IDs")
    targets = [record for record in records if record["pair_id"] == pair_id]
    return len(targets) == 1 and targets[0]["status"] == "resolved"


def _validate_failed_state(
    state: object,
    matrix: Mapping[str, object],
    from_pair: str,
    matrix_sha256: str,
    project_root: Path,
) -> int:
    if type(state) is not dict or set(state) != STATE_KEYS:
        raise ValueError("suite state does not match the exact schema")
    if state.get("status") != "failed":
        raise ValueError("resume requires a previously failed suite state")
    if (
        state.get("protocol_version") != matrix["protocol_version"]
        or type(state.get("matrix_sha256")) is not str
        or state.get("matrix_sha256") != matrix_sha256
        or type(state.get("suite_run_id")) is not str
        or not state["suite_run_id"].strip()
        or state.get("execution_mode")
        not in ("dual_gpu_crossover", "sequential_triplet")
        or type(state.get("started_at_utc")) is not str
        or not state["started_at_utc"].strip()
        or type(state.get("completed_at_utc")) is not str
        or not state["completed_at_utc"].strip()
    ):
        raise ValueError("suite state identity or terminal fields are invalid")
    stored = state.get("entries")
    matrix_entries = matrix["entries"]
    if type(stored) is not list or not isinstance(matrix_entries, list):
        raise ValueError("suite state entries are invalid")
    if len(stored) != len(matrix_entries):
        raise ValueError("suite state does not match the frozen matrix")
    failed_indices = []
    for index, (item, entry) in enumerate(zip(stored, matrix_entries)):
        if type(item) is not dict:
            raise ValueError("suite state item is invalid")
        if set(item) != STATE_ENTRY_KEYS:
            raise ValueError("suite state item keys do not match exact schema")
        if (
            item.get("pair_id") != entry["pair_id"]
            or item.get("config_id") != entry.get("config_id")
            or item.get("output_path") != entry["output_path"]
        ):
            raise ValueError("suite state order does not match the matrix")
        status = item.get("status")
        if status not in ("completed", "failed", "not_started"):
            raise ValueError("suite state item status is invalid")
        if status == "completed":
            if (
                type(item.get("started_at_utc")) is not str
                or not item["started_at_utc"].strip()
                or type(item.get("completed_at_utc")) is not str
                or not item["completed_at_utc"].strip()
                or type(item.get("exit_status")) is not int
                or item["exit_status"] != 0
                or type(item.get("audit_sha256")) is not str
                or re.fullmatch(r"[0-9a-f]{64}", item["audit_sha256"]) is None
                or type(item.get("proof_sha256")) is not str
                or re.fullmatch(r"[0-9a-f]{64}", item["proof_sha256"]) is None
                or item.get("error") is not None
            ):
                raise ValueError("completed state item fields are invalid")
        elif status == "failed":
            audit_hash = item.get("audit_sha256")
            proof_hash = item.get("proof_sha256")
            if (
                type(item.get("started_at_utc")) is not str
                or not item["started_at_utc"].strip()
                or type(item.get("completed_at_utc")) is not str
                or not item["completed_at_utc"].strip()
                or type(item.get("exit_status")) is not int
                or type(item.get("error")) is not str
                or not item["error"].strip()
                or (
                    audit_hash is not None
                    and (
                        type(audit_hash) is not str
                        or re.fullmatch(r"[0-9a-f]{64}", audit_hash) is None
                    )
                )
                or (
                    proof_hash is not None
                    and (
                        type(proof_hash) is not str
                        or re.fullmatch(r"[0-9a-f]{64}", proof_hash) is None
                    )
                )
            ):
                raise ValueError("failed state item fields are invalid")
        else:
            if any(
                item.get(field) is not None
                for field in (
                    "started_at_utc",
                    "completed_at_utc",
                    "exit_status",
                    "audit_sha256",
                    "proof_sha256",
                    "error",
                )
            ):
                raise ValueError("not_started state item has forged terminal fields")
        if item.get("status") == "failed":
            failed_indices.append(index)
    if len(failed_indices) != 1:
        raise ValueError("suite state must contain exactly one failed item")
    failed = failed_indices[0]
    if from_pair != matrix_entries[failed]["pair_id"]:
        raise ValueError("--from-pair must name the recorded failed item")
    if any(item.get("status") != "completed" for item in stored[:failed]):
        raise ValueError("resume cannot skip an earlier unfinished item")
    if any(item.get("status") != "not_started" for item in stored[failed + 1 :]):
        raise ValueError("resume cannot skip a later started item")
    for index, (item, entry) in enumerate(zip(stored[:failed], matrix_entries[:failed])):
        relative = str(entry["output_path"])
        terminal_name = _terminal_filename(entry)
        files = _secure_output_files(
            project_root,
            relative,
            (terminal_name, "audit.json", "suite_proof.json"),
        )
        accepted, digest = _parse_audit(files["audit.json"])
        if not accepted or digest != item["audit_sha256"]:
            raise ValueError("completed prefix audit proof mismatch at " + str(index))
        proof_bytes = files["suite_proof.json"]
        if hashlib.sha256(proof_bytes).hexdigest() != item["proof_sha256"]:
            raise ValueError("completed prefix suite proof hash mismatch")
        try:
            proof = json.loads(proof_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("completed prefix suite proof is invalid") from exc
        expected_proof = {
            "protocol_version": state["protocol_version"],
            "matrix_sha256": state["matrix_sha256"],
            "suite_run_id": state["suite_run_id"],
            "pair_id": entry["pair_id"],
            "config_id": entry.get("config_id"),
            "phase": entry["phase"],
            "output_path": entry["output_path"],
            "execution_mode": state["execution_mode"],
            "audit_sha256": item["audit_sha256"],
            "terminal_filename": terminal_name,
            "terminal_sha256": hashlib.sha256(files[terminal_name]).hexdigest(),
        }
        if type(proof) is not dict or set(proof) != PROOF_KEYS or proof != expected_proof:
            raise ValueError("completed prefix suite proof binding mismatch")
        try:
            terminal = json.loads(files[terminal_name].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("completed prefix terminal config is invalid") from exc
        if type(terminal) is not dict or terminal.get("status") != "completed":
            raise ValueError("completed prefix is not terminal completed")
        if entry["phase"] == "dev":
            if terminal.get("runner_valid") is not True:
                raise ValueError("completed dev prefix runner is not valid")
        elif (
            terminal.get("variant") != entry["variant"]
            or terminal.get("dataset_kind") != entry["dataset_kind"]
        ):
            raise ValueError("completed confirmatory prefix identity mismatch")
        output_path = _project_path(project_root, relative)
        manifest_path = _project_path(
            project_root, str(entry["manifest_path"])
        )
        if entry["phase"] == "dev":
            fresh_audit = audit_triplet(output_path, manifest_path)
        else:
            fresh_audit = audit_single_config(entry, output_path, project_root)
        if type(fresh_audit) is not dict or fresh_audit.get("accepted") is not True:
            raise ValueError("completed prefix fails fresh full audit")
        freshly_accepted, fresh_digest = _read_audit_at(project_root, relative)
        if (
            not freshly_accepted
            or fresh_digest != item["audit_sha256"]
            or fresh_digest != proof["audit_sha256"]
        ):
            raise ValueError("completed prefix fresh audit hash mismatch")
    failed_item = stored[failed]
    assert isinstance(failed_item, dict)
    failed_hash = failed_item.get("audit_sha256")
    if failed_hash is not None:
        _, actual_hash = _read_audit_at(
            project_root, str(matrix_entries[failed]["output_path"])
        )
        if actual_hash != failed_hash:
            raise ValueError("failed item audit hash mismatch")
    return failed


def _archive_failed_output(path: Path, root: Path) -> None:
    _reject_symlink_chain(path, root)
    if not path.exists():
        return
    index = 1
    while True:
        archive = Path(str(path) + ".failed-attempt-" + str(index))
        _reject_symlink_chain(archive, root)
        if not archive.exists():
            os.replace(path, archive)
            return
        index += 1


def _fail(
    state: Dict[str, object],
    item: Dict[str, object],
    state_path: Path,
    root: Path,
    stdout: TextIO,
    message: str,
) -> int:
    item["status"] = "failed"
    item["error"] = message
    item["completed_at_utc"] = _utc_now()
    state["status"] = "failed"
    state["completed_at_utc"] = _utc_now()
    _atomic_json(state_path, state, root)
    print("FAILED " + str(item["pair_id"]), file=stdout)
    return 1


def run_suite(
    matrix_path: Path = DEFAULT_MATRIX_PATH,
    *,
    project_root: Path = PROJECT_ROOT,
    status_path: Optional[Path] = None,
    pair_runner: Optional[Callable[[Mapping[str, object], Path, str], object]] = None,
    single_runner: Optional[Callable[[Mapping[str, object], Path, str], object]] = None,
    runner: Optional[Callable[[Mapping[str, object], Path, str], object]] = None,
    execution_mode_resolver: Optional[Callable[[str], str]] = None,
    from_pair: Optional[str] = None,
    resolution_log: Optional[Path] = None,
    stdout: Optional[TextIO] = None,
    python: str = sys.executable,
    backend: str = "vllm",
    model: Optional[str] = None,
    dataset_paths: Optional[Mapping[str, str]] = None,
) -> int:
    """Run all 18 entries in order, stopping at the first unaudited result."""
    matrix, matrix_sha256 = _load_matrix_and_hash(Path(matrix_path))
    root = Path(project_root).absolute()
    state_path = Path(status_path or root / "artifacts/turnvega/20260720/suite_status.json").absolute()
    output = stdout or sys.stdout
    if runner is not None and (pair_runner is not None or single_runner is not None):
        raise ValueError("generic runner cannot be mixed with phase runners")
    selected_pair_runner = pair_runner or runner
    selected_single_runner = single_runner or runner
    entries = matrix["entries"]
    assert isinstance(entries, list)
    try:
        _reject_symlink_chain(state_path, root)
        if from_pair is None:
            if resolution_log is not None:
                raise ValueError("--resolution-log requires --from-pair")
            if state_path.exists():
                raise ValueError("suite status already exists")
            resolver = execution_mode_resolver or (
                lambda requested: resolve_suite_execution_mode(requested, python)
            )
            mode = resolver(str(matrix["execution_mode"]))
            if mode not in ("dual_gpu_crossover", "sequential_triplet"):
                raise ValueError("execution mode was not frozen before formal runs")
            state = _new_state(
                matrix, mode, matrix_sha256, secrets.token_hex(16)
            )
            _atomic_json(state_path, state, root)
            start_index = 0
        else:
            if from_pair not in [entry["pair_id"] for entry in entries]:
                raise ValueError("--from-pair does not exist in the matrix")
            if resolution_log is None:
                raise ValueError("resume requires --resolution-log")
            if not _resolution_allows(Path(resolution_log), from_pair):
                raise ValueError("resolution log has no matching resolved record")
            state = _load_regular_json(state_path, "suite status")
            start_index = _validate_failed_state(
                state,
                matrix,
                from_pair,
                matrix_sha256,
                root,
            )
            mode = state.get("execution_mode")
            if mode not in ("dual_gpu_crossover", "sequential_triplet"):
                raise ValueError("stored execution mode is not frozen")
            failed_output = _project_path(
                root, str(entries[start_index]["output_path"])
            )
            _archive_failed_output(failed_output, root)
            stored_entries = state["entries"]
            assert isinstance(stored_entries, list)
            for index in range(start_index, len(entries)):
                stored_entries[index] = _state_entry(entries[index])
            state["status"] = "prepared"
            state["completed_at_utc"] = None
            _atomic_json(state_path, state, root)
        if selected_pair_runner is None:
            selected_pair_runner = lambda entry, output_path, mode: run_with_task5(
                entry,
                output_path,
                mode,
                project_root=root,
                python=python,
                backend=backend,
                model=model,
                dataset_paths=dataset_paths,
            )
        if selected_single_runner is None:
            selected_single_runner = lambda entry, output_path, mode: run_single_config(
                entry,
                output_path,
                mode,
                project_root=root,
                python=python,
                backend=backend,
                model=model,
                dataset_paths=dataset_paths,
            )
        state["status"] = "running"
        _atomic_json(state_path, state, root)
        stored_entries = state["entries"]
        assert isinstance(stored_entries, list)
        for index in range(start_index, len(entries)):
            entry = entries[index]
            item = stored_entries[index]
            assert isinstance(item, dict)
            try:
                output_path = _project_path(root, str(entry["output_path"]))
                _ensure_empty_output(output_path, root)
                item["status"] = "running"
                item["started_at_utc"] = _utc_now()
                _atomic_json(state_path, state, root)
                active_runner = (
                    selected_pair_runner
                    if entry["phase"] == "dev"
                    else selected_single_runner
                )
                exit_status = _normalize_exit_status(
                    active_runner(entry, output_path, str(mode))
                )
                item["exit_status"] = exit_status
                if exit_status != 0:
                    return _fail(
                        state,
                        item,
                        state_path,
                        root,
                        output,
                        "runner exited " + str(exit_status),
                    )
                _project_path(root, str(entry["output_path"]))
                accepted, digest = _read_audit_at(
                    root, str(entry["output_path"])
                )
                item["audit_sha256"] = digest
                if not accepted:
                    return _fail(
                        state,
                        item,
                        state_path,
                        root,
                        output,
                        "audit accepted=false",
                    )
                item["proof_sha256"] = _write_suite_proof(
                    entry, root, state, digest
                )
                item["status"] = "completed"
                item["completed_at_utc"] = _utc_now()
                item["error"] = None
                _atomic_json(state_path, state, root)
            except BaseException as exc:
                if item.get("started_at_utc") is None:
                    item["started_at_utc"] = _utc_now()
                if item.get("exit_status") is None:
                    item["exit_status"] = 1
                return _fail(
                    state,
                    item,
                    state_path,
                    root,
                    output,
                    type(exc).__name__ + ": " + str(exc),
                )
        state["status"] = "completed"
        state["completed_at_utc"] = _utc_now()
        _atomic_json(state_path, state, root)
        return 0
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print("SUITE ERROR: " + str(exc), file=output)
        return 1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX_PATH))
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--status-path")
    parser.add_argument("--from-pair")
    parser.add_argument("--resolution-log")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--backend", choices=["mlx", "vllm"], default="vllm")
    parser.add_argument("--model")
    parser.add_argument("--task2-dataset-path")
    parser.add_argument("--task3-dataset-path")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = Path(args.project_root)
    dataset_paths = {
        key: value
        for key, value in (
            ("task2", args.task2_dataset_path),
            ("task3", args.task3_dataset_path),
        )
        if value
    }
    try:
        return run_suite(
            Path(args.matrix),
            project_root=root,
            status_path=Path(args.status_path) if args.status_path else None,
            from_pair=args.from_pair,
            resolution_log=(
                Path(args.resolution_log) if args.resolution_log else None
            ),
            python=args.python,
            backend=args.backend,
            model=args.model,
            dataset_paths=dataset_paths,
        )
    except (OSError, ValueError, TypeError) as exc:
        print("SUITE ERROR: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

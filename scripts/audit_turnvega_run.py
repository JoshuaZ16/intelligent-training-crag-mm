#!/usr/bin/env python3
"""Strictly audit a completed TurnVEGA triplet."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple


TRACE_FILENAME = "agent_trace_v5.jsonl"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_FAMILY_LOCK_ROOT = PROJECT_ROOT / ".turnvega-family-locks"
ANCHOR_EQUIVALENCE_MIN = 0.98
ANCHOR_ACCURACY_DELTA_MAX = 0.025
V5_REQUIRED_FIELDS = frozenset(
    {
        "trace_schema",
        "experiment_variant",
        "run_id",
        "interaction_id",
        "dataset_kind",
        "session_key",
        "turn_index",
        "query",
        "answer",
        "status",
        "search_call_count",
        "history_mode",
        "question_frame",
        "image_needed",
        "history_needed",
        "web_needed",
        "entity_candidates_before",
        "entity_candidates_after",
        "candidate_queries",
        "query_family",
        "candidate_budget",
        "atomic_evidence",
        "source_clusters",
        "relation_coverage",
        "circularity_flags",
        "answerability_scores",
        "typed_conflicts",
        "evidence_packet",
        "evidence_packet_sha256",
        "evidence_token_count",
        "memory_state_before",
        "memory_state_after",
        "provisional_claims",
        "verified_claims",
        "quarantined_claims",
        "state_version",
    }
)
V5_STRING_FIELDS = frozenset(
    {
        "trace_schema",
        "experiment_variant",
        "run_id",
        "interaction_id",
        "dataset_kind",
        "session_key",
        "query",
        "answer",
        "status",
        "history_mode",
        "query_family",
        "evidence_packet_sha256",
    }
)
V5_INT_FIELDS = frozenset(
    {
        "turn_index",
        "search_call_count",
        "candidate_budget",
        "evidence_token_count",
        "state_version",
    }
)
V5_BOOL_FIELDS = frozenset({"image_needed", "history_needed", "web_needed"})
V5_DICT_FIELDS = frozenset(
    {
        "question_frame",
        "relation_coverage",
        "answerability_scores",
        "evidence_packet",
        "memory_state_before",
        "memory_state_after",
    }
)
V5_LIST_FIELDS = V5_REQUIRED_FIELDS.difference(
    V5_STRING_FIELDS | V5_INT_FIELDS | V5_BOOL_FIELDS | V5_DICT_FIELDS
)
EXPECTED_INVENTORIES = {
    "sequential_triplet": [
        ("anchor_before", "anchor_before", 0),
        ("variant", "variant", 0),
        ("anchor_after", "anchor_after", 0),
    ],
    "dual_gpu_crossover": [
        ("anchor_before", "round_a/anchor", 0),
        ("variant_round_a", "round_a/variant", 1),
        ("variant_round_b", "round_b/variant", 0),
        ("anchor_after", "round_b/anchor", 1),
    ],
}


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_symlink():
        raise FileExistsError("refusing symlink audit output: " + str(path))
    temporary = path.parent / (
        "." + path.name + ".tmp-" + str(os.getpid()) + "-" + str(time.time_ns())
    )
    data = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _open_dirfd_chain(root: Path, relative: Path):
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    descriptors: List[int] = []
    try:
        current = os.open(Path(root), flags)
        descriptors.append(current)
        for component in relative.parts:
            if component in ("", "."):
                continue
            if component == ".." or "/" in component:
                raise ValueError("unsafe inventory path component")
            current = os.open(component, flags, dir_fd=current)
            descriptors.append(current)
        yield current
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _read_text_at(directory_fd: int, filename: str) -> str:
    if not filename or filename in (".", "..") or "/" in filename:
        raise ValueError("unsafe security filename")
    descriptor = os.open(
        filename,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("security input must be a regular file")
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            return handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _trace_rows_text(text: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for line_number, line in enumerate(
        text.splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("trace line " + str(line_number) + " is not an object")
        rows.append(value)
    return rows


def _session_key(row: Mapping[str, object]) -> str:
    return str(row.get("session_key") or row.get("session_id") or "")


def _ordered_identity(row: Mapping[str, object]) -> Tuple[object, ...]:
    return (
        _session_key(row),
        row.get("turn_index", 0),
        str(row.get("interaction_id") or ""),
        str(row.get("query") or ""),
    )


def _turn_key(row: Mapping[str, object], dataset_kind: str) -> Tuple[object, ...]:
    if dataset_kind == "task3":
        return (
            _session_key(row),
            row.get("turn_index", 0),
            str(row.get("interaction_id") or ""),
        )
    return (str(row.get("interaction_id") or ""),)


def _normalize_answer(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE).strip()


def _accuracy_text(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    try:
        payload = json.loads(text)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("all"), dict):
        return None
    value = payload["all"].get("accuracy")
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    ):
        return float(value)
    return None


def _trace_types_valid(row: Mapping[str, object]) -> bool:
    return (
        all(type(row.get(field)) is str for field in V5_STRING_FIELDS)
        and all(type(row.get(field)) is int for field in V5_INT_FIELDS)
        and all(type(row.get(field)) is bool for field in V5_BOOL_FIELDS)
        and all(type(row.get(field)) is dict for field in V5_DICT_FIELDS)
        and all(type(row.get(field)) is list for field in V5_LIST_FIELDS)
    )


def _validate_pair_summaries(
    triplet_dir: Path, execution_mode: object, reasons: List[str]
) -> bool:
    if execution_mode != "dual_gpu_crossover":
        return True
    expected = {
        "round_a": {"anchor_gpu": 0, "variant_gpu": 1},
        "round_b": {"variant_gpu": 0, "anchor_gpu": 1},
    }
    valid = True
    for round_name, assignments in expected.items():
        try:
            with _open_dirfd_chain(
                triplet_dir, Path(round_name)
            ) as round_fd:
                pair = json.loads(_read_text_at(round_fd, "pair_summary.json"))
        except (OSError, ValueError, json.JSONDecodeError):
            reasons.append(round_name + " pair summary is missing or invalid")
            valid = False
            continue
        if not isinstance(pair, dict):
            reasons.append(round_name + " pair summary is not an object")
            valid = False
            continue
        required = {
            "status": "completed",
            "execution_mode": "dual_gpu_crossover",
            "pair_valid": True,
            "oom_detected": False,
        }
        required.update(assignments)
        typed_contract = (
            type(pair.get("status")) is str
            and type(pair.get("execution_mode")) is str
            and type(pair.get("pair_valid")) is bool
            and type(pair.get("oom_detected")) is bool
            and all(type(pair.get(key)) is int for key in assignments)
        )
        if not typed_contract or any(
            pair.get(key) != value for key, value in required.items()
        ):
            valid = False
            reasons.append(round_name + " pair summary contract mismatch")
        for role in assignments:
            status_key = role.replace("_gpu", "_exit_status")
            if type(pair.get(status_key)) is not int or pair.get(status_key) != 0:
                valid = False
                reasons.append(round_name + " pair side did not exit zero")
    return valid


def _canonical_lock_path(experiment_family: str) -> Path:
    digest = hashlib.sha256(experiment_family.encode("utf-8")).hexdigest()
    return Path(CANONICAL_FAMILY_LOCK_ROOT).absolute() / (digest + ".json")


def _path_chain_has_symlink(path: Path) -> bool:
    absolute = Path(path).absolute()
    for candidate in [absolute, *absolute.parents]:
        try:
            if stat.S_ISLNK(candidate.lstat().st_mode):
                return True
        except FileNotFoundError:
            continue
    return False


def _audit_triplet_impl(triplet_dir: Path, manifest_path: Path) -> Dict[str, object]:
    triplet_dir = Path(triplet_dir)
    manifest_path = Path(manifest_path)
    reasons: List[str] = []
    runner_valid = True
    scope_valid = True
    schema_valid = True
    budget_valid = True

    if triplet_dir.is_symlink() or not triplet_dir.is_dir():
        raise ValueError("triplet directory must be a real directory")
    summary_path = triplet_dir / "triplet_summary.json"
    with _open_dirfd_chain(triplet_dir, Path(".")) as triplet_fd:
        summary = json.loads(_read_text_at(triplet_fd, "triplet_summary.json"))
    if not isinstance(summary, dict):
        raise ValueError("triplet_summary.json must contain an object")
    execution_mode = summary.get("execution_mode")
    if execution_mode not in ("dual_gpu_crossover", "sequential_triplet"):
        scope_valid = False
        reasons.append("runner did not freeze one explicit execution_mode")
    if summary.get("experiment_family") != "turnvega":
        runner_valid = False
        scope_valid = False
        reasons.append("runner summary experiment_family mismatch")
    lock_identity = summary.get("family_lock_identity")
    lock_path_hash = summary.get("family_lock_path_sha256")
    canonical_lock = _canonical_lock_path("turnvega")
    expected_lock_path_hash = hashlib.sha256(
        str(canonical_lock.resolve()).encode("utf-8")
    ).hexdigest()
    if (
        lock_identity != "turnvega"
        or not isinstance(lock_path_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", lock_path_hash) is None
        or lock_path_hash != expected_lock_path_hash
    ):
        scope_valid = False
        reasons.append("runner family lock metadata differs from canonical path")
    if (
        not canonical_lock.is_file()
        or canonical_lock.is_symlink()
        or _path_chain_has_symlink(canonical_lock)
    ):
        scope_valid = False
        reasons.append("canonical family lock is missing or not a regular file")
    else:
        try:
            with _open_dirfd_chain(
                canonical_lock.parent, Path(".")
            ) as lock_fd:
                canonical_payload = json.loads(
                    _read_text_at(lock_fd, canonical_lock.name)
                )
        except (OSError, ValueError, json.JSONDecodeError):
            canonical_payload = None
        if (
            not isinstance(canonical_payload, dict)
            or canonical_payload.get("experiment_family") != "turnvega"
            or canonical_payload.get("family_id") != "turnvega"
            or canonical_payload.get("execution_mode") != execution_mode
        ):
            scope_valid = False
            reasons.append("canonical family lock family/mode mismatch")
    anchor_variant = summary.get("anchor_variant")
    target_variant = summary.get("variant")
    if (
        not isinstance(anchor_variant, str)
        or not anchor_variant
        or not isinstance(target_variant, str)
        or not target_variant
    ):
        schema_valid = False
        reasons.append("runner anchor/target variant identity is missing")
    if (
        summary.get("status") != "completed"
        or summary.get("runner_valid") is not True
    ):
        runner_valid = False
        reasons.append("runner summary is not completed and valid")
    if summary.get("oom_detected"):
        runner_valid = False
        reasons.append("OOM was detected by the runner")

    with _open_dirfd_chain(manifest_path.parent, Path(".")) as manifest_fd:
        manifest_text = _read_text_at(manifest_fd, manifest_path.name)
    manifest = json.loads(manifest_text)
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("manifest must be a non-empty ordered list")
    if not all(isinstance(row, dict) for row in manifest):
        raise ValueError("manifest rows must be objects")
    expected_count = len(manifest)
    manifest_sha = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    expected_order = [_ordered_identity(row) for row in manifest]
    manifest_kind = "task3" if any(
        "session_key" in row or "turn_index" in row for row in manifest
    ) else "task2"
    manifest_keys = [_turn_key(row, manifest_kind) for row in manifest]
    if len(set(manifest_keys)) != expected_count:
        schema_valid = False
        reasons.append("manifest contains duplicate turn IDs")

    run_specs = summary.get("runs")
    if not isinstance(run_specs, list) or not run_specs:
        runner_valid = False
        scope_valid = False
        schema_valid = False
        budget_valid = False
        reasons.append("runner summary has no run inventory")
        run_specs = []
    expected_inventory = EXPECTED_INVENTORIES.get(execution_mode, [])
    actual_inventory = []
    for spec in run_specs:
        if isinstance(spec, dict):
            actual_inventory.append(
                (
                    spec.get("logical_run"),
                    spec.get("path"),
                    spec.get("gpu"),
                )
            )
            if (
                type(spec.get("logical_run")) is not str
                or type(spec.get("path")) is not str
                or type(spec.get("gpu")) is not int
                or type(spec.get("exit_status")) is not int
            ):
                scope_valid = False
                reasons.append("run inventory contains invalid scalar types")
    if actual_inventory != expected_inventory:
        runner_valid = False
        scope_valid = False
        reasons.append("run inventory does not exactly match execution mode")
    if not _validate_pair_summaries(triplet_dir, execution_mode, reasons):
        runner_valid = False
        scope_valid = False

    seen_paths = set()
    seen_run_ids = set()
    valid_runs: Dict[str, Dict[str, object]] = {}
    budget_totals = []
    for position, spec in enumerate(run_specs):
        if not isinstance(spec, dict):
            scope_valid = False
            reasons.append("run inventory entry is not an object")
            continue
        logical_name = str(spec.get("logical_run") or "run-" + str(position))
        relative = Path(str(spec.get("path") or ""))
        if (
            relative.is_absolute()
            or str(relative) in ("", ".")
            or any(part in ("", ".", "..") for part in relative.parts)
        ):
            scope_valid = False
            reasons.append(logical_name + " escapes triplet scope or is a symlink")
            continue
        path_identity = relative.as_posix()
        if path_identity in seen_paths:
            scope_valid = False
            reasons.append("duplicate run path in runner inventory")
            continue
        seen_paths.add(path_identity)
        try:
            with _open_dirfd_chain(triplet_dir, relative) as run_fd:
                config = json.loads(_read_text_at(run_fd, "run_config.json"))
                rows = _trace_rows_text(_read_text_at(run_fd, TRACE_FILENAME))
                try:
                    scores_text = _read_text_at(run_fd, "scores_dictionary.json")
                except FileNotFoundError:
                    scores_text = None
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            schema_valid = False
            reasons.append(logical_name + " has unreadable JSON: " + str(exc))
            continue
        if not isinstance(config, dict):
            schema_valid = False
            reasons.append(logical_name + " run_config is not an object")
            continue

        completed = config.get("status") == "completed"
        if not completed or spec.get("exit_status") != 0:
            runner_valid = False
            reasons.append(logical_name + " failed and was excluded from audit metrics")
            continue
        run_id = config.get("run_id")
        if not isinstance(run_id, str) or not run_id or run_id in seen_run_ids:
            scope_valid = False
            reasons.append(logical_name + " has missing or duplicate run_id")
        else:
            seen_run_ids.add(run_id)
        if (
            config.get("experiment_family") != "turnvega"
            or config.get("execution_mode") != execution_mode
            or config.get("triplet_role") != logical_name
            or type(config.get("assigned_gpu")) is not int
            or config.get("assigned_gpu") != spec.get("gpu")
        ):
            scope_valid = False
            reasons.append(logical_name + " run family/mode/role/GPU contract mismatch")
        if (
            config.get("family_lock_identity") != "turnvega"
            or config.get("family_lock_path_sha256") != expected_lock_path_hash
        ):
            scope_valid = False
            reasons.append(logical_name + " family lock metadata mismatch")
        if config.get("manifest_sha256") != manifest_sha:
            schema_valid = False
            reasons.append(logical_name + " manifest hash mismatch")
        if config.get("trace_schema") != "v5":
            schema_valid = False
            reasons.append(logical_name + " config is not Trace v5")
        if (
            type(config.get("manifest_row_count")) is not int
            or config.get("manifest_row_count") != expected_count
        ):
            schema_valid = False
            reasons.append(logical_name + " manifest count mismatch")
        if len(rows) != expected_count:
            schema_valid = False
            reasons.append(logical_name + " Trace v5 count mismatch")

        row_fields_ok = True
        identities = []
        keys = []
        search_total = 0
        candidate_total = 0
        dataset_kind = str(config.get("dataset_kind") or manifest_kind)
        expected_variant = (
            anchor_variant if logical_name.startswith("anchor_") else target_variant
        )
        if config.get("dataset_kind") != manifest_kind:
            schema_valid = False
            row_fields_ok = False
            reasons.append(logical_name + " dataset_kind differs from manifest kind")
        if config.get("variant") != expected_variant:
            schema_valid = False
            row_fields_ok = False
            reasons.append(logical_name + " variant differs from role assignment")
        configured_candidate_raw = config.get("candidate_budget")
        max_search_raw = config.get("max_search_calls")
        if (
            type(configured_candidate_raw) is not int
            or configured_candidate_raw < 0
            or type(max_search_raw) is not int
            or max_search_raw < 0
        ):
            budget_valid = False
            schema_valid = False
            row_fields_ok = False
            reasons.append(logical_name + " config budget fields have invalid types")
        for row_index, row in enumerate(rows):
            missing = V5_REQUIRED_FIELDS.difference(row)
            if missing:
                row_fields_ok = False
                reasons.append(
                    logical_name
                    + " trace row "
                    + str(row_index)
                    + " missing required fields: "
                    + ",".join(sorted(missing))
                )
            if row.get("trace_schema") != "v5" or row.get("status") != "ok":
                row_fields_ok = False
                reasons.append(logical_name + " contains non-v5 or failed trace row")
            if not _trace_types_valid(row):
                row_fields_ok = False
                schema_valid = False
                reasons.append(logical_name + " trace row has invalid v5 field types")
            if row.get("run_id") != run_id:
                row_fields_ok = False
                reasons.append(logical_name + " trace run_id mismatch")
            if row.get("dataset_kind") != config.get("dataset_kind"):
                row_fields_ok = False
                schema_valid = False
                reasons.append(logical_name + " trace dataset_kind mismatch")
            if row.get("experiment_variant") != config.get("variant"):
                row_fields_ok = False
                schema_valid = False
                reasons.append(logical_name + " trace experiment_variant mismatch")
            try:
                packet_payload = json.dumps(
                    row.get("evidence_packet"),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
                expected_packet_hash = hashlib.sha256(packet_payload).hexdigest()
            except (TypeError, ValueError):
                expected_packet_hash = ""
            declared_packet_hash = row.get("evidence_packet_sha256")
            if (
                type(declared_packet_hash) is not str
                or re.fullmatch(r"[0-9a-fA-F]{64}", declared_packet_hash) is None
                or declared_packet_hash.lower() != expected_packet_hash
            ):
                row_fields_ok = False
                schema_valid = False
                reasons.append(logical_name + " evidence_packet_sha256 mismatch")
            identities.append(_ordered_identity(row))
            keys.append(_turn_key(row, dataset_kind))
            try:
                search_raw = row.get("search_call_count")
                candidate_raw = row.get("candidate_budget")
                if type(search_raw) is not int or type(candidate_raw) is not int:
                    raise ValueError
                search_count = search_raw
                candidate_count = candidate_raw
                search_total += search_count
                candidate_total += candidate_count
                if search_count < 0 or candidate_count < 0:
                    raise ValueError
                if type(max_search_raw) is not int or type(configured_candidate_raw) is not int:
                    raise ValueError
                max_search = max_search_raw
                configured_candidate = configured_candidate_raw
                if search_count > max_search or candidate_count != configured_candidate:
                    budget_valid = False
                    reasons.append(logical_name + " exceeds or changes fixed budget")
            except (TypeError, ValueError):
                budget_valid = False
                schema_valid = False
                row_fields_ok = False
                reasons.append(logical_name + " has invalid budget values")
        if identities != expected_order:
            row_fields_ok = False
            reasons.append(logical_name + " ordered turn identity/query differs from manifest")
        if len(set(keys)) != len(keys):
            row_fields_ok = False
            reasons.append(logical_name + " contains duplicate turn IDs")
        if not row_fields_ok:
            schema_valid = False
        budget_totals.append((search_total, candidate_total))
        valid_runs[logical_name] = {
            "path": relative.as_posix(),
            "config": config,
            "rows": rows,
            "accuracy": _accuracy_text(scores_text),
            "schema_ok": row_fields_ok and len(rows) == expected_count,
        }

    if budget_totals and len(set(budget_totals)) != 1:
        budget_valid = False
        reasons.append("total search/candidate budgets differ across runs")
    if len(valid_runs) != len(run_specs):
        budget_valid = False

    before = valid_runs.get("anchor_before")
    after = valid_runs.get("anchor_after")
    anchor_equivalence_rate: Optional[float] = None
    anchor_accuracy_delta: Optional[float] = None
    anchor_drift = True
    if before and after and before["schema_ok"] and after["schema_ok"]:
        before_rows = before["rows"]
        after_rows = after["rows"]
        assert isinstance(before_rows, list) and isinstance(after_rows, list)
        if expected_count:
            equivalent = sum(
                _normalize_answer(left.get("answer"))
                == _normalize_answer(right.get("answer"))
                for left, right in zip(before_rows, after_rows)
            )
            anchor_equivalence_rate = equivalent / expected_count
        before_accuracy = before["accuracy"]
        after_accuracy = after["accuracy"]
        if isinstance(before_accuracy, float) and isinstance(after_accuracy, float):
            anchor_accuracy_delta = abs(after_accuracy - before_accuracy)
        else:
            reasons.append("anchor accuracy is missing or invalid in scores_dictionary.json all")
    else:
        reasons.append("two completed schema-valid anchor runs are required")

    if anchor_equivalence_rate is not None and anchor_accuracy_delta is not None:
        anchor_drift = (
            anchor_equivalence_rate + 1e-12 < ANCHOR_EQUIVALENCE_MIN
            or anchor_accuracy_delta - 1e-12 > ANCHOR_ACCURACY_DELTA_MAX
        )
    if anchor_drift:
        reasons.append("anchor drift exceeds equivalence or accuracy threshold")

    accepted = (
        runner_valid
        and scope_valid
        and schema_valid
        and budget_valid
        and not anchor_drift
    )
    result: Dict[str, object] = {
        "runner_valid": runner_valid,
        "scope_isolation_valid": scope_valid,
        "schema_valid": schema_valid,
        "budget_valid": budget_valid,
        "anchor_equivalence_rate": anchor_equivalence_rate,
        "anchor_accuracy_delta": anchor_accuracy_delta,
        "anchor_drift": anchor_drift,
        "accepted": accepted,
        "reasons": reasons,
    }
    _atomic_json(triplet_dir / "audit.json", result)
    return result


def audit_triplet(triplet_dir: Path, manifest_path: Path) -> Dict[str, object]:
    triplet_dir = Path(triplet_dir)
    try:
        return _audit_triplet_impl(triplet_dir, Path(manifest_path))
    except BaseException as exc:
        result: Dict[str, object] = {
            "runner_valid": False,
            "scope_isolation_valid": False,
            "schema_valid": False,
            "budget_valid": False,
            "anchor_equivalence_rate": None,
            "anchor_accuracy_delta": None,
            "anchor_drift": True,
            "accepted": False,
            "reasons": [type(exc).__name__ + ": " + str(exc)],
        }
        if triplet_dir.is_dir() and not triplet_dir.is_symlink():
            _atomic_json(triplet_dir / "audit.json", result)
        return result


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triplet-dir", required=True)
    parser.add_argument("--manifest-path", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    result = audit_triplet(Path(args.triplet_dir), Path(args.manifest_path))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Build deterministic TurnVEGA Task 2 and Task 3 manifests."""

import argparse
import hashlib
import json
import os
import random
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path


DEFAULT_SEED = 20260720
TASK2_SPLITS = (
    ("t2_cal40.json", 40),
    ("t2_dev80.json", 80),
    ("t2_test120.json", 120),
)
TASK3_SPLITS = (
    ("t3_cal20.json", 20),
    ("t3_dev40.json", 40),
    ("t3_test60.json", 60),
)


def canonical_json_bytes(value):
    """Return canonical UTF-8 JSON with exactly one trailing newline."""
    text = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return (text + "\n").encode("utf-8")


def _first(value, default=""):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


def _stable_text(value):
    value = _first(value)
    return "" if value is None else str(value)


def _turn_field(row, field):
    turns = row.get("turns")
    if isinstance(turns, dict) and field in turns:
        return _first(turns.get(field))
    if isinstance(turns, (list, tuple)) and turns:
        first_turn = turns[0]
        if isinstance(first_turn, dict) and field in first_turn:
            return _first(first_turn.get(field))
    return _first(row.get(field))


def _turn_count(row):
    explicit = _first(row.get("total_turn_count"), None)
    if explicit is not None:
        try:
            return int(explicit)
        except (TypeError, ValueError):
            pass
    turns = row.get("turns")
    if isinstance(turns, (list, tuple)):
        return len(turns)
    if isinstance(turns, dict):
        lengths = [
            len(value)
            for value in turns.values()
            if isinstance(value, (list, tuple))
        ]
        return max(lengths, default=0)
    return 0


def _task2_candidate(row, source_index):
    manifest_row = {
        "source_index": source_index,
        "session_id": _stable_text(row.get("session_id")),
        "interaction_id": _stable_text(_turn_field(row, "interaction_id")),
        "query_category": _stable_text(_turn_field(row, "query_category")),
        "domain": _stable_text(_turn_field(row, "domain")),
        "query": _stable_text(_turn_field(row, "query")),
    }
    stratum = (
        manifest_row["domain"],
        manifest_row["query_category"],
        _stable_text(_turn_field(row, "image_quality")),
        _stable_text(_turn_field(row, "dynamism")),
    )
    return manifest_row, stratum


def _task3_candidate(row, source_index):
    turn_count = _turn_count(row)
    manifest_row = {
        "source_index": source_index,
        "session_id": _stable_text(row.get("session_id")),
        "interaction_id": _stable_text(_turn_field(row, "interaction_id")),
        "query_category": _stable_text(_turn_field(row, "query_category")),
        "domain": _stable_text(_turn_field(row, "domain")),
        "query": _stable_text(_turn_field(row, "query")),
        "turn_count": turn_count,
    }
    is_ego = row.get("is_ego", False)
    if not isinstance(is_ego, bool):
        is_ego = False
    stratum = (
        manifest_row["domain"],
        turn_count,
        is_ego,
        _stable_text(_turn_field(row, "image_quality")),
    )
    return manifest_row, stratum


def _legacy_identifiers(rows):
    if not isinstance(rows, list):
        raise ValueError("legacy manifest must be a list of dict rows")
    source_indices = set()
    session_ids = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("legacy manifest rows must be dict values")
        has_source = "source_index" in row
        has_session = "session_id" in row
        if not has_source and not has_session:
            raise ValueError(
                "legacy manifest row must contain source_index or session_id"
            )
        if has_source:
            source_index = row["source_index"]
            if type(source_index) is not int:
                raise ValueError("legacy source_index must be an int")
            source_indices.add(source_index)
        if has_session:
            session_id = row["session_id"]
            if not isinstance(session_id, str) or not session_id.strip():
                raise ValueError("legacy session_id must be a non-empty string")
            session_ids.add(session_id)
    return source_indices, session_ids


def _eligible_candidates(
    rows,
    converter,
    dataset_kind,
    legacy_sources,
    legacy_sessions,
):
    candidates = []
    seen_sources = set()
    seen_sessions = set()
    assigned_source_index_count = 0
    for assigned_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{dataset_kind} rows must be mappings")
        if "source_index" in row:
            source_index = row["source_index"]
            if type(source_index) is not int:
                raise ValueError(f"{dataset_kind} source_index must be an int")
        else:
            source_index = assigned_index
            assigned_source_index_count += 1
        session_id = row.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError(
                f"{dataset_kind} session_id must be a non-empty string"
            )
        if source_index in seen_sources:
            raise ValueError(f"{dataset_kind} duplicate source_index")
        if session_id in seen_sessions:
            raise ValueError(f"{dataset_kind} duplicate session_id")
        seen_sources.add(source_index)
        seen_sessions.add(session_id)

        manifest_row, stratum = converter(row, source_index)
        if source_index in legacy_sources or session_id in legacy_sessions:
            continue
        identity = (
            dataset_kind,
            source_index,
            session_id,
        )
        candidates.append((manifest_row, stratum, identity))
    return candidates, assigned_source_index_count


def _stratified_round_robin(candidates, required_count, seed, task_name):
    if len(candidates) < required_count:
        raise ValueError(
            f"{task_name} requires {required_count} eligible rows; "
            f"found {len(candidates)}"
        )

    buckets = defaultdict(list)
    for manifest_row, stratum, identity in candidates:
        buckets[stratum].append((identity, manifest_row))

    rng = random.Random(seed)
    strata = sorted(
        buckets,
        key=lambda key: json.dumps(
            key,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
    )
    for key in strata:
        buckets[key].sort(key=lambda candidate: candidate[0])
        rng.shuffle(buckets[key])
    rng.shuffle(strata)

    offsets = {key: 0 for key in strata}
    selected = []
    while len(selected) < required_count:
        for key in strata:
            offset = offsets[key]
            if offset < len(buckets[key]):
                selected.append(buckets[key][offset][1])
                offsets[key] = offset + 1
                if len(selected) == required_count:
                    break
    return selected


def _slice_splits(selected, split_spec):
    result = {}
    start = 0
    for name, count in split_spec:
        result[name] = selected[start : start + count]
        start += count
    return result


def _build_manifest_bundle(
    single_rows,
    multi_rows,
    legacy_manifest_rows,
    seed=DEFAULT_SEED,
):
    legacy_sources, legacy_sessions = _legacy_identifiers(legacy_manifest_rows)
    task2_candidates, task2_assigned_count = _eligible_candidates(
        single_rows,
        _task2_candidate,
        "task2",
        legacy_sources,
        legacy_sessions,
    )
    task3_candidates, task3_assigned_count = _eligible_candidates(
        multi_rows,
        _task3_candidate,
        "task3",
        legacy_sources,
        legacy_sessions,
    )

    task2 = _stratified_round_robin(
        task2_candidates,
        sum(count for _, count in TASK2_SPLITS),
        seed,
        "Task 2",
    )
    selected_task2_sessions = {row["session_id"] for row in task2}
    task3_candidates = [
        candidate
        for candidate in task3_candidates
        if candidate[0]["session_id"] not in selected_task2_sessions
    ]
    task3 = _stratified_round_robin(
        task3_candidates,
        sum(count for _, count in TASK3_SPLITS),
        seed,
        "Task 3",
    )
    manifests = _slice_splits(task2, TASK2_SPLITS)
    manifests.update(_slice_splits(task3, TASK3_SPLITS))
    manifests["t2_main40.json"] = manifests["t2_test120.json"][-40:]
    manifests["t3_main40.json"] = manifests["t3_test60.json"][-40:]
    metadata = {
        "assigned_source_index_count": (
            task2_assigned_count + task3_assigned_count
        ),
        "assigned_source_index_count_by_dataset": {
            "task2": task2_assigned_count,
            "task3": task3_assigned_count,
        },
    }
    return manifests, metadata


def build_manifests(
    single_rows,
    multi_rows,
    legacy_manifest_rows,
    seed=DEFAULT_SEED,
):
    """Purely build all eight manifests from in-memory row iterables."""
    manifests, _ = _build_manifest_bundle(
        single_rows,
        multi_rows,
        legacy_manifest_rows,
        seed=seed,
    )
    return manifests


def _pair_overlap_summary(manifests):
    summary = {}
    for names in (
        tuple(name for name, _ in TASK2_SPLITS),
        tuple(name for name, _ in TASK3_SPLITS),
    ):
        for left_name, right_name in combinations(names, 2):
            left = manifests[left_name]
            right = manifests[right_name]
            source_overlap = {
                row["source_index"] for row in left
            } & {row["source_index"] for row in right}
            session_overlap = {
                row["session_id"] for row in left
            } & {row["session_id"] for row in right}
            summary[f"{left_name}|{right_name}"] = {
                "source_overlap_count": len(source_overlap),
                "session_overlap_count": len(session_overlap),
            }
    return summary


def _manifest_summary(manifests, seed, metadata):
    test_names = {
        "t2_main40.json": "t2_test120.json",
        "t3_main40.json": "t3_test60.json",
    }
    derived = {}
    for main_name, test_name in test_names.items():
        main_rows = manifests[main_name]
        test_bytes = canonical_json_bytes(manifests[test_name])
        derived[main_name] = {
            "source_manifest": test_name,
            "source_manifest_sha256": hashlib.sha256(test_bytes).hexdigest(),
            "source_indices": [row["source_index"] for row in main_rows],
            "session_ids": [row["session_id"] for row in main_rows],
        }
    task2_sessions = {
        row["session_id"]
        for name, _ in TASK2_SPLITS
        for row in manifests[name]
    }
    task3_sessions = {
        row["session_id"]
        for name, _ in TASK3_SPLITS
        for row in manifests[name]
    }
    return {
        "seed": seed,
        "counts": {name: len(rows) for name, rows in manifests.items()},
        "split_pair_overlap_count": _pair_overlap_summary(manifests),
        "cross_task_session_overlap_count": len(
            task2_sessions & task3_sessions
        ),
        "source_index_namespace": (
            "dataset-local; cross-task source indices are not compared"
        ),
        **metadata,
        "derived_main40": derived,
    }


def _safe_write(path, payload):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("failed to write staging file")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path):
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prepare_output_directory(output_path, publication_names):
    if output_path.is_symlink():
        raise ValueError("output_dir must not be a symlink")
    if output_path.exists() and not output_path.is_dir():
        raise ValueError("output_dir must be a directory")
    output_path.mkdir(parents=True, exist_ok=True)
    for name in publication_names:
        if (output_path / name).is_symlink():
            raise ValueError(f"publication target must not be a symlink: {name}")


def write_manifests(
    single_rows,
    multi_rows,
    legacy_manifest_rows,
    output_dir,
    seed=DEFAULT_SEED,
):
    """Build manifests and write canonical JSON, hashes, and a summary."""
    manifests, metadata = _build_manifest_bundle(
        single_rows,
        multi_rows,
        legacy_manifest_rows,
        seed=seed,
    )
    summary = _manifest_summary(manifests, seed, metadata)
    output_path = Path(output_dir)
    manifest_names = list(manifests)
    hash_names = [f"{name}.sha256" for name in manifest_names]
    publication_names = (
        manifest_names
        + hash_names
        + ["manifest_summary.json", "MANIFESTS_COMPLETE"]
    )
    _prepare_output_directory(output_path, publication_names)

    staging_path = Path(
        tempfile.mkdtemp(prefix=".turnvega-manifests-", dir=output_path)
    )
    try:
        for name, rows in manifests.items():
            payload = canonical_json_bytes(rows)
            _safe_write(staging_path / name, payload)
            digest = hashlib.sha256(payload).hexdigest().encode("ascii")
            _safe_write(staging_path / f"{name}.sha256", digest + b"\n")
        summary_payload = canonical_json_bytes(summary)
        _safe_write(staging_path / "manifest_summary.json", summary_payload)
        summary_digest = hashlib.sha256(summary_payload).hexdigest()
        _safe_write(
            staging_path / "MANIFESTS_COMPLETE",
            (summary_digest + "\n").encode("ascii"),
        )
        _fsync_directory(staging_path)

        marker_path = output_path / "MANIFESTS_COMPLETE"
        if marker_path.exists():
            marker_path.unlink()
            _fsync_directory(output_path)
        for name in manifest_names + hash_names:
            os.replace(staging_path / name, output_path / name)
        os.replace(
            staging_path / "manifest_summary.json",
            output_path / "manifest_summary.json",
        )
        _fsync_directory(output_path)
        os.replace(
            staging_path / "MANIFESTS_COMPLETE",
            marker_path,
        )
        _fsync_directory(output_path)
    finally:
        shutil.rmtree(staging_path, ignore_errors=True)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build deterministic TurnVEGA evaluation manifests."
    )
    parser.add_argument("--single-dataset-path", required=True)
    parser.add_argument("--multi-dataset-path", required=True)
    parser.add_argument("--legacy-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from datasets import load_dataset

    single_rows = load_dataset(
        "parquet",
        data_files=args.single_dataset_path,
        split="train",
    )
    multi_rows = load_dataset(
        "parquet",
        data_files=args.multi_dataset_path,
        split="train",
    )
    with Path(args.legacy_manifest).open(encoding="utf-8") as handle:
        legacy_rows = json.load(handle)
    write_manifests(
        single_rows,
        multi_rows,
        legacy_rows,
        args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

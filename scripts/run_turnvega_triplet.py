#!/usr/bin/env python3
"""Run a reciprocal TurnVEGA anchor/variant/anchor triplet."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple


EXECUTION_MODES = ("dual_gpu_crossover", "sequential_triplet")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_FAMILY = "turnvega"
CANONICAL_FAMILY_LOCK_ROOT = PROJECT_ROOT / ".turnvega-family-locks"
MIN_RAM_BYTES = 6 * 1024**3
MAX_GPU_MEMORY_MIB = 500
OOM_MARKERS = (
    "cuda out of memory",
    "gpu out of memory",
    "cpu out of memory",
    "oom killer",
    "oom-kill",
    "killed process",
)
SWAP_THRASH_PAGES_PER_SECOND = 1024
_last_swap_sample: Optional[Tuple[int, int, float]] = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def resolve_execution_mode(
    requested_mode: str,
    *,
    pre_formal_smoke: bool,
    dual_gpu_smoke_ok: Optional[bool],
) -> str:
    """Resolve auto once, before any formal/dev/test output is created."""
    if requested_mode in EXECUTION_MODES:
        return requested_mode
    if requested_mode != "auto":
        raise ValueError("unknown execution mode: " + requested_mode)
    if not pre_formal_smoke:
        raise ValueError("auto execution mode is allowed only for pre-formal smoke")
    if dual_gpu_smoke_ok is None:
        raise ValueError("auto execution mode requires a dual-GPU smoke result")
    return "dual_gpu_crossover" if dual_gpu_smoke_ok else "sequential_triplet"


def _ensure_empty_output(path: Path) -> None:
    if path.is_symlink():
        raise FileExistsError("refusing symlink output directory: " + str(path))
    if not path.exists():
        return
    if not path.is_dir():
        raise FileExistsError("output path is not a directory: " + str(path))
    first = next(path.iterdir(), None)
    if first is not None:
        raise FileExistsError("output directory must be empty; found: " + str(first))


def canonical_family_lock_path() -> Path:
    root = Path(CANONICAL_FAMILY_LOCK_ROOT).absolute()
    digest = hashlib.sha256(EXPERIMENT_FAMILY.encode("utf-8")).hexdigest()
    return root / (digest + ".json")


def _reject_symlink_chain(path: Path) -> None:
    absolute = Path(path).absolute()
    chain = [absolute, *absolute.parents]
    for candidate in reversed(chain):
        try:
            mode = candidate.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise FileExistsError("refusing symlink in canonical lock path")


def _freeze_family_mode(mode: str) -> Path:
    lock_path = canonical_family_lock_path()
    _reject_symlink_chain(lock_path.parent)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.parent.is_symlink() or lock_path.is_symlink():
        raise FileExistsError("refusing symlink family mode lock")
    guard_path = lock_path.with_suffix(".guard")
    if guard_path.is_symlink():
        raise FileExistsError("refusing symlink family mode guard")
    guard_descriptor = os.open(
        guard_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        fcntl.flock(guard_descriptor, fcntl.LOCK_EX)
        payload = {
            "experiment_family": "turnvega",
            "family_id": EXPERIMENT_FAMILY,
            "execution_mode": mode,
            "frozen_at_utc": _utc_now(),
        }
        try:
            descriptor = os.open(
                lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
        except FileExistsError:
            locked = json.loads(lock_path.read_text(encoding="utf-8"))
            if (
                locked.get("experiment_family") != "turnvega"
                or locked.get("family_id") != EXPERIMENT_FAMILY
            ):
                raise ValueError("family mode lock identity mismatch")
            if locked.get("execution_mode") != mode:
                raise ValueError("cannot mix execution modes in one experiment family")
        else:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(
                    (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode(
                        "utf-8"
                    )
                )
                handle.flush()
                os.fsync(handle.fileno())
            directory_descriptor = os.open(
                lock_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        fcntl.flock(guard_descriptor, fcntl.LOCK_UN)
        os.close(guard_descriptor)
    return lock_path


def _family_lock_metadata() -> Tuple[str, str]:
    lock_path = canonical_family_lock_path()
    path_hash = hashlib.sha256(str(lock_path.resolve()).encode("utf-8")).hexdigest()
    return EXPERIMENT_FAMILY, path_hash


def detect_swap_thrashing(
    previous: Tuple[int, int, float],
    current: Tuple[int, int, float],
    minimum_page_rate: int = SWAP_THRASH_PAGES_PER_SECOND,
) -> bool:
    """Return true only for sustained bidirectional swap-in and swap-out."""
    elapsed = current[2] - previous[2]
    if elapsed <= 0:
        return False
    swap_in_rate = max(0, current[0] - previous[0]) / elapsed
    swap_out_rate = max(0, current[1] - previous[1]) / elapsed
    return swap_in_rate >= minimum_page_rate and swap_out_rate >= minimum_page_rate


def _parse_macos_vm_stat(text: str) -> Dict[str, object]:
    page_match = re.search(r"page size of (\d+) bytes", text)
    if page_match is None:
        raise ValueError("vm_stat page size is missing")
    page_size = int(page_match.group(1))
    values: Dict[str, int] = {}
    for line in text.splitlines():
        match = re.match(r"([^:]+):\s*(\d+)\.", line.strip())
        if match:
            values[match.group(1)] = int(match.group(2))
    available_pages = sum(
        values.get(name, 0)
        for name in ("Pages free", "Pages inactive", "Pages speculative")
    )
    return {
        "ram_available_bytes": available_pages * page_size,
        "swap_pages": (values.get("Pageins", 0), values.get("Pageouts", 0)),
    }


def _system_resource_snapshot() -> Dict[str, object]:
    global _last_swap_sample
    snapshot: Dict[str, object] = {
        "ram_available_bytes": None,
        "swap_thrashing": False,
        "gpu_memory_mib": {},
    }
    try:
        import psutil

        snapshot["ram_available_bytes"] = int(psutil.virtual_memory().available)
        swap = psutil.swap_memory()
        current_swap = (
            int(swap.sin) // 4096,
            int(swap.sout) // 4096,
            time.monotonic(),
        )
        if _last_swap_sample is not None:
            snapshot["swap_thrashing"] = detect_swap_thrashing(
                _last_swap_sample, current_swap
            )
        _last_swap_sample = current_swap
    except (ImportError, AttributeError, OSError, ValueError):
        pass
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                snapshot["ram_available_bytes"] = int(line.split()[1]) * 1024
                break
    vmstat = Path("/proc/vmstat")
    if vmstat.exists():
        values = {}
        for line in vmstat.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(" ")
            if key in ("pswpin", "pswpout"):
                values[key] = int(value.strip())
        if "pswpin" in values and "pswpout" in values:
            current_swap = (
                values["pswpin"],
                values["pswpout"],
                time.monotonic(),
            )
            if _last_swap_sample is not None:
                snapshot["swap_thrashing"] = detect_swap_thrashing(
                    _last_swap_sample, current_swap
                )
            _last_swap_sample = current_swap
    if snapshot["ram_available_bytes"] is None and platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["vm_stat"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            parsed = _parse_macos_vm_stat(result.stdout)
            snapshot["ram_available_bytes"] = parsed["ram_available_bytes"]
            swap_in, swap_out = parsed["swap_pages"]
            current_swap = (int(swap_in), int(swap_out), time.monotonic())
            if _last_swap_sample is not None:
                snapshot["swap_thrashing"] = detect_swap_thrashing(
                    _last_swap_sample, current_swap
                )
            _last_swap_sample = current_swap
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        memories: Dict[int, int] = {}
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            index, used = [item.strip() for item in line.split(",", 1)]
            memories[int(index)] = int(float(used))
        snapshot["gpu_memory_mib"] = memories
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return snapshot


def _resource_stop_reason(snapshot: Mapping[str, object]) -> Optional[str]:
    available = snapshot.get("ram_available_bytes")
    if available is not None and int(available) < MIN_RAM_BYTES:
        return "RAM available is below 6 GiB"
    if bool(snapshot.get("swap_thrashing")):
        return "swap thrashing detected"
    return None


def _wait_for_gpu_clear(
    resource_probe: Callable[[], Mapping[str, object]],
    gpu_indices: Sequence[int],
    timeout_seconds: float,
) -> Optional[str]:
    deadline = time.monotonic() + max(timeout_seconds, 0)
    while True:
        try:
            snapshot = resource_probe()
        except BaseException as exc:
            reason = "resource probe failed: " + type(exc).__name__ + ": " + str(exc)
            break
        stop_reason = _resource_stop_reason(snapshot)
        if stop_reason:
            return stop_reason
        raw = snapshot.get("gpu_memory_mib") or {}
        memories = {int(key): int(value) for key, value in dict(raw).items()}
        telemetry_complete = all(index in memories for index in gpu_indices)
        if telemetry_complete and all(
            memories[index] < MAX_GPU_MEMORY_MIB for index in gpu_indices
        ):
            return None
        if time.monotonic() >= deadline:
            if not telemetry_complete:
                return "GPU memory telemetry is missing between rounds"
            return "GPU memory did not fall below 500 MiB between rounds"
        time.sleep(1)


def _scan_log_incremental(path: Path, state: Dict[str, object]) -> bool:
    offset = int(state.get("log_offset", 0))
    tail = state.get("log_tail", b"")
    if not isinstance(tail, bytes):
        tail = b""
    found = False
    with path.open("rb") as handle:
        handle.seek(offset)
        while True:
            chunk = handle.read(64 * 1024)
            if not chunk:
                break
            combined = tail + chunk.lower()
            if any(marker.encode("utf-8") in combined for marker in OOM_MARKERS):
                found = True
            tail = combined[-128:]
            offset += len(chunk)
    state["log_offset"] = offset
    state["log_tail"] = tail
    return found


def _finalize_run_config(
    run_dir: Path,
    execution_mode: str,
    role: str,
    gpu: int,
    family_lock_identity: str,
    family_lock_path_sha256: str,
) -> None:
    config_path = run_dir / "run_config.json"
    if (
        not config_path.exists()
        or config_path.is_symlink()
        or not config_path.is_file()
    ):
        raise ValueError("missing non-symlink regular run_config.json")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid run_config.json: " + str(exc)) from exc
    if not isinstance(config, dict):
        raise ValueError("run_config.json must contain an object")
    if config.get("experiment_family") != "turnvega":
        raise ValueError("run_config experiment_family mismatch")
    existing = config.get("execution_mode")
    if existing not in (None, execution_mode):
        raise ValueError("cannot mix execution modes in one experiment family")
    if config.get("triplet_role") not in (None, role):
        raise ValueError("run_config triplet_role mismatch")
    if config.get("assigned_gpu") not in (None, gpu):
        raise ValueError("run_config assigned_gpu mismatch")
    config.update(
        execution_mode=execution_mode,
        triplet_role=role,
        assigned_gpu=gpu,
        family_lock_identity=family_lock_identity,
        family_lock_path_sha256=family_lock_path_sha256,
    )
    _atomic_json(config_path, config)
    if config.get("status") != "completed":
        raise ValueError("run_config status must be terminal completed")


def _launch(
    command: Sequence[str],
    run_dir: Path,
    gpu: int,
):
    run_dir.mkdir(parents=True, exist_ok=False)
    log_handle = (run_dir / "stdout_stderr.log").open("w", encoding="utf-8")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    run_suffix = "-".join(run_dir.parts[-2:]).replace("_", "-")
    materialized_command = [
        part.replace("{output_dir}", str(run_dir)).replace(
            "{run_suffix}", run_suffix
        )
        for part in command
    ]
    started_at = time.time()
    try:
        process = subprocess.Popen(
            materialized_command,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except BaseException:
        log_handle.close()
        raise
    return process, log_handle, started_at


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return bool(_group_member_pids(pgid))
    return True


def _group_member_pids(pgid: int) -> List[int]:
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,pgid="],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    members = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2 and int(fields[1]) == pgid:
            members.append(int(fields[0]))
    return members


def _signal_process_group(pgid: int, signal_number: int) -> None:
    try:
        os.killpg(pgid, signal_number)
        return
    except ProcessLookupError:
        return
    except PermissionError:
        pass
    for pid in _group_member_pids(pgid):
        try:
            os.kill(pid, signal_number)
        except (ProcessLookupError, PermissionError):
            pass


def _terminate_process_group(process: subprocess.Popen) -> None:
    pgid = process.pid
    if _process_group_exists(pgid):
        _signal_process_group(pgid, signal.SIGTERM)
    deadline = time.monotonic() + 0.5
    while _process_group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(0.02)
    if _process_group_exists(pgid):
        _signal_process_group(pgid, signal.SIGKILL)
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        _signal_process_group(pgid, signal.SIGKILL)
        process.wait()
    deadline = time.monotonic() + 2
    while _process_group_exists(pgid) and time.monotonic() < deadline:
        _signal_process_group(pgid, signal.SIGKILL)
        time.sleep(0.02)


def _monitor_processes(
    entries: Sequence[Dict[str, object]],
    resource_probe: Callable[[], Mapping[str, object]],
    poll_interval_seconds: float = 0.05,
) -> Tuple[Optional[str], bool]:
    reason: Optional[str] = None
    oom_detected = False
    try:
        while True:
            running = False
            for entry in entries:
                process = entry["process"]
                assert isinstance(process, subprocess.Popen)
                status = process.poll()
                if status is None:
                    running = True
                log_path = entry["run_dir"] / "stdout_stderr.log"
                marker_found = (
                    log_path.exists() and _scan_log_incremental(log_path, entry)
                )
                if marker_found:
                    entry["oom_found"] = True
                if marker_found or status in (-9, 137):
                    oom_detected = True
                    reason = "CPU/GPU OOM detected while round was running"
                    break
                if status is not None and status != 0:
                    reason = "one side exited nonzero while peer was running"
                    break
            if reason:
                break
            snapshot = resource_probe()
            reason = _resource_stop_reason(snapshot)
            if reason:
                break
            if not running:
                break
            time.sleep(poll_interval_seconds)
    except BaseException as exc:
        reason = "monitor failed: " + type(exc).__name__ + ": " + str(exc)
    finally:
        for entry in entries:
            process = entry["process"]
            assert isinstance(process, subprocess.Popen)
            if reason:
                _terminate_process_group(process)
            else:
                process.wait()
                if _process_group_exists(process.pid):
                    _terminate_process_group(process)
    for entry in entries:
        process = entry["process"]
        handle = entry["handle"]
        run_dir = entry["run_dir"]
        assert isinstance(process, subprocess.Popen)
        status = process.wait()
        handle.close()
        entry["exit_status"] = status
        (run_dir / "exit_status.txt").write_text(
            str(status) + "\n", encoding="utf-8"
        )
        if (
            bool(entry.get("oom_found"))
            or _scan_log_incremental(run_dir / "stdout_stderr.log", entry)
            or status in (-9, 137)
        ):
            oom_detected = True
    return reason, oom_detected


def _run_pair(
    pair_dir: Path,
    left_name: str,
    left_command: Sequence[str],
    left_gpu: int,
    right_name: str,
    right_command: Sequence[str],
    right_gpu: int,
    execution_mode: str,
    *,
    left_role: Optional[str] = None,
    right_role: Optional[str] = None,
    resource_probe: Optional[Callable[[], Mapping[str, object]]] = None,
    require_run_config: bool = True,
) -> Dict[str, object]:
    pair_dir.mkdir(parents=True, exist_ok=False)
    prepared = {
        "status": "prepared",
        "execution_mode": execution_mode,
        "pair_valid": False,
        "oom_detected": False,
        "reasons": [],
    }
    _atomic_json(pair_dir / "pair_summary.json", prepared)
    left_dir = pair_dir / left_name
    right_dir = pair_dir / right_name
    entries: List[Dict[str, object]] = []
    reasons: List[str] = []
    try:
        left_process, left_handle, left_started = _launch(
            left_command, left_dir, left_gpu
        )
        entries.append(
            {
                "process": left_process,
                "handle": left_handle,
                "run_dir": left_dir,
                "log_offset": 0,
                "log_tail": b"",
            }
        )
        right_process, right_handle, right_started = _launch(
            right_command, right_dir, right_gpu
        )
        entries.append(
            {
                "process": right_process,
                "handle": right_handle,
                "run_dir": right_dir,
                "log_offset": 0,
                "log_tail": b"",
            }
        )
        monitor_reason, oom_detected = _monitor_processes(
            entries, resource_probe or _system_resource_snapshot
        )
        if monitor_reason:
            reasons.append(monitor_reason)
    except BaseException as exc:
        for entry in entries:
            process = entry["process"]
            assert isinstance(process, subprocess.Popen)
            _terminate_process_group(process)
            handle = entry["handle"]
            handle.close()
            run_dir = entry["run_dir"]
            status = process.wait()
            entry["exit_status"] = status
            (run_dir / "exit_status.txt").write_text(
                str(status) + "\n", encoding="utf-8"
            )
        reasons.append(type(exc).__name__ + ": " + str(exc))
        oom_detected = False
        right_started = left_started if entries else time.time()
    statuses = {
        left_name: entries[0].get("exit_status") if entries else None,
        right_name: entries[1].get("exit_status") if len(entries) > 1 else None,
    }
    if require_run_config:
        lock_identity, lock_path_hash = _family_lock_metadata()
        for name, run_dir, role, gpu in (
            (left_name, left_dir, left_role or left_name, left_gpu),
            (right_name, right_dir, right_role or right_name, right_gpu),
        ):
            try:
                _finalize_run_config(
                    run_dir,
                    execution_mode,
                    role,
                    gpu,
                    lock_identity,
                    lock_path_hash,
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                reasons.append(name + " config invalid: " + str(exc))
    pair_valid = (
        len(entries) == 2
        and statuses[left_name] == 0
        and statuses[right_name] == 0
        and not reasons
        and not oom_detected
    )
    summary: Dict[str, object] = {
        "status": "completed" if pair_valid else "failed",
        "execution_mode": execution_mode,
        left_name + "_gpu": left_gpu,
        right_name + "_gpu": right_gpu,
        left_name + "_exit_status": statuses[left_name],
        right_name + "_exit_status": statuses[right_name],
        "start_delta_seconds": abs(right_started - left_started),
        "oom_detected": oom_detected,
        "pair_valid": pair_valid,
        "reasons": reasons,
        "completed_at_utc": _utc_now(),
    }
    _atomic_json(pair_dir / "pair_summary.json", summary)
    return summary


def _run_one(
    command: Sequence[str],
    run_dir: Path,
    gpu: int,
    execution_mode: str,
    role: str,
    resource_probe: Callable[[], Mapping[str, object]],
) -> Dict[str, object]:
    process, handle, started = _launch(command, run_dir, gpu)
    entry: Dict[str, object] = {
        "process": process,
        "handle": handle,
        "run_dir": run_dir,
        "log_offset": 0,
        "log_tail": b"",
    }
    reason, oom_detected = _monitor_processes([entry], resource_probe)
    reasons = [reason] if reason else []
    try:
        lock_identity, lock_path_hash = _family_lock_metadata()
        _finalize_run_config(
            run_dir,
            execution_mode,
            role,
            gpu,
            lock_identity,
            lock_path_hash,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        reasons.append("config invalid: " + str(exc))
    return {
        "exit_status": entry["exit_status"],
        "oom_detected": oom_detected,
        "started_at_unix": started,
        "reasons": reasons,
    }


def _derive_role_variants(
    triplet_dir: Path, runs: Sequence[Mapping[str, object]]
) -> Tuple[str, str]:
    anchors = set()
    variants = set()
    for run in runs:
        role = str(run["logical_run"])
        config = json.loads(
            (triplet_dir / str(run["path"]) / "run_config.json").read_text(
                encoding="utf-8"
            )
        )
        value = config.get("variant")
        if not isinstance(value, str) or not value:
            raise ValueError(role + " run_config variant is required")
        if role.startswith("anchor_"):
            anchors.add(value)
        elif role.startswith("variant"):
            variants.add(value)
    if len(anchors) != 1 or len(variants) != 1:
        raise ValueError("anchor and variant roles must each use one variant")
    return next(iter(anchors)), next(iter(variants))


def _run_triplet_commands_impl(
    commands: Mapping[str, Sequence[str]],
    triplet_dir: Path,
    *,
    execution_mode: str,
    resource_probe: Optional[Callable[[], Mapping[str, object]]] = None,
    gpu_clear_timeout_seconds: float = 120,
    family_root: Optional[Path] = None,
    family_id: str = "turnvega",
    require_gpu_telemetry: Optional[bool] = None,
) -> Dict[str, object]:
    """Run a frozen triplet mode and always preserve a terminal summary."""
    if execution_mode not in EXECUTION_MODES:
        raise ValueError("execution mode must be resolved before formal output")
    required = {"anchor_before", "variant", "anchor_after"}
    if set(commands) != required:
        raise ValueError("commands must contain anchor_before, variant, anchor_after")
    triplet_dir = Path(triplet_dir)
    _ensure_empty_output(triplet_dir)
    canonical_root = Path(CANONICAL_FAMILY_LOCK_ROOT).resolve()
    if family_root is not None and Path(family_root).resolve() != canonical_root:
        raise ValueError("family_root must equal the canonical lock root")
    if family_id != EXPERIMENT_FAMILY:
        raise ValueError("family_id must equal the canonical experiment family")
    _freeze_family_mode(execution_mode)
    lock_identity, lock_path_hash = _family_lock_metadata()
    triplet_dir.mkdir(parents=True, exist_ok=True)
    probe = resource_probe or _system_resource_snapshot
    summary: Dict[str, object] = {
        "experiment_family": "turnvega",
        "family_lock_identity": lock_identity,
        "family_lock_path_sha256": lock_path_hash,
        "execution_mode": execution_mode,
        "require_gpu_telemetry": (
            execution_mode == "dual_gpu_crossover"
            if require_gpu_telemetry is None
            else require_gpu_telemetry
        ),
        "status": "prepared",
        "runner_valid": False,
        "oom_detected": False,
        "runs": [],
        "reasons": [],
        "started_at_utc": _utc_now(),
    }
    _atomic_json(triplet_dir / "triplet_summary.json", summary)

    def fail(reason: str, oom: bool = False) -> Dict[str, object]:
        reasons = summary["reasons"]
        assert isinstance(reasons, list)
        reasons.append(reason)
        summary["oom_detected"] = bool(summary["oom_detected"] or oom)
        summary["status"] = "failed"
        summary["runner_valid"] = False
        summary["completed_at_utc"] = _utc_now()
        _atomic_json(triplet_dir / "triplet_summary.json", summary)
        return summary

    initial_reason = _resource_stop_reason(probe())
    if initial_reason:
        return fail(initial_reason)

    runs = summary["runs"]
    assert isinstance(runs, list)
    if execution_mode == "dual_gpu_crossover":
        rounds = (
            (
                "round_a",
                "anchor",
                commands["anchor_before"],
                0,
                "variant",
                commands["variant"],
                1,
                "anchor_before",
                "variant_round_a",
            ),
            (
                "round_b",
                "variant",
                commands["variant"],
                0,
                "anchor",
                commands["anchor_after"],
                1,
                "variant_round_b",
                "anchor_after",
            ),
        )
        for (
            round_name,
            left_name,
            left_command,
            left_gpu,
            right_name,
            right_command,
            right_gpu,
            left_logical,
            right_logical,
        ) in rounds:
            pair = _run_pair(
                triplet_dir / round_name,
                left_name,
                left_command,
                left_gpu,
                right_name,
                right_command,
                right_gpu,
                execution_mode,
                left_role=left_logical,
                right_role=right_logical,
                resource_probe=probe,
            )
            runs.extend(
                [
                    {
                        "logical_run": left_logical,
                        "path": str(Path(round_name) / left_name),
                        "gpu": left_gpu,
                        "exit_status": pair[left_name + "_exit_status"],
                    },
                    {
                        "logical_run": right_logical,
                        "path": str(Path(round_name) / right_name),
                        "gpu": right_gpu,
                        "exit_status": pair[right_name + "_exit_status"],
                    },
                ]
            )
            if not pair["pair_valid"]:
                pair_reasons = pair.get("reasons") or []
                return fail(
                    round_name
                    + " failed: "
                    + "; ".join(str(reason) for reason in pair_reasons),
                    bool(pair["oom_detected"]),
                )
            clear_reason = _wait_for_gpu_clear(
                probe, (0, 1), gpu_clear_timeout_seconds
            )
            if clear_reason:
                return fail(clear_reason)
    else:
        for logical_name in ("anchor_before", "variant", "anchor_after"):
            run_dir = triplet_dir / logical_name
            result = _run_one(
                commands[logical_name],
                run_dir,
                0,
                execution_mode,
                logical_name,
                probe,
            )
            runs.append(
                {
                    "logical_run": logical_name,
                    "path": logical_name,
                    "gpu": 0,
                    "exit_status": result["exit_status"],
                }
            )
            if result["exit_status"] != 0 or result["reasons"]:
                return fail(
                    logical_name
                    + " failed: "
                    + "; ".join(str(reason) for reason in result["reasons"]),
                    bool(result["oom_detected"]),
                )
            must_check_gpu = bool(require_gpu_telemetry)
            if must_check_gpu:
                clear_reason = _wait_for_gpu_clear(
                    probe, (0,), gpu_clear_timeout_seconds
                )
                if clear_reason:
                    return fail(clear_reason)

    anchor_variant, target_variant = _derive_role_variants(triplet_dir, runs)
    summary["anchor_variant"] = anchor_variant
    summary["variant"] = target_variant
    summary["status"] = "completed"
    summary["runner_valid"] = True
    summary["completed_at_utc"] = _utc_now()
    _atomic_json(triplet_dir / "triplet_summary.json", summary)
    return summary


def run_triplet_commands(
    commands: Mapping[str, Sequence[str]],
    triplet_dir: Path,
    *,
    execution_mode: str,
    resource_probe: Optional[Callable[[], Mapping[str, object]]] = None,
    gpu_clear_timeout_seconds: float = 120,
    family_root: Optional[Path] = None,
    family_id: str = "turnvega",
    require_gpu_telemetry: Optional[bool] = None,
) -> Dict[str, object]:
    try:
        return _run_triplet_commands_impl(
            commands,
            triplet_dir,
            execution_mode=execution_mode,
            resource_probe=resource_probe,
            gpu_clear_timeout_seconds=gpu_clear_timeout_seconds,
            family_root=family_root,
            family_id=family_id,
            require_gpu_telemetry=require_gpu_telemetry,
        )
    except BaseException as exc:
        summary_path = Path(triplet_dir) / "triplet_summary.json"
        if not summary_path.is_file() or summary_path.is_symlink():
            raise
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            summary = {
                "experiment_family": "turnvega",
                "execution_mode": execution_mode,
                "runs": [],
                "reasons": [],
            }
        reasons = summary.get("reasons")
        if not isinstance(reasons, list):
            reasons = []
        reasons.append(type(exc).__name__ + ": " + str(exc))
        summary.update(
            status="failed",
            runner_valid=False,
            reasons=reasons,
            completed_at_utc=_utc_now(),
        )
        _atomic_json(summary_path, summary)
        return summary


def run_dual_gpu_smoke(
    smoke_command: Sequence[str],
    *,
    resource_probe: Optional[Callable[[], Mapping[str, object]]] = None,
) -> bool:
    """Concurrently exercise both isolated GPU sides before resolving auto."""
    probe = resource_probe or _system_resource_snapshot
    snapshot = probe()
    memories = dict(snapshot.get("gpu_memory_mib") or {})
    resources_ok = (
        _resource_stop_reason(snapshot) is None
        and len(memories) >= 2
        and all(
        int(memories.get(index, MAX_GPU_MEMORY_MIB)) < MAX_GPU_MEMORY_MIB
        for index in (0, 1)
        )
    )
    if not resources_ok:
        return False
    with tempfile.TemporaryDirectory(prefix="turnvega-dual-smoke-") as tmp:
        pair = _run_pair(
            Path(tmp) / "pair",
            "gpu0",
            smoke_command,
            0,
            "gpu1",
            smoke_command,
            1,
            "dual_gpu_crossover",
            resource_probe=probe,
            require_run_config=False,
        )
    return bool(pair["pair_valid"] and not pair["oom_detected"])


def _dual_gpu_smoke(python: str) -> bool:
    smoke_code = (
        "import torch;"
        "assert torch.cuda.is_available();"
        "x=torch.empty(1,device='cuda');"
        "assert x.is_cuda"
    )
    return run_dual_gpu_smoke([python, "-c", smoke_code])


def _experiment_command(
    args: argparse.Namespace, variant: str, run_id: str, output_dir: str
) -> List[str]:
    command = [
        args.python,
        "scripts/turnvega_experiment.py",
        "--dataset-kind",
        args.dataset_kind,
        "--variant",
        variant,
        "--manifest-path",
        args.manifest_path,
        "--candidate-passages",
        str(args.candidate_passages),
        "--prompt-evidence",
        str(args.prompt_evidence),
        "--max-search-calls",
        str(args.max_search_calls),
        "--trace-schema",
        "v5",
        "--run-id",
        run_id,
        "--output-dir",
        output_dir,
        "--backend",
        args.backend,
    ]
    if args.model:
        command.extend(["--model", args.model])
    if args.dataset_path:
        command.extend(["--dataset-path", args.dataset_path])
    return command


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triplet-dir", required=True)
    parser.add_argument(
        "--execution-mode",
        choices=["auto", *EXECUTION_MODES],
        required=True,
    )
    parser.add_argument("--pre-formal-smoke", action="store_true")
    parser.add_argument("--family-root")
    parser.add_argument("--family-id", default="turnvega")
    parser.add_argument("--dataset-kind", choices=["task2", "task3"], required=True)
    parser.add_argument("--anchor-variant", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--candidate-passages", type=int, required=True)
    parser.add_argument("--prompt-evidence", type=int, required=True)
    parser.add_argument("--max-search-calls", type=int, required=True)
    parser.add_argument("--backend", choices=["mlx", "vllm"], default="vllm")
    parser.add_argument("--model")
    parser.add_argument("--dataset-path")
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    dual_ok = (
        _dual_gpu_smoke(args.python) if args.execution_mode == "auto" else None
    )
    mode = resolve_execution_mode(
        args.execution_mode,
        pre_formal_smoke=args.pre_formal_smoke,
        dual_gpu_smoke_ok=dual_ok,
    )
    root = Path(args.triplet_dir)
    commands = {
        "anchor_before": _experiment_command(
            args,
            args.anchor_variant,
            args.run_prefix + "-anchor-before",
            "{output_dir}",
        ),
        "variant": _experiment_command(
            args,
            args.variant,
            args.run_prefix + "-variant-{run_suffix}",
            "{output_dir}",
        ),
        "anchor_after": _experiment_command(
            args,
            args.anchor_variant,
            args.run_prefix + "-anchor-after",
            "{output_dir}",
        ),
    }
    summary = run_triplet_commands(
        commands,
        root,
        execution_mode=mode,
        family_root=Path(args.family_root) if args.family_root else None,
        family_id=args.family_id,
        require_gpu_telemetry=(args.backend == "vllm"),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["runner_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

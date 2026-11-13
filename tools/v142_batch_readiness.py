#!/usr/bin/env python3
"""Resource safety and preregistered six-run native OFF/ON matrix.

Default is ALWAYS --dry-audit: never spawn IsaacLab, modify controllers,
create USD prims, take GPU locks or modify source data. GPU occupied =>
fail-closed. The six-run table is a contract, not evidence that runs happened.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import os
from v142_native_pair_ledger import frozen, safe_write_once, historical_recheck
from v142_human_data_qualification import review_assets

ROOT = Path(__file__).resolve().parents[1]
CASES = ("frozen_v32", "cmu_10_02", "cmu_10_05")
MODES = ("off", "on")
GPU_LOCK = Path("/home/you/projects/kinetickick/experiments/kinetickick_physics_gpu.lock")


def execute_cmd(*cmd):
    return subprocess.check_output(cmd, text=True, timeout=8, stderr=subprocess.STDOUT).strip()


def gpu_occupants():
    out = execute_cmd("nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                      "--format=csv,noheader")
    rows = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3 or not parts[0].isdigit():
            raise ValueError("UNPARSEABLE_GPU_OCCUPANT")
        rows.append({"pid": int(parts[0]), "executable": parts[1],
                     "used_memory": parts[2]})
    return rows


def readiness(*, free_gib, gpu_rows, branch, git_dirty, lock_exists):
    cfg = frozen()
    if branch != "version/v0.1.142":
        raise ValueError("WRONG_NATIVE_EVIDENCE_BRANCH")
    jobs = [{"policy": case, "observer_mode": mode,
             "source_original_200hz_expected": cfg["original_cases"][case],
             "new_physical_run_complete": False}
            for case in CASES for mode in MODES]
    if len(jobs) != 6 or len(set((j["policy"], j["observer_mode"]) for j in jobs)) != 6:
        raise ValueError("BROKEN_PREREGISTERED_BATCH")
    blockers = []
    if free_gib < cfg["min_free_disk_gib"]:
        blockers.append("DISK_BELOW_FROZEN_20GIB")
    if gpu_rows:
        blockers.append("GPU_OCCUPIED_BY_EXISTING_JOB")
    if not lock_exists:
        blockers.append("ORIGINAL_GPU_INTERLOCK_FILE_NOT_FOUND")
    if git_dirty:
        blockers.append("SOURCE_DIRTY_CANNOT_START_NATIVE_RUN")
    # No CLI execution mode by design. GPU may become free, but "ready" is
    # only a PROPOSAL; native calibration and real provenance still required.
    return {
        "branch": branch, "free_disk_gib": round(float(free_gib), 3),
        "gpu_occupants": gpu_rows,
        "original_lock_exists": lock_exists,
        "blockers": blockers,
        "can_start_real_native_now": False,
        "would_pass_minimal_resource_preflight": not blockers,
        "calibration_runs_predeclared": ["isolated_contact_positive_on",
                                         "isolated_contact_positive_off",
                                         "isolated_separated_negative_on"],
        "original_G1_paired_jobs": jobs,
        "original_G1_physx_runs_planned": len(jobs),
        "original_G1_physx_runs_executed_by_this_tool": 0,
        "native_collider_pair_certified": False,
        "native_observer_read_only_invariance_certified": False,
        "new_strategy_14s_kick_success_rate": None,
        "frozen_config_sha": cfg["parent_sha"],
    }


def inspect():
    branch = execute_cmd("git", "-C", str(ROOT), "branch", "--show-current")
    # Only source dirty matters; ignored outputs in reports never block CPU.
    dirty = bool(execute_cmd("git", "-C", str(ROOT), "status",
                             "--porcelain", "--untracked-files=no"))
    # Prevent source/build artifacts from being mistaken for root evidence.
    free = shutil.disk_usage(ROOT).free / (1024 ** 3)
    rows = gpu_occupants()
    return readiness(free_gib=free, gpu_rows=rows, branch=branch,
                     git_dirty=dirty, lock_exists=GPU_LOCK.exists())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cfg = frozen()
    result = {
        "version": cfg["version"],
        "intent": "READ_ONLY_RESOURCE_AND_SOURCE_AUDIT_NOT_SIMULATION",
        "preflight": inspect(),
        "historical_physics": historical_recheck(),
        "human_data": review_assets(),
        "new_physx_runs": 0,
        "high_speed_natural_instep_learned": False,
    }
    s = safe_write_once(args.output, result)
    print(json.dumps({"output": str(args.output), "sha256": s,
                      "blocked": bool(result["preflight"]["blockers"]),
                      "blockers": result["preflight"]["blockers"],
                      "historical_frames": result["historical_physics"]["historical_physics_5ms_frames"],
                      "human_clips": result["human_data"]["clip_count"],
                      "new_physx_runs": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()

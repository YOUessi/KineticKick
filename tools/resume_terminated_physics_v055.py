#!/usr/bin/env python3
"""Resume immutable v55 treatment matrix after a *valid* early physical stop.

This is a scorer/recorder amendment, not a control/protocol or training change.
The r4 launcher rejected case 10 because a genuine original G1 nonfoot stop
at 1.64s produced only 329 x 200Hz frames. DO NOT delete, overwrite or replay
that negative: ingest it under its original JSON SHA and console log.
"""
from __future__ import annotations
import fcntl
import json
import os
import subprocess
import sys
import time
import numpy as np
from pathlib import Path
from run_pair_dynamics_batch_v055 import ROOT, CONFIG, PROGRESS, LOCK, freeze, sha

def admit(report: dict, case: dict) -> tuple[int, bool]:
    if report.get("root_fixed") is not False or report.get("post_initialization_state_writes"):
        raise ValueError("Original free-root/non-mutating PhysX required")
    if report.get("seed") != case["seed"]:
        raise ValueError("Incorrect seed in original output")
    a = report.get("joint_strike", {})
    if not np.allclose(a.get("ball_offset_xy", []), case["ball_offset_xy_m"],
                       atol=1e-8, rtol=0):
        raise ValueError("Wrong frozen ball placement")
    if abs(a.get("velocity_y_delta", float("nan")) - case["velocity_y_delta_mps"]) > 1e-8:
        raise ValueError("Wrong frozen initial velocity")
    if report.get("policy_sha256") != sha(ROOT / json.loads(CONFIG.read_text())["policies"][case["policy"]]):
        raise ValueError("Frozen policy has changed")
    rows = report.get("traces", [[]])[0]
    if len(rows) < 20 or any(len(r.get("joint_pos", [])) != 29 or
                             len(r.get("joint_vel", [])) != 29 for r in rows):
        raise ValueError("Missing actual original G1 29DoF motion")
    t = np.asarray([r["time_s"] for r in rows], dtype=float)
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Nonfinite or nonmonotone original physics time")
    term = report.get("termination_reasons")
    complete = len(rows) == 2800 and abs(t[-1] - 14) < 1e-5 and term == [[]]
    early_nonfoot = bool(term == [["physical_nonfoot"]] and t[-1] < 14)
    # The original runner appends one epsilon-time terminal state (t+1us)
    # only on early physical termination. This is NOT another 200Hz step.
    intervals = np.diff(t)
    regular = intervals if complete else intervals[:-1]
    if (not np.allclose(regular, .005, atol=1e-5, rtol=0) or
        (early_nonfoot and not (0 < intervals[-1] <= .00501))):
        raise ValueError("Missing 200Hz physics cadence (except terminal sentinel)")
    if not (complete or early_nonfoot):
        raise ValueError("Only complete 14s or actual original physical_nonfoot stop admissible")
    if early_nonfoot and (report["records"][0]["recovered"] or
                          not report.get("nonfoot_contact_diagnostics")):
        raise ValueError("Early stop missing actual original named nonfoot evidence")
    if "nonfoot_contact_diagnostics" not in report:
        raise ValueError("Missing original nonfoot sensor recording")
    return len(rows), complete

def run():
    if "env_isaaclab" not in sys.executable:
        raise RuntimeError("Original IsaacLab runtime required")
    c = json.loads(CONFIG.read_text())
    freeze(c)  # unchanged r4 treatment/source SHA, not a re-registered experiment
    p = json.loads(PROGRESS.read_text())
    if p["protocol_sha256"] != sha(CONFIG):
        raise ValueError("Original r4 protocol SHA changed")
    history = p["runs"]
    if len(history) > len(c["cases"]):
        raise ValueError("Invalid previous progress count")
    for i, r in enumerate(history):
        case = c["cases"][i]
        if r["id"] != case["id"] or r["sha256"] != sha(ROOT / case["out"]):
            raise ValueError("Original recorded evidence changed")

    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a") as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError("Another KineticKick physics batch holds GPU lock") from e
        for i, case in enumerate(c["cases"][len(history):], start=len(history)):
            freeze(c)
            out = ROOT / case["out"]
            log = out.with_suffix(".log")
            existing = out.exists() or log.exists()
            if existing and not (out.exists() and log.exists()):
                raise FileExistsError("Partial evidence exists; manual review required: " + str(out))
            if not existing:
                check_gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                    "--format=csv,noheader"], capture_output=True, text=True)
                if check_gpu.returncode != 0 or check_gpu.stdout.strip():
                    raise RuntimeError("Other GPU workload present; refuse to launch")
                fs = os.statvfs(ROOT)
                if fs.f_bavail * fs.f_frsize < 20 * 1024**3:
                    raise RuntimeError("Less than 20GB free; protect existing evidence")
                args = [sys.executable, str(ROOT / "tools/run_placement_strike_v055.py"),
                        "--config", str(ROOT / c["policies"][case["policy"]]),
                        "--output", str(out), "--ball-offset",
                        *[str(x) for x in case["ball_offset_xy_m"]],
                        "--velocity-delta", str(case["velocity_y_delta_mps"]),
                        "--eval-seed", str(case["seed"]), "--contact-diagnostics"]
                print("V055_R5_REAL_PHYSX_BEGIN", i+1, len(c["cases"]), case["id"], flush=True)
                start = time.monotonic()
                with log.open("x") as fd:
                    proc = subprocess.run(args, cwd=ROOT, env=dict(os.environ,
                        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                        MKL_NUM_THREADS="1",
                        PYTHONPATH=":".join(["tools", str(ROOT /
                            "third_party/HumanoidSoccer/source/whole_body_tracking")])),
                        stdout=fd, stderr=subprocess.STDOUT)
                if proc.returncode != 0 or not out.exists():
                    raise RuntimeError("Original G1 physical task failed (unscored), never overwrite: " + str(log))
            else:
                start = None
                print("V055_R5_ADMIT_EXISTING_VALID_NEGATIVE", case["id"], flush=True)
            report = json.loads(out.read_text())
            n, full = admit(report, case)
            r = dict(id=case["id"], out=case["out"], sha256=sha(out),
                     size_bytes=out.stat().st_size,
                     wall_s=round(time.monotonic()-start,2) if start else None,
                     physics_samples=n,
                     termination=report["termination_reasons"],
                     full_14s=full,
                     early_original_nonfoot_stop=not full,
                     original_preexisting_failure_preserved=existing,
                     scoring_amendment_only=True,
                     pair_manifold_events=None, collider_path_events=None,
                     new_pair_instrumentation_not_activated=True)
            history.append(r)
            p["runs"] = history
            p["scoring_amendment"] = {
                "reason": "r4 incorrectly excluded valid early 329-frame original physical_nonfoot stop",
                "treatment_config_sha256_unchanged": sha(CONFIG),
                "original_r4_launcher_sha256_unchanged": sha(ROOT / "tools/run_pair_dynamics_batch_v055.py"),
                "existing_negative_files_never_overwritten": True,
                "admission_rule": "original 200Hz 29DoF 14s complete OR terminated physical_nonfoot with named net force"}
            PROGRESS.write_text(json.dumps(p, indent=2, ensure_ascii=False)+"\n")
            print("V055_R5_REAL_PHYSX_END", json.dumps(r), flush=True)
    print("V055_R5_BATCH_COMPLETE",len(history),flush=True)

if __name__ == "__main__":
    run()

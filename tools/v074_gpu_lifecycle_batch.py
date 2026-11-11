#!/usr/bin/env python3
"""GPU mutex, immutable evidence and bounded Isaac lifecycle in fresh worktree."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("/home/you/projects/kinetickick")
V72 = Path("/home/you/projects/kinetickick-wt-v072")
PYTHON = Path("/home/you/conda_env/env_isaaclab/bin/python")
PROTOCOL = ROOT / "configs/v074_frozen_native_lifecycle_research.json"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def preflight(protocol: dict) -> float:
    if protocol["version"] != "v0.1.74":
        raise ValueError("WRONG_VERSION")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != "version/v0.1.74":
        raise ValueError("WRONG_BRANCH")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise ValueError("TRACKED_DIRTY")
    for name, expected in protocol["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("SOURCE_DRIFT:" + name)
    for name, expected in protocol["external_v72_sha256"].items():
        if sha(V72 / name) != expected:
            raise ValueError("V72_SOURCE_DRIFT:" + name)
    for name, expected in protocol["original_physx_sha256"].items():
        if sha(SOURCE_ROOT / name) != expected:
            raise ValueError("ORIGINAL_PHYSX_CHANGED:" + name)
    disk = os.statvfs(str(ROOT))
    gib = disk.f_bavail * disk.f_frsize / (1024**3)
    if gib < protocol["free_disk_gib_minimum"]:
        raise RuntimeError("INSUFFICIENT_DISK:" + str(gib))
    app = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=15, check=True)
    if app.stdout.strip():
        raise RuntimeError("GPU_OCCUPIED:" + app.stdout.strip())
    return gib

def classify(text: str, code: int, timeout: bool) -> str:
    if "APP_CLOSE_RETURN" in text and code == 0:
        return "NORMAL_EIGHT_STEP_SHUTDOWN" if "EIGHT_REAL_SIM_STEPS_DONE_NOT_G1" in text else "NORMAL_WITHOUT_ALL_STEPS"
    if "APP_CLOSE_ENTER" in text and "APP_CLOSE_RETURN" not in text and timeout:
        return "APP_CLOSE_ENTERED_NEVER_RETURNED_WITHIN_WATCHDOG"
    if "APP_CLOSE_RETURN" in text and timeout:
        return "APP_CLOSE_RETURNED_INTERPRETER_EXIT_HUNG"
    if "SIM_STEP_8_RETURN" in text and timeout:
        return "EIGHT_STEPS_DONE_PRE_CLOSE_OR_FINALIZATION_TIMEOUT"
    if timeout:
        return "ISAAC_INIT_OR_SIM_TIMEOUT"
    return "ISAAC_PROBE_NONZERO"

def execute(variant: str, timeout_s: int, protocol: dict, free_gib: float) -> dict:
    logpath = ROOT / ("reports/v074_lifecycle_" + variant + ".log")
    outpath = ROOT / ("reports/v074_lifecycle_" + variant + ".json")
    if logpath.exists() or outpath.exists():
        raise FileExistsError("WRITE_ONCE_EVIDENCE_COLLISION:" + variant)
    cmd = [str(PYTHON), str(ROOT / "tools/v074_isaac_close_probe.py"), "--variant", variant]
    env = dict(os.environ, PYTHONUNBUFFERED="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="0")
    started = time.monotonic()
    timeout = False
    with logpath.open("x") as log:
        log.write("V074_WRAPPER_START " + repr(cmd) + "\n")
        log.flush()
        p = subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
        try:
            code = p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timeout = True
            os.kill(p.pid, signal.SIGUSR1)
            time.sleep(2.0)
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
            try:
                code = p.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                code = p.wait(timeout=5.0)
            log.write("\nV074_OWNED_CHILD_WATCHDOG_TIMEOUT_PROBE_NOT_G1\n")
            log.flush()
    raw = logpath.read_text(errors="replace")
    verdict = classify(raw, code, timeout)
    data = dict(version="v0.1.74", variant=variant, classification=verdict,
                timeout=timeout, return_code=code, elapsed_wall_s=round(time.monotonic()-started, 3),
                child_pid=p.pid, raw_sha256=sha(logpath), raw_bytes=logpath.stat().st_size,
                original_protocol_sha256=sha(PROTOCOL),
                freeze_commit="2000bc3dbfd6e4faba7ca84eb77718c0c0f89e96",
                source_probe_sha256=sha(ROOT / "tools/v074_isaac_close_probe.py"),
                source_runner_sha256=sha(Path(__file__)),
                disk_free_GiB=round(free_gib, 2), real_isaac_steps=raw.count("_RETURN"),
                original_g1_physics_episodes=0, scored_kicks=0,
                app_close_return="APP_CLOSE_RETURN" in raw,
                real_original_native_G1_contact_pair_certified=False,
                next_stage_allowed=bool(code==0 and "APP_CLOSE_RETURN" in raw and
                                        "EIGHT_REAL_SIM_STEPS_DONE_NOT_G1" in raw))
    with outpath.open("x") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("V074_LIFECYCLE_RESULT " + json.dumps(data, ensure_ascii=False), flush=True)
    return data

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--only", choices=("baseline", "cleanup", "all"), default="all")
    args=p.parse_args()
    c=json.loads(PROTOCOL.read_text())
    locks=Path(c["gpu_lock"])
    if not locks.exists():
        raise FileNotFoundError("existing G1 GPU mutex not found")
    with locks.open("r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stages=c["stages"]
        variants=["baseline","cleanup"] if args.only=="all" else [args.only]
        summary=[]
        for variant in variants:
            gib=preflight(c)
            row=next(x for x in stages if x.get("probe")==variant)
            summary.append(execute(variant, row["duration_max_s"], c, gib))
            # Both probes are diagnostics; do not start G1 if either cannot close.
            if variant=="baseline" and summary[-1]["next_stage_allowed"]:
                print("V074_BASELINE_HEALTHY_NO_NEED_FOR_CLEANUP_DIAGNOSTIC",flush=True)
                break
        print("V074_FINAL_DIAGNOSTIC",json.dumps([
            [x["variant"],x["classification"]] for x in summary]), flush=True)

if __name__=="__main__":
    main()

#!/usr/bin/env python3
"""v72 SHA-locked, single-process, GPU-mutex isolated Isaac boot probe.

Unmodified v71 smoke source, no policy/body/PhysX changes. A timeout here is
NOT a soccer episode. Exclusive evidence, fail closed; never retries itself.
"""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/v072_runtime_triage_and_native_pair.json"
LOG = ROOT / "reports/v072_kit_empty_headless.log"
RESULT = ROOT / "reports/v072_kit_empty_headless_result.json"
GPU_MUTEX = Path("/home/you/projects/kinetickick/experiments/kinetickick_physics_gpu.lock")
ISAAC_PYTHON = Path("/home/you/conda_env/env_isaaclab/bin/python")
SHA = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def preflight(*, check_gpu=True):
    c = json.loads(PROTOCOL.read_text())
    if c["version"] != "v0.1.72" or c["parent_sha"] != "ca2e3f51a7d30467bb6416b9abc0e7468cac2155":
        raise RuntimeError("PROTOCOL_REF_MISMATCH")
    if subprocess.check_output(["git", "branch", "--show-current"], text=True, cwd=ROOT).strip() != "version/v0.1.72":
        raise RuntimeError("VERSION_BRANCH_MISMATCH")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True, cwd=ROOT).strip():
        raise RuntimeError("TRACKED_DIRTY_SOURCE")
    if Path(sys.executable).resolve() != ISAAC_PYTHON.resolve():
        raise RuntimeError("ISAAC_PYTHON_NOT_PINNED")
    if any(SHA(ROOT / name) != sha for name, sha in c["sources_sha256"].items()):
        raise RuntimeError("SOURCE_DRIFT")
    disk = os.statvfs(str(ROOT))
    free_gb = disk.f_bavail * disk.f_frsize / 1024**3
    if free_gb < 20:
        raise RuntimeError("DISK_BELOW_20_GIB")
    if check_gpu:
        p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                           text=True, capture_output=True, timeout=12)
        if p.returncode or p.stdout.strip():
            raise RuntimeError("GPU_OCCUPIED:" + p.stdout.strip())
    if LOG.exists() or RESULT.exists():
        raise FileExistsError("Probe evidence already exists: never overwrite")
    return c, free_gb

def classify(log, rc, timed_out):
    if "KINETICKICK_ISAAC_SMOKE_OK" in log:
        if rc == 0:
            return "ISAAC_CONTEXT_EIGHT_STEPS_VALID"
        return "ISAAC_EIGHT_STEPS_DONE_UNCLEAN_SHUTDOWN_OR_FINALIZATION_NOT_G1"
    if timed_out:
        return "ISAAC_BOOT_OR_SIM_STEP_TIMEOUT_NOT_G1"
    if "[INFO][AppLauncher]" not in log:
        return "PYTHON_OR_APP_LAUNCHER_BEFORE_KIT"
    return "ISAAC_CONTEXT_EXIT_NONZERO_NOT_G1"

def run():
    GPU_MUTEX.parent.mkdir(parents=True, exist_ok=True)
    # Use precisely the same mutex as the existing v70/69 G1 jobs.
    with GPU_MUTEX.open("a") as h:
        fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c, free_gb = preflight()
        timeout_s = c["stages"][0]["max_wall_s"]
        cmd = [str(ISAAC_PYTHON), str(ROOT / c["stages"][0]["script"])]
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   PYTHONUNBUFFERED="1")
        start = time.monotonic()
        timed_out = False
        with LOG.open("x") as handle:
            try:
                result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle,
                                        stderr=subprocess.STDOUT, timeout=timeout_s)
                rc = result.returncode
            except subprocess.TimeoutExpired:
                rc = 124
                timed_out = True
                handle.write("V072_TIMEOUT_AFTER_OR_BEFORE_SMOKE_MARKER_NO_G1_EVIDENCE\n")
        raw = LOG.read_text(errors="replace")
        report = dict(schema_version=1,classification=classify(raw,rc,timed_out),
                      is_actual_G1_physics=False, actual_G1_episodes=0,
                      original_controller_changed=False, original_physx_changed=False,
                      rc=rc, timeout=timed_out, max_wall_s=timeout_s,
                      elapsed_s=round(time.monotonic()-start,3), free_GiB_at_start=round(free_gb,2),
                      frozen_protocol_sha256=SHA(PROTOCOL), frozen_source_git_sha=c["parent_sha"],
                      original_smoke_sha256=SHA(ROOT/c["stages"][0]["script"]),
                      raw_log_sha256=SHA(LOG), raw_log_bytes=LOG.stat().st_size,
                      kit_app_launcher_reached="[INFO][AppLauncher]" in raw,
                      Kit_logging_reached="Logging to file:" in raw,
                      smoke_completed="KINETICKICK_ISAAC_SMOKE_OK" in raw,
                      scientific_limitation="A smoke pass cannot attest native G1, contact manifold, or shooting ability")
        with RESULT.open("x") as f:
            json.dump(report,f,indent=2)
            f.write("\n")
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__ == "__main__":
    run()

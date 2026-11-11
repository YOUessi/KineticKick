#!/usr/bin/env python3
"""Write-once, GPU-mutexed, preregistered G1 liveness / stack evidence.

No controller, IsaacLab, USD or original evidence is changed. Diagnostic
timing is not a physical no-perturbation certification.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path("/home/you/projects/kinetickick")
V076 = Path("/home/you/projects/kinetickick-wt-v076")
CONFIG = ROOT / "configs/v077_frozen_boot_diagnostics.json"
FROZEN_SHA = "f299cae1a9d044d99e2c936f3c75c02d10605c264e87a134719ba1d25ef2a928"
PYTHON = "/home/you/conda_env/env_isaaclab/bin/python"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validated_config() -> dict:
    if sha(CONFIG) != FROZEN_SHA:
        raise RuntimeError("PROTOCOL_SHA_DRIFT")
    c = json.loads(CONFIG.read_text())
    if c["version"] != "v0.1.77":
        raise RuntimeError("VERSION_DRIFT")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != "version/v0.1.77":
        raise RuntimeError("WRONG_BRANCH")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    ).strip()
    if dirty:
        raise RuntimeError("TRACKED_SOURCE_DIRTY")
    if not subprocess.check_output(
        ["git", "merge-base", "--is-ancestor", c["base_worktree_sha"], "HEAD"], cwd=ROOT
    ) == b"":
        raise RuntimeError("INCORRECT_BASE")
    for rel, expected in c["source_sha256"].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError("SOURCE_SHA_DRIFT:" + rel)
    for rel, expected in c["original_physics_sha256"].items():
        if sha(ORIGINAL / rel) != expected:
            raise RuntimeError("ORIGINAL_PHYSICS_DRIFT:" + rel)
    for rel, expected in c["prior_v076_failure_audits"].items():
        if sha(V076 / "reports" / rel) != expected:
            raise RuntimeError("PRIOR_G1_DIAGNOSTIC_DRIFT:" + rel)
    raw = json.loads((ROOT / c["case_config"]).read_text())
    for rel, expected in raw["frozen_files_sha256"].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError("CASE_FROZEN_INPUT_SHA_DRIFT:" + rel)
    for key in ("template", "reference", "expert_plan"):
        if sha(ROOT / raw[key]) != raw[key + "_sha256"]:
            raise RuntimeError("CASE_SUPPORT_DRIFT:" + key)
    sv = os.statvfs(ROOT)
    free_gib = sv.f_bavail * sv.f_frsize / 1024**3
    if free_gib < c["minimum_free_gib"]:
        raise RuntimeError("DISK_BELOW_20_GIB")
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name",
         "--format=csv,noheader"],
        capture_output=True, check=True, text=True, timeout=12,
    )
    if result.stdout.strip():
        raise RuntimeError("GPU_ALREADY_OWNED:" + result.stdout)
    return c


def classify(text: str, valid_episode: bool, timed_out: bool) -> dict:
    markers = {
        "python_trampoline_started": "V077_DIAG_SIGNAL_REGISTERED" in text,
        "isaac_environment_constructed": "Completed setting up the environment" in text,
        "policy_network_created": "Actor RNN:" in text,
        "original_g1_report_marker": "CLOSED_LOOP_STRIKE_RESULT" in text,
        "post_physics_shutdown_guard": "V076_POST_PHYSICS_STOP_CALLBACK_GUARD_TRUE" in text,
        "has_python_stack_dump": "Current thread" in text or "Thread " in text and "File " in text,
        "has_exception_traceback": "Traceback (most recent call last)" in text,
    }
    blocked = "NO_CERTIFIED_EPISODE" if not valid_episode else "PHYSX_EPISODE_PRESENT_DIAGNOSTIC_ONLY"
    stack = re.findall(r'\s*File "([^"\n]+)", line (\d+), in ([^\n]+)', text)
    return {
        "markers": markers,
        "outcome": blocked,
        "timed_out": timed_out,
        "stack_tail": [dict(file=f, line=int(n), function=fn.strip()) for f, n, fn in stack[-24:]],
        "stack_frame_count": len(stack),
    }


def is_valid_g1_report(path: Path) -> tuple[bool, dict]:
    if not path.exists():
        return False, {"reason": "NO_PHYSICS_REPORT"}
    try:
        p = json.loads(path.read_text())
        traces = p.get("traces", [])
        rec = p.get("records", [])
        names = p.get("joint_names", [])
        length = len(traces[0]) if traces else 0
        valid = len(names) == 29 and length >= 2 and bool(rec) and p.get("sequence_records") is not None
        return bool(valid), {"trace_len": length, "joint_count": len(names),
                             "record_count": len(rec), "reason": "COMPLETE_JSON" if valid else "INVALID_JSON_SCHEMA"}
    except (OSError, ValueError, TypeError, KeyError, IndexError) as e:
        return False, {"reason": "BAD_REPORT:" + type(e).__name__}


def diagnose() -> dict:
    c = validated_config()
    prefix = ROOT / "reports" / c["run_namespace"]
    logfile, output, sidecar = (
        prefix.with_suffix(".log"), prefix.with_suffix(".physx.json"),
        prefix.with_suffix(".audit.json"),
    )
    # Path.with_suffix would replace the namespace portion if it contains a dot;
    # avoid re-use of any existing old scientific record.
    if any(p.exists() for p in (logfile, output, sidecar)):
        raise FileExistsError("WRITE_ONCE_OUTPUT_ALREADY_EXISTS:" + str(prefix))
    lockfile = Path(c["gpu_lock"])
    if not lockfile.is_file():
        raise FileNotFoundError("EXISTING_GPU_MUTEX_MISSING")
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(ROOT / "tools") + ":" + str(ROOT / "third_party/HumanoidSoccer/source/whole_body_tracking"),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    })
    command = [PYTHON, str(ROOT / "tools/v077_g1_stack_child.py"),
               "--config", str(ROOT / c["case_config"]),
               "--output", str(output), "--contact-diagnostics", "--eval-seed", "1201"]
    with lockfile.open("r+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("GPU_LOCK_HELD_BY_OTHER_JOB")
        # second check inside lock to guard between preflight and spawn
        gpus = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                        "--format=csv,noheader"], text=True).strip()
        if gpus:
            raise RuntimeError("GPU_ACQUIRED_BY_OTHER_PROCESS:" + gpus)
        with logfile.open("x") as log:
            log.write("V077_FROZEN_PARENT_SHA=" + c["base_worktree_sha"] + "\n")
            log.write("V077_FROZEN_CONFIG_SHA=" + FROZEN_SHA + "\n")
            log.write("V077_CHILD_ARGS=" + repr(command) + "\n")
            log.flush()
            start = time.monotonic()
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            sent = []
            deadline = c["child_timeout_s"]
            timed_out = False
            while proc.poll() is None:
                elapsed = time.monotonic() - start
                for t in c["stack_signal_s"]:
                    if t not in sent and elapsed >= t:
                        try:
                            os.kill(proc.pid, signal.SIGUSR1)
                            sent.append(t)
                            log.write(f"V077_STACK_SIGNAL_AFTER_{t}_S\n")
                            log.flush()
                        except ProcessLookupError:
                            pass
                if elapsed >= deadline:
                    timed_out = True
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=4)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                    break
                time.sleep(0.5)
            code = proc.wait(timeout=10)
            elapsed = time.monotonic() - start
            log.write(f"V077_CHILD_EXIT_CODE={code} WATCHDOG={timed_out} ELAPSED={elapsed:.2f}\n")
    valid, report = is_valid_g1_report(output)
    if timed_out or code != 0:
        valid = False
    txt = logfile.read_text(errors="replace")
    outcome = classify(txt, valid, timed_out)
    payload = {
        "version": c["version"], "protocol_sha256": FROZEN_SHA,
        "run_namespace": c["run_namespace"], "original_controller_sha256": c["source_sha256"]["tools/v076_run_placement_strike.py"],
        "gpu_original_g1_attempt_count": 1,
        "physics_episode_certified": valid,
        "strict_shot_success_rate": None,
        "return_code": code, "watchdog": timed_out, "wall_seconds": round(elapsed, 2),
        "signal_schedule_s": c["stack_signal_s"], "signals_delivered": sent,
        "raw_output_sha256": sha(output) if output.exists() else None,
        "log_sha256": sha(logfile), "g1_schema": report,
        "liveness": outcome, "native_collider_pair_certified": False,
        "no_disturbance_physx_parity_certified": False,
    }
    with sidecar.open("x") as out:
        json.dump(payload, out, ensure_ascii=False, indent=2)
        out.write("\n")
    print("V077_G1_LIVENESS_AUDIT", json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


if __name__ == "__main__":
    diagnose()

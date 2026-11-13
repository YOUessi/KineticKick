#!/usr/bin/env python3
"""v151 six original-G1 OFF/ON runs: immutable plan, safe optional execution.

Dry-run is the default. Importing this module never starts IsaacLab/GPU.
The historical v54 records are measured trajectories; planned runs are NOT results.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("/home/you/projects/kinetickick")
PROTOCOL = ROOT / "configs/v151_frozen_native_parity_campaign.json"
FROZEN_SHA = "2d835f72658c712dbc045a30cd97e13510db2234"
RUNNER = ROOT / "tools/v149_run_original_anyphase_readonly.py"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def frozen():
    raw = subprocess.check_output(
        ["git", "-C", str(ROOT), "show",
         FROZEN_SHA + ":configs/v151_frozen_native_parity_campaign.json"])
    if raw != PROTOCOL.read_bytes():
        raise ValueError("V151_FROZEN_PROTOCOL_DRIFT")
    cfg = json.loads(raw)
    if (cfg.get("version") != "v0.1.151"
        or cfg.get("parent_sha") != "9d9575662db2b6434f30b35c93a457a889425f58"
        or cfg.get("physical_dt_s") != .005 or cfg.get("original_dof") != 29
        or cfg.get("policies") != ["frozen_v32","cmu_10_02","cmu_10_05"]
        or cfg.get("observer_modes") != ["off","on"]
        or cfg.get("seed") != 1201 or cfg.get("live_trial_count") != 6
        or cfg.get("new_gpu_trials_when_foreign_context") != 0
        or cfg.get("history_expected_rows") !=
            {"frozen_v32":2800,"cmu_10_02":2800,"cmu_10_05":1028}
        or cfg.get("strict_kick_gates", {}).get("legal_instep_impulse_fraction_min") != .95
        or cfg.get("strict_kick_gates", {}).get("ball_speed_mps_min") != 3.5
        or cfg.get("minimum_disk_free_gib") != 20
        or not all(cfg.get(k) is True for k in (
            "source_only_not_native_manifold", "off_on_exact_full_physics_required",
            "native_positive_negative_controls_required",
            "same_native_tick_missing_means_unknown", "no_events_is_not_absence",
            "do_not_apply_usd_schema","never_modify_original_controller","outputs_write_once"))):
        raise ValueError("V151_ASSUMPTIONS_DRIFT")
    return cfg


def source_inventory(source_root=SOURCE_ROOT):
    cfg = frozen()
    source_root = Path(source_root)
    original = source_root / cfg["historical_protocol"]
    expected_sha = "add8693911cdd2489f3fb944adb77698238ddca50f14867a2a388b31eaeeeefa"
    if sha(original) != expected_sha:
        raise ValueError("V151_ORIGINAL_V54_PROTOCOL_CHANGED")
    historical = json.loads(original.read_text())
    if list(historical["original_three_center_policies"]) != cfg["policies"]:
        raise ValueError("V151_ORIGINAL_POLICY_ORDER_CHANGED")
    checked = []
    for name in cfg["policies"]:
        original_config = historical["original_three_center_policies"][name]["config"]
        p = source_root / original_config
        expected = historical["immutable_sha256"][original_config]
        if sha(p) != expected:
            raise ValueError("V151_ORIGINAL_POLICY_SHA_CHANGED:" + name)
        data = json.loads(p.read_text())
        for key in ("template","reference","expert_plan"):
            path = source_root / data[key]
            if sha(path) != data[key + "_sha256"]:
                raise ValueError("V151_ORIGINAL_POLICY_DEPENDENCY_DRIFT:" + name + ":" + key)
        if data.get("seed") != 2601:
            raise ValueError("V151_ORIGINAL_SEED_DRIFT")
        checked.append(dict(policy=name, config=original_config, config_sha256=expected,
                            expected_history_rows=cfg["history_expected_rows"][name]))
    if not RUNNER.is_file():
        raise FileNotFoundError("V151_ORIGINAL_READONLY_RUNNER_MISSING")
    # Only immutable parent v149 runner, never mutate original policy.
    baseline = subprocess.check_output(
        ["git","-C",str(ROOT),"show",
         "9d9575662db2b6434f30b35c93a457a889425f58:tools/v149_run_original_anyphase_readonly.py"])
    if baseline != RUNNER.read_bytes():
        raise ValueError("V151_PARENT_READONLY_RUNNER_CHANGED")
    return checked


def make_campaign():
    cfg = frozen()
    inventory = source_inventory()
    rows = []
    for item in inventory:
        for mode in cfg["observer_modes"]:
            name = item["policy"] + "_" + mode
            output = ROOT / "reports" / ("v151_" + name + "_original.json")
            sidecar = ROOT / "reports" / ("v151_" + name + "_native_sidecar.json")
            log = ROOT / "reports" / ("v151_" + name + "_run.log")
            args = [
                "/home/you/conda_env/env_isaaclab/bin/python",str(RUNNER.resolve()),
                "--config",str((ROOT/item["config"]).resolve()),
                "--output",str(output.resolve()),
                "--eval-seed","1201", "--ball-offset","0","0",
                "--velocity-delta","0","--contact-diagnostics",
                "--v149-observer-mode",mode,"--v149-sidecar",str(sidecar.resolve())]
            rows.append(dict(
                name=name,policy=item["policy"], mode=mode,seed=1201,
                original_v54_policy_config=item["config"],
                original_v54_policy_sha256=item["config_sha256"],
                expected_historical_rows=item["expected_history_rows"],
                output=str(output), sidecar=str(sidecar), log=str(log),
                argv=args, status="PLANNED_NOT_EXECUTED",
                new_real_physx_episode=False))
    assert len(rows) == cfg["live_trial_count"]
    return dict(version=cfg["version"],frozen_sha=FROZEN_SHA,
                parent_sha=cfg["parent_sha"],original_source_verified=True,
                original_runner_sha256=sha(RUNNER),original_dof=29,
                physical_dt_s=.005,runs=rows,run_count=len(rows),
                physically_executed=0,new_real_physx_success=None,
                strict_instep_physical_success=None,
                left_wrist_hip_manifold="UNKNOWN",gpu_safety_not_waived=True,
                human_pro_style_learned=False)


def safety_preflight(*, gpu_table=None, disk_bytes=None, lock_exists=None):
    cfg = frozen()
    if disk_bytes is None:
        disk_bytes = shutil.disk_usage(ROOT).free
    if disk_bytes < cfg["minimum_disk_free_gib"] * 1024**3:
        raise RuntimeError("V151_DISK_BELOW_20GIB_NO_PHYSICS")
    if lock_exists is None:
        lock_exists = (SOURCE_ROOT / "experiments/kinetickick_physics_gpu.lock").is_file()
    if not lock_exists:
        raise RuntimeError("V151_SHARED_GPU_MUTEX_MISSING")
    if gpu_table is None:
        p = subprocess.run(["nvidia-smi","--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader"],capture_output=True,text=True,timeout=15)
        if p.returncode:
            raise RuntimeError("V151_UNKNOWN_GPU_STATE")
        gpu_table=p.stdout
    if gpu_table.strip():
        raise RuntimeError("V151_FOREIGN_GPU_CONTEXT_NO_PHYSICS")
    return dict(eligible_only_at_this_instant=True,shared_mutex_must_be_locked_by_runner=True,
                free_gib=disk_bytes/(1024**3))


def available_output_paths(plan):
    all_paths=[]
    for row in plan["runs"]:
        all_paths.extend([Path(row[k]) for k in ("output","sidecar","log")])
    if len(all_paths)!=len(set(all_paths)):
        raise ValueError("V151_DUPLICATE_OUTPUT_PATHS")
    for p in all_paths:
        if p.exists() or p.is_symlink() or not p.parent.is_dir():
            raise FileExistsError("V151_OUTPUT_EXISTS_OR_UNSAFE:" + str(p))
    return True


def execute_sequential():
    # Safety gate + entire campaign output uniqueness BEFORE any run;
    # the original v149 runner additionally obtains nonblocking exclusive flock.
    plan=make_campaign()
    safety_preflight()
    available_output_paths(plan)
    for row in plan["runs"]:
        safety_preflight()  # Do not start another trial if GPU is no longer idle.
        with Path(row["log"]).open("x",encoding="utf8") as log:
            result=subprocess.run(row["argv"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("V151_PHYSICS_RUN_FAILED_PRESERVE_EVIDENCE:" + row["name"])
        for k in ("output","sidecar"):
            if not Path(row[k]).is_file():
                raise RuntimeError("V151_PHYSICS_MISSING_WRITE_ONCE_FILE:" + row["name"] + ":" + k)
    return "ALL_SIX_PHYSICS_EXECUTED_UNAUDITED"


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,help="Unique local write-once CPU plan JSON")
    ap.add_argument("--execute-physics",action="store_true",
                    help="EXPLICIT ONLY: preflight+6 original GPU trials, then independent audit required")
    a=ap.parse_args(argv)
    if a.execute_physics:
        print(execute_sequential())
        return
    plan=make_campaign()
    try:
        plan["gpu_preflight"]=safety_preflight()
    except RuntimeError as e:
        plan["gpu_preflight"]={"allowed":False,"reason":str(e)}
    if a.output:
        from v142_native_pair_ledger import safe_write_once
        plan["plan_file_sha256"]=None
        out=Path(a.output)
        h=safe_write_once(out,plan)
        print(json.dumps({"write_once_plan":str(out),"sha256":h,"runs":6,
                          "physx_executed":0,"gpu_preflight":plan["gpu_preflight"]}))
    else:
        print(json.dumps(plan,indent=2))


if __name__=="__main__":
    main()

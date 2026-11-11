#!/usr/bin/env python3
"""Safely triage ONLY completed write-once v070 GPU job rows on CPU.

Never opens a log of the currently executing child, never kills processes,
never asserts TIMEOUT == a negative G1 physics episode.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_v071_native_event_ledger import InvalidSource, sha256

def parse_completed_run(run:dict, reports:Path) -> dict:
    name=run.get("id")
    if not isinstance(name,str) or not name.startswith("trial_"):
        raise InvalidSource("INVALID_COMPLETED_V070_ID")
    log=reports/("v070_"+name+".log")
    if not log.is_file() or sha256(log)!=run.get("log_sha256"):
        raise InvalidSource("MUTATED_COMPLETED_V070_LOG_"+name)
    logtext=log.read_text(errors="replace")
    raw=reports/("v070_"+name+".json")
    expected=run.get("raw_sha256")
    if expected and (not raw.is_file() or sha256(raw)!=expected):
        raise InvalidSource("MUTATED_COMPLETED_V070_RAW_"+name)
    if not expected and raw.exists():
        raise InvalidSource("UNREGISTERED_V070_RAW_"+name)
    markers=dict(
        argv_cuda=("V058_BACKEND_ARG_FORWARDED cuda:0" in logtext),
        isaac_kit_loader=("[INFO][AppLauncher]" in logtext),
        simulation_app_loading=("Loading user config located at:" in logtext),
        scene_cuda_env=("Environment device    : cuda:0" in logtext),
        real_native_return=("V058_NATIVE_RETURN cuda:0" in logtext),
        raw_written=bool(expected),
        wait_240_timeout=(run.get("return_code")==124),
        seen_gui_context_failure=("GLFW initialization failed" in logtext),
        seen_vulkan_driver_cache_warning=("DriverShaderCacheManager::init()" in logtext),
    )
    if expected:
        stage="PHYSICS_OUTPUT_PRESENT_REQUIRES_FULL_NATIVE_AUDIT"
    elif markers["scene_cuda_env"] and markers["real_native_return"]:
        stage="PHYSICS_RETURNED_BUT_OUTPUT_ABSENT_UNSCORABLE"
    elif markers["simulation_app_loading"] and not markers["scene_cuda_env"]:
        stage="ISAAC_KIT_BOOT_OR_SCENE_INIT_NO_G1_EPISODE"
    elif markers["isaac_kit_loader"]:
        stage="APP_LAUNCHER_STARTED_BUT_NO_NATIVE_SCENE"
    elif markers["argv_cuda"]:
        stage="BEFORE_ISAAC_APP_LOAD"
    else:
        stage="FAILED_BEFORE_GPU_NATIVE_PROCESS"
    return dict(
        id=name,classification=stage,
        frozen_runner_status=run.get("status"),
        child_return_code=run.get("return_code"),
        elapsed_wall_s=run.get("wall_s"),original_log_sha256=sha256(log),
        native_output_sha256=expected,
        markers=markers,
        is_certified_scored_G1_29DoF=bool(expected) and stage.startswith("PHYSICS_OUTPUT"),
        caution="GLFW/cache warnings are NOT a demonstrated root cause; never score absent physics"
    )

def audit_inflight(progress:Path, expected_total:int=12) -> dict:
    p=json.loads(progress.read_text())
    if p.get("version")!="v0.1.70":
        raise InvalidSource("WRONG_V070_PROGRESS")
    rows=p.get("runs",[])
    if not isinstance(rows,list) or len(rows)>expected_total:
        raise InvalidSource("V070_PROGRESS_COUNT_INVALID")
    ids=[r.get("id") for r in rows]
    if len(ids)!=len(set(ids)):
        raise InvalidSource("DUPLICATE_V070_COMPLETED_ID")
    results=[parse_completed_run(r,progress.parent) for r in rows]
    failures=sum(not x["is_certified_scored_G1_29DoF"] for x in results)
    stages={}
    for r in results:
        stages[r["classification"]]=stages.get(r["classification"],0)+1
    return dict(
        version="v0.1.71",
        source_inflight_v070_sha256=sha256(progress),
        v070_completed_rows_snapshot=len(results),
        v070_registered_total=expected_total,
        v070_completed_all=(len(results)==expected_total),
        v070_completed_rows_without_scorable_original_29DoF=failures,
        v070_completed_rows_with_raw_to_qualify=len(results)-failures,
        actual_v071_gpu_physical_runs=0,
        probable_hang_root_cause_certified=False,
        diagnosis="Snapshot ONLY; v070 owner still controls GPU. Missing raw/return is startup failure, not simulated G1 physical defeat.",
        stage_distribution=stages,
        entries=results
    )

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--v070-progress",type=Path,default=Path("/home/you/projects/kinetickick-wt-v070/reports/v070_batch_progress.json"))
    ap.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"reports/v071_independent_v070_launch_snapshot.json")
    args=ap.parse_args()
    v=audit_inflight(args.v070_progress)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f:
        json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write("\n")
    print("V071_V070_SNAPSHOT",v["v070_completed_rows_snapshot"],"/",v["v070_registered_total"],
          "NO_SCORED_G1",v["v070_completed_rows_without_scorable_original_29DoF"])
    print("STAGES",json.dumps(v["stage_distribution"]))
if __name__=="__main__":
    main()

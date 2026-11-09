#!/usr/bin/env python3
"""v46 frozen, one-shot, no tuning original-G1/PhysX complete landing study.

First zero center must have exact shared-physics parity. The research design
then prespecifies 2 previously known diagnoses and 6 NEW continuous physical
ball/initial-velocity triples, each with original frozen v41 and v46. No
selection of successful episodes and no reusing the previous test outcomes.
"""
from __future__ import annotations
from pathlib import Path
import hashlib,json,subprocess,sys,time,argparse
ROOT=Path(__file__).resolve().parents[1]
STUDY="configs/landing_transfer_study_v046.json"
def digest(rel):
    h=hashlib.sha256()
    with (ROOT/rel).open("rb") as f:
        for buf in iter(lambda:f.read(1<<20),b""):h.update(buf)
    return h.hexdigest()

def make_jobs(study):
    jobs=[]
    for case in study["old_diagnostics"]:
        jobs.append(dict(id=case["id"],controller="landing_v46",
          config=study["v46_active_config"],runner=study["active_runner"],
          offset=case["ball_offset_xy_m"],velocity=case["velocity_y_delta_mps"],output=case["output"],
          provenance="already_analyzed_old_development"))
    for case in study["six_new_ball_conditions"]:
        for control in ("original_v41","landing_v46"):
            jobs.append(dict(id=case["id"],controller=control,
              config=study["v41_config"] if control=="original_v41" else study["v46_active_config"],
              runner=study["baseline_runner"] if control=="original_v41" else study["active_runner"],
              offset=case["ball_offset_xy_m"],velocity=case["velocity_y_delta_mps"],
              output=case["outputs"][control],provenance="new_exact_precommitted_ball_parameters"))
    if len(jobs)!=14 or len({x["output"] for x in jobs})!=14:
        raise ValueError("Expected 2 old and 12 newly paired original PhysX tasks")
    return jobs

def execute():
    study=json.loads((ROOT/STUDY).read_text())
    if study["version"]!="0.1.46" or study["total_prespecified_physx_runs"]!=15 or not study["no_gain_fitting_after_physical_results"]:
        raise ValueError("Original frozen study changed")
    for p,h in study["source_sha256"].items():
        if digest(p)!=h:raise ValueError("Frozen original G1 source drift: "+p)
    zero=json.loads((ROOT/study["zero_audit"]).read_text())
    if not zero["zero_parity"]["all_shared_original_physics_identical"] or not zero["zero_parity"]["original_action_stream_identical"]:
        raise ValueError("Original exact zero test must pass before active physical experiment")
    jobs=make_jobs(study)
    for row in jobs:
        if (ROOT/row["output"]).exists():raise FileExistsError("Never rerun original condition: "+row["output"])
    progress=ROOT/"reports/v046_original_physx_progress.json"
    if progress.exists():raise FileExistsError(progress)
    if "/env_isaaclab/" not in sys.executable:raise ValueError("Use original IsaacLab python")
    done=[]
    for i,row in enumerate(jobs,1):
        for p,h in study["source_sha256"].items():
            if digest(p)!=h:raise ValueError("Source changed during registered trial "+p)
        dest=ROOT/row["output"]
        start=time.monotonic()
        args=[sys.executable,str(ROOT/row["runner"]),"--config",str(ROOT/row["config"]),
           "--ball-offset",str(row["offset"][0]),str(row["offset"][1]),
           "--velocity-delta",str(row["velocity"]),"--output",str(dest)]
        print("V046_PHYSX_START",i,len(jobs),row["id"],row["controller"],row["offset"],row["velocity"],flush=True)
        with dest.with_suffix(".log").open("w") as log:
            result=subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
        entry=dict(index=i,id=row["id"],controller=row["controller"],original_physx_report=row["output"],
             process_exit=int(result.returncode),seconds=round(time.monotonic()-start,3),
             bytes=dest.stat().st_size if dest.exists() else 0,
             sha256=digest(row["output"]) if dest.exists() else None)
        done.append(entry)
        progress.write_text(json.dumps(dict(version="0.1.46",
           frozen_study_sha256=digest(STUDY),completed=len(done),planned=len(jobs),
           original_runs=done),indent=2)+"\n")
        print("V046_PHYSX_COMPLETE",json.dumps(entry),flush=True)
        if result.returncode!=0 or not dest.exists():
            raise RuntimeError("Physical simulator execution failure. Do not relax original joint limits or rerun successful cases")
    print("V046_ALL_PREDECLARED_PHYSX_RUNS_DONE",len(done),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--execute",action="store_true")
    a=p.parse_args()
    cfg=json.loads((ROOT/STUDY).read_text())
    make_jobs(cfg)
    if a.execute:execute()
    else: print("V046_REGISTERED_TRIAL_LAYOUT_VERIFIED",len(cfg["six_new_ball_conditions"]))

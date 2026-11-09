#!/usr/bin/env python3
"""Execute one predeclared novel-ball-offset/velocity PhysX study, NO fitting.

Each new continuous ball-parameter combination is run once using:
(1) frozen v32 controller and (2) frozen v41 style controller.
We do NOT adjust configs or parameters after observing any outcome.
The source study is created, verified and signed before the first simulation.
"""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/"configs/novel_ball_generalization_v042.json"

def sha(path):
 h=hashlib.sha256()
 with (ROOT/path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def verify(study):
    if study["version"]!="0.1.42" or study["number_of_new_parameter_combinations"]!=6:
        raise ValueError("Unexpected frozen study design")
    for rel,digest in study["source_sha256"].items():
        if sha(rel)!=digest:
            raise ValueError("Frozen source drift "+rel)
    if not study["no_adaptive_parameter_search"] or not study["no_retraining"]:
        raise ValueError("Only run frozen previously trained controller")
    if len(study["trials"])!=6:
        raise ValueError("Exactly six randomized predeclared combinations")
    if len({tuple(case["ball_offset_xy_m"]+[case["velocity_y_delta_mps"]]) for case in study["trials"]})!=6:
        raise ValueError("Repeated condition")
    for k in study["trials"]:
        x,y=k["ball_offset_xy_m"];v=k["velocity_y_delta_mps"]
        if not all(isinstance(z,(float,int)) for z in (x,y,v)) or max(abs(x),abs(y))>.15 or abs(v)>.30:
            raise ValueError("Original runner physical bounds exceeded")
    return True

def run():
    study=json.loads(STUDY.read_text())
    verify(study)
    python=sys.executable
    if not python.endswith("/python") or "env_isaaclab" not in python:
        raise ValueError("Must execute through isolated original IsaacLab python")
    if not shutil_which("nvidia-smi"):
        raise ValueError("GPU not available")
    progress=ROOT/"reports/v042_novel_run_progress.json"
    if progress.exists():raise FileExistsError(progress)
    all_trials=[
        (case,slot) for case in study["trials"] for slot in ("original_v32","style_v41")
    ]
    for case,slot in all_trials:
        output=ROOT/case["outputs"][slot]
        if output.exists():raise FileExistsError(f"Refuse to overwrite prior physical evidence {output}")
    complete=[]
    for idx,(case,slot) in enumerate(all_trials):
        verify(study)
        report=ROOT/case["outputs"][slot]
        log=report.with_suffix(".log")
        report.parent.mkdir(parents=True,exist_ok=True)
        cfg=study["controllers"][slot]["config"]
        x,y=case["ball_offset_xy_m"]
        v=case["velocity_y_delta_mps"]
        cmd=[
            python,str(ROOT/"tools/run_placement_strike.py"),
            "--config",str(ROOT/cfg),
            "--output",str(report),
            "--ball-offset",str(x),str(y),
            "--velocity-delta",str(v)
        ]
        print("V042_PHYSX_RUN_BEGIN",idx+1,"of",len(all_trials),case["id"],slot,
              "ball_offset",[x,y],"velocity_y_delta",v,flush=True)
        started=time.monotonic()
        with log.open("w") as f:
            result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=False)
        secs=time.monotonic()-started
        item=dict(id=case["id"],controller=slot,output=case["outputs"][slot],
                  cli_exit=int(result.returncode),
                  duration_seconds=round(secs,3),physical_bytes=report.stat().st_size if report.exists() else 0,
                  physical_sha256=sha(case["outputs"][slot]) if report.exists() else None)
        complete.append(item)
        progress.write_text(json.dumps(dict(version="0.1.42",
            started_from_frozen_study_sha256=hashlib.sha256(STUDY.read_bytes()).hexdigest(),
            planned_total=len(all_trials),completed=len(complete),trials=complete),indent=2)+"\n")
        print("V042_PHYSX_RUN_END",json.dumps(item),flush=True)
        if result.returncode!=0 or not report.exists():
            raise RuntimeError("Original physical runner itself failed. Preserve log and stop without selecting cases.")
    print("V042_ALL_PREDECLARED_RUNS_COMPLETE",len(complete),flush=True)

def shutil_which(binary):
    import shutil
    return shutil.which(binary)
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--run",action="store_true")
    args=ap.parse_args()
    if args.run:run()
    else:
        verify(json.loads(STUDY.read_text()))
        print("V042_FROZEN_STUDY_VALIDATED_NO_PHYSICS")

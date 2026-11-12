#!/usr/bin/env python3
"""Frozen CPU/CUDA PhysX-native positive/negative isolation; write-once audited data."""
from __future__ import annotations
import fcntl, hashlib, json, os, shutil, signal, subprocess, time
from pathlib import Path
from v083_run_physics_batch import PYTHON

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v120_contact_frozen.json"
MASTER = ROOT / "reports/v120_contact_master.audit.json"
SOURCE = ROOT / "tools/v120_contact_phase_probe.py"
FROZEN = (
    "configs/v120_contact_frozen.json",
    "docs/V120_CONTACT_PHASE_FROZEN_PROTOCOL.md",
    "tools/v120_contact_batch.py",
    "tests/test_v120_contact_phase.py",
    "tools/v120_contact_phase_probe.py",
    "tools/v118_native_full_batch.py",
)

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def git(*arg: str) -> str:
    return subprocess.check_output(("git",*arg),cwd=ROOT,text=True).strip()

def parity(a: dict,b: dict) -> dict:
    ta=a.get("runtime_z_vz")
    tb=b.get("runtime_z_vz")
    same=bool(ta and tb and ta==tb and
              a.get("box_position_w")==b.get("box_position_w"))
    return {"recorded_pose_velocity_parity":same,
            "frames":len(ta or []), "same_backend":a.get("device")==b.get("device")}

def diagnose(d: dict) -> dict:
    return {
        "runtime_steps":d.get("actual_runtime_frames",0),
        "native_pair_witness":bool(d.get("native_pair_witness")),
        "raw_native_pair_events":d.get("native_counts",{}).get("native_pair_events",0),
        "callback_headers":d.get("native_counts",{}).get("callback_headers",0),
        "full_callback_headers":d.get("native_full_counts",{}).get("callback_headers",0),
        "poll_headers":d.get("native_counts",{}).get("poll_headers",0),
        "full_poll_headers":d.get("native_full_counts",{}).get("poll_headers",0),
        "tensor_contact_counts":d.get("tensor_contact_counts",0),
        "tensor_peak_force_N":d.get("tensor_peak_force_N",0.0),
        "observed_positive_dynamics":d.get("observed_collision_dynamics",False),
        "observed_separation_dynamics":d.get("observed_noncollision_dynamics",False),
        "stage":d.get("stage"),
        "errors":d.get("errors",[]),
        "full_errors":d.get("native_full_counts",{}).get("errors",[])
    }

def preflight() -> dict:
    cfg=json.loads(CONFIG.read_text())
    assert cfg["version"]=="v0.1.120"
    assert cfg["steps"]==180 and cfg["dt_seconds"]==.005
    assert cfg["parent_sha"] in git("log","--format=%H").splitlines()
    assert cfg["original_G1_motor_changed"] is False
    if git("branch","--show-current")!="version/v0.1.120":raise RuntimeError("WRONG_BRANCH")
    if git("status","--porcelain","--untracked-files=no"):raise RuntimeError("FROZEN_SOURCE_DIRTY")
    if MASTER.exists() or MASTER.is_symlink():raise FileExistsError("MASTER_ALREADY_EXISTS")
    for file in FROZEN:
        if (ROOT/file).read_bytes()!=subprocess.check_output(
            ("git","show","HEAD:"+file),cwd=ROOT):
            raise RuntimeError("SOURCE_DRIFT:"+file)
    names=[r["name"] for r in cfg["matrix"]]
    assert names==["cuda_positive_off","cuda_positive_on","cuda_spawner_on","cuda_rootshape_on",
                  "cuda_postreset_on","cuda_render_on","cuda_separated_on"]
    assert cfg["timeout_seconds"]==150 and cfg["min_free_gib"]==20
    if shutil.disk_usage(ROOT).free<20*1024**3:raise RuntimeError("LOW_DISK")
    if subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name",
                 "--format=csv,noheader"],text=True).strip():
        raise RuntimeError("FOREIGN_GPU_JOB")
    for x in cfg["matrix"]:
        for ext in (".json",".log",".audit.json"):
            p=ROOT/"reports"/("v120_native_"+x["name"]+ext)
            if p.exists() or p.is_symlink():raise FileExistsError(str(p))
    return cfg

def run_one(cfg:dict,case:dict) -> dict:
    name=case["name"]; base=ROOT/"reports"/("v120_native_"+name)
    raw=Path(str(base)+".json"); log=Path(str(base)+".log")
    audit=Path(str(base)+".audit.json")
    cmd=[PYTHON,str(SOURCE),"--device",case["device"],
         "--report-mode",case["report"],"--capture",case["capture"],
         "--layout",case["layout"],"--steps","180","--output",str(raw),
         "--authoring",case["authoring"],"--step-mode",case["step_mode"],
         "--subscribe-phase",case["subscribe_phase"]]
    begin=time.monotonic(); timed=False
    with log.open("x") as f:
        f.write("FROZEN_SOURCE_SHA="+git("rev-parse","HEAD")+"\n")
        f.write("FROZEN_INPUT_SHA256="+sha(CONFIG)+"\n")
        f.write("FROZEN_PROBE_SHA256="+sha(SOURCE)+"\n");f.flush()
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",PYTHONUNBUFFERED="1",
              OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1"),
            start_new_session=True)
        try:rc=child.wait(timeout=cfg["timeout_seconds"])
        except subprocess.TimeoutExpired:
            timed=True; os.killpg(child.pid,signal.SIGTERM)
            try:rc=child.wait(timeout=6)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=6)
            f.write("CHILD_ONLY_150S_WATCHDOG\n")
    data={}
    if raw.exists():
        try:data=json.loads(raw.read_text())
        except (ValueError,OSError):pass
    diag=diagnose(data)
    certified=bool(rc==0 and not timed and diag["runtime_steps"]==180
                   and "exception" not in data and data.get("device")==case["device"]
                   and data.get("layout")==case["layout"])
    row=dict(**case,source_sha=git("rev-parse","HEAD"),
             certified_physics=certified,diag=diag,exit_code=rc,timeout=timed,
             raw_sha=sha(raw) if raw.exists() else None,
             log_sha=sha(log),elapsed_seconds=round(time.monotonic()-begin,2))
    with audit.open("x") as f:json.dump(row,f,indent=2);f.write("\n")
    print("V120_NATIVE_RUN",json.dumps({k:v for k,v in row.items() if k!="diag"}),
          "diag",json.dumps(diag)[:1300],flush=True)
    return row

def main():
    cfg=preflight()
    lock=Path(cfg["global_lock"]);assert lock.exists()
    done=[]; data={}
    with lock.open("r+") as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("GPU_LOCK_HELD")
        for case in cfg["matrix"]:
            if shutil.disk_usage(ROOT).free<20*1024**3:break
            if subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name",
                   "--format=csv,noheader"],text=True).strip():break
            row=run_one(cfg,case);done.append(row)
            if not row["certified_physics"]:break
            data[case["name"]]=json.loads(
                (ROOT/"reports"/("v120_native_"+case["name"]+".json")).read_text())
    pair=parity(data["cuda_positive_off"],data["cuda_positive_on"]) if all(
        k in data for k in ("cuda_positive_off","cuda_positive_on")) else None
    variants={name:dict(physics_positive=bool(data.get(name,{}).get("observed_collision_dynamics")),
            native=bool(data.get(name,{}).get("native_pair_witness")),
            tensor=bool(data.get(name,{}).get("tensor_pair_witness")),
            callback_headers=data.get(name,{}).get("native_counts",{}).get("callback_headers"),
            full_headers=data.get(name,{}).get("native_full_counts",{}).get("callback_headers"))
        for name in (r["name"] for r in cfg["matrix"])}
    summary=dict(version="v0.1.120",source_sha=git("rev-parse","HEAD"),
        protocol_sha256=sha(CONFIG),
        completed_isolated_physx_runs=sum(r["certified_physics"] for r in done),
        real_200hz_physics_steps=sum(r["diag"]["runtime_steps"] for r in done if r["certified_physics"]),
        positive_off_on_parity=pair,variants=variants,
        original_G1_shots=0,original_G1_motor_changed=False,
        valid_native_actor_collider_witness=any(v["native"] for v in variants.values()),
        results=done,not_run=[x["name"] for x in cfg["matrix"][len(done):]])
    with MASTER.open("x") as f:json.dump(summary,f,indent=2);f.write("\n")
    print("V120_NATIVE_MASTER",json.dumps({k:v for k,v in summary.items()
                  if k!="results"}),flush=True)
if __name__=="__main__":main()

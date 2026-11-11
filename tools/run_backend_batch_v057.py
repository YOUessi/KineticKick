#!/usr/bin/env python3
"""v057 frozen CPU/GPU original-G1 contact backend qualification, write-once output.

Separate physical backends are *not* interchangeable. Reports are not policy
training data. The contact callback may return 0 headers: do not infer no pair.
"""
from __future__ import annotations
import fcntl, hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/physx_g1_backend_pair_v057.json"
LOCK=ROOT/"reports/v057_protocol_lock.json"
PROGRESS=ROOT/"reports/v057_batch_progress.json"
GPU_LOCK=ROOT/"experiments/kinetickick_physics_gpu.lock"
PYTHON=Path("/home/you/conda_env/env_isaaclab/bin/python")
BASE="4714393ed447ef569bc1ebe757f6665929e24a32"

def digest(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):h.update(chunk)
    return h.hexdigest()

def ensure_frozen(config:dict)->str:
    if config["version"]!="v0.1.57" or config["parent_commit"]!=BASE:
        raise ValueError("Wrong preregistered version/parent")
    if len(config["cases"])!=6 or len(config["toy_cases"])!=4:
        raise ValueError("Missing one or more preregistered physical controls")
    if len({c["output"] for c in config["cases"]+config["toy_cases"]})!=10:
        raise ValueError("Output collision")
    for name,expect in config["source_hashes"].items():
        if digest(ROOT/name)!=expect:raise RuntimeError("Original source modified: "+name)
    return digest(CFG)

def validate_machine():
    available=os.statvfs(ROOT)
    free=available.f_bavail*available.f_frsize/(1024**3)
    if free<20:raise RuntimeError(f"Disk safety stop: {free:.1f}GiB")
    cmd=["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"]
    r=subprocess.run(cmd,encoding="utf-8",capture_output=True,timeout=15)
    if r.returncode or r.stdout.strip():raise RuntimeError("Foreign GPU compute process or nvidia-smi failed")

def eval_cmd(case,kind):
    if kind=="g1":
        return [str(PYTHON),str(ROOT/"tools/run_placement_strike_v057.py"),
            "--config",str(ROOT/config["policies"][case["policy"]]),
            "--output",str(ROOT/case["output"]),
            "--ball-offset",*map(str,case["ball_offset_xy_m"]),
            "--velocity-delta",str(case["velocity_y_delta_mps"]),
            "--eval-seed",str(case["seed"]),
            "--sim-device",case["backend"],
            "--contact-diagnostics","--pair-dynamics-v055","--native-readonly-pair-v056"]
    return [str(PYTHON),str(ROOT/"tools/probe_physx_pair_isolated_v056.py"),
            "--control",case["control"],"--sim-device",case["backend"],
            "--output",str(ROOT/case["output"])]

def accept_g1(data,case):
    if data.get("seed")!=case["seed"] or data.get("root_fixed") is not False:
        raise ValueError("G1 root/seed changed")
    if data.get("policy_sha256")!=digest(ROOT/config["policies"][case["policy"]]):
        raise ValueError("G1 policy hash mismatch")
    if data.get("v057_requested_physics_backend")!=case["backend"]:
        raise ValueError("Backend mismatch")
    if data.get("joint_names") is None or len(data["joint_names"])!=29:
        raise ValueError("Original 29DoF order missing")
    for name in ("traces","records","termination_reasons","stride_shot_success",
                 "physx_native_report_v056","physx_pair_dynamics_v055"):
        if name not in data:raise ValueError("Missing original audit: "+name)
    frames=data["traces"][0]
    if len(frames)<35 or any(len(x["joint_vel"])!=29 for x in frames):
        raise ValueError("Missing full 200Hz 29DoF physics")
    obs=data["physx_pair_dynamics_v055"]
    if len(obs["frames"])<30:raise ValueError("Missing wholebody 200Hz observer")
    native=data["physx_native_report_v056"]
    return dict(frames=len(frames),observer_frames=len(obs["frames"]),
                strict_success=bool(data["stride_shot_success"].get("success",False))
                if isinstance(data["stride_shot_success"],dict) else bool(data["stride_shot_success"]),
                native_headers=native["native_headers"],
                reporter_errors=native["errors"],
                wrist_hip_direct_events=native["wrist_hip_direct_path_events"],
                stop=data["termination_reasons"])

def accept_toy(data,case):
    if data["case"]!=case["control"] or data["sim_dt_s"]!=.005:
        raise ValueError("Toy mismatch")
    return dict(peak_N=data["filtered_pair_peak_N"],contact_events=data["total_probe_events"],
                native_pair_reported=data["native_pair_proven"],errors=data["errors"])

if __name__=="__main__":
    if Path(sys.executable).resolve()!=PYTHON.resolve():
        raise RuntimeError("Run with pinned IsaacLab Python")
    config=json.loads(CFG.read_text())
    protocol_hash=ensure_frozen(config)
    if PROGRESS.exists():
        state=json.loads(PROGRESS.read_text())
        if state["protocol_sha256"]!=protocol_hash:raise RuntimeError("Progress protocol drift")
    else:state=dict(protocol_sha256=protocol_hash,runs=[])
    GPU_LOCK.parent.mkdir(parents=True,exist_ok=True)
    with GPU_LOCK.open("a") as mutex:
        fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
        validate_machine()
        if LOCK.exists():
            saved=json.loads(LOCK.read_text())
            if saved["protocol_sha256"]!=protocol_hash:raise RuntimeError("Locked protocol changed")
        else:
            with LOCK.open("x") as f:json.dump({"parent_sha":BASE,"protocol_sha256":protocol_hash,
                "number_G1":6,"number_toy":4,"created_before_physics":True},f,indent=2)
        if not PROGRESS.exists():
            with PROGRESS.open("x") as f:json.dump(state,f,indent=2)
        runlist=[("g1",x) for x in config["cases"]]+[("toy",x) for x in config["toy_cases"]]
        for idx,(kind,case) in enumerate(runlist):
            ensure_frozen(config);validate_machine()
            if idx<len(state["runs"]):
                earlier=state["runs"][idx]
                if earlier["id"]!=case["id"]:raise RuntimeError("Progress ordering mismatch")
                if earlier.get("report_sha256") and digest(ROOT/case["output"])!=earlier["report_sha256"]:
                    raise RuntimeError("Historical report modified")
                print("V057_PRESERVED",earlier["id"],earlier["status"],flush=True)
                continue
            dest=ROOT/case["output"];log=dest.with_suffix(".log")
            if dest.exists() or log.exists():raise FileExistsError("Write-once output collision: "+str(dest))
            env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",PYTHONPATH=":".join(["tools",
                    str(ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking")]))
            argv=eval_cmd(case,kind)
            start=time.monotonic()
            print("V057_START",idx+1,len(runlist),kind,case["id"],flush=True)
            with log.open("x") as stream:
                try:
                    result=subprocess.run(argv,cwd=ROOT,env=env,stdout=stream,
                        stderr=subprocess.STDOUT,timeout=540 if kind=="g1" else 180)
                    code=result.returncode
                except subprocess.TimeoutExpired:
                    code=124
                    stream.write("OWN_CHILD_TIMED_OUT; raw evidence if present remains untouched.\n")
            row=dict(id=case["id"],kind=kind,backend=case["backend"],
                wall_seconds=round(time.monotonic()-start,1),exit_code=code,
                log_sha256=digest(log))
            if dest.exists():
                row["report_sha256"]=digest(dest)
                try:
                    data=json.loads(dest.read_text())
                    row.update(accept_g1(data,case) if kind=="g1" else accept_toy(data,case))
                    row["status"]="PHYSICAL_VALIDATED" if code==0 else "WRITTEN_BUT_CHILD_EXIT_ABNORMAL"
                except (KeyError,ValueError,TypeError,IndexError) as e:
                    row.update(status="UNSCORABLE_RAW_PRESERVED",accept_error=str(e))
            else:row["status"]="NO_COMPLETE_PHYSICS"
            state["runs"].append(row)
            temp=PROGRESS.with_suffix(".tmp")
            temp.write_text(json.dumps(state,indent=2,ensure_ascii=False)+"\n")
            os.replace(temp,PROGRESS)
            print("V057_END",case["id"],row["status"],row.get("strict_success"),
                row.get("native_headers"),flush=True)
        print("V057_BATCH_COMPLETE",len(state["runs"]),
            sum(x["status"]=="PHYSICAL_VALIDATED" for x in state["runs"]),flush=True)

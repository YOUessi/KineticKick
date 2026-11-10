#!/usr/bin/env python3
"""SHA-frozen 15-run original PhysX G1 controls and holdout negatives.

The optional new manifold observer was rejected during three *non-scored*
preflight runs. Use the known successful named-contact-only original v54
recorder path. This does not collect pair or manifold truth.
"""
from __future__ import annotations
import hashlib,json,os,subprocess,sys,time,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/physx_pair_dynamics_v055.json"
PROGRESS=ROOT/"reports/v055_pair_batch_progress.json"
LOCK=ROOT/"experiments/kinetickick_physics_gpu.lock"
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for buf in iter(lambda:f.read(1<<20),b""):h.update(buf)
    return h.hexdigest()
def freeze(c):
    if c["version"]!="0.1.55" or c["parent_sha"]!="3cac1736f72218cf7bb864059f3e0a21c08b7da6":raise ValueError("Wrong parent")
    if len(c["cases"])!=15 or len({v["id"] for v in c["cases"]})!=15:raise ValueError("Must run full 15 cases")
    if not c["no_training"] or not c["read_only_observer"]:raise ValueError("Not read-only")
    for p,h in c["immutable_sha256"].items():
        if sha(ROOT/p)!=h:raise ValueError("Source drift: "+p)
    return True
def run():
    if "env_isaaclab" not in sys.executable:raise RuntimeError("Run on original IsaacLab Python")
    c=json.loads(CONFIG.read_text());freeze(c)
    history=[]
    if PROGRESS.exists():
        progress=json.loads(PROGRESS.read_text())
        if progress["protocol_sha256"]!=sha(CONFIG):raise RuntimeError("Protocol modified after freezing")
        history=progress["runs"]
    LOCK.parent.mkdir(exist_ok=True,parents=True)
    with LOCK.open("w") as guard:
        try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("A KineticKick physics GPU batch already has lock")
        for index,case in enumerate(c["cases"]):
            out=ROOT/case["out"];log=out.with_suffix(".log")
            if index<len(history):
                row=history[index]
                if row["id"]!=case["id"] or sha(out)!=row["sha256"]:raise ValueError("Invalid prior evidence (no overwrite)")
                continue
            if out.exists() or log.exists():raise FileExistsError("Existing physical result must not be overwritten: "+case["out"])
            freeze(c)
            # Do not start if another CUDA compute job is using the shared 4090.
            gpu=subprocess.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            if gpu.returncode or gpu.stdout.strip():raise RuntimeError("GPU external compute job present; refuse to launch")
            usage=os.statvfs(str(ROOT));free=usage.f_bavail*usage.f_frsize
            if free < 20*1024**3:raise RuntimeError("Disk under 20GB; protect original evidence")
            s=c["policies"][case["policy"]]
            env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
                PYTHONPATH=":".join(["tools",str(ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking")]))
            cmd=[sys.executable,str(ROOT/"tools/run_placement_strike_v055.py"),
                "--config",str(ROOT/s),"--output",str(out),
                "--ball-offset",*[str(v) for v in case["ball_offset_xy_m"]],
                "--velocity-delta",str(case["velocity_y_delta_mps"]),
                 "--eval-seed",str(case["seed"]),"--contact-diagnostics"]
            print("V055_REAL_PHYSX_BEGIN",index+1,len(c["cases"]),case["id"],flush=True)
            st=time.monotonic()
            with log.open("w") as f:
                p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            if p.returncode or not out.exists():
                raise RuntimeError(f"V055 physical case failed {case['id']}; inspect ignored {log} and repair without overwriting")
            d=json.loads(out.read_text())
            if len(d["traces"][0])<500 or len(d["traces"][0][0]["joint_pos"])!=29 or "nonfoot_contact_diagnostics" not in d:
                raise RuntimeError("No valid original 200Hz 29DoF physics run recorded")
            record=dict(id=case["id"],out=case["out"],sha256=sha(out),
                size_bytes=out.stat().st_size,wall_s=round(time.monotonic()-st,2),
                physics_samples=len(d["traces"][0]),termination=d["termination_reasons"],
                pair_manifold_events=None,collider_path_events=None,
                new_pair_instrumentation_not_activated=True)
            history.append(record)
            PROGRESS.write_text(json.dumps(dict(protocol_sha256=sha(CONFIG),runs=history),indent=2)+"\n")
            print("V055_REAL_PHYSX_END",json.dumps(record),flush=True)
    print("V055_REAL_PHYSX_BATCH_COMPLETE",len(history),flush=True)
if __name__=="__main__":run()

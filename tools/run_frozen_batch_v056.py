#!/usr/bin/env python3
"""v0.1.56 frozen original G1/PhysX matrix, no source or physical evidence overwrite.

Twelve true seeded initial-condition rollouts plus three unchanged-center
native-contact-callback trials. Sequential GPU exclusively, 20 GiB floor.
"""
import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/physx_contact_safety_v056.json"
PROTO=ROOT/"reports/v056_protocol_lock.json"
PROGRESS=ROOT/"reports/v056_batch_progress.json"
GPU_LOCK=ROOT/"experiments/kinetickick_physics_gpu.lock"

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

def verify(c):
    if c["version"]!="v0.1.56" or c["parent_commit"]!="052925e14031917f5bfad33705116d2429fecfab":
        raise ValueError("Incorrect frozen parent")
    if len(c["cases"])!=15 or len({case["id"] for case in c["cases"]})!=15:
        raise ValueError("Incomplete frozen cases")
    for rel,expected in c["immutable_sha256"].items():
        if sha(ROOT/rel)!=expected:raise ValueError("Source SHA drift: "+rel)
    return sha(CFG)

def reserve(protocol_sha):
    if not PROTO.exists():
        with PROTO.open("x") as fd:
            json.dump(dict(version="v0.1.56",config_sha256=protocol_sha,
                parent_sha="052925e14031917f5bfad33705116d2429fecfab",
                sealed_utc_by_process=True),fd,indent=2)
    if json.loads(PROTO.read_text())["config_sha256"]!=protocol_sha:
        raise ValueError("Protocol changed since registration; refuse to execute")

def gpu_ok():
    free=os.statvfs(ROOT)
    if free.f_bavail*free.f_frsize<20*(1024**3):
        raise RuntimeError("Disk available below 20GiB; physical run is blocked")
    g=subprocess.run(["nvidia-smi","--query-compute-apps=pid",
          "--format=csv,noheader"],capture_output=True,text=True)
    if g.returncode or g.stdout.strip():
        raise RuntimeError("Foreign GPU compute process; do not interfere")

def accept(report,case):
    if report.get("seed")!=case["seed"] or report.get("root_fixed") is not False:
        raise ValueError("Not original free-root seeded G1")
    if report.get("perturb_reset") is not case["perturb_reset"]:
        raise ValueError("Seeded reset perturbation absent")
    if report.get("policy_sha256")!=sha(ROOT/config["policies"][case["policy"]]):
        raise ValueError("Policy SHA differs from frozen")
    if report["joint_strike"]["ball_offset_xy"]!=case["ball_offset_xy_m"]:
        raise ValueError("Ball XY differs")
    if abs(report["joint_strike"]["velocity_y_delta"]-case["velocity_y_delta_mps"])>1e-9:
        raise ValueError("Ball velocity delta differs")
    rows=report["traces"][0]
    if len(rows)<35 or any(len(row["joint_vel"])!=29 for row in rows):
        raise ValueError("Missing 200Hz real 29DoF evidence")
    if "nonfoot_contact_diagnostics" not in report:
        raise ValueError("No named sensor evidence")
    if case["native_report"] and "physx_native_report_v056" not in report:
        raise ValueError("No reporter audit (do not claim pair proof)")
    return dict(physics_frames=len(rows),
                termination=report["termination_reasons"],
                actual_perturb_reset=report["perturb_reset"],
                native_reporter_on=case["native_report"])

if __name__=="__main__":
    if "env_isaaclab" not in sys.executable:
        raise RuntimeError("Only the original G1 IsaacLab Python is admitted")
    config=json.loads(CFG.read_text())
    protocol_sha=verify(config)
    GPU_LOCK.parent.mkdir(exist_ok=True,parents=True)
    with GPU_LOCK.open("a") as guard:
        try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise RuntimeError("Physics GPU lock held") from e
        reserve(protocol_sha)
        if PROGRESS.exists():
            progress=json.loads(PROGRESS.read_text())
            if progress["protocol_sha256"]!=protocol_sha:
                raise ValueError("Progress protocol drift")
        else:
            progress=dict(protocol_sha256=protocol_sha,runs=[])
            with PROGRESS.open("x") as f:json.dump(progress,f,indent=2)
        history=progress["runs"]
        for idx,case in enumerate(config["cases"]):
            verify(config);reserve(protocol_sha)
            if idx<len(history):
                prev=history[idx]
                if prev["id"]!=case["id"]:
                    raise ValueError("Progress order drift")
                if prev.get("out_sha256") and sha(ROOT/case["output"])!=prev["out_sha256"]:
                    raise ValueError("Existing physical evidence changed")
                print("V056_PRESERVED",case["id"],prev["status"],flush=True)
                continue
            out=ROOT/case["output"];log=out.with_suffix(".log")
            if out.exists() or log.exists():
                raise FileExistsError("Do not overwrite preexisting data: "+str(out))
            gpu_ok()
            env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",PYTHONPATH=":".join(["tools",
                 str(ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking")]))
            cmd=[sys.executable,str(ROOT/"tools/run_placement_strike_v056.py"),
                 "--config",str(ROOT/config["policies"][case["policy"]]),
                 "--output",str(out),"--ball-offset",
                 *map(str,case["ball_offset_xy_m"]),
                 "--velocity-delta",str(case["velocity_y_delta_mps"]),
                 "--eval-seed",str(case["seed"]),"--contact-diagnostics"]
            if case["perturb_reset"]:cmd+=["--perturb-reset-v056"]
            if case["native_report"]:cmd+=["--native-readonly-pair-v056"]
            print("V056_REAL_PHYSX_BEGIN",idx+1,len(config["cases"]),case["id"],flush=True)
            start=time.monotonic()
            with log.open("x") as stream:
                try:
                    process=subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,
                              stderr=subprocess.STDOUT,timeout=300)
                    exit_code=process.returncode
                except subprocess.TimeoutExpired:
                    exit_code=124
                    stream.write("V056_EVAL_TIMEOUT 300 seconds, terminated, never overwrite\n")
            row=dict(id=case["id"],policy=case["policy"],cohort=case["cohort"],
                     exit_code=exit_code,wall_s=round(time.monotonic()-start,1),
                     log_sha256=sha(log))
            if exit_code!=0 or not out.exists():
                row.update(status="FAILED_UNSCORED_NO_COMPLETE_PHYSICS",out_sha256=None)
                print("V056_REAL_PHYSX_UNSCORED",case["id"],exit_code,flush=True)
            else:
                try:
                    report=json.loads(out.read_text())
                    row.update(status="PHYSICAL_REPORT_VALIDATED",
                        out_sha256=sha(out),**accept(report,case))
                    print("V056_REAL_PHYSX_VALIDATED",case["id"],
                          row["physics_frames"],row["termination"],flush=True)
                except Exception as error:
                    row.update(status="UNSCORABLE_OUTPUT_PRESERVED",
                        out_sha256=sha(out),admission_error=str(error))
                    print("V056_UNSCORABLE",case["id"],str(error),flush=True)
            history.append(row)
            with PROGRESS.open("w") as fd:
                json.dump(progress,fd,indent=2)
                fd.write("\n")
                fd.flush();os.fsync(fd.fileno())
        print("V056_FROZEN_REAL_PHYSX_BATCH_COMPLETE",
              len(history),sum(r["status"]=="PHYSICAL_REPORT_VALIDATED" for r in history),flush=True)

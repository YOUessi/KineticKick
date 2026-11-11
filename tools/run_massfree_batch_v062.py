#!/usr/bin/env python3
"""v062 preregistered independent original G1 PhysX/CPU staged tensor probes.

All physical episodes share the exact frozen v060 policy, seed, ball position
and unchanged v058 physics/termination logic. Cases are sequential and
write-once; failures must not be relabeled as losses/successes.
"""
from __future__ import annotations
import fcntl, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/g1_massfree_probe_v062.json"
LOCK=ROOT/"reports/v062_protocol_lock.json"
PROGRESS=ROOT/"reports/v062_batch_progress.json"
MUTEX=ROOT/"experiments/kinetickick_physics_gpu.lock"
PYTHON=Path("/home/you/conda_env/env_isaaclab/bin/python")
DEVICE=re.compile(r"Environment device\s*:\s*(cpu|cuda:\d+)")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def verify_protocol(c):
    if c["version"]!="v0.1.62" or c["parent_sha"]!="1b9a69063b68a04d41ea64223c30d198f68b08f6":
        raise RuntimeError("Unexpected parent branch or version")
    assert len(c["cases"])==8
    ids=[x["id"] for x in c["cases"]]
    if len(set(ids))!=8 or len({x["output"] for x in c["cases"]})!=8:
        raise RuntimeError("Duplicate output paths")
    for x in c["cases"]:
        if not x["output"].startswith("reports/v062_") or x["probe"] not in (
            "safe_full","mass_no_mul","mass_clone","mass_after"):
            raise RuntimeError("Unregistered probe")
        if x["backend"] not in ("cpu","cuda:0") or x["seed"]!=1201 or x["policy"] not in c["policies"]:
            raise RuntimeError("Unregistered backend, seed or policy")
    for p,digest in {**c["source_hashes"],**c["policy_hashes"]}.items():
        if sha(ROOT/p)!=digest:raise RuntimeError("Frozen source drift "+p)
    return sha(CFG)

def check_environment():
    v=os.statvfs(ROOT)
    free=v.f_bavail*v.f_frsize/1024**3
    if free<20:raise RuntimeError(f"Disk below 20GiB: {free:.2f}")
    gpu=subprocess.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],
                       capture_output=True,text=True,timeout=12)
    if gpu.returncode or gpu.stdout.strip():raise RuntimeError("GPU computer already busy")
    return free

def verify_output(c,case,raw,log,rc):
    from audit_v060_observer_factorial import assert_complete_physics_fields
    if rc!=0 or "V058_NATIVE_RETURN" not in log or "V058_BACKEND_ARG_FORWARDED" not in log:
        raise ValueError("No clean original G1 execution")
    devices=DEVICE.findall(log)
    if not devices or devices[0]!=case["backend"] or raw.get("v057_requested_physics_backend")!=case["backend"]:
        raise ValueError("Physical backend drift")
    if raw.get("root_fixed") is not False or raw.get("seed")!=1201:
        raise ValueError("Not true free-root original G1")
    if raw.get("policy_sha256")!=c["policy_hashes"][c["policies"][case["policy"]]]:
        raise ValueError("Control strategy drift")
    frames=raw.get("traces",[[]])[0]
    if len(frames)<35 or len(raw.get("joint_names",[]))!=29:
        raise ValueError("Missing original 200Hz G1 29DoF evidence")
    assert_complete_physics_fields(frames)
    observer=raw.get("safe_body_v062")
    if not observer or observer["mode"]!=case["probe"] or observer["actuator_writes"] or observer["physical_writes"]:
        raise ValueError("Stage probe missing or physical writes asserted")
    if observer["actual_200hz_rows"]<30 or len(observer["rows"])!=observer["actual_200hz_rows"]:
        raise ValueError("Stage capture missing actual physics")
    for x in observer["rows"]:
        if not isinstance(x["t"],(int,float)):raise ValueError("Invalid probe time")
    success=raw.get("stride_shot_success")
    if not isinstance(success,bool):raise ValueError("Missing strict all-gates evaluation")
    return dict(strict_success=success,trace_rows=len(frames),
                contact_s=raw["records"][0].get("contact_time_s"),
                goal=raw["records"][0].get("goal"),
                termination=raw["termination_reasons"],
                probe_frames=observer["actual_200hz_rows"],
                last_successful_stage=observer["last_successful_stage"])

def main():
    if Path(sys.executable).resolve()!=PYTHON.resolve():
        raise RuntimeError("Use the original IsaacLab pinned Python")
    if subprocess.run(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT,
                      capture_output=True,text=True).stdout.strip():
        raise RuntimeError("Tracked Git worktree dirty; lock experiments before editing")
    c=json.loads(CFG.read_text())
    digest=verify_protocol(c)
    if PROGRESS.exists():
        state=json.loads(PROGRESS.read_text())
        if state["protocol_sha256"]!=digest:raise RuntimeError("Existing v062 protocol mismatch")
    else:
        state={"version":"v0.1.62","protocol_sha256":digest,"runs":[]}
    MUTEX.parent.mkdir(parents=True,exist_ok=True)
    with MUTEX.open("a") as lock_handle:
        fcntl.flock(lock_handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check_environment()
        if LOCK.exists():
            if json.loads(LOCK.read_text())["protocol_sha256"]!=digest:
                raise RuntimeError("Locked preregistration mismatch")
        else:
            with LOCK.open("x") as f:
                json.dump(dict(protocol_sha256=digest,parent_sha=c["parent_sha"],
                               frozen_before_real_PhysX=True,planned_count=8),f,indent=2)
        if not PROGRESS.exists():
            with PROGRESS.open("x") as f:json.dump(state,f,indent=2)
        for i,case in enumerate(c["cases"]):
            verify_protocol(c);check_environment()
            path=ROOT/case["output"];log=path.with_suffix(".log")
            if i<len(state["runs"]):
                old=state["runs"][i]
                if old["id"]!=case["id"] or not log.exists() or sha(log)!=old["log_sha256"]:
                    raise RuntimeError("Prior raw trial log drift")
                if path.exists()!=bool(old.get("raw_sha256")) or path.exists() and sha(path)!=old["raw_sha256"]:
                    raise RuntimeError("Prior raw G1 evidence drift")
                print("V062_PRESERVED",old["id"],old["status"],flush=True)
                continue
            if path.exists() or log.exists():raise FileExistsError(path)
            cmd=[str(PYTHON),str(ROOT/"tools/run_placement_strike_v062.py"),
                 "--config",str(ROOT/c["policies"][case["policy"]]),
                 "--output",str(path),"--eval-seed","1201","--ball-offset","0","0",
                 "--velocity-delta","0","--sim-device",case["backend"],"--safe-body-v062",case["probe"]]
            env=dict(os.environ,PYTHONPATH=":".join(["tools",str(ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking")]),
                     OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
            print("V062_START",i+1,len(c["cases"]),case["id"],flush=True)
            start=time.monotonic()
            with log.open("x") as handle:
                try:
                    proc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,timeout=240)
                    rc=proc.returncode
                except subprocess.TimeoutExpired:
                    rc=124
                    handle.write("\nV062_TIMEOUT_240_no_new_G1_evidence\n")
            logstr=log.read_text(errors="replace")
            observed=DEVICE.findall(logstr)
            row=dict(id=case["id"],backend=case["backend"],observed_backend=observed[0] if observed else None,
                     probe=case["probe"],policy=case["policy"],seed=case["seed"],
                     code=rc,elapsed_wall_s=round(time.monotonic()-start,2),log_sha256=sha(log))
            if path.exists():
                row["raw_sha256"]=sha(path)
                try:
                    row.update(verify_output(c,case,json.loads(path.read_text()),logstr,rc))
                    row["status"]="VALID_G1"
                except (ValueError,TypeError,KeyError,IndexError) as ex:
                    row.update(status="INVALID_G1",reason=str(ex))
            else:
                stages=[x.strip() for x in logstr.splitlines() if "V062_" in x]
                row.update(status="NO_G1_OUTPUT",reason="No original 200Hz G1 JSON",
                           last_stage_log=stages[-1] if stages else None)
            state["runs"].append(row)
            tmp=PROGRESS.with_suffix(".tmp")
            tmp.write_text(json.dumps(state,indent=2,ensure_ascii=False)+"\n")
            os.replace(tmp,PROGRESS)
            print("V062_END",row["id"],row["status"],row.get("strict_success"),row.get("last_stage_log"),flush=True)
    print("V062_COMPLETE",len(state["runs"]),sum(x["status"]=="VALID_G1" for x in state["runs"]),flush=True)

if __name__=="__main__":main()

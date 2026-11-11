#!/usr/bin/env python3
"""Frozen 4 contexts x 2 policies original G1 PhysX batch, immutable evidence.

This is synthetic coordinated wholebody ABNORMAL-ACTION / negative-control
qualification. It is not human learned or a naturalness comparison.
"""
from __future__ import annotations
import fcntl,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/"configs/g1_phase_batch_v063.json"
LOCK=ROOT/"reports/v063_protocol_lock.json"
PROGRESS=ROOT/"reports/v063_batch_progress.json"
MUTEX=ROOT/"experiments/kinetickick_physics_gpu.lock"
PYTHON=Path("/home/you/conda_env/env_isaaclab/bin/python")
BACKEND=re.compile(r"Environment device\s*:\s*(cpu|cuda:\d+)")
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def environment_ok():
    v=os.statvfs(ROOT);free=v.f_bavail*v.f_frsize/1024**3
    if free<20:raise RuntimeError("Low disk (<20GiB): "+str(round(free,2)))
    s=subprocess.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],
        capture_output=True,text=True,timeout=15)
    if s.returncode or s.stdout.strip():raise RuntimeError("GPU compute busy: "+s.stdout)
    return free
def config():
    c=json.loads(PROTOCOL.read_text())
    if c["version"]!="v0.1.63" or c["parent_sha"]!="ea492310e9a88cf2a43909c464472ce09ec87805":
        raise ValueError("Protocol ancestry not frozen")
    if len(c["cases"])!=8 or len({x["id"] for x in c["cases"]})!=8:
        raise ValueError("Eight independent write-once trial paths required")
    for name,expected in c["source_sha256"].items():
        if sha(ROOT/name)!=expected:raise RuntimeError("SOURCE_DRIFT "+name)
    for name,expected in c["policy_sha256"].items():
        if sha(ROOT/name)!=expected:raise RuntimeError("POLICY_DRIFT "+name)
    for x in c["cases"]:
        if not x["output"].startswith("reports/v063_") or not x["output"].endswith(".json"):
            raise ValueError("Unsafe result path")
        if x["policy"] not in ("v32","synth") or x["seed"] not in (1201,3107,3119,3137):
            raise ValueError("Protocol trial drift")
    return c
def check_real(case,raw,log,rc,c):
    from audit_v060_observer_factorial import assert_complete_physics_fields
    if rc!=0 or "V058_NATIVE_RETURN" not in log or "V058_BACKEND_ARG_FORWARDED" not in log:
        raise ValueError("No clean G1 physical execution")
    found=BACKEND.findall(log)
    if not found or found[0]!="cuda:0" or raw.get("v057_requested_physics_backend")!="cuda:0":
        raise ValueError("Physical backend mismatch")
    if raw.get("root_fixed") is not False or raw.get("seed")!=case["seed"]:
        raise ValueError("Original free-root G1 seed mismatch")
    if len(raw.get("joint_names",[]))!=29:raise ValueError("29DoF original joint order missing")
    from wholebody_safety_interface_v055 import G1_JOINT_ORDER
    if tuple(raw["joint_names"])!=G1_JOINT_ORDER:raise ValueError("Actual 29DoF order changed")
    if raw.get("policy_sha256")!=c["policy_sha256"][c["policies"][case["policy"]]]:
        raise ValueError("Frozen policy config wrong")
    frame=raw.get("traces",[[]])[0]
    if len(frame)<30:raise ValueError("Missing real 200Hz physics")
    assert_complete_physics_fields(frame)
    obs=raw.get("safe_body_v062")
    if not isinstance(obs,dict) or obs["mode"]!="safe_full" or len(obs["rows"])!=len(frame):
        raise ValueError("200Hz wholebody observer not valid")
    max_clock_offset=0.
    for i,(y,x) in enumerate(zip(frame,obs["rows"])):
        if len(y["joint_pos"])!=29 or len(x["joint_pos_rad"])!=29:
            raise ValueError("29DoF measured state absent")
        if max(abs(float(a)-float(b)) for a,b in zip(y["joint_pos"],x["joint_pos_rad"]))>1e-6:
            raise ValueError("Observed 29DoF position not synchronized")
        if max(abs(float(a)-float(b)) for a,b in zip(y["joint_vel"],x["joint_vel_rad_s"]))>1e-6:
            raise ValueError("Observed 29DoF velocity not synchronized")
        if abs(x["state_aligned_time_s"]-y["time_s"])>1e-7 or x["state_alignment_source_index"]!=i:
            raise ValueError("Physical state clock is not aligned")
        max_clock_offset=max(max_clock_offset,abs(x["state_clock_difference_s"]))
    if max_clock_offset<.009 or max_clock_offset>.011:
        raise ValueError("Observed old callback offset changed; clock parity must be requalified")
    if case["policy"]=="synth":
        z=raw.get("phase_v063_summary")
        if not z or z["hypothesis"]!="synthetic_ablation_not_professional_human_learned":
            raise ValueError("Wrong phase candidate or experimental claim")
    elif "phase_v063_summary" in raw:
        raise ValueError("Baseline unexpectedly activated new controller")
    if type(raw.get("stride_shot_success")) is not bool:raise ValueError("Missing strict all-gate verdict")
    return dict(status="VALID_G1",strict_pass=raw["stride_shot_success"],
        trace_rows=len(frame),physical_end_s=frame[-1]["time_s"],
        contact_s=raw["records"][0].get("contact_time_s"),
        termination=raw["termination_reasons"],goal=raw["records"][0].get("goal"),
        actuation=raw.get("phase_v063_summary"),max_clock_offset_s=max_clock_offset)
def main():
    if Path(sys.executable).resolve()!=PYTHON.resolve():raise RuntimeError("Use pinned IsaacLab")
    branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()
    if branch!="version/v0.1.63":raise RuntimeError("Do not run on unrelated branch")
    if subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT,text=True).strip():
        raise RuntimeError("Tracked source dirty before physical experiment")
    c=config();digest=sha(PROTOCOL)
    if PROGRESS.exists():
        state=json.loads(PROGRESS.read_text())
        if state["protocol_sha256"]!=digest:raise RuntimeError("Preexisting experiment drift")
    else:state=dict(protocol_sha256=digest,version=c["version"],runs=[])
    MUTEX.parent.mkdir(exist_ok=True,parents=True)
    with MUTEX.open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        environment_ok()
        if LOCK.exists():
            if json.loads(LOCK.read_text())["protocol_sha256"]!=digest:
                raise RuntimeError("Locked protocol changed")
        else:
            LOCK.write_text(json.dumps(dict(protocol_sha256=digest,
                frozen_git_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                frozen_before_G1_real_PhysX=True,planned_count=len(c["cases"])),indent=2)+"\n")
        if not PROGRESS.exists():PROGRESS.write_text(json.dumps(state,indent=2)+"\n")
        for i,case in enumerate(c["cases"]):
            config();environment_ok()
            dest=ROOT/case["output"];log=dest.with_suffix(".log")
            if i<len(state["runs"]):
                old=state["runs"][i]
                if (old["id"]!=case["id"] or not log.exists() or sha(log)!=old["log_sha256"]
                    or dest.exists()!=bool(old.get("raw_sha256"))
                    or dest.exists() and sha(dest)!=old["raw_sha256"]):
                    raise RuntimeError("PREVIOUS_EVIDENCE_DRIFT: "+case["id"])
                print("PRESERVED",old["id"],old["status"],flush=True);continue
            if dest.exists() or log.exists():raise FileExistsError(dest)
            cmd=[str(PYTHON),str(ROOT/"tools/run_placement_strike_v063.py"),
                "--config",str(ROOT/c["policies"][case["policy"]]),
                "--output",str(dest),"--eval-seed",str(case["seed"]),
                "--sim-device","cuda:0","--safe-body-v062","safe_full",
                "--ball-offset",*[str(x) for x in case["offset_xy"]],
                "--velocity-delta",str(case["velocity_delta_mps"])]
            if case["perturb_reset"]:cmd.append("--perturb-reset-v056")
            env=dict(os.environ,PYTHONPATH=":".join(["tools",str(ROOT/"third_party/HumanoidSoccer/source/whole_body_tracking")]),
                OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
            print("V063_BEGIN",i+1,len(c["cases"]),case["id"],flush=True)
            start=time.monotonic()
            with log.open("x") as out:
                try:
                    p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=240)
                    rc=p.returncode
                except subprocess.TimeoutExpired:
                    rc=124
                    out.write("\nV063_TIMEOUT_240_UNSCORABLE\n")
            readlog=log.read_text(errors="replace")
            row=dict(id=case["id"],case=case,return_code=rc,
                wall_s=round(time.monotonic()-start,2),log_sha256=sha(log))
            if dest.exists():
                row["raw_sha256"]=sha(dest)
                try:row.update(check_real(case,json.loads(dest.read_text()),readlog,rc,c))
                except Exception as ex:row.update(status="INVALID_G1",reason=str(ex))
            else:row.update(status="NO_G1_OUTPUT",
                reason="No original G1 raw physics; never score pass/fail",
                last_console=readlog[-1200:])
            state["runs"].append(row)
            tmp=PROGRESS.with_suffix(".tmp")
            tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n")
            os.replace(tmp,PROGRESS)
            print("V063_END",row["id"],row["status"],row.get("strict_pass"),
                row.get("actuation",{}).get("actual_max_joint_delta_rad") if row.get("actuation") else None,flush=True)
    print("V063_COMPLETE",len(state["runs"]),sum(r["status"]=="VALID_G1" for r in state["runs"]),flush=True)
if __name__=="__main__":main()

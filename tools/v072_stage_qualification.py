#!/usr/bin/env python3
"""No-retry lifecycle triage: compare v67 success, v70 timeouts, v72 smoke.

Read-only, source-hashed, no GPU. 'AppLauncher' / warning matches are not
evidence of G1 physics; smoke marker shows steps happened before hang.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
V70=Path("/home/you/projects/kinetickick-wt-v070/reports")
V67=Path("/home/you/projects/kinetickick-wt-v067/reports")
RESULT=ROOT/"reports/v072_kit_empty_headless_result.json"
LOG=ROOT/"reports/v072_kit_empty_headless.log"
OUT=ROOT/"reports/v072_independent_boot_lifecycle.json"
SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def stage(text,returncode,*,smoke=False):
    """Check exact milestones, never infer G1 success from Kit startup."""
    return dict(app_launcher="[INFO][AppLauncher]" in text,
                kit_log="Logging to file:" in text,
                glfw_warn="GLFW initialization failed" in text,
                shader_warn="DriverShaderCacheManager" in text,
                env_cuda="Environment device    : cuda:0" in text,
                g1_native_return="V058_NATIVE_RETURN cuda:0" in text,
                smoke_finished_eight_steps="KINETICKICK_ISAAC_SMOKE_OK" in text if smoke else False,
                subprocess_exit_ok=returncode==0)

def validate_smoke(meta,text):
    if meta["raw_log_sha256"]!=SHA(LOG):raise RuntimeError("SMOKE_LOG_SOURCE_DRIFT")
    x=stage(text,meta["rc"],smoke=True)
    if x["smoke_finished_eight_steps"] and meta["timeout"]:
        x["diagnosis"]="SMOKE_STEPPED_EIGHT_TIMES_THEN_EXIT_OR_APP_CLOSE_TIMEOUT"
        x["caveat"]="success marker precedes finally: app.close(); stack proof of exact blocking call not recorded"
    elif x["smoke_finished_eight_steps"] and x["subprocess_exit_ok"]:
        x["diagnosis"]="SMOKE_STEP_AND_NORMAL_SHUTDOWN"
    elif meta["timeout"]:
        x["diagnosis"]="SMOKE_TIMEOUT_STAGE_UNRESOLVED"
    else:x["diagnosis"]="SMOKE_NONZERO_OR_PREINIT"
    x["g1_episodes"]=0
    return x

def audit_v70(progress_path=V70/"v070_batch_progress.json"):
    progress=json.loads(progress_path.read_text())
    rows=[]
    for row in progress.get("runs",[]):
        name=row["id"]
        if "/" in name or ".." in name:raise ValueError("unsafe case name")
        log=V70/("v070_"+name+".log")
        if SHA(log)!=row["log_sha256"]:raise RuntimeError("v70_LOG_SOURCE_DRIFT "+name)
        text=log.read_text(errors="replace")
        x=stage(text,row["return_code"])
        x.update(id=name,raw_written=bool(row.get("raw_sha256")),
                 timed_out_240s=row["return_code"]==124,
                 claimed_native_valid=row["status"]=="VALID_G1")
        if x["claimed_native_valid"] and not (x["g1_native_return"] and x["env_cuda"] and x["raw_written"]):
            raise ValueError("unsubstantiated PhysX certification in progress")
        rows.append(x)
    return rows

def audit_v67_reference():
    p=V67/"v067_new_stance_a_v32.log"
    t=p.read_text(errors="replace")
    x=stage(t,0)
    if not (x["g1_native_return"] and x["env_cuda"]):
        raise ValueError("Selected historical G1 reference is not native")
    return x

def run():
    raw=LOG.read_text(errors="replace")
    meta=json.loads(RESULT.read_text())
    smoke=validate_smoke(meta,raw)
    v70=audit_v70()
    ref=audit_v67_reference()
    data=dict(version="v0.1.72",source="read-only log-forensic boot+shutdown distinction",
              freeze_protocol_sha256=meta["frozen_protocol_sha256"],
              historical_v67_reference_log_stage=ref,
              v70_earlier_jobs=len(v70),v70_timeout_without_original_G1=sum(x["timed_out_240s"] and not x["raw_written"] for x in v70),
              v70_native_G1_valid=sum(x["claimed_native_valid"] for x in v70),
              v70_log_warning_glfw_count=sum(x["glfw_warn"] for x in v70),
              v70_logs=v70,
              v72_new_smoke_gpu_processes=1,v72_smoke=smoke,
              v72_new_G1_episodes=0,verified_cuda_wrist_hip_manifolds=0,
              scientific_inference="Normal 8-step smoke was printed before shutdown timeout. v70 stalls are not football failures. Common GLFW/Shader warnings also occur in working v67 and are not a proven cause.",
              necessary_next_trace="isolate AppLauncher/SceneCfg first-frame transitions and app.close() with timestamped begin/end markers before restarting original G1", 
              stop_on_stage_zero_failure=True)
    with OUT.open("x") as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write("\n")
    print("V072_LIFECYCLE","v70_no_g1",data["v70_timeout_without_original_G1"],
          "smoke",smoke["diagnosis"],"v67_glfw_warn",ref["glfw_warn"],
          "sha",SHA(OUT))
    return data
if __name__=="__main__":run()

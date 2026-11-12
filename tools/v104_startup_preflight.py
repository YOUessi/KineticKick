#!/usr/bin/env python3
"""Read-only provenance for an earlier G1 pre-step failure; no Isaac Sim run.

A passing static preflight is NOT evidence that G1 successfully initialized.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
V102=Path("/home/you/projects/kinetickick-wt-v102")
FAILED_LOG_SHA="88397c045028558d3c189cac27c090b96b7e5e0a38f720a5181de5fa5843296f"
def sha(p:Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()
def describe_log(text):
    actor_count=text.count("Actor MLP:")
    stop_stack=text.count("simulation_app.py\", line 581 in close")
    if not stop_stack:
        stop_stack=text.count("simulation_app.py") if "_app_control_on_stop_handle_fn" in text else 0
    return dict(actor_mlp_blocks=actor_count,
                shutdown_stop_callback_witness="_app_control_on_stop_handle_fn" in text and "close" in text,
                shutdown_thread_stack_samples=stop_stack,
                physics_ready="FREE_SHOOTING_READY" in text,
                original_physics_raw_trace_mark="CLOSED_LOOP_STRIKE_RESULT" in text,
                pre_close_python_exception_captured=("V103_PENDING_ORIGINAL_TRACEBACK_BEGIN" in text),
                true_root_exception_unknown=("V103_PENDING_ORIGINAL_TRACEBACK_BEGIN" not in text))
def inspect(previous=V102, output=None):
    log=previous/"reports/v102_L_v32_off.log"
    if not log.is_file():raise FileNotFoundError("MISSING_V102_PHYSX_FAILURE_LOG")
    got=sha(log)
    if got!=FAILED_LOG_SHA:raise ValueError("V102_FAILURE_LOG_SHA_MISMATCH")
    observation=describe_log(log.read_text(errors="replace"))
    if observation["physics_ready"] or observation["original_physics_raw_trace_mark"]:
        raise ValueError("SOURCE_LOG_ALREADY_HAS_PHYSICS")
    original_sources=("tools/v076_eval_guarded.py","tools/v080_run_poll_strike.py",
                      "tools/v101_trace_launcher.py")
    source={}
    for p in original_sources:
        a=ROOT/p;b=previous/p
        if not a.is_file() or not b.is_file() or sha(a)!=sha(b):
            raise ValueError("UNPINNED_ORIGINAL_G1_SOURCE:"+p)
        source[p]=sha(a)
    evidence={
       "original_policy_ONNX":ROOT/".pretrained_checkpoints/paid_official/policy_30000.onnx",
       "original_G1_URDF":ROOT/"configs/paid_original/paid_main_resolved.urdf",
       "original_120hz_CMU":ROOT/"experiments/cmu_real_120hz_phase_v052/10_02/g1_joint_delta_50hz.npz",
       "original_RL_checkpoint":ROOT/"experiments/post_shot_v032/ppo_s3201/last_model.pt",
    }
    artifacts={}
    for name,p in evidence.items():
        if not p.is_file():raise FileNotFoundError("MISSING_LOCAL_UNMODIFIED_EVIDENCE:"+name)
        artifacts[name]=dict(sha256=sha(p),bytes=p.stat().st_size,resolved_path=str(p.resolve()))
    cfg=json.loads((ROOT/"configs/v104_frozen_pairing.json").read_text())
    if cfg["gpu_allowed"] or not cfg["historical_only"]:
        raise ValueError("CPU_FORENSIC_PROTOCOL_DRIFT")
    z=dict(version="v0.1.104",original_v102_failed_log_sha256=got,
        source_unchanged_sha256=source,local_ignored_evidence=artifacts,
        confirmed_pre_step_failure=not observation["physics_ready"],
        last_witness_is_shutting_down=observation["shutdown_stop_callback_witness"],
        **observation,
        interpretation="Shutdown callback stack is observed; upstream initiating exception NOT recovered. Static asset checks do not prove Isaac Sim initialization.",
        newly_executed_G1_runs=0,newly_executed_PhysX_runs=0,
        g1_safety_and_human_coordination_gates_unchanged=True,
        not_a_native_actor_collider_pair_proof=True)
    if output:
        if output.exists():raise FileExistsError("WRITE_ONCE_FORENSIC_REPORT")
        with output.open("x") as f:json.dump(z,f,indent=2);f.write("\n")
    return z
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();z=inspect(output=a.output)
    print("V104_STARTUP",json.dumps({
        "confirmed_pre_step_failure":z["confirmed_pre_step_failure"],
        "last_witness_is_shutting_down":z["last_witness_is_shutting_down"],
        "actor_mlp_blocks":z["actor_mlp_blocks"],
        "true_root_exception_unknown":z["true_root_exception_unknown"],
        "assets_verified":len(z["local_ignored_evidence"])
    }))

from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from v096_contact_probe import native_positive
import v091_independent_dynamics as g1
g1.ROOT=Path("/home/you/projects/kinetickick-wt-v091")
def sha(p:Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def check_frame_trace(raw:dict,row:dict):
    if raw.get("actual_runtime_frames")!=180 or raw.get("steps_requested")!=180:raise ValueError("MISSING_TRUE_180_STEP_PHYSICS")
    for k in ("device","layout","sensor_mode","box_mode","capture","report_mode"):
        if raw.get(k)!=row[k]:raise ValueError("FROZEN_CASE_FIELD_MISMATCH_"+k)
    if raw.get("dt_s")!=.005 or raw.get("original_G1") is not False:raise ValueError("WRONG_PHYSICS_ORIGINAL_G1")
    z=np.asarray(raw.get("runtime_z_vz"),dtype=float)
    if z.shape!=(180,3) or not np.isfinite(z).all() or not np.array_equal(z[:,0],np.arange(1,181)):raise ValueError("BAD_200HZ_STATE")
    if row["layout"]=="separated":
        if not raw.get("observed_noncollision_dynamics") or raw.get("tensor_pair_witness") or raw.get("tensor_contact_counts")!=0:
            raise ValueError("SEPARATION_FALSE_POSITIVE")
    elif not raw.get("observed_collision_dynamics"):
        raise ValueError("ACTUAL_PHYSICS_CONTACT_NOT_OBSERVED")
    stage=raw["stage"]
    if not stage["all_colliders"]["ProbeBall"] or not stage["all_colliders"]["ProbeBox"]:
        raise ValueError("MISSING_REAL_USD_SHAPE_INVENTORY")
    expected=(row["report_mode"]=="on" and row["sensor_mode"]!="collider")
    if stage["ball_report"] is not expected or stage["box_report"] is not expected:
        raise ValueError("PHYSX_REPORT_SCHEMA_MISMATCH")
    if raw.get("native_pair_witness") != any(native_positive(x) for x in raw.get("native_events",[])):
        raise ValueError("UNVERIFIED_NATIVE_CONTACT_REPORT")
    # A tensor filter/path witness is NOT a native collision manifold.
    if raw.get("tensor_pair_witness"):
        if row["device"]!="cuda:0" or row["layout"]!="colliding" or row["capture"] not in ("tensor","both"):
            raise ValueError("IMPOSSIBLE_TENSOR_POSITIVE")
        points=raw.get("tensor_contact_points",[])
        if not points or not all(len(p["point_w"])==3 and len(p["normal_w"])==3 and
                                  np.isfinite(p["point_w"]+p["normal_w"]).all() and
                                  p["force_magnitude_N"]>0 for p in points):
            raise ValueError("TENSOR_POSITIVE_INVALID_POINT")
    return z
def parity(raw1:dict,raw2:dict):
    x=np.asarray(raw1["runtime_z_vz"],float)
    y=np.asarray(raw2["runtime_z_vz"],float)
    if raw1["device"]!=raw2["device"] or raw1["layout"]!=raw2["layout"] or raw1["box_mode"]!=raw2["box_mode"]:
        raise ValueError("DIFFERENT_MECHANICS_NOT_PARITY")
    return dict(fields=["step","sphere_z_m","sphere_vz_m","box_xyz_m"],
       max_absolute_saved_state_difference=float(np.max(np.abs(x-y))),
       exact_saved_state=bool(np.array_equal(x,y) and raw1["box_position_w"]==raw2["box_position_w"]),
       unrecorded_internal_states_compared=False)
def phase_dynamics(case:str):
    r,(t,vel,lim,loads,com,nonfoot,football,impact)=g1.inspect(case)
    if impact is None:raise ValueError("NO_REAL_CONTACT_FOR_ALIGN")
    mask=(t>=impact-.4-1e-6)&(t<=min(t[-1],impact+1.25)+1e-6)
    if not np.any(mask):raise ValueError("MISSING_TRUE_200HZ_DYNAMICS")
    tt=t[mask];vv=vel[mask];ll=loads[mask];nf=nonfoot[mask]
    groups={}
    names=json.loads((g1.ROOT/"reports"/("v091_"+case+".json")).read_text())["joint_names"]
    for name,joints in g1.GROUPS.items():
        ix=np.asarray([names.index(j) for j in joints])
        energy=np.sum(np.square(vv[:,ix]/lim[ix]),axis=1)
        groups[name]=dict(normalized_speed_energy_mean=float(energy.mean()),
           phase_peak_relative_impact_s=float(tt[np.argmax(energy)]-impact))
    # Same support thresholds and original sensor evidence as v91. No forceplate claim.
    result=dict(case=case,sha256=r["original_raw_sha256"],real_G1_frames=r["real_200hz_frames"],
        first_foot_ball_contact_s=float(impact),postimpact_window_fully_observed=r["contact_window_post1p25_complete"],
        window_200hz_samples=len(tt),window_final_relative_impact_s=float(tt[-1]-impact),
        left_foot_support_80N_fraction=float(np.mean(ll[:,0]>=80)),
        right_foot_support_80N_fraction=float(np.mean(ll[:,1]>=80)),
        double_support_80N_fraction=float(np.mean((ll[:,0]>=80)&(ll[:,1]>=80))),
        nonfoot_peak_in_event_window_N=float(np.max(nf)),
        max_all29_true_measured_qdot_over_limit=float(np.max(np.abs(vv)/lim)),
        group_phase_proxies=groups,strict_original_pass=r["independent_gates"]["strict_pass"],
        original_failed_hard_gates=r["independent_gates"]["failed"],
        root_height_not_fall_proof=True,original_actor_collider_pair_authenticated=False,
        human_elite_motion_ground_truth=False)
    return result
def audit(path:Path):
    if path.exists():raise FileExistsError("WRITE_ONCE_INDEPENDENT_AUDIT")
    cfg=json.loads((ROOT/"configs/v096_frozen_matrix.json").read_text())
    batchpath=ROOT/"reports/v096_frozen_contact_batch.audit.json"
    batch=json.loads(batchpath.read_text())
    if batch["version"]!=cfg["version"] or batch["frozen_commit"]!="3480862b90b7a391c0ae9976095b6aa10c4c10fc":
        raise ValueError("FROZEN_COMMIT_DRIFT")
    if batch["protocol_sha256"]!=sha(ROOT/"configs/v096_frozen_matrix.json"):
        raise ValueError("PROTOCOL_HASH_DRIFT")
    if len(batch["trials"])!=len(cfg["matrix"]) or batch["unexecuted"]:
        raise ValueError("NOT_COMPLETE_PRE_REGISTERED_PHYSICS_MATRIX")
    cases={};raws={};sources={}
    for row,b in zip(cfg["matrix"],batch["trials"]):
        if b["case"]!=row["case"] or b["timeout"] or b["rc"]!=0 or not b["recorded_runtime_evidence"] or not b["physically_confirmed"]:
            raise ValueError("NOT_PHYSICALLY_QUALIFIED_"+row["case"])
        stem=ROOT/"reports"/("v096_"+row["case"])
        rawpath=stem.with_suffix(".json");log=stem.with_suffix(".log");receipt=stem.with_suffix(".audit.json")
        if not all(x.is_file() for x in (rawpath,log,receipt)):raise ValueError("MISSING_WRITE_ONCE_EVIDENCE")
        if sha(rawpath)!=b["raw_sha256"] or sha(log)!=b["log_sha256"] or json.loads(receipt.read_text())!=b:
            raise ValueError("ORIGINAL_EVIDENCE_SHA_MISMATCH")
        raw=json.loads(rawpath.read_text())
        check_frame_trace(raw,row)
        cases[row["case"]]=dict(step_count=180,native_actor_collider_manifold_certified=bool(raw.get("native_pair_witness")),
           gpu_tensor_filtered_pair_witness=bool(raw.get("tensor_pair_witness")),
           force_peak_N=raw.get("tensor_peak_force_N"),
           contact_callback_calls=raw.get("native_counts",{}).get("callback_calls"),
           captured_true_native_headers=raw.get("native_counts",{}).get("callback_headers"),
           sensor_mode=raw["sensor_mode"],box_mode=raw["box_mode"],device=raw["device"],
           observed_collision=raw.get("observed_collision_dynamics"),
           observed_separation=raw.get("observed_noncollision_dynamics"))
        raws[row["case"]]=raw
        sources[row["case"]]=dict(raw_sha256=sha(rawpath),log_sha256=sha(log),receipt_sha256=sha(receipt))
    parities={a+"__"+b:parity(raws[a],raws[b]) for a,b in cfg["parity_pairs"]}
    if not all(v["exact_saved_state"] for v in parities.values()):
        raise ValueError("OFF_ON_STATE_PARITY_FAILED")
    if not cases["cuda_spawn_dual"]["gpu_tensor_filtered_pair_witness"] or cases["cuda_separated_dual"]["gpu_tensor_filtered_pair_witness"]:
        raise ValueError("INVALID_PHYSICAL_TENSOR_POSITIVE_OR_NEGATIVE")
    historical=[phase_dynamics(c) for c in g1.CASES]
    result=dict(version="v0.1.96",frozen_commit=batch["frozen_commit"],
      protocol_sha256=sha(ROOT/"configs/v096_frozen_matrix.json"),batch_sha256=sha(batchpath),
      isolated_200hz_physx_calibration_runs=len(cases),isolated_true_200hz_physx_steps=180*len(cases),
      authentic_new_original_G1_shooting_runs=0,strict_pass_new_G1=None,
      validated_native_actor_collider_pair_positive=sum(v["native_actor_collider_manifold_certified"] for v in cases.values()),
      validated_tensor_filtered_positive=sum(v["gpu_tensor_filtered_pair_witness"] for v in cases.values()),
      all_five_observed_state_parity=parities,immutable_sources=sources,cases=cases,
      historical_real_original_G1_runs=len(historical),historical_strict_pass=sum(x["strict_original_pass"] for x in historical),
      original_200hz_29dof_event_aligned_coupling=historical,
      actual_G1_wrist_hip_pair_manifold_proven=False,new_29dof_control_physically_safety_certified=False,
      temporal_motion_learning_is_elite_human_truth=False)
    with path.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print("V096_INDEPENDENT",json.dumps(dict(physx_calibrations=len(cases),actual_steps=180*len(cases),
      native_true_pair_count=result["validated_native_actor_collider_pair_positive"],
      tensor_witnesses=result["validated_tensor_filtered_positive"],
      parity_pass=sum(v["exact_saved_state"] for v in parities.values()),
      historical_G1_count=len(historical),historical_G1_strict_pass=result["historical_strict_pass"],
      new_G1_shots=0)),flush=True)
    return result
if __name__=="__main__":
    audit(ROOT/"reports/v096_independent_evidence.audit.json")


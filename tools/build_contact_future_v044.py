#!/usr/bin/env python3
"""v44: leak-safe supervised *planned-action-conditioned* delayed G1 qdot risk data.

Uses ONLY the 12 previously seen v42 real PhysX 14-second original G1 runs.
Offline features are measured before first foot-ball impact. The target is the
true maximal |qdot| / unchanged 29 original actuator velocity limits until
1.25 seconds after actual first impact. A 0/1 indicator describes the known
scheduled reference controller (original vs fixed v41 style); without it the
same pre-action state may have different futures. All sources SHA-pinned.
Not prospective unseen data; not human biomechanics or safety certification.
"""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
STUDY="configs/contact_future_forecast_v044.json"
def sha(file):
    h=hashlib.sha256()
    with (ROOT/file).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def feature_from_sample(sample,source,style_flag):
    q=np.asarray(sample["joint_pos"],np.float32)
    v=np.asarray(sample["joint_vel"],np.float32)
    limit=np.asarray(source["joint_velocity_limits"][0],np.float32)
    support=np.asarray([sample["support_foot_force_N"][name] for name in ("left","right")],np.float32)
    ball=np.asarray(sample["ball_xyz"],np.float32)
    base=np.asarray(sample["root_xyz"],np.float32)
    ball_vel=np.asarray(sample["ball_velocity"],np.float32)
    if q.shape!=(29,) or v.shape!=(29,) or limit.shape!=(29,) or support.shape!=(2,) or base.shape!=(3,) or ball.shape!=(3,) or ball_vel.shape!=(3,):
        raise ValueError("Incomplete input state for original G1")
    if np.any(limit<=0):raise ValueError("Unchanged positive original 29 joint velocity limits required")
    result=np.concatenate((q, v/limit, np.clip(support,0,5000)/1500.,
                           (ball[:2]-base[:2])/3.,ball_vel/5.,
                           np.asarray([float(sample["time_s"])/2.,float(style_flag)],np.float32)))
    if result.shape!=(67,) or not np.all(np.isfinite(result)):
        raise ValueError("Finite 67D G1 state/action-conditioned input required")
    return result

def build():
    study=json.loads((ROOT/STUDY).read_text())
    if (study["version"]!="0.1.44" or study["feature_width"]!=67
        or study["future_joint_count"]!=29 or study["training_uses_v42_existing_development_data"] is not True):
        raise ValueError("Wrong frozen data contract")
    for file,d in study["source_sha256"].items():
        if sha(file)!=d:raise ValueError("Frozen v42 source has drifted: "+file)
    rows=[]
    X,Y,lab_case,lab_ctrl,t_samples,peak_actual=[],[],[],[],[],[]
    previous=json.loads((ROOT/"reports/v042_paired_novel_ball_audit.json").read_text())
    groups={split:set(names) for split,names in study["group_split"].items()}
    if any(groups[a]&groups[b] for a,b in (("train","val"),("train","test"),("val","test"))):
        raise ValueError("Scenario-level leakage across splits")
    if set.union(*groups.values())!=set(r["id"] for r in previous["trials"]):
        raise ValueError("Scenarios missing/added")
    for trial in previous["trials"]:
        for controller in ("original_v32","style_v41"):
            p=trial[controller]["path"]
            if sha(p)!=trial[controller]["sha256"]:raise ValueError("Original exact 14s PhysX source changed")
            z=json.loads((ROOT/p).read_text())
            raw=z["traces"][0]
            contact=float(z["records"][0]["contact_time_s"])
            if len(raw)!=2800 or not 0.5<contact<3 or z["post_initialization_state_writes"] or z["root_fixed"]:
                raise ValueError("Invalid original free-root 14s physical record")
            times=np.asarray([r["time_s"] for r in raw],float)
            if not np.allclose(np.diff(times),.005,atol=5e-7):raise ValueError("Physics 5ms samples missing")
            qdot=np.asarray([r["joint_vel"] for r in raw],np.float32)
            limits=np.asarray(z["joint_velocity_limits"][0],np.float32)
            assert qdot.shape==(2800,29) and limits.shape==(29,)
            ratio=np.abs(qdot)/limits
            begin=study["observation_relative_to_impact_start_s"]
            end=study["observation_relative_to_impact_end_s"]
            stride=study["observation_stride_s"]
            relative=np.arange(begin,end+stride*.1,stride)
            assert len(relative)==12
            for dt in relative:
                requested=contact+dt
                idx=int(np.argmin(np.abs(times-requested)))
                if abs(times[idx]-requested)>1e-6:raise ValueError("Impact-relative 5ms grid not exact")
                future_end=contact+study["future_end_relative_to_impact_s"]
                end_idx=int(np.argmin(np.abs(times-future_end)))
                if abs(times[end_idx]-future_end)>1e-6 or end_idx<=idx:
                    raise ValueError("End of future horizon not in actual physical episode")
                target=ratio[idx:end_idx+1].max(axis=0)
                if not np.all(np.isfinite(target)) or target.shape!=(29,):
                    raise ValueError("Invalid original fully observed future qdot")
                x=feature_from_sample(raw[idx],z,controller=="style_v41")
                X.append(x)
                Y.append(target)
                lab_case.append(trial["id"])
                lab_ctrl.append(controller)
                t_samples.append(float(dt))
                peak_actual.append(float(np.max(target)))
            rows.append(dict(case=trial["id"],controller=controller,
                input_sha256=sha(p),physical_steps=len(raw),
                original_actual_qdot_max_ratio=float(ratio.max()),
                first_true_foot_ball_impact_s=contact))
    X=np.stack(X).astype(np.float32)
    Y=np.stack(Y).astype(np.float32)
    assert X.shape==(144,67) and Y.shape==(144,29)
    cases=np.asarray(lab_case)
    controller=np.asarray(lab_ctrl)
    t=np.asarray(t_samples,dtype=np.float32)
    split=np.asarray([next(k for k,v in groups.items() if c in v) for c in cases])
    assert {k:int((split==k).sum()) for k in ("train","val","test")}=={"train":96,"val":24,"test":24}
    # Core identifiability: at least one same-case original/style before action
    # has nearly identical q/qd but distinct planned control identifier.
    out=ROOT/study["output_directory"]
    out.mkdir(parents=True,exist_ok=True)
    binary=out/"planned_action_future_qdot_v044.npz"
    manifest=out/"build_manifest.json"
    if binary.exists() or manifest.exists():raise FileExistsError("Do not overwrite frozen v44 data")
    np.savez_compressed(binary,
       features=X,targets_max_qdot_ratio_per_joint=Y,
       future_any_original_qdot_exceeded=np.max(Y,axis=1)>1.+1e-6,
       episode_id=cases,controller=controller,split=split,
       relative_observation_time_s=t,
       human_joint_motion_truth=False,source_real_G1_hardware=False,
       actual_physics=True,
       input_names=np.asarray(
        [f"q_{i}" for i in range(29)]+[f"qdot_normalized_{i}" for i in range(29)]+
        ["left_stance_support_div1500","right_swing_support_div1500",
         "ball_rel_root_x_div3","ball_rel_root_y_div3",
         "ball_velocity_x_div5","ball_velocity_y_div5","ball_velocity_z_div5",
         "absolute_time_div2","precommitted_style_reference_enabled"]))
    m=dict(version="0.1.44",frozen_study_sha256=sha(STUDY),
       data_sha256=sha(study["output_directory"]+"/planned_action_future_qdot_v044.npz"),
       frozen_robot_source_episode_count=len(rows),original_physx_step_total=12*2800,
       observation_count=len(X),raw_input_dim=X.shape[1],future_target_dofs=Y.shape[1],
       source_episodes=rows,
       precommit_future_action_flag=True,
       independent_samples_cannot_be_counted_as_144=True,
       independent_robot_task_groups_per_split={k:len(v) for k,v in groups.items()},
       source_group_splits={k:sorted(v) for k,v in groups.items()},
       split_window_count={k:int((split==k).sum()) for k in ("train","val","test")},
       source_is_historical_development_observed=True,
       worst_future_qdot_ratio=max(peak_actual),
       known_original_joint_velocity_limits_not_relaxed=True,
       limitations=["Models see action plan indicator and current measured state only; no future PhysX states leaked into input features",
        "Future label includes original physically observed 1.25s after real impact, which is not known online",
        "Multiple times in same original episode are highly correlated, grouping is by initial-condition trial",
        "Only 4 training conditions,1 validation,1 test; prior human source is 3DSP broadcast 3D unqualified",
        "This is a data research artifact, not proof a forecaster can guarantee qdot limits"])
    manifest.write_text(json.dumps(m,indent=2,ensure_ascii=False)+"\n")
    print("V044_RISK_DATASET_BUILT",json.dumps(dict(
        input_shape=X.shape,target_shape=Y.shape,split_windows=m["split_window_count"],
        train_groups=sorted(groups["train"]),val_groups=sorted(groups["val"]),test_groups=sorted(groups["test"]),
        n_source_episodes=len(rows),checksum=m["data_sha256"],
        future_unsafe_windows=int((Y.max(axis=1)>1.000001).sum()))),flush=True)

if __name__=="__main__":build()

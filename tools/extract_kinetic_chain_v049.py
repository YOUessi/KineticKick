#!/usr/bin/env python3
"""v49 contact-aligned real G1 full-body *kinematic chain* evidence.

Audit actual IsaacLab/PhysX foot velocity decomposition (base/hip/knee/ankle)
and first-contact ball/foot speeds, support, joint qdot peak times, estimated
PD joint power and 29-joint speeds. No simulation, policy tuning or action
injection. A signed velocity contribution is NOT an impulse work/energy share
or a verified human proximal-distal muscle power transfer.
"""
from pathlib import Path
import hashlib,json,math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CONFIG="configs/kinetic_chain_observability_v049.json"
REL=np.round(np.arange(-60,71)*.005,8)  # -300ms to +350ms incl impact

def sha(path):
 h=hashlib.sha256()
 with (ROOT/path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def vector(row,name,n):
 v=np.asarray(row[name],dtype=np.float64)
 if v.shape!=(n,) or not np.isfinite(v).all():raise ValueError(f"Malformed original physical signal {name}")
 return v

def first_contact_index(rows,reported):
 if reported is None:raise ValueError("Cannot claim biomechanical ball contact when no physical event exists")
 time=np.array([row["time_s"] for row in rows],float)
 available=np.flatnonzero((time>=reported-.00501)&(time<=reported+.01501)&
                      (np.asarray([r["contact_force_N"] for r in rows])>2.))
 if len(available)==0:raise ValueError("No original force-positive contact close to recorded impact")
 first=min(available,key=lambda k:abs(time[k]-reported))
 if abs(time[first]-reported)>.01501:raise ValueError("Impact time disagreement")
 return first

def aligned_indices(rows,t_contact,relative):
 t=np.asarray([r["time_s"] for r in rows],float)
 requested=t_contact+relative
 idx=np.rint((requested-t[0])/.005).astype(int)
 if idx.min()<0 or idx.max()>=len(t):
  raise ValueError("Original original 5ms physical trace does not span full pre and post contact window")
 if np.max(abs(t[idx]-requested))>5e-6 or not np.all(np.diff(idx)==1):
  raise ValueError("Non-contiguous or re-interpolated raw physical window")
 return idx

def analyze_one(raw,config_name,kind):
 r=json.loads((ROOT/raw).read_text())
 if r["root_fixed"] or r["post_initialization_state_writes"]:
  raise ValueError("True non-fixed original G1 PhysX, no state rewriting")
 if r["policy_sha256"]!=sha(config_name) or r["joint_strike"]["config_sha256"]!=sha(config_name):
  raise ValueError("Wrong original G1 physical controller")
 if r["joint_strike"]["checkpoint_sha256"]!=sha("experiments/post_shot_v032/ppo_s3201/last_model.pt"):
  raise ValueError("Original recovery PPO swapped")
 rows=r["traces"][0]
 if len(rows)==0 or any(abs(z["physics_dt_s"]-.005)>1e-7 for z in rows):
  raise ValueError("Original G1 5ms physics only")
 contact=r["records"][0]["contact_time_s"]
 k=first_contact_index(rows,contact)
 idx=aligned_indices(rows,contact,REL)
 current=[rows[int(i)] for i in idx]
 name=r["joint_names"];names={n:i for i,n in enumerate(name)}
 if len(name)!=29 or len(names)!=29:raise ValueError("Not original G1 joint semantics")
 required=("right_hip_pitch_joint","right_knee_joint","right_ankle_pitch_joint",
           "waist_yaw_joint","waist_pitch_joint","right_shoulder_pitch_joint",
           "left_ankle_pitch_joint","left_ankle_roll_joint")
 if not all(n in names for n in required):raise ValueError("Missing body-part joint")
 q=np.asarray([z["joint_pos"] for z in current],float)
 qdot=np.asarray([z["joint_vel"] for z in current],float)
 tau=np.asarray([z["joint_torque_estimate"] for z in current],float)
 foot=np.asarray([z["right_foot_point_velocity"] for z in current],float)
 ball=np.asarray([z["ball_velocity"] for z in current],float)
 ball_spin=np.asarray([z["ball_angular_velocity"] for z in current],float)
 support=np.asarray([[z["support_foot_force_N"][side] for side in ("left","right")] for z in current],float)
 root=np.asarray([z["root_xyz"] for z in current],float)
 com=np.asarray([z["whole_body_com_velocity"] for z in current],float)
 limits=np.asarray(r["joint_velocity_limits"][0],float)
 if any(x.shape!=s for x,s in (
   (q,(len(REL),29)),(qdot,(len(REL),29)),(tau,(len(REL),29)),(foot,(len(REL),3)),
   (ball,(len(REL),3)),(ball_spin,(len(REL),3)),(support,(len(REL),2)),
   (root,(len(REL),3)),(com,(len(REL),3)),(limits,(29,)))):
  raise ValueError("Full-body original PhysX shape mismatch")
 if not all(np.isfinite(x).all() for x in (q,qdot,tau,foot,ball,ball_spin,support,root,com,limits)):
  raise ValueError("Malformed original physical measured motion")
 if np.any(limits<=0):raise ValueError("Original G1 qdot limits missing")
 contributions={}
 for part in ("base","hip","knee","ankle"):
  v=np.asarray([z["foot_velocity_contributions"][part] for z in current],float)
  if v.shape!=(len(REL),3) or not np.isfinite(v).all():
   raise ValueError("Missing original real-world foot point velocity decompositions "+part)
  contributions[part]=v
 summed=np.sum(np.stack(list(contributions.values())),axis=0)
 residual=np.linalg.norm(summed-foot,axis=1)
 if residual.max()>1e-5:
  raise ValueError("Recorded foot-point physics velocity decomposition is inconsistent")
 near=int(np.argmin(abs(REL)))
 if REL[near]!=0:
  raise ValueError("Time zero must be original impact measurement")
 # Original robot moves toward decreasing world Y. Ball goal also toward
 # -Y; this is an observed WORLD signed foot-velocity component, not a
 # player anatomical instep-normal or contact material-frame velocity.
 goal_direction=np.asarray([0.,-1.,0.])
 point_speed=float(np.linalg.norm(foot[near]))
 forward=float(foot[near]@goal_direction)
 components_signed={part:float(v[near]@goal_direction) for part,v in contributions.items()}
 if abs(sum(components_signed.values())-forward)>1e-5:
  raise ValueError("Claimed whole-body foot velocity decomposition invalid")
 # Peak joint speed phase is NOT joint mechanical power phase; explicitly
 # return both metrics. The earliest first contact clock is measured by PhysX.
 pre=(REL>=-.30)&(REL<=0)
 post=(REL>0)&(REL<=.35)
 def peak_speed(joint):
  j=names[joint]
  local=np.abs(qdot[pre,j])
  step=int(np.argmax(local))
  when=float(REL[pre][step])
  peak=float(local[step])
  pmax=float(np.max(np.abs(tau[pre,j]*qdot[pre,j])))
  return dict(absolute_qdot_peak_rad_s=peak,
              peak_relative_contact_s=when,
              maximum_absolute_clipped_PD_power_W=pmax,
              joint="G1 "+joint,
              does_not_prove_mechanical_causality=True)
 chain={key:peak_speed(key) for key in required}
 # Differences between start and after measured first contact, not launch
 # velocities from assistant-injected ball speed.
 ball_after=ball[np.argmin(abs(REL-.025))]
 ball_before=ball[np.argmin(abs(REL+.025))]
 energy_delta=.5*.43*float((ball_after@ball_after)-(ball_before@ball_before))
 pre_directions=[chain[n]["peak_relative_contact_s"] for n in
             ("right_hip_pitch_joint","right_knee_joint","right_ankle_pitch_joint")]
 timing_order=bool(pre_directions[0]<=pre_directions[1]<=pre_directions[2])
 power=tau*qdot
 group={
  "right_leg":[i for i,n in enumerate(name) if n.startswith("right_") and any(k in n for k in ("hip","knee","ankle"))],
  "left_leg":[i for i,n in enumerate(name) if n.startswith("left_") and any(k in n for k in ("hip","knee","ankle"))],
  "waist":[i for i,n in enumerate(name) if n.startswith("waist_")],
  "arms":[i for i,n in enumerate(name) if any(k in n for k in ("shoulder","elbow","wrist"))]}
 assert sum(map(len,group.values()))==29
 group_power={}
 for g,joints in group.items():
  positive=np.maximum(power[:,joints],0)
  negative=np.maximum(-power[:,joints],0)
  group_power[g]=dict(estimated_positive_PD_work_pre_J=float(positive[pre].sum()*.005),
     estimated_negative_PD_work_pre_J=float(negative[pre].sum()*.005),
     max_absolute_joint_speed_ratio_pre=float((abs(qdot[pre][:,joints])/limits[joints]).max()),
     does_not_represent_human_muscle_work=True)
 record=r["records"][0]
 line=dict(version="0.1.49",source=raw,source_sha256=sha(raw),controller=kind,
    actual_full_original_PhysX_steps=len(rows),true_first_foot_ball_contact_s=float(contact),
    contact_index_original_physics=int(k),
    measured_surface=record["measured_surface"],
    lawful_right_instep=record["measured_surface"]=="instep" and record["surface_match"] is True,
    actual_instep_impulse_Ns=float(record["contact_impulse_by_surface_Ns"].get("instep",0)),
    original_foot_contact_peak_force_N=float(record["peak_filtered_contact_force_N"]),
    original_ball_goal=bool(record["goal"]),
    original_recovered=bool(record["recovered"]),
    standing_s=float(record["standing_seconds"]),
    original_forward_foot_contact_speed_mps=forward,
    original_3d_foot_point_speed_mps=point_speed,
    actual_ball_translation_energy_delta_25ms_J=energy_delta,
    original_forward_velocity_from_parts_mps=components_signed,
    foot_velocity_world_decomposition_max_numerical_error_mps=float(residual.max()),
    original_hip_knee_ankle_speed_peak_order_example=timing_order,
    original_right_joint_speed_peak_phases=chain,
    original_positive_PD_work_estimate_by_body=group_power,
    max_actual_qdot_over_limit_ratio_in_event_window=float((abs(qdot)/limits).max()),
    true_original_inst_angle_not_verified=True,
    original_human_pose_groundtruth_lacking=True,
    interpretation="Signed component to G1 foot velocity (world -Y) and original PD qdot, NOT energy transfer contributions, human explosiveness, naturalness or anatomy."
    )
 arrays=dict(q=q,qdot=qdot,tau=tau,
      foot_point_velocity=foot,ball_velocity=ball,ball_angular_velocity=ball_spin,
      left_right_support=support,root_xyz=root,com_velocity=com,
      qdot_over_limit=abs(qdot)/limits,
      base_contribution=contributions["base"],hip_contribution=contributions["hip"],
      knee_contribution=contributions["knee"],ankle_contribution=contributions["ankle"],
      original_power_W_estimated=power)
 return line,arrays,name,limits

def build():
    s=json.loads((ROOT/CONFIG).read_text())
    for f,h in s["source_sha256"].items():
        if sha(f)!=h:raise ValueError("v49 original registered source drift "+f)
    if s["version"]!="0.1.49" or s["no_new_PhysX_episodes"] is not True:
        raise ValueError("Only read-only original G1 true physical records permitted")
    all_arrays={}
    reports=[]
    joint_names=None
    joint_limits=None
    prior=json.loads((ROOT/"reports/v048_preimpact_full_physics_audit.json").read_text())
    for c in prior["cases"]:
        for model in ("original_v32","learned_preimpact_v48"):
            raw=c[model]["original_source"]
            config=s["original_config"] if model=="original_v32" else s["trained_preimpact_config"]
            detail,trace,labels,limits=analyze_one(raw,config,model)
            if joint_names is None:joint_names=labels;joint_limits=limits
            if labels!=joint_names or not np.array_equal(limits,joint_limits):
                raise ValueError("Original G1 body and actuator limit drift")
            reports.append(dict(case=c["id"],**detail))
            for key,v in trace.items():all_arrays.setdefault(key,[]).append(v.astype(np.float32))
    if len(reports)!=12:raise ValueError("Only original six source conditions paired, 12 original physical runs")
    payload={k:np.stack(v) for k,v in all_arrays.items()}
    if any(v.shape[0]!=12 or v.shape[1]!=len(REL) for v in payload.values()):
        raise ValueError("Incomplete contiguous original PhysX 5ms data")
    out=ROOT/s["output_dir"]
    out.mkdir(parents=True,exist_ok=True)
    npz=out/"kinetic_chain_physx_v049.npz"
    audit=out/"kinetic_chain_audit_v049.json"
    if npz.exists() or audit.exists():raise FileExistsError("Do not recompute after result review")
    np.savez_compressed(npz,**payload,
       relative_real_contact_s=REL,
       source_case=np.asarray([r["case"] for r in reports]),
       controller=np.asarray([r["controller"] for r in reports]),
       original_G1_joint_names=np.asarray(joint_names),
       original_G1_joint_speed_limits_rad_s=joint_limits,
       physical_human_foot_biomechanics_verified=np.asarray(False),
       original_PhysX_data_only=np.asarray(True))
    meta=dict(version="0.1.49",source_protocol_sha256=sha(CONFIG),
       archived_original_v48_audit_sha256=sha("reports/v048_preimpact_full_physics_audit.json"),
       v49_physics_npz_sha256=sha(s["output_dir"]+"/kinetic_chain_physx_v049.npz"),
       data_is_original_PhysX_5ms_no_model_fitting=True,
       first_ball_foot_contact_as_time_zero=True,
       original_relative_time_window_s=[float(REL[0]),float(REL[-1])],
       original_contact_aligned_physics_samples_per_episode=len(REL),
       independent_ball_cases=6,
       original_source_episode_count=12,
       joint_names=joint_names,
       limitations=[
         "Only G1 real IsaacLab physics, NOT hardware nor calibrated professional human instep mechanics",
         "3DSP prior is broadcast-estimated 3D with no impact timestamps, anatomical joint centers, ground force, muscle, ball speed truth",
         "Joint qdot peak ordering does not establish causal kinetic chain of momentum transmission; foot velocity sums verify only kinematic decomposition",
         "PD torque estimate times actual G1 qdot is not real motor/human muscle mechanical work",
         "G1 link and joint coordinate system differs from professional human skeleton",
         "Only six known v48 exact ball conditions already evaluated; no new blind test, seed or generalization",
         "Visible ball exit speed and ground load cannot replace human-rated naturalness"],
       episodes=reports)
    audit.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n")
    print("V049_TRUE_G1_KINETIC_CHAIN_DATA",json.dumps(dict(
       episodes=len(reports),window_physics_samples=len(REL),signals=sorted(payload),
       checksum=meta["v49_physics_npz_sha256"],
       forward_velocity_contributions_at_impact=[dict(case=r["case"],controller=r["controller"],
           hip_mps=r["original_forward_velocity_from_parts_mps"]["hip"],
           knee_mps=r["original_forward_velocity_from_parts_mps"]["knee"],
           ankle_mps=r["original_forward_velocity_from_parts_mps"]["ankle"],
           ball_speed_3d_foot_mps=r["original_3d_foot_point_speed_mps"],
           ball_translation_energy_delta25ms_J=r["actual_ball_translation_energy_delta_25ms_J"])
           for r in reports]),ensure_ascii=False),flush=True)

if __name__=="__main__":build()

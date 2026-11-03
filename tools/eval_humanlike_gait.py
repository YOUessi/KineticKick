"""Quantitative G1 gait baseline audit (before human-like walking fine-tune).

Isaac Lab 2.1 / NVIDIA pretrained G1 37DOF, fixed velocity command, free root.
No visual motion judgment is claimed from a scalar metric.
Foot sole contact is approximate using ankle-roll link; contact needs per-patch
instrumentation before statements about heel-strike or toe-off.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True,enable_cameras=False).app
import torch
import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import isaaclab_tasks
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg_PLAY
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg

ROOT=Path("/home/you/projects/kinetickick")
CKPT=ROOT/".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-G1-v0/checkpoint.pt"

TRACKED_JOINTS=(
    "left_hip_pitch_joint","right_hip_pitch_joint",
    "left_knee_joint","right_knee_joint",
    "left_ankle_pitch_joint","right_ankle_pitch_joint",
    "left_shoulder_pitch_joint","right_shoulder_pitch_joint",
    "torso_joint",
)
FOOT_LINKS=("left_ankle_roll_link","right_ankle_roll_link")

def percentile(seq,p):
    if not seq:return None
    s=sorted(seq)
    v=p*(len(s)-1)
    i=int(v);j=min(i+1,len(s)-1)
    return s[i]*(1-(v-i))+s[j]*(v-i)

def correlation(x,y):
    import numpy as np
    a=np.asarray(x,dtype=float)
    b=np.asarray(y,dtype=float)
    if len(a)<4 or a.std()<1e-6 or b.std()<1e-6:return None
    return float(np.corrcoef(a,b)[0,1])

def contiguous_segments(flags):
    out=[];s=None
    for i,flag in enumerate(list(flags)+[False]):
        if flag and s is None:s=i
        if not flag and s is not None:
            out.append((s,i))
            s=None
    return out

def compute_metrics(rows, step_dt, desired_speed):
    import numpy as np
    if len(rows)<30:raise ValueError("At least 30 time samples are needed")
    # Ignore first .5 s settling; exclude reset scenes from gait metrics.
    samples=[r for r in rows if r["t_s"]>=.5 and r["done"]==0]
    if not samples:raise ValueError("No valid steady-state samples")
    la=[r["left_contact_N"]>45. for r in samples]
    ra=[r["right_contact_N"]>45. for r in samples]
    single_left=sum(l and not rr for l,rr in zip(la,ra))/len(la)
    single_right=sum(rr and not l for l,rr in zip(la,ra))/len(la)
    double=sum(l and rr for l,rr in zip(la,ra))/len(la)
    flight=sum(not l and not rr for l,rr in zip(la,ra))/len(la)
    out={
        "valid_seconds":round(len(samples)*step_dt,3),
        "command_forward_mps":desired_speed,
        "root_forward_speed_mean_mps":float(np.mean([r["root_vx"] for r in samples])),
        "root_forward_velocity_mae_mps":float(np.mean([abs(r["root_vx"]-desired_speed) for r in samples])),
        "root_height_median_m":percentile([r["root_z"] for r in samples],.5),
        "root_height_min_m":min(r["root_z"] for r in samples),
        "torso_upright_tilt_deg_median":percentile([r["root_tilt_deg"] for r in samples],.5),
        "torso_upright_tilt_deg_p95":percentile([r["root_tilt_deg"] for r in samples],.95),
        "left_foot_contact_fraction":sum(la)/len(la),
        "right_foot_contact_fraction":sum(ra)/len(ra),
        "double_support_fraction":double,
        "left_only_support_fraction":single_left,
        "right_only_support_fraction":single_right,
        "no_support_fraction":flight,
        "foot_contact_threshold_N":45.,
        "joint_knee_angle_rad_note":"values are simulator joint coordinates; map to anatomical knee-flexion convention before cross-robot/human comparison",
        "foot_slip_metric_note":"foot LINK horizontal velocity when link carries >=45N; not precise sole-patch slip",
    }
    for side,grounded in (("left",la),("right",ra)):
        stance_q=[abs(r[f"{side}_knee"]) for r,flag in zip(samples,grounded) if flag]
        swing_q=[abs(r[f"{side}_knee"]) for r,flag in zip(samples,grounded) if not flag]
        out[f"{side}_knee_abs_stance_median_rad"]=percentile(stance_q,.5)
        out[f"{side}_knee_abs_swing_median_rad"]=percentile(swing_q,.5)
        out[f"{side}_knee_abs_stance_p90_rad"]=percentile(stance_q,.9)
        out[f"{side}_knee_abs_swing_p90_rad"]=percentile(swing_q,.9)
        out[f"{side}_foot_link_slip_speed_p90_mps"]=percentile(
            [r[f"{side}_foot_xy_speed"] for r,flag in zip(samples,grounded) if flag],.9)
        out[f"{side}_foot_swing_clearance_proxy_median_m"]=percentile(
            [r[f"{side}_foot_z"] for r,flag in zip(samples,grounded) if not flag],.5)
        out[f"{side}_stance_bout_lengths_s"]=[
            round((end-start)*step_dt,3)
            for start,end in contiguous_segments(grounded) if (end-start)>=2]
        out[f"{side}_swing_bout_lengths_s"]=[
            round((end-start)*step_dt,3)
            for start,end in contiguous_segments(not x for x in grounded) if (end-start)>=2]
    out["left_right_hip_pitch_correlation"]=correlation(
        [r["left_hip_pitch"] for r in samples],
        [r["right_hip_pitch"] for r in samples])
    out["left_right_shoulder_pitch_correlation"]=correlation(
        [r["left_shoulder_pitch"] for r in samples],
        [r["right_shoulder_pitch"] for r in samples])
    out["left_hip_vs_right_shoulder_correlation"]=correlation(
        [r["left_hip_pitch"] for r in samples],
        [r["right_shoulder_pitch"] for r in samples])
    out["right_hip_vs_left_shoulder_correlation"]=correlation(
        [r["right_hip_pitch"] for r in samples],
        [r["left_shoulder_pitch"] for r in samples])
    out["observed_episode_resets"]=sum(r["done"] for r in rows)
    out["metrics_cannot_alone_prove_human_likeness"]=True
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=str,default=str(CKPT))
    parser.add_argument("--vx",type=float,default=.5)
    parser.add_argument("--steps",type=int,default=430)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--output",type=str,default="reports/human_walk_baseline_v050.csv")
    args=parser.parse_args()
    if not (0<=args.vx<=1.0): raise ValueError("--vx must be in [0,1]")
    checkpoint=Path(args.checkpoint)
    if not checkpoint.exists():raise FileNotFoundError(checkpoint)
    cfg=G1FlatEnvCfg_PLAY()
    cfg.scene.num_envs=1
    cfg.seed=args.seed
    cfg.sim.device="cuda:0"
    cfg.episode_length_s=32.
    cfg.commands.base_velocity.debug_vis=False
    cfg.commands.base_velocity.debug_vis=False
    cfg.commands.base_velocity.ranges.lin_vel_x=(args.vx,args.vx)
    cfg.commands.base_velocity.ranges.lin_vel_y=(0.,0.)
    cfg.commands.base_velocity.ranges.ang_vel_z=(0.,0.)
    cfg.commands.base_velocity.ranges.heading=(0.,0.)
    cfg.commands.base_velocity.heading_command=False
    cfg.commands.base_velocity.rel_standing_envs=0.0
    cfg.commands.base_velocity.resampling_time_range=(100.,100.)
    cfg.observations.policy.enable_corruption=False
    ppo=G1FlatPPORunnerCfg()
    raw=gym.make("Isaac-Velocity-Flat-G1-Play-v0",cfg=cfg)
    env=RslRlVecEnvWrapper(raw,clip_actions=ppo.clip_actions)
    runner=OnPolicyRunner(env,ppo.to_dict(),log_dir=None,device="cuda:0")
    runner.load(str(checkpoint),load_optimizer=False)
    policy=runner.get_inference_policy(device="cuda:0")
    robot=env.unwrapped.scene["robot"]
    cf=env.unwrapped.scene["contact_forces"]
    ji={name:i for i,name in enumerate(robot.joint_names)}
    bi={name:i for i,name in enumerate(robot.body_names)}
    ci={name:i for i,name in enumerate(cf.body_names)}
    assert all(k in ji for k in TRACKED_JOINTS)
    assert all(k in bi and k in ci for k in FOOT_LINKS)
    obs,_=env.get_observations()
    step_dt=env.unwrapped.step_dt
    rows=[]
    start=robot.data.root_pos_w[0].detach().cpu().tolist()
    for step in range(args.steps):
        with torch.inference_mode():
            act=policy(obs)
            obs,reward,done,extra=env.step(act)
        q=robot.data.joint_pos[0].detach().cpu().tolist()
        xyz=robot.data.root_pos_w[0].detach().cpu().tolist()
        vel=robot.data.root_lin_vel_w[0].detach().cpu().tolist()
        grav=robot.data.projected_gravity_b[0].detach().cpu().tolist()
        row={
            "step":step,"t_s":round(step*step_dt,5),
            "root_x":float(xyz[0]),"root_y":float(xyz[1]),
            "root_z":float(xyz[2]),"root_vx":float(vel[0]),
            "root_vy":float(vel[1]),
            "root_tilt_deg":math.degrees(math.acos(max(-1.,min(1.,-grav[2])))),
            "done":int(done[0].item()),
        }
        for joint in TRACKED_JOINTS:
            prefix=joint.replace("_joint","")
            row[prefix]=float(q[ji[joint]])
        fv=robot.data.body_lin_vel_w[0].detach().cpu().tolist()
        fp=robot.data.body_pos_w[0].detach().cpu().tolist()
        forces=cf.data.net_forces_w[0].detach().cpu().tolist()
        for side in ("left","right"):
            link=side+"_ankle_roll_link"
            row[f"{side}_foot_xy_speed"]=math.hypot(*fv[bi[link]][:2])
            row[f"{side}_foot_z"]=float(fp[bi[link]][2])
            row[f"{side}_contact_N"]=math.sqrt(sum(x*x for x in forces[ci[link]]))
        rows.append(row)
        if step%90==0 or row["done"]:
            print("GAIT_DIAG_STEP",step,
                  "root",[round(x,3) for x in xyz],
                  "q_knee",[round(row[f"{s}_knee"],3) for s in ("left","right")],
                  "contact",[round(row[f"{s}_contact_N"]) for s in ("left","right")],
                  "done",row["done"],flush=True)
        if row["done"]:
            # No silently mixing a physically reset episode into a single gait cycle.
            print("GAIT_DIAG_TERMINATED",step,flush=True)
            break
    output=ROOT/args.output;output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="") as file:
        wr=csv.DictWriter(file,fieldnames=list(rows[0].keys()))
        wr.writeheader();wr.writerows(rows)
    metrics=compute_metrics(rows,step_dt,args.vx)
    metrics["seed"]=args.seed
    metrics["steps_recorded"]=len(rows)
    metrics["root_forward_displacement_m"]=float(rows[-1]["root_x"]-start[0])
    metrics["joint_mapping"]= {name:ji[name] for name in TRACKED_JOINTS}
    metrics["body_mapping"]= {name:bi[name] for name in FOOT_LINKS}
    metrics["checkpoint"]=str(checkpoint)
    metrics["source_rows_csv"]=str(output)
    summary=output.with_suffix(".json")
    summary.write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
    print("GAIT_DIAG_COMPLETED",json.dumps(metrics,ensure_ascii=False),flush=True)
    env.close()

if __name__=="__main__":
    try:main()
    finally:app.close()

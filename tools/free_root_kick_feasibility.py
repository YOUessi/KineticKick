"""First actual free-root football strike feasibility test using learned G1 gait + scripted limb action.

An engineering prototype, NOT a trained human-like soccer skill.
Requires accurate G1 37DoF joint action mapping; computes action=(q_desired-q_default)/0.5.
The pre-trained PPO controls other joints and recovery. Ball is real PhysX rigid body.
Run headless; reports clear failure if no FOOT-BALL physical contact.
"""
from __future__ import annotations
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True).app
import argparse,json,math
from pathlib import Path
import gymnasium as gym
import torch
import isaaclab_tasks
import isaaclab.sim as sim
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg_PLAY
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg

ROOT=Path("/home/you/projects/kinetickick")
WEIGHTS=ROOT/".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-G1-v0/checkpoint.pt"
DT=.02

def ease(x):
    x=min(1.,max(0.,x))
    return x*x*(3-2*x)
def create(ball_x=.23,ball_y=-.12, num_envs=1):
    cfg=G1FlatEnvCfg_PLAY()
    cfg.scene.num_envs=num_envs
    cfg.seed=42
    cfg.episode_length_s=40.
    cfg.sim.device="cuda:0"
    cfg.observations.policy.enable_corruption=False
    cfg.events.reset_base.params["pose_range"]={
      "x":(0.,0.),"y":(0.,0.),"yaw":(0.,0.)}
    cfg.events.reset_base.params["velocity_range"]={
      "x":(0.,0.),"y":(0.,0.),"z":(0.,0.),
      "roll":(0.,0.),"pitch":(0.,0.),"yaw":(0.,0.)}
    cfg.commands.base_velocity.ranges.lin_vel_x=(0.,0.)
    cfg.commands.base_velocity.ranges.lin_vel_y=(0.,0.)
    cfg.commands.base_velocity.ranges.ang_vel_z=(0.,0.)
    cfg.commands.base_velocity.heading_command=False
    cfg.commands.base_velocity.ranges.heading=(0.,0.)
    cfg.commands.base_velocity.resampling_time_range=(100.,100.)
    cfg.scene.football=RigidObjectCfg(
       prim_path="{ENV_REGEX_NS}/SoccerBall",
       spawn=sim.SphereCfg(
        radius=.11,
        collision_props=sim.CollisionPropertiesCfg(),
        rigid_props=sim.RigidBodyPropertiesCfg(),
        mass_props=sim.MassPropertiesCfg(mass=.43)),
       init_state=RigidObjectCfg.InitialStateCfg(pos=(ball_x,ball_y,.113)),
    )
    cfg.scene.foot_ball_sensor=ContactSensorCfg(
       prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
       update_period=0.,history_length=2,
       filter_prim_paths_expr=["{ENV_REGEX_NS}/SoccerBall"])
    runner_cfg=G1FlatPPORunnerCfg()
    env=RslRlVecEnvWrapper(
        gym.make("Isaac-Velocity-Flat-G1-Play-v0",cfg=cfg),
        clip_actions=runner_cfg.clip_actions)
    runner=OnPolicyRunner(env,runner_cfg.to_dict(),log_dir=None,device="cuda:0")
    runner.load(str(WEIGHTS))
    policy=runner.get_inference_policy(device="cuda:0")
    return env,policy

def execute(ball_x,ball_y,magnitude=1.0, total_steps=160):
    env,policy=create(ball_x,ball_y)
    raw=env.unwrapped
    robot=raw.scene["robot"]
    ball=raw.scene["football"]
    term=raw.action_manager.get_term("joint_pos")
    names=list(term._joint_names)
    ids={n:i for i,n in enumerate(names)}
    expected=["right_hip_pitch_joint","right_knee_joint","right_ankle_pitch_joint"]
    assert all(x in ids for x in expected),names
    assert len(names)==37 and len(robot.joint_names)==37
    assert abs(term._scale-.5)<1.e-6
    print("G1_KICK_ACTUATED_JOINTS",names,flush=True)
    sensor=raw.scene["foot_ball_sensor"]
    cmd=raw.command_manager.get_term("base_velocity")
    obs,_=env.get_observations()
    print("PROBE_OBS_OK",tuple(obs.shape),"ball",ball.data.root_pos_w[0].cpu().tolist(),flush=True)
    # Action-manager joint order is DIFFERENT from robot data order. Resolve
    # each name explicitly; CUDA-to-host reads are done ONCE, never each step.
    names_robot={name:i for i,name in enumerate(robot.joint_names)}
    q0_cpu=robot.data.default_joint_pos[0].detach().cpu().tolist()
    limits_cpu=robot.data.soft_joint_pos_limits[0].detach().cpu().tolist()
    targets_info={name:(ids[name],q0_cpu[names_robot[name]],
                          limits_cpu[names_robot[name]]) for name in expected}
    print("G1_KICK_CORRECT_ACTION_JOINT_MAPPING",targets_info,flush=True)
    phase_states=[]
    first_hit=None
    max_force=0.
    max_hspeed=0.
    min_base_z=99.
    reset_count=0
    root_start=robot.data.root_pos_w[0].detach().cpu().tolist()
    for step in range(total_steps):
        if step<2: print("PROBE_START_ITERATION",step,flush=True)
        t=step*DT
        cmd.vel_command_b[:]=0.
        cmd.is_standing_env[:]=False
        cmd.time_left[:]=1000.
        # Inference tensors cannot be modified inplace outside InferenceMode.
        # Clone OUTSIDE it to create a writable standard CUDA tensor for
        # policy+strike action residual blending.
        with torch.inference_mode():
            inf_actions=policy(obs)
        actions=inf_actions.clone()
        if step<2: print("PROBE_POLICY_FORWARD_DONE",step,flush=True)
        # Phase: neutral stance->pre-pose->upper-leg swing->relax/recover.
        # Actual ball motion depends on PhysX contact; no ball teleport/velocity injection.
        if t<.35:
            amount=0.0
            hp=.20
            knee=1.20
            ankle=.64
        elif t<.75:
            amount=ease((t-.35)/.40)
            hp=.20
            knee=1.20
            ankle=.64
        elif t<1.05:
            amount=1.
            hp=.20
            knee=1.20
            ankle=.64
        elif t<1.34:
            amount=1.
            hh=ease((t-1.05)/.29)
            kk=ease((t-1.13)/.21)
            hp=.20+(-1.05-.20)*hh*magnitude
            knee=1.20+(.80-1.20)*kk*magnitude
            ankle=.64
        elif t<1.67:
            amount=1.
            hp=.20+(-1.05-.20)*magnitude
            knee=1.20+(.80-1.20)*magnitude
            ankle=.64
        elif t<2.08:
            amount=1.-ease((t-1.67)/.41)
            hp=.20+(-1.05-.20)*magnitude
            knee=1.20+(.80-1.20)*magnitude
            ankle=.64
        else:
            amount=0.
            hp=0.;knee=0.;ankle=0.
        if step<2:print("PROBE_PRE_OVERRIDES",step,"amount",amount,flush=True)
        for name,target in zip(expected,[hp,knee,ankle]):
            if amount<=0.0001:
                continue  # avoid any CUDA action override when pure gait policy
            if step<2:print("PROBE_OVERRIDE_START",step,name,flush=True)
            i,ref,(lo,hi)=targets_info[name]
            v=max(float(lo)+.01,min(float(hi)-.01,float(target)))
            desired=(v-ref)/.5
            actions[:,i]=(1-amount)*actions[:,i]+amount*desired
            if step<2:print("PROBE_OVERRIDE_DONE",step,name,flush=True)
        if step<2: print("PROBE_BEFORE_PHYSICS_STEP",step,flush=True)
        with torch.inference_mode():
            obs,_,dones,_=env.step(actions)
        if step<2: print("PROBE_AFTER_PHYSICS_STEP",step,flush=True)
        f=sensor.data.force_matrix_w
        force=float(torch.linalg.vector_norm(f).item()) if f is not None and f.numel() else 0.
        max_force=max(max_force,force)
        v=ball.data.root_lin_vel_w[0].detach().cpu().tolist()
        hs=math.hypot(v[0],v[1])
        max_hspeed=max(hs,max_hspeed)
        base=robot.data.root_pos_w[0].detach().cpu().tolist()
        min_base_z=min(min_base_z,base[2])
        done=bool(dones[0].item())
        if done:reset_count+=1
        if force>2. and first_hit is None:
            first_hit={"step":step,"t":round(t,3),
                       "vxyz":[round(x,3) for x in v],
                       "force_N":round(force,2),
                       "root_xyz":[round(x,3) for x in base]}
            print("FREE_ROOT_BALL_IMPACT",json.dumps(first_hit),flush=True)
        if step%12==0 or done:
            print("FREE_ROOT_STEP",step,
                 "phase",("swing" if 1.05<t<1.67 else ("prepose" if .35<t<1.05 else "walk_stance")),
                 "root",[round(x,3) for x in base],
                 "ball",[round(x,3) for x in ball.data.root_pos_w[0].cpu().tolist()],
                 "ball_v",[round(x,3) for x in v],
                 "force",round(force,2),"done",done,flush=True)
        if done:break
    final=ball.data.root_pos_w[0].detach().cpu().tolist()
    summary={
       "test":"UNTRAINED_INTEGRATION_FREE_ROOT",
       "ball_start":[ball_x,ball_y,.113],"root_start":root_start,
       "first_contact":first_hit,
       "max_filtered_contact_force_N":round(max_force,3),
       "max_ball_horizontal_speed_mps":round(max_hspeed,3),
       "min_root_height_m":round(min_base_z,4),
       "reset_count":reset_count,
       "ball_final_xyz":[round(x,3) for x in final],
       "sim_steps":step+1,
       "conclusion":"not a trained shot; explicit control intervention under free root, compare stability"
    }
    p=ROOT/"reports/free_root_kick_feasibility.json"
    p.write_text(json.dumps(summary,indent=2))
    print("FREE_ROOT_KICK_TEST_DONE",json.dumps(summary),flush=True)
    env.close()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--ball-x",type=float,default=.23)
    parser.add_argument("--ball-y",type=float,default=-.12)
    parser.add_argument("--intensity",type=float,default=.7)
    parser.add_argument("--steps",type=int,default=160)
    p=parser.parse_args()
    execute(p.ball_x,p.ball_y,p.intensity,total_steps=p.steps)

if __name__=="__main__":
    try:main()
    finally:app.close()

#!/usr/bin/env python3
"""KineticKick: REAL, free-root PAiD 29DoF G1 physical policy live viewer.

Runs the saved model in the exact Stage-I single-condition Isaac Lab tracking
task. The control is learned PPO, not hand-authored animation, root-fixing,
direct ball velocity injection, or a kinematic human-reference viewer.

No ball is present: this is PHYSICAL MOTION TRACKING, NOT COMPLETED SOCCER.
"""
from __future__ import annotations
import argparse
import csv
import json
import pathlib
import sys
import time
import traceback
import types

parser = argparse.ArgumentParser()
parser.add_argument("--max-steps", type=int, default=0,
                    help="0 = show GUI until closed; positive = bounded smoke")
parser.add_argument("--checkpoint", choices=["BEST", "PHASE", "BASELINE"], default="BEST")
cli = parser.parse_args()

from isaaclab.app import AppLauncher
app = AppLauncher(headless=False, enable_cameras=False).app

import gymnasium as gym
import omni.ui as ui
import torch
import isaaclab.sim as sim
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

ROOT = pathlib.Path("/home/you/projects/kinetickick")
UPSTREAM = ROOT / "third_party/HumanoidSoccer/source/whole_body_tracking"
URDF = ROOT / "configs/paid_original/paid_main_resolved.urdf"
MOTION = ROOT / "third_party/HumanoidSoccer/motions/soccer-standard/soccer-standard-001_right.npz"
WEIGHTS = {
    "BEST": ROOT / "experiments/paid29_stage1_contact_phase/seed84_n96_iter200_phase108_148_contactv1/last_model.pt",
    "PHASE": ROOT / "experiments/paid29_stage1_phase_focus/seed84_n96_iter200_phase108_148_zero0p4/last_model.pt",
    "BASELINE": ROOT / "experiments/paid29_stage1_original/seed74_n64_iter160/last_model.pt",
}
for key, path in WEIGHTS.items():
    if not path.is_file():
        raise FileNotFoundError(f"{key}: {path}")

sys.path.insert(0, str(UPSTREAM))
shim = types.ModuleType("soccer.assets")
shim.ASSET_DIR = str(UPSTREAM/"soccer/assets")
sys.modules["soccer.assets"] = shim

class Panel:
    def __init__(self, state):
        self.window = ui.Window("KineticKick | G1 PHYSICAL PPO LIVE", width=560, height=355, visible=True)
        self.window.position_x = 845
        self.window.position_y = 70
        with self.window.frame:
            with ui.VStack(spacing=6):
                ui.Label("G1 29DoF PHYSICS-BASED MOTION TRACKING", height=24,
                         style={"font_size":16})
                ui.Label("Learned PPO | Floating base | Gravity + ground contacts")
                ui.Label("NO FOOTBALL: Stage I kick-motion tracking, NOT physical shooting",
                         style={"color":0xFF55B8FF})
                with ui.HStack(height=28):
                    ui.Button("BEST 260/260", clicked_fn=lambda:self.choose(state,"BEST"))
                    ui.Button("PHASE 143/260", clicked_fn=lambda:self.choose(state,"PHASE"))
                    ui.Button("EARLY BASELINE", clicked_fn=lambda:self.choose(state,"BASELINE"))
                with ui.HStack(height=28):
                    ui.Button("PLAY/PAUSE", clicked_fn=lambda:self.pause(state))
                    ui.Button("RESTART", clicked_fn=lambda:self.restart(state))
                    ui.Button("SLOW", clicked_fn=lambda:self.pacing(state,"SLOW"))
                    ui.Button("NORMAL", clicked_fn=lambda:self.pacing(state,"NORMAL"))
                with ui.HStack(height=28):
                    ui.Button("SIDE CAMERA",clicked_fn=lambda:self.cam(state,"SIDE"))
                    ui.Button("THREE-QUARTER",clicked_fn=lambda:self.cam(state,"ANGLE"))
                    ui.Button("FRONT CAMERA",clicked_fn=lambda:self.cam(state,"FRONT"))
                self.model=ui.Label("CHECKPOINT: starting",height=22)
                self.clock=ui.Label("Reference: --/260 | --/5.20 s",height=22)
                self.root=ui.Label("Root -- | torso height error --",height=22)
                self.contacts=ui.Label("Support forces --",height=22)
                self.status=ui.Label("Initial physical rollout pending...",height=22)
                ui.Label("Unperturbed 1-condition preview; visual switching is NOT an A/B study.")
    @staticmethod
    def choose(s,key):
        s["requested_checkpoint"]=key
    @staticmethod
    def pause(s):
        s["paused"]=not s["paused"]
    @staticmethod
    def restart(s):
        s["restart"]=True
    @staticmethod
    def pacing(s,value):
        s["pace"]=value
    @staticmethod
    def cam(s,value):
        s["camera"]=value
        s["camera_changed"]=True

def main():
    import soccer.tasks
    from soccer.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
    from soccer.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    from soccer_field_visual import create_field_visuals

    cfg=G1FlatEnvCfg()
    cfg.scene.num_envs=1
    cfg.scene.env_spacing=2.7
    cfg.sim.device="cuda:0"
    cfg.seed=501
    cfg.commands.motion.motion_files=[str(MOTION)]
    cfg.commands.motion.debug_vis=False
    cfg.scene.contact_forces.debug_vis=False
    cfg.scene.robot.spawn.asset_path=str(URDF)
    cfg.scene.robot.spawn.fix_base=False
    cfg.scene.terrain.visual_material=None
    cfg.scene.sky_light.spawn=sim.DomeLightCfg(intensity=1100)
    cfg.observations.policy.enable_corruption=False
    # Strict frozen reference condition; train rewards are not changed.
    for key in cfg.commands.motion.pose_range:
        cfg.commands.motion.pose_range[key]=(0.,0.)
    for key in cfg.commands.motion.velocity_range:
        cfg.commands.motion.velocity_range[key]=(0.,0.)
    cfg.commands.motion.joint_position_range=(0.,0.)
    cfg.terminations.ee_body_pos.params["threshold"]=0.25
    cfg.events.physics_material=None
    cfg.events.add_joint_default_pos=None
    cfg.events.base_com=None
    cfg.events.base_external_force_torque=None
    cfg.events.push_robot=None

    agent=G1FlatPPORunnerCfg()
    agent.num_steps_per_env=24
    env=RslRlVecEnvWrapper(gym.make("Tracking-Flat-G1-v0",cfg=cfg),
                           clip_actions=agent.clip_actions)
    raw=env.unwrapped
    robot=raw.scene["robot"]
    motion=raw.command_manager.get_term("motion")
    sensor=raw.scene["contact_forces"]
    foot_sensor_ids=[sensor.body_names.index(k) for k in
                     ("left_ankle_roll_link","right_ankle_roll_link")]
    assert len(robot.joint_names)==29 and len(robot.body_names)==30
    print("LIVE_29DOF_PHYSICAL_ROBOT_READY",True,"root_fixed",cfg.scene.robot.spawn.fix_base,flush=True)
    create_field_visuals()
    for wname in ("Simulation Settings","Content","Console","Isaac Sim Assets [Beta]"):
        pane=ui.Workspace.get_window(wname)
        if pane is not None:
            pane.visible=False
    state={
        "model":cli.checkpoint,"requested_checkpoint":None,
        "paused":False,"restart":True,"camera":"ANGLE","camera_changed":True,
        "pace":"NORMAL","episode":0,"steps":0,"show_failure":""
    }
    panel=Panel(state)
    runner=OnPolicyRunner(env,agent.to_dict(),log_dir=None,device="cuda:0")
    policy=None
    obs=None
    step=0
    absolute_steps=0
    last_episode_summary=None
    out=ROOT/"reports/g1_paid_stage1_live_physics_health.json"
    state["requested_checkpoint"]=cli.checkpoint
    while app.is_running():
        if state["requested_checkpoint"] is not None:
            name=state["requested_checkpoint"]
            state["requested_checkpoint"]=None
            runner.load(str(WEIGHTS[name]),load_optimizer=False)
            policy=runner.get_inference_policy(device="cuda:0")
            state["model"]=name
            state["restart"]=True
            print("LIVE_POLICY_LOADED",name,str(WEIGHTS[name]),flush=True)
        if state["restart"]:
            obs,_=env.reset()
            # Critical: manager's body reference cache is zero after reset until update.
            motion.time_steps[:]=-1
            motion._update_command()
            assert int(motion.time_steps[0])==0
            step=0
            state["episode"]+=1
            state["restart"]=False
            state["show_failure"]="New physical episode"
            print("LIVE_PHYSICS_EPISODE_START",state["episode"],
                  state["model"],"reference_frame",int(motion.time_steps[0]),flush=True)
        if state["camera_changed"] or absolute_steps%15==0:
            xyz=robot.data.root_pos_w[0].detach().cpu().tolist()
            if state["camera"]=="SIDE":
                eye=[xyz[0]+0.40,xyz[1]-3.10,1.37]
            elif state["camera"]=="FRONT":
                eye=[xyz[0]+3.30,xyz[1]+0.15,1.28]
            else:
                eye=[xyz[0]+1.90,xyz[1]-2.50,1.55]
            raw.sim.set_camera_view(eye=eye,target=[xyz[0],xyz[1],.72])
            state["camera_changed"]=False
        if state["paused"]:
            app.update()
            if cli.max_steps and absolute_steps>=cli.max_steps:
                break
            time.sleep(.03)
            continue
        if step>=260:
            state["show_failure"]="FULL 260/260 tracking segment completed; resetting"
            last_episode_summary={"episode":state["episode"],"model":state["model"],
                "reference_steps":260,"complete":True,"physical_ppo":True,
                "ball_present":False,"time":time.strftime("%Y-%m-%d %H:%M:%S")}
            out.write_text(json.dumps(last_episode_summary,indent=2))
            print("LIVE_PHYSICS_FULL_SEGMENT",json.dumps(last_episode_summary),flush=True)
            panel.status.text="Completed full 260/260 physical tracking. Replaying..."
            app.update()
            time.sleep(.45)
            state["restart"]=True
            continue
        reference_idx=int(motion.time_steps[0])
        torso_z_ref=float(motion.anchor_pos_w[0,2])
        torso_z_actual=float(motion.robot_anchor_pos_w[0,2])
        # Use no_grad rather than inference_mode for Isaac Lab env.step:
        # the latter creates inference tensors in manager buffers, which
        # cannot be reset in-place outside inference_mode on replay.
        with torch.no_grad():
            act=policy(obs)
            obs,reward,done,extras=env.step(act)
        step+=1
        absolute_steps+=1
        if absolute_steps%10==0 or bool(done[0]):
            forces=sensor.data.net_forces_w[0,foot_sensor_ids,2].detach().cpu().tolist()
            root=robot.data.root_pos_w[0].detach().cpu().tolist()
            err=abs(torso_z_ref-torso_z_actual)
            panel.model.text=f"CHECKPOINT: {state['model']} | Episode #{state['episode']} | {state['pace']}"
            panel.clock.text=f"PHYSICAL CONTROL STEP: {step}/260 | clip {reference_idx/50:.2f}/5.20s"
            panel.root.text=f"ROOT z {root[2]:.3f}m | TORSO z-error {err:.3f}m"
            panel.contacts.text=f"L/R vertical foot load {forces[0]:.0f}N / {forces[1]:.0f}N"
            panel.status.text=state["show_failure"]
            print("LIVE_POLICY_TICK","model",state["model"],"step",step,
                  "root_z",round(root[2],3),"torso_dz",round(err,3),
                  "Fz",[round(x,1) for x in forces],flush=True)
        if bool(done[0]):
            causes={k:bool(raw.termination_manager.get_term(k)[0])
                    for k in raw.termination_manager.active_terms}
            state["show_failure"]=f"Reset at {step}/260, reason {', '.join(k for k,v in causes.items() if v)}"
            last_episode_summary={"episode":state["episode"],"model":state["model"],
                "reference_steps":step,"complete":False,"causes":causes,
                "physical_ppo":True,"ball_present":False}
            out.write_text(json.dumps(last_episode_summary,indent=2))
            print("LIVE_PHYSICS_EARLY_RESET",json.dumps(last_episode_summary),flush=True)
            app.update()
            time.sleep(.35)
            state["restart"]=True
        if cli.max_steps and absolute_steps>=cli.max_steps:
            break
        time.sleep(.027 if state["pace"]=="SLOW" else .005)
    print("LIVE_POLICY_VIEWER_FINISHED","total_steps",absolute_steps,flush=True)
    env.close()

if __name__=="__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("LIVE_POLICY_VIEWER_FAILED",flush=True)
        raise
    finally:
        app.close()

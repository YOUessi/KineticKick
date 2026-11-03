"""Visible Isaac Lab demo: Unitree G1, soccer-size rigid ball, and basic joint actuation.

WARNING: anchored-base model viewer and scripted test animation.
NOT an RL-trained soccer/kicking/stability policy, and NOT PAiD's 29-DoF G1.
"""
import math
import time

from isaaclab.app import AppLauncher

app = AppLauncher(headless=False, enable_cameras=False).app

import omni.ui as ui
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab_assets.robots.unitree import G1_CFG


def scene():
    ground = sim_utils.GroundPlaneCfg(color=(0.24, 0.38, 0.23))
    ground.func("/World/TurfGround", ground)
    light = sim_utils.DomeLightCfg(intensity=2400.0, color=(0.93, 0.94, 1.0))
    light.func("/World/SceneSky", light)
    sun = sim_utils.DistantLightCfg(intensity=1200.0, color=(1.0, 0.95, 0.82))
    sun.func("/World/SceneSun", sun)

    ball = sim_utils.SphereCfg(
        radius=0.11,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.97, 0.81, 0.19),
            roughness=0.6,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.43),
    )
    ball.func("/World/SoccerBall", ball, translation=(0.58, -0.16, 0.11))

    cfg = G1_CFG.copy()
    # Only to demonstrate a standing asset and actuated joints without a policy.
    # Root anchoring is prohibited for actual balance/soccer success experiments.
    cfg.spawn.articulation_props.fix_root_link = True
    cfg.prim_path = "/World/UnitreeG1"
    robot = Articulation(cfg)
    return robot


def main():
    sim = SimulationContext(SimulationCfg(dt=1 / 60.0))
    sim.set_camera_view(eye=[2.20, -3.40, 1.36], target=[0.12, 0.0, 0.79])
    robot = scene()
    sim.reset()
    print("KINETICKICK_GUI_READY: G1 and soccer ball visible in Isaac Sim", flush=True)
    print("KINETICKICK_G1", len(robot.joint_names), len(robot.body_names), flush=True)

    # Save real USD scene for reopening (external USD mesh references may need Nucleus).
    try:
        target = "/home/you/projects/kinetickick/media/kinetickick_G1_stage.usda"
        ok = omni.usd.get_context().get_stage().GetRootLayer().Export(target)
        print("KINETICKICK_STAGE_SAVED", target, ok, flush=True)
    except Exception as err:
        print("KINETICKICK_STAGE_SAVE_WARN", str(err)[:300], flush=True)

    try:
        popup = ui.Workspace.get_window("Simulation Settings")
        if popup:
            popup.visible = False
    except Exception:
        pass

    joint_ids = {name: idx for idx, name in enumerate(robot.joint_names)}
    dt = sim.get_physics_dt()
    step = 0
    while app.is_running():
        # Slight repetitive joint actuation only: not a natural football skill.
        q = robot.data.default_joint_pos.clone()
        phase = 2 * math.pi * (step * dt / 2.5)
        targets = {
            "right_hip_pitch_joint": 0.28 * math.sin(phase),
            "right_knee_joint": 0.22 * max(0.0, math.sin(phase)),
            "left_shoulder_pitch_joint": 0.11 * math.sin(phase),
            "right_shoulder_pitch_joint": -0.11 * math.sin(phase),
        }
        for name, delta in targets.items():
            if name in joint_ids:
                q[:, joint_ids[name]] += delta
        robot.set_joint_position_target(q)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        step += 1
        time.sleep(0.01)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()

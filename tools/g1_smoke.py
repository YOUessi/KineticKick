"""Boot a G1 articulation and export simulator joint/body ordering.

This is a smoke test, not a trained soccer policy or PAiD schema proof.
"""
import json
from pathlib import Path

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab_assets.robots.unitree import G1_CFG

    sim = SimulationContext(SimulationCfg(dt=0.005, device="cuda:0"))
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    robot = Articulation(G1_CFG.replace(prim_path="/World/Robot"))
    sim.reset()
    for _ in range(10):
        sim.step()
    schema = {
        "description": "Built-in Isaac Lab G1 USD order, NOT yet verified for PAiD URDF NPZ",
        "joint_names": list(robot.joint_names),
        "body_names": list(robot.body_names),
        "num_joints": len(robot.joint_names),
        "num_bodies": len(robot.body_names),
    }
    path = Path("/home/you/projects/kinetickick/configs/g1_builtin_runtime_schema.json")
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print("KINETICKICK_G1_SMOKE_OK", len(robot.joint_names), len(robot.body_names), flush=True)
finally:
    app.close()

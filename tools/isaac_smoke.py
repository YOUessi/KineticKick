"""Minimal deterministic Isaac Lab headless sanity check; no training or data edits."""
from isaaclab.app import AppLauncher

launcher = AppLauncher(headless=True)
app = launcher.app

try:
    from isaaclab.sim import SimulationCfg, SimulationContext
    cfg = SimulationCfg(dt=0.02)
    sim = SimulationContext(cfg)
    sim.reset()
    for _ in range(8):
        sim.step()
    print("KINETICKICK_ISAAC_SMOKE_OK", flush=True)
finally:
    app.close()

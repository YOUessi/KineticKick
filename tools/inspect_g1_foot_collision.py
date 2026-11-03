"""Inspect 37-DoF G1's physical foot collider and local geometry on Tang."""
from isaaclab.app import AppLauncher
simulation_app=AppLauncher(headless=True).app
import json
from pathlib import Path
from pxr import UsdGeom, UsdPhysics
import omni.usd
import isaaclab.sim as sim
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg,SimulationContext
from isaaclab_assets.robots.unitree import G1_CFG
try:
    ctx=SimulationContext(SimulationCfg(dt=1/120))
    c=G1_CFG.copy()
    c.prim_path="/World/G1"
    c.spawn.articulation_props.fix_root_link=True
    r=Articulation(c)
    ctx.reset()
    stage=omni.usd.get_context().get_stage()
    cache=UsdGeom.BBoxCache(0.0,[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy],useExtentsHint=False)
    root=stage.GetPrimAtPath("/World/G1/right_ankle_roll_link")
    print("FOOT_ROOT_VALID",root.IsValid(),flush=True)
    if root.IsValid():
        output=[]
        for p in UsdGeom.PrimRange(root):
            path=str(p.GetPath())
            coll=bool(p.HasAPI(UsdPhysics.CollisionAPI))
            if coll or p==root or "foot" in path.lower() or "visuals" in path.lower():
                rec={"path":path,"type":p.GetTypeName(),"has_collision":coll}
                if coll:
                    try:
                        bb=cache.ComputeLocalBound(p).GetBox()
                        rec["bbox_min"]=list(bb.GetMin())
                        rec["bbox_max"]=list(bb.GetMax())
                    except Exception as e:rec["bbox_error"]=str(e)
                output.append(rec)
        print("FOOT_COLLISION_COUNT",sum(o["has_collision"] for o in output),flush=True)
        for obj in output[:55]: print("COLLIDER",json.dumps(obj),flush=True)
        Path("/home/you/projects/kinetickick/reports/g1_foot_collision_geometry.json").write_text(json.dumps(output,indent=2))
finally:
    simulation_app.close()

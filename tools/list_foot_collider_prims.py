"""List actual USD collision prims underneath built-in G1 foot without bounding calculation."""
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True).app
from pxr import UsdPhysics, UsdGeom
import omni.usd
import isaaclab.sim as sim
from isaaclab.assets import Articulation
from isaaclab_assets.robots.unitree import G1_CFG
from isaaclab.sim import SimulationCfg,SimulationContext
try:
    ctx=SimulationContext(SimulationCfg(dt=1/120))
    cfg=G1_CFG.copy()
    cfg.prim_path="/World/G1"
    cfg.spawn.articulation_props.fix_root_link=True
    bot=Articulation(cfg)
    ctx.reset()
    root=omni.usd.get_context().get_stage().GetPrimAtPath("/World/G1/right_ankle_roll_link")
    levels=[(root,0)]
    while levels:
        prim,depth=levels.pop(0)
        if depth>6:continue
        c=bool(prim.HasAPI(UsdPhysics.CollisionAPI))
        if c or depth<3:
            print("FOOT_PRIM",depth,str(prim.GetPath()),prim.GetTypeName(),"COLLISION",c,flush=True)
        if c and prim.GetTypeName()=="Mesh":
            mesh=UsdGeom.Mesh(prim)
            pts=mesh.GetPointsAttr().Get()
            if pts:
                dims=[[min(float(p[i]) for p in pts),max(float(p[i]) for p in pts)] for i in range(3)]
                print("FOOT_MESH_BOUNDS",str(prim.GetPath()),dims,"num_verts",len(pts),flush=True)
            xf=UsdGeom.Xformable(prim)
            print("FOOT_MESH_XFORM",[(str(op.GetOpName()),str(op.Get())) for op in xf.GetOrderedXformOps()],flush=True)
        levels.extend((p,depth+1) for p in prim.GetChildren())
finally:
    app.close()

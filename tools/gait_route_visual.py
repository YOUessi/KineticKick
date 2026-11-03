"""Visual A→B walking target for human-derived G1 straight gait reference.

Only render-only floor route geometry; no hidden steering controller,
physics actors, artificial root velocity, or added soccer balls.
"""
from __future__ import annotations


def create_straight_lane():
    from pxr import Gf,UsdGeom
    import omni.usd
    stage=omni.usd.get_context().get_stage()
    root_path="/World/StraightGaitRoute"
    root=UsdGeom.Xform.Define(stage,root_path)
    colors={
      "line":Gf.Vec3f(.91,.93,.86),
      "start":Gf.Vec3f(.22,.79,.51),
      "finish":Gf.Vec3f(1.00,.67,.09),
      "middle":Gf.Vec3f(.85,.87,.82),
    }
    def box(name,pos,scale,color):
        prim=UsdGeom.Cube.Define(stage,root_path+"/"+name)
        prim.CreateSizeAttr(1.)
        xf=UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xf.AddScaleOp().Set(Gf.Vec3f(*scale))
        prim.CreateDisplayColorAttr([colors[color]])
        return prim
    # Visual footpath line centered on the trajectory, not a steering command.
    box("CenterLine",(3.2,0.,.008),(7.3,.025,.003),"middle")
    box("StartLine",(0.,0.,.009),(.065,1.3,.005),"start")
    for i in range(1,7):
        box("MeterTick_"+str(i),(float(i),-.56,.009),(.045,.18,.005),"line")
    marker=UsdGeom.Xform.Define(stage,root_path+"/FinishMarker")
    mark_xf=UsdGeom.Xformable(marker)
    mark_xf.AddTranslateOp().Set(Gf.Vec3d(4.0,0.,0.))
    finish=UsdGeom.Cube.Define(stage,root_path+"/FinishMarker/FinishLine")
    finish.CreateSizeAttr(1.)
    fx=UsdGeom.Xformable(finish)
    fx.AddTranslateOp().Set(Gf.Vec3d(0.,0.,.012))
    fx.AddScaleOp().Set(Gf.Vec3f(.095,1.45,.007))
    finish.CreateDisplayColorAttr([colors["finish"]])
    print("GAIT_ROUTE_RENDER_ONLY","start_x",0.,
          "finish_dynamic",True,"added_colliders",0,"added_bodies",0,flush=True)
    return marker


def update_route_finish(finish_marker,world_x):
    from pxr import Gf,UsdGeom
    xf=UsdGeom.Xformable(finish_marker)
    trans=[x for x in xf.GetOrderedXformOps() if str(x.GetOpName())=="xformOp:translate"]
    if len(trans)!=1:raise RuntimeError("Finish marker transform is not initialized")
    trans[0].Set(Gf.Vec3d(float(world_x),0.,0.))

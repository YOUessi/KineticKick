"""World helpers for live KineticKick G1 research demos.

Imported only AFTER starting Isaac Sim with AppLauncher. All shoe patches are
PROTOTYPE geometry. Mechanical fit/actuation on a real G1 is not yet verified.
"""
from __future__ import annotations
import math

from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
import omni.usd

GOAL_X=3.15
GOAL_WIDTH=1.6
GOAL_HEIGHT=0.85

# All locations in right_ankle_roll_link frame. The original bare foot collider
# is a cube centered (0.0359, 0, -0.0252), size (0.2031,0.0655,0.0185).
# Shoe patches deliberately extend beyond its faces so their labels can be read
# from per-collider PhysX contact reports. Sizes approximate a robot sports shoe.
BOOT_PATCHES={
    "toe":((0.148, 0, -0.025),(.033,.085,.041),(.95,.80,.11)),
    "instep":((0.057, 0,-0.006),(.118,.078,.024),(.16,.48,.96)),
    "medial":((0.042, 0.046, -0.026),(.155,.022,.045),(.08,.85,.72)),
    "lateral":((0.042,-0.046, -0.026),(.155,.022,.045),(.99,.42,.15)),
    "heel":((-0.075,0,-0.026),(.035,.078,.043),(.27,.27,.31)),
}

def _cube(stage, path,xyz,scale,color,collision=True):
    geom=UsdGeom.Cube.Define(stage,path)
    geom.CreateSizeAttr(1.0)
    xf=UsdGeom.Xformable(geom.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*xyz))
    xf.AddScaleOp().Set(Gf.Vec3f(*scale))
    geom.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collision:
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    return geom.GetPrim()


def attach_training_football_shoe(root_path="/World/UnitreeG1/right_ankle_roll_link"):
    """Physically attach 5 separately named convex box colliders to G1 right foot.

    The foot body remains the same articulated rigid link, and the original
    barefoot collider remains active for conservative mass/support behavior.
    Multiple colliders can report which region was involved in a collision.
    """
    stage=omni.usd.get_context().get_stage()
    base=str(root_path)+"/KineticShoe"
    UsdGeom.Xform.Define(stage,base)
    foot_link=stage.GetPrimAtPath(root_path)
    report=PhysxSchema.PhysxContactReportAPI.Apply(foot_link)
    report.CreateThresholdAttr().Set(0.0)
    paths={}
    for label,(pos,size,color) in BOOT_PATCHES.items():
        path=f"{base}/Patch_{label}"
        _cube(stage,path,pos,size,color,collision=True)
        paths[label]=path
    print("KINETIC_SHOE_PATCHES",paths,flush=True)
    return paths


def create_goal(goal_x=GOAL_X, initial_y=0.0):
    stage=omni.usd.get_context().get_stage()
    root=UsdGeom.Xform.Define(stage,"/World/ResearchGoal")
    xform=UsdGeom.Xformable(root)
    xform.AddTranslateOp().Set(Gf.Vec3d(float(goal_x),float(initial_y),0.0))
    cobalt=(.06,.33,.92)
    _cube(stage,"/World/ResearchGoal/LeftPost",(0,-GOAL_WIDTH/2,GOAL_HEIGHT/2),(.075,.08,GOAL_HEIGHT),cobalt)
    _cube(stage,"/World/ResearchGoal/RightPost",(0,GOAL_WIDTH/2,GOAL_HEIGHT/2),(.075,.08,GOAL_HEIGHT),cobalt)
    _cube(stage,"/World/ResearchGoal/Crossbar",(0,0,GOAL_HEIGHT),(.075,GOAL_WIDTH+.08,.075),cobalt)
    # horizontal orientation markers to make the target obvious in 3D
    _cube(stage,"/World/ResearchGoal/LeftMarker",(-.03,-GOAL_WIDTH/2,.03),(.16,.12,.045),(.98,.90,.17),False)
    _cube(stage,"/World/ResearchGoal/RightMarker",(-.03,GOAL_WIDTH/2,.03),(.16,.12,.045),(.98,.90,.17),False)
    return root.GetPrim()


def move_goal(goal_prim,y):
    root=UsdGeom.Xformable(goal_prim)
    ops=root.GetOrderedXformOps()
    if not ops:
        raise RuntimeError("Goal root lacks authored transform")
    for op in ops:
        if str(op.GetOpName())=="xformOp:translate":
            # Preserve the goal X chosen at scene creation; older behavior
            # silently teleported all goals back to the default x=3.15 m.
            xyz=op.Get()
            op.Set(Gf.Vec3d(float(xyz[0]),float(y),float(xyz[2])))
            return
    raise RuntimeError("Goal root lacks translate op")


def locate_goal_in_usd(stage=None):
    """Scene-graph locator, NOT an RGB/camera-based goal recognizer."""
    if stage is None:stage=omni.usd.get_context().get_stage()
    root=stage.GetPrimAtPath("/World/ResearchGoal")
    if not root.IsValid():
        return None
    w=UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    xyz=w.ExtractTranslation()
    return (float(xyz[0]),float(xyz[1]),float(xyz[2])+GOAL_HEIGHT/2)


def classify_ball_crossing(previous_xyz,current_xyz,goal_xyz):
    """Check ball center crossing goal front plane; excludes post/crossbar collisions.

    Returns None if no x-plane crossing, bool otherwise. Accuracy is constrained
    to rigid sphere radius 0.11m (so use smaller usable width/height).
    """
    if previous_xyz is None:return None
    x=goal_xyz[0]
    dx=current_xyz[0]-previous_xyz[0]
    if dx<=1e-8 or previous_xyz[0]>x or current_xyz[0]<x:return None
    u=(x-previous_xyz[0])/dx
    y=previous_xyz[1]+u*(current_xyz[1]-previous_xyz[1])
    z=previous_xyz[2]+u*(current_xyz[2]-previous_xyz[2])
    ball_radius=.11
    return (abs(y-goal_xyz[1])< GOAL_WIDTH/2-ball_radius
            and ball_radius-.008<=z<GOAL_HEIGHT-ball_radius)


def estimated_goal_cross_y(name):
    """Calibrated rough crossing coordinate from early smoke tests;
    NOT trained vision; valid only at specified starting ball locations."""
    if name=="instep": return -.124
    if name=="inside":return -.558
    if name=="outside":return .105
    raise ValueError(name)


def choose_pose_by_goal(goal_xyz,mode_names=("instep","inside","outside")):
    """Preliminary model-based choice from prior physics rollouts.

    It reads a goal already located in 3D stage and selects nearest predicted
    ball crossing; no camera detection, no guarantee of scoring.
    """
    candidates={name:abs(estimated_goal_cross_y(name)-goal_xyz[1]) for name in mode_names}
    return min(candidates,key=candidates.get), candidates

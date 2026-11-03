"""KineticKick Stage 3A soccer scene: one ball, pitch and a visible real goal.

G1 in this stage is the existing 29DoF human-motion *kinematic* reference.
The ball is a separately instantiated PhysX rigid sphere (0.43kg, r=0.11m),
with two render-only black/white panel meshes sharing its rigid transform.
Goal frames have static collision geometry; turf/lines/net/ready mark are
render-only. No shoes, extra balls, waypoint spheres or robot actuators.
"""
from __future__ import annotations

from pxr import Gf,UsdGeom,UsdShade,UsdPhysics,Sdf
import omni.usd
import isaaclab.sim as sim
from isaaclab.assets import RigidObject,RigidObjectCfg
from classic_soccer_visual import attach_classic_soccer_visual

BALL_XY=(4.80,0.0)
READY_XY=(4.20,0.0)
GOAL_XY=(8.80,0.0)
BALL_RADIUS=.11
BALL_MASS=.43
GOAL_WIDTH=3.0
GOAL_HEIGHT=1.70

def scene_material(stage,base,name,rgb,roughness=.82):
    mat=UsdShade.Material.Define(stage,f"{base}/Mat_{name}")
    surf=UsdShade.Shader.Define(stage,f"{base}/Shader_{name}")
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    surf.CreateInput("roughness",Sdf.ValueTypeNames.Float).Set(float(roughness))
    surf.CreateInput("metallic",Sdf.ValueTypeNames.Float).Set(0.)
    mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(),"surface")
    return mat

def create_soccer_walkup_scene():
    stage=omni.usd.get_context().get_stage()
    root="/World/KineticKickSoccerField"
    UsdGeom.Xform.Define(stage,root)
    materials={
        "turf1":scene_material(stage,root,"turf1",(.033,.26,.105)),
        "turf2":scene_material(stage,root,"turf2",(.039,.31,.131)),
        "white":scene_material(stage,root,"white",(.96,.97,.95),.42),
        "ready":scene_material(stage,root,"ready",(1.00,.70,.08)),
        "start":scene_material(stage,root,"start",(.24,.88,.90)),
    }
    def block(path,pos,scale,material,physical=False):
        prim=UsdGeom.Cube.Define(stage,path)
        prim.CreateSizeAttr(1.)
        xf=UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xf.AddScaleOp().Set(Gf.Vec3f(*scale))
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(materials[material])
        if physical:
            UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
        return prim
    def deco(name,pos,scale,kind="white"):
        return block(f"{root}/{name}",pos,scale,kind)

    # Noncolliding turf overlays the physical ground plane (one ground collider).
    # Pitch lines are an athletic training diagram, not FIFA regulation layout.
    for k in range(20):
        deco(f"TurfStripe_{k:02d}",(-4.0+(k+.5)*1.0,0.,.0015),
             (1.005,12.,.003),"turf1" if k%2 else "turf2")
    deco("TouchlineLeft",(6.0,4.65,.006),(14.0,.035,.002))
    deco("TouchlineRight",(6.0,-4.65,.006),(14.0,.035,.002))
    deco("BackLine",(13.,0.,.006),(.035,9.30,.002))
    deco("FrontLine",(-1.,0.,.006),(.035,9.30,.002))
    deco("Halfway",(4.,0.,.006),(.030,9.30,.002))
    # Goal-area and penalty-box boundaries align with x=8.8 practice goal.
    deco("GoalLine",(GOAL_XY[0],0.,.007),(.030,9.3,.002))
    deco("PenaltyLineX",(6.65,0.,.007),(.030,5.1,.002))
    deco("PenaltyLineY1",(7.73,-2.55,.007),(2.15,.028,.002))
    deco("PenaltyLineY2",(7.73,+2.55,.007),(2.15,.028,.002))
    deco("ReadyLine",(READY_XY[0],0.,.012),(.08,.90,.005),"ready")
    deco("ReadyCenter",(READY_XY[0],0.,.015),(.42,.035,.004),"ready")
    # A is mode-dependent. Set it each time a new motion clip is loaded.
    start=block(root+"/StartLine",(-.45,0.,.012),(.07,.92,.005),"start")
    start_xf=UsdGeom.Xformable(start)
    start_translate=next(op for op in start_xf.GetOrderedXformOps() if "translate" in str(op.GetOpName()))

    goal=f"{root}/TrainingGoal"
    UsdGeom.Xform.Define(stage,goal)
    gw=GOAL_WIDTH/2
    blocks=[
        ("LeftUpright",(GOAL_XY[0],-gw,GOAL_HEIGHT/2),(.085,.085,GOAL_HEIGHT)),
        ("RightUpright",(GOAL_XY[0],gw,GOAL_HEIGHT/2),(.085,.085,GOAL_HEIGHT)),
        ("Crossbar",(GOAL_XY[0],0.,GOAL_HEIGHT),(.085,GOAL_WIDTH+.085,.085)),
        ("RearLeft",(GOAL_XY[0]+.64,-gw,.85),(.055,.055,GOAL_HEIGHT)),
        ("RearRight",(GOAL_XY[0]+.64,gw,.85),(.055,.055,GOAL_HEIGHT)),
        ("RearCrossbar",(GOAL_XY[0]+.64,0.,GOAL_HEIGHT),(.055,GOAL_WIDTH,.055)),
    ]
    for name,pos,size in blocks:
        block(f"{goal}/{name}",pos,size,"white",physical=True)
    net=UsdGeom.BasisCurves.Define(stage,goal+"/NetRenderOnly")
    pts=[];cnt=[]
    def netseg(a,b):
        pts.extend([Gf.Vec3f(*a),Gf.Vec3f(*b)])
        cnt.append(2)
    for i in range(21):
        y=-gw+GOAL_WIDTH*i/20
        netseg((GOAL_XY[0]+.64,y,0.),(GOAL_XY[0]+.64,y,GOAL_HEIGHT))
        netseg((GOAL_XY[0],y,GOAL_HEIGHT),(GOAL_XY[0]+.64,y,GOAL_HEIGHT))
    for i in range(13):
        z=GOAL_HEIGHT*i/12
        netseg((GOAL_XY[0]+.64,-gw,z),(GOAL_XY[0]+.64,gw,z))
    net.CreateTypeAttr(UsdGeom.Tokens.linear)
    net.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    net.CreatePointsAttr(pts)
    net.CreateCurveVertexCountsAttr(cnt)
    net.CreateWidthsAttr([.006])
    net.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    UsdShade.MaterialBindingAPI.Apply(net.GetPrim()).Bind(materials["white"])

    # A single physically meaningful soccer ball. Its base sphere gets a
    # classic 12-pentagon/20-hexagon *non-colliding* rendered skin.
    ball=RigidObject(RigidObjectCfg(
        prim_path="/World/MatchFootball",
        spawn=sim.SphereCfg(
            radius=BALL_RADIUS,
            rigid_props=sim.RigidBodyPropertiesCfg(),
            collision_props=sim.CollisionPropertiesCfg(),
            mass_props=sim.MassPropertiesCfg(mass=BALL_MASS),
            visual_material=sim.PreviewSurfaceCfg(
                diffuse_color=(.026,.025,.024),roughness=.69)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(BALL_XY[0],BALL_XY[1],BALL_RADIUS+.003)),
    ))
    skin=attach_classic_soccer_visual("/World/MatchFootball",radius=BALL_RADIUS)
    report={
        "football_rigid_body_prim":"/World/MatchFootball",
        "football_count":1,
        "football_mass_kg":BALL_MASS,
        "football_radius_m":BALL_RADIUS,
        "soccer_panels_black_pentagons":skin["pentagons"],
        "soccer_panels_white_hexagons":skin["hexagons"],
        "ball_xy":BALL_XY,"ready_xy":READY_XY,"goal_xy":GOAL_XY,
        "goal_width_m":GOAL_WIDTH,"goal_height_m":GOAL_HEIGHT,
        "goal_static_colliders":len(blocks),
        "visual_waypoint_spheres":0,
        "note":"G1 reference motion is kinematic only, ball genuinely PhysX; no kick."
    }
    print("SOCCER_WALKUP_SCENE_CREATED",report,flush=True)
    return ball,start_translate,report

def move_start_line(translate_op,x):
    translate_op.Set(Gf.Vec3d(float(x),0.,.012))

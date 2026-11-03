"""Visual training football pitch and lightweight net, with ZERO added colliders.

The original Isaac Lab terrain stays the ONLY ground contact surface.
Decorations are mere USD visual meshes and line curves, not physical objects.
"""
from __future__ import annotations

def create_field_visuals():
    from pxr import Gf, UsdGeom, UsdShade, Sdf
    import omni.usd
    stage=omni.usd.get_context().get_stage()
    base="/World/TrainingPitchVisual"
    UsdGeom.Xform.Define(stage,base)
    colors={
        "grass_a":(.045,.225,.103),
        "grass_b":(.056,.264,.122),
        "line":(.93,.95,.91),
        "target":(.88,.72,.12)
    }

    def material(name,color):
        m=UsdShade.Material.Define(stage,base+"/Material_"+name)
        sh=UsdShade.Shader.Define(stage,base+"/Shader_"+name)
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput("roughness",Sdf.ValueTypeNames.Float).Set(.95)
        sh.CreateInput("metallic",Sdf.ValueTypeNames.Float).Set(0.)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),"surface")
        return m
    mats={k:material(k,c) for k,c in colors.items()}

    def noncolliding_cube(name,at,scale,color):
        p=UsdGeom.Cube.Define(stage,base+"/"+name)
        p.CreateSizeAttr(1.0)
        xf=UsdGeom.Xformable(p)
        xf.AddTranslateOp().Set(Gf.Vec3d(*at))
        xf.AddScaleOp().Set(Gf.Vec3f(*scale))
        UsdShade.MaterialBindingAPI.Apply(p.GetPrim()).Bind(mats[color])
        return p
    # Extend the green, noncolliding turf far beyond the training lines so
    # the camera sees a field instead of grey checkerboard scenery.
    x0=-8.0
    stripes=20
    dx=20.0/stripes
    for i in range(stripes):
        center=x0+(i+.5)*dx
        noncolliding_cube("TurfStripe_%02d"%i,
            (center,0,.0015),(dx+.004,16.0,.0030),
            "grass_a" if i%2==0 else "grass_b")

    white=("line")
    noncolliding_cube("SideLine_North",(1.75,3.05,.005),(10.5,.026,.002),white)
    noncolliding_cube("SideLine_South",(1.75,-3.05,.005),(10.5,.026,.002),white)
    noncolliding_cube("BackLine_Left",(-3.5,0,.005),(.026,6.1,.002),white)
    noncolliding_cube("GoalLine",(5.0,0,.005),(.026,6.1,.002),white)
    noncolliding_cube("Halfway",(1.0,0,.005),(.020,6.1,.002),white)
    # Training penalty box, not FIFA regulation measurements.
    noncolliding_cube("PenaltyBoxTop",(3.75,1.5,.005),(2.5,.024,.002),white)
    noncolliding_cube("PenaltyBoxBottom",(3.75,-1.5,.005),(2.5,.024,.002),white)
    noncolliding_cube("PenaltyBoxBack",(2.5,0,.005),(.024,3.0,.002),white)

    report={"visual_stripes":stripes,"extra_ground_collision_prims":0,
            "training_pitch_extent_m":[20.0,16.0],
            "note":"visuals only; ground physics unchanged"}
    print("FOOTBALL_PITCH_VISUAL_READY",report,flush=True)
    return report


def decorate_goal_net(goal_path="/World/ResearchGoal"):
    from pxr import Gf, UsdGeom, UsdShade, Sdf
    import omni.usd
    stage=omni.usd.get_context().get_stage()
    root=stage.GetPrimAtPath(goal_path)
    if not root.IsValid(): raise RuntimeError("Goal not found "+goal_path)
    w=1.6
    h=.85
    depth=.32
    points=[]
    counts=[]
    def line(a,b):
        points.extend([Gf.Vec3f(*a),Gf.Vec3f(*b)])
        counts.append(2)
    for i in range(13):
        y=-w/2+w*i/12
        line((depth,y,0),(depth,y,h))
        line((0,y,h),(depth,y,h))
    for j in range(9):
        z=h*j/8
        line((depth,-w/2,z),(depth,w/2,z))
        line((0,-w/2,z),(depth,-w/2,z))
        line((0,w/2,z),(depth,w/2,z))
    curve=UsdGeom.BasisCurves.Define(stage,goal_path+"/MeshNetVisual")
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    curve.CreatePointsAttr(points)
    curve.CreateCurveVertexCountsAttr(counts)
    curve.CreateWidthsAttr([.004])
    curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    curve.CreateDisplayColorAttr([Gf.Vec3f(.88,.9,.92)])

    m=UsdShade.Material.Define(stage,goal_path+"/NetMaterial")
    shader=UsdShade.Shader.Define(stage,goal_path+"/NetShader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.89,.9,.91))
    shader.CreateInput("roughness",Sdf.ValueTypeNames.Float).Set(.85)
    m.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),"surface")
    UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(m)
    report={"net_line_segments":len(counts),"net_depth_m":depth,
            "physics_colliders_added":0}
    print("TRAINING_GOAL_NET_VISUAL_READY",report,flush=True)
    return report


def create_ground_target_markers():
    """Two thin, noncolliding X-shaped ground drawings, never round markers."""
    from pxr import Gf,UsdGeom,UsdShade,Sdf
    import omni.usd
    stage=omni.usd.get_context().get_stage()
    base="/World/FootPlacementGuides"
    UsdGeom.Xform.Define(stage,base)
    out={}
    for name,color in (("Approach",(.95,.85,.12)),("SupportFoot",(.20,.89,.92))):
        marker=UsdGeom.Xform.Define(stage,base+"/"+name)
        for tag,rot in (("Slash",45.),("Backslash",-45.)):
            child=UsdGeom.Cube.Define(stage,base+"/"+name+"/"+tag)
            child.CreateSizeAttr(1.)
            transform=UsdGeom.Xformable(child)
            transform.AddTranslateOp().Set(Gf.Vec3d(0.,0.,.006))
            transform.AddRotateZOp().Set(rot)
            transform.AddScaleOp().Set(Gf.Vec3f(.15,.015,.0018))
            child.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdGeom.Xformable(marker).AddTranslateOp().Set(Gf.Vec3d(0.,0.,0.))
        out[name]=marker
    print("GROUND_X_GUIDES_READY",list(out),flush=True)
    return out


def move_ground_target_marker(marker,xy):
    from pxr import Gf,UsdGeom
    ops=UsdGeom.Xformable(marker).GetOrderedXformOps()
    op=next(x for x in ops if "translate" in str(x.GetOpName()))
    op.Set(Gf.Vec3d(float(xy[0]),float(xy[1]),0.0))


def tone_down_head_render_material(head_path="/World/envs/env_0/Robot/head_link"):
    """Neutral, robot-like G1 head material, without replacing articulation/colliders.

    Only a cosmetic override. IsaacLab's original joint chain and existing head
    collision shape remain untouched. If the strong binding is rejected, log
    and leave the official visual asset alone.
    """
    from pxr import Gf,UsdShade,Sdf
    import omni.usd
    stage=omni.usd.get_context().get_stage()
    prim=stage.GetPrimAtPath(head_path)
    if not prim.IsValid():
        print("G1_HEAD_VISUAL_STYLING_SKIPPED",head_path,flush=True)
        return False
    path="/World/RobotRenderStyle"
    m=UsdShade.Material.Define(stage,path+"/Graphite")
    shader=UsdShade.Shader.Define(stage,path+"/GraphiteShader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.21,.26,.31))
    shader.CreateInput("roughness",Sdf.ValueTypeNames.Float).Set(.47)
    shader.CreateInput("metallic",Sdf.ValueTypeNames.Float).Set(.18)
    m.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),"surface")
    binding=UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(m,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    print("G1_HEAD_NEUTRAL_VISUAL_APPLIED",head_path,flush=True)
    return True


def inspect_head_visual_subtree(head_path="/World/envs/env_0/Robot/head_link"):
    """Diagnostic only: identify which geometry contains neon colors."""
    from pxr import Usd,UsdGeom
    import omni.usd
    stage=omni.usd.get_context().get_stage()
    root=stage.GetPrimAtPath(head_path)
    if not root.IsValid():
        print("HEAD_INSPECT_MISSING",head_path,flush=True)
        return
    limit=60
    for i,p in enumerate(Usd.PrimRange(root)):
        if i>=limit: break
        name=p.GetName()
        tp=p.GetTypeName()
        mat=p.GetAttribute("material:binding")
        col=p.GetAttribute("primvars:displayColor")
        color=str(col.Get())[:180] if col and col.IsValid() else ""
        binding=str(mat.Get())[:180] if mat and mat.IsValid() else ""
        print("HEAD_VIS_SUBPRIM",i,str(p.GetPath()),tp,
              "instance",p.IsInstanceProxy(),"display_color",color,
              "material",binding,flush=True)


def replace_neon_g1_head_visual(head_path="/World/envs/env_0/Robot/head_link"):
    """Optional render-only G1-inspired graphite shell; no joint or physics changes.

    The source minimal G1 USD has bright green/blue head visuals that make the
    athletic scene difficult to judge. Hide only its original VISUAL child and
    author a restrained graphite shell + visor under the same articulated link.
    The exact cosmetics are not a manufacturer-verified CAD asset.
    """
    from pxr import Gf,UsdGeom,UsdShade,Sdf
    import omni.usd
    stage=omni.usd.get_context().get_stage()
    head=stage.GetPrimAtPath(head_path)
    visuals=stage.GetPrimAtPath(head_path+"/visuals")
    if not (head.IsValid() and visuals.IsValid()):
        print("HEAD_NEUTRAL_REPLACEMENT_NOT_SUPPORTED",head_path,flush=True)
        return False

    # Only the rendered original head visuals are hidden. The original
    # head_link physics primitive, articulation, collision and mass persist.
    visual_img=UsdGeom.Imageable(visuals)
    visual_img.MakeInvisible()

    root=head_path+"/NeutralHeadVisual"
    UsdGeom.Xform.Define(stage,root)
    mat=UsdShade.Material.Define(stage,root+"/GraphiteMat")
    shader=UsdShade.Shader.Define(stage,root+"/GraphiteSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(.18,.21,.235))
    shader.CreateInput("roughness",Sdf.ValueTypeNames.Float).Set(.55)
    shader.CreateInput("metallic",Sdf.ValueTypeNames.Float).Set(.22)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),"surface")
    shell=UsdGeom.Capsule.Define(stage,root+"/Shell")
    shell.CreateRadiusAttr(.106)
    shell.CreateHeightAttr(.10)
    UsdShade.MaterialBindingAPI.Apply(shell.GetPrim()).Bind(mat)

    visor_mat=UsdShade.Material.Define(stage,root+"/VisorMat")
    vs=UsdShade.Shader.Define(stage,root+"/VisorSurface")
    vs.CreateIdAttr("UsdPreviewSurface")
    vs.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(.045,.062,.080))
    vs.CreateInput("roughness",Sdf.ValueTypeNames.Float).Set(.24)
    vs.CreateInput("metallic",Sdf.ValueTypeNames.Float).Set(.28)
    visor_mat.CreateSurfaceOutput().ConnectToSource(vs.ConnectableAPI(),"surface")
    face=UsdGeom.Cube.Define(stage,root+"/FrontVisor")
    xf=UsdGeom.Xformable(face)
    xf.AddTranslateOp().Set(Gf.Vec3d(.087,0.,.015))
    xf.AddScaleOp().Set(Gf.Vec3f(.018,.15,.08))
    UsdShade.MaterialBindingAPI.Apply(face.GetPrim()).Bind(visor_mat)
    print("NEUTRAL_G1_HEAD_RENDER_ONLY_READY",
          "original_visuals_hidden",visuals.GetPath(),
          "new_visual_shell",root,
          "added_rigidbodies",0,"added_colliders",0,flush=True)
    return True

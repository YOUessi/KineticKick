"""Non-colliding field and goal matching the measured goal aperture."""
from __future__ import annotations


def create_shooting_visuals(ball_xy, goal_xy, *, forward=None, target_x=None):
    import math
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    base = '/World/ShootingVisuals'
    UsdGeom.Xform.Define(stage, base)

    def cube(path, position, size, color):
        shape = UsdGeom.Cube.Define(stage, path)
        shape.CreateSizeAttr(1.)
        transform = UsdGeom.Xformable(shape)
        transform.AddTranslateOp().Set(Gf.Vec3d(*position))
        transform.AddScaleOp().Set(Gf.Vec3f(*size))
        shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    for i in range(16):
        cube(f'{base}/Grass{i}', (-7.5+i, -2., .001), (1., 16., .002),
             (.055, .27 if i % 2 else .23, .10))
    dx, dy = forward if forward is not None else (goal_xy[0]-ball_xy[0], goal_xy[1]-ball_xy[1])
    angle = math.degrees(math.atan2(dx, -dy))
    goal = UsdGeom.Xform.Define(stage, base+'/Goal')
    transform = UsdGeom.Xformable(goal)
    transform.AddTranslateOp().Set(Gf.Vec3d(*goal_xy, 0.))
    transform.AddRotateZOp().Set(angle)
    white = (.95, .95, .92)
    for name, x in (('Left', -.83), ('Right', .83)):
        cube(f'{base}/Goal/{name}', (x, 0., .55), (.06, .06, 1.1), white)
    cube(base+'/Goal/Crossbar', (0., 0., 1.13), (1.72, .06, .06), white)
    cube(base+'/Goal/Line', (0., 0., .003), (3.2, .025, .002), white)
    if target_x is not None:
        # A horizontal ±15 cm target interval; it does not claim height control.
        cube(base+'/Target', (target_x, goal_xy[1], .006), (.30, .16, .008), (1., .64, .05))
        for side in (-1, 1):
            cube(f'{base}/TargetEdge{side}', (target_x+side*.15, goal_xy[1], .18),
                 (.012, .012, .36), (1., .64, .05))
    # Local -Y is behind the aperture; the net is a visual reference only.
    points, counts = [], []
    for i in range(17):
        x = -.8 + i*.1
        points.extend([Gf.Vec3f(x, -.4, .01), Gf.Vec3f(x, -.4, 1.1)])
        counts.append(2)
    for j in range(12):
        z = j*.1
        points.extend([Gf.Vec3f(-.8, -.4, z), Gf.Vec3f(.8, -.4, z)])
        counts.append(2)
    net = UsdGeom.BasisCurves.Define(stage, base+'/Goal/Net')
    net.CreateTypeAttr(UsdGeom.Tokens.linear)
    net.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    net.CreatePointsAttr(points)
    net.CreateCurveVertexCountsAttr(counts)
    net.CreateWidthsAttr([.004])
    net.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    net.CreateDisplayColorAttr([Gf.Vec3f(*white)])

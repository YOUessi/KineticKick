"""Classic soccer visual skin: 12 black pentagons + 20 white hexagons.

The ONLY physics football remains a 0.43kg radius=0.11m spherical rigid body.
This module adds 2 noncolliding USD render meshes under that dynamic ball prim.
No additional spheres, body masses, or collision geometries are added.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.spatial import ConvexHull


def soccer_panel_polygons():
    phi=(1+math.sqrt(5))/2
    verts=[]
    for a in (-1,1):
        for b in (-1,1):
            verts.extend([(0,a,b*phi),(a,b*phi,0),(a*phi,0,b)])
    v=np.array(verts,dtype=np.float64)
    v/=np.linalg.norm(v,axis=1)[:,None]
    faces=ConvexHull(v).simplices
    sides=[set() for _ in range(len(v))]
    oriented=[]
    for face in faces:
        a,b,c=map(int,face)
        va,vb,vc=v[[a,b,c]]
        if np.dot(np.cross(vb-va,vc-va),va)<0:
            b,c=c,b
        oriented.append((a,b,c))
        sides[a].update([b,c]);sides[b].update([a,c]);sides[c].update([a,b])
    assert len(oriented)==20, len(oriented)

    def near(i,j):
        p=2*v[i]+v[j]
        return p/np.linalg.norm(p)
    hexes=[]
    for a,b,c in oriented:
        poly=np.array([near(a,b),near(b,a),near(b,c),
                       near(c,b),near(c,a),near(a,c)])
        hexes.append(poly)
    pents=[]
    for i,neighbors in enumerate(sides):
        assert len(neighbors)==5, (i,neighbors)
        normal=v[i]
        first=np.array([1.,0.,0.])
        if abs(np.dot(normal,first))>.90:first=np.array([0.,1.,0.])
        tangent=first-normal*np.dot(normal,first)
        tangent/=np.linalg.norm(tangent)
        bitangent=np.cross(normal,tangent)
        candidates=[]
        for j in neighbors:
            p=near(i,j)
            q=p-normal*np.dot(p,normal)
            ang=math.atan2(np.dot(q,bitangent),np.dot(q,tangent))
            candidates.append((ang,p))
        poly=np.array([p for a,p in sorted(candidates,key=lambda x:x[0])])
        pents.append(poly)

    def signed_orientation(poly):
        n=np.cross(poly[1]-poly[0],poly[2]-poly[0])
        return float(np.dot(n,poly[0]))
    hexes=[p if signed_orientation(p)>0 else p[::-1] for p in hexes]
    pents=[p if signed_orientation(p)>0 else p[::-1] for p in pents]
    return pents,hexes


def triangle_spherical_panels(polygons,radius=.1105,edge_shrink=.976,
                              radial_subdiv=3,edge_subdiv=5):
    points=[];normals=[];indices=[];counts=[]
    def sphere(v,r):
        return v/np.linalg.norm(v)*r
    for polygon in polygons:
        center=sphere(np.mean(polygon,axis=0),1.)
        rim=np.array([sphere(center+(point-center)*edge_shrink,1.) for point in polygon])
        boundary=[]
        for i in range(len(rim)):
            a,b=rim[i],rim[(i+1)%len(rim)]
            for j in range(edge_subdiv):
                t=j/edge_subdiv
                boundary.append(sphere((1-t)*a+t*b,1.))
        boundary=np.array(boundary)
        m=len(boundary)
        rings=[np.array([center.copy() for _ in range(m)])]
        for step in range(1,radial_subdiv+1):
            ratio=step/radial_subdiv
            rings.append(np.array([sphere(center*(1-ratio)+edge*ratio,1.)
                                   for edge in boundary]))
        old=len(points)
        for ring in rings:
            for pt in ring:
                points.append((pt*radius).tolist())
                normals.append(pt.tolist())
        for step in range(1,radial_subdiv+1):
            for j in range(m):
                a=old+(step-1)*m+j
                b=old+(step-1)*m+((j+1)%m)
                c=old+step*m+j
                d=old+step*m+((j+1)%m)
                for tri in ((a,c,d),(a,d,b)):
                    a0,a1,a2=[np.array(points[k]) for k in tri]
                    if np.dot(np.cross(a1-a0,a2-a0),a0)<0:
                        tri=(tri[0],tri[2],tri[1])
                    indices.extend(tri)
                    counts.append(3)
    return points,normals,counts,indices


def attach_classic_soccer_visual(ball_prim_path,radius=.11):
    from pxr import Gf, UsdGeom, UsdShade, Sdf, UsdPhysics
    import omni.usd
    stage=omni.usd.get_context().get_stage()
    root=stage.GetPrimAtPath(ball_prim_path)
    if not root.IsValid(): raise RuntimeError("Missing physics ball prim: "+ball_prim_path)
    visual_scope=ball_prim_path+"/ClassicSoccerVisual"
    UsdGeom.Xform.Define(stage,visual_scope)
    pents,hexes=soccer_panel_polygons()

    def mat(name,rgb,rough):
        m=UsdShade.Material.Define(stage,visual_scope+"/"+name+"Material")
        shader=UsdShade.Shader.Define(stage,visual_scope+"/"+name+"Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        shader.CreateInput("roughness",Sdf.ValueTypeNames.Float).Set(float(rough))
        shader.CreateInput("metallic",Sdf.ValueTypeNames.Float).Set(0.0)
        m.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),"surface")
        return m

    def mesh(name,panels,material):
        pp,nn,cc,ii=triangle_spherical_panels(
            panels,radius=radius*1.007,edge_shrink=.970,
            radial_subdiv=3,edge_subdiv=6)
        path=visual_scope+"/"+name
        ge=UsdGeom.Mesh.Define(stage,path)
        ge.CreatePointsAttr([Gf.Vec3f(*pt) for pt in pp])
        ge.CreateFaceVertexCountsAttr(cc)
        ge.CreateFaceVertexIndicesAttr(ii)
        ge.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        ge.CreateNormalsAttr([Gf.Vec3f(*n) for n in nn])
        ge.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        UsdShade.MaterialBindingAPI.Apply(ge.GetPrim()).Bind(material)
        assert not ge.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
        return {"path":path,"polygons":len(panels),"triangles":len(cc),"points":len(pp)}

    white=mat("White",(0.94,0.96,0.94),.69)
    dark=mat("Black",(0.009,0.010,0.014),.78)
    h=mesh("WhiteHexPanels",hexes,white)
    p=mesh("BlackPentPanels",pents,dark)
    result={"physical_ball":ball_prim_path,"radius_m":radius,
            "pentagons":p["polygons"],"hexagons":h["polygons"],
            "meshes":[h,p],"physics_colliders_added":0}
    print("SOCCER_VISUAL_READY",result,flush=True)
    return result


if __name__=="__main__":
    p,h=soccer_panel_polygons()
    assert len(p)==12 and len(h)==20
    for group in (p,h):
        for poly in group:
            assert 0.999 < min(np.linalg.norm(x) for x in poly)<1.001
    for group in (p,h):
        pts,nrm,counts,indices=triangle_spherical_panels(group)
        assert len(indices)==3*len(counts)
        assert len(pts)==len(nrm)
    print("SOCCER_GEOMETRY_VALIDATED",len(p),len(h))

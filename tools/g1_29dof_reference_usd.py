"""Render-only G1 29DoF LAFAN1 walking reference in Isaac Sim.

No rigid body, collision, controller or any physics claim. Exact 29DoF URDF FK,
read from 30-FPS kinematically retargeted human motion. Source meshes are STL.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from g1_retarget_kinematics import (
  load_robot_tree,origin_transform,binary_stl_sample,fk_g1_pose,URDF
)

class G1HumanReferenceVisual:
    def __init__(self,root_prim="/World/G1HumanMotionReference29",
                 max_triangles_per_visual=4000):
        from pxr import Gf,UsdGeom,UsdPhysics,UsdShade,Sdf
        import omni.usd
        self.stage=omni.usd.get_context().get_stage()
        self.root_link,self.links,self.joints=load_robot_tree()
        self.root=UsdGeom.Xform.Define(self.stage,root_prim)
        self.ops={}
        def make_material(label,color):
            m=UsdShade.Material.Define(self.stage,root_prim+"/"+label+"_material")
            shader=UsdShade.Shader.Define(self.stage,root_prim+"/"+label+"_shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(color)
            shader.CreateInput("roughness",Sdf.ValueTypeNames.Float).Set(.62)
            shader.CreateInput("metallic",Sdf.ValueTypeNames.Float).Set(0.)
            shader.CreateInput("opacity",Sdf.ValueTypeNames.Float).Set(1.0)
            m.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),"surface")
            return m
        self.meshes=0
        self.triangles=0
        self.root_path=root_prim
        col_body=Gf.Vec3f(.74,.78,.82)
        col_dark=Gf.Vec3f(.19,.25,.30)
        col_arm=Gf.Vec3f(.55,.63,.70)
        materials={
          "body":make_material("body",col_body),
          "dark":make_material("dark",col_dark),
          "arms":make_material("arms",col_arm),
        }
        for link_name,link_elem in self.links.items():
            # Every movable visual is a separate Xform with its world transform.
            # No PhysX articulation/Collider is authored anywhere.
            link=UsdGeom.Xform.Define(self.stage,root_prim+"/"+link_name)
            op=UsdGeom.Xformable(link.GetPrim()).AddTransformOp()
            self.ops[link_name]=op
            all_visual=link_elem.findall("visual")
            for visual_index,visual in enumerate(all_visual):
                mesh_el=visual.find("./geometry/mesh")
                if mesh_el is None:continue
                rel=mesh_el.attrib["filename"]
                file_path=(URDF.parent/rel).resolve()
                if not file_path.is_file():raise FileNotFoundError(file_path)
                points=binary_stl_sample(file_path,max_triangles_per_visual).astype(np.float64)
                if "scale" in mesh_el.attrib:
                    scale=np.fromstring(mesh_el.attrib["scale"],sep=" ")
                    if len(scale)!=3:raise ValueError("URDF mesh scale")
                    points*=scale[None,:]
                xf=origin_transform(visual.find("origin"))
                points=(points@xf[:3,:3].T)+xf[:3,3]
                mesh=UsdGeom.Mesh.Define(self.stage,
                      f"{root_prim}/{link_name}/Mesh_{visual_index}")
                pts=[Gf.Vec3f(*map(float,v)) for v in points]
                triangles=len(pts)//3
                mesh.CreatePointsAttr(pts)
                mesh.CreateFaceVertexCountsAttr([3]*triangles)
                mesh.CreateFaceVertexIndicesAttr(list(range(len(pts))))
                mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                mesh.CreateDoubleSidedAttr(True)
                style=("dark" if "ankle" in link_name or "head" in link_name else
                       ("arms" if any(x in link_name for x in ("shoulder","wrist","elbow")) else "body"))
                col={"dark":col_dark,"arms":col_arm,"body":col_body}[style]
                mesh.CreateDisplayColorAttr([col])
                mesh.CreateDisplayOpacityAttr([1.0])
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(materials[style])
                assert not mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
                assert not mesh.GetPrim().HasAPI(UsdPhysics.RigidBodyAPI)
                self.meshes+=1
                self.triangles+=triangles
        print("REFERENCE29_RENDER_ONLY_LOADED",
              "root",self.root_link,
              "links",len(self.links),
              "usd_visual_meshes",self.meshes,
              "render_triangles",self.triangles,
              "physics_rigidbodies_added",0,
              "collision_prims_added",0,flush=True)

    @staticmethod
    def gf_matrix(np_T):
        from pxr import Gf
        # SciPy/NumPy transforms act on COLUMN vectors, Gf Matrix4d on rows.
        return Gf.Matrix4d(*map(float,np_T.T.reshape(16)))

    def update(self,frame,origin_xy=None,ref_start_xy=None):
        row=np.asarray(frame,dtype=np.float64)
        if row.shape!=(36,):raise ValueError("Expected 36 LAFAN1 retargeted coordinates")
        rootxyz=row[:3].copy()
        if origin_xy is not None:
            if ref_start_xy is None:raise ValueError("Expected source initial XY offset")
            rootxyz[:2]+=np.asarray(origin_xy)-np.asarray(ref_start_xy)
        T=fk_g1_pose(rootxyz,row[3:7],row[7:],self.root_link,self.joints)
        for name,op in self.ops.items():
            op.Set(self.gf_matrix(T[name]))
        return rootxyz

    def visible(self,on=True):
        from pxr import UsdGeom
        p=UsdGeom.Imageable(self.root.GetPrim())
        p.MakeVisible() if on else p.MakeInvisible()

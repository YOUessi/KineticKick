"""Pure numerical FK/URDF reader for G1 29DoF retargeted LAFAN1 CSV.

Strictly kinematic: NEVER implies a dynamically stable humanoid policy.
Can be tested using Anaconda Python without booting Isaac Sim.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import struct,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path("/home/you/projects/kinetickick")
URDF=ROOT/"data/assets/g1_29dof_lafan1/robot_description/g1/g1_29dof_rev_1_0.urdf"
DATA=ROOT/"data/human_motion/lafan1_g1_29dof"
JOINTS=(
    "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint",
    "left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint",
    "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint",
    "right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint","waist_roll_joint","waist_pitch_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
    "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
    "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint",
)
assert len(JOINTS)==29


def origin_transform(origin):
    a=np.eye(4)
    if origin is None:return a
    xyz=np.fromstring(origin.attrib.get("xyz","0 0 0"),sep=" ")
    rpy=np.fromstring(origin.attrib.get("rpy","0 0 0"),sep=" ")
    if len(xyz)!=3 or len(rpy)!=3:raise ValueError("URDF origin invalid")
    a[:3,:3]=Rotation.from_euler("xyz",rpy).as_matrix()
    a[:3,3]=xyz
    return a


@dataclass
class RobotJoint:
    name:str
    kind:str
    parent:str
    child:str
    origin:np.ndarray
    axis:np.ndarray


def load_robot_tree():
    parsed=ET.parse(URDF).getroot()
    links={l.attrib["name"]:l for l in parsed.findall("link")}
    joints=[]
    for node in parsed.findall("joint"):
        axis_el=node.find("axis")
        axis=np.fromstring(axis_el.attrib["xyz"],sep=" ") if axis_el is not None else np.array([1.,0.,0.])
        if len(axis)!=3:raise ValueError("Invalid joint axis")
        parent=node.find("parent").attrib["link"]
        child=node.find("child").attrib["link"]
        if parent not in links or child not in links:raise ValueError("Missing linked body")
        joints.append(RobotJoint(node.attrib["name"],node.attrib["type"],parent,child,
                        origin_transform(node.find("origin")),axis))
    child_links={j.child for j in joints}
    roots=[n for n in links if n not in child_links]
    if len(roots)!=1:raise ValueError(("Expected one URDF root link",roots))
    if set(j.name for j in joints if j.kind!="fixed")!=set(JOINTS):
        raise ValueError("URDF active joints don't match G1 LAFAN CSV names")
    ordered=[]
    known={roots[0]}
    pending=joints[:]
    while pending:
        ready=[j for j in pending if j.parent in known]
        if not ready:raise RuntimeError("URDF joint graph loop/disconnection")
        for j in ready:
            ordered.append(j)
            known.add(j.child)
            pending.remove(j)
    return roots[0],links,ordered


def fk_g1_pose(root_pos,root_quat_xyzw,joint_values,root_link,joints):
    if len(joint_values)!=29:raise ValueError("Expected exactly 29 joints")
    state=dict(zip(JOINTS,joint_values))
    base=np.eye(4)
    base[:3,:3]=Rotation.from_quat(root_quat_xyzw).as_matrix()
    base[:3,3]=root_pos
    world={root_link:base}
    for j in joints:
        child_T=world[j.parent] @ j.origin
        if j.kind in ("revolute","continuous"):
            rotation=Rotation.from_rotvec(j.axis*float(state[j.name])).as_matrix()
            r=np.eye(4);r[:3,:3]=rotation
            child_T=child_T@r
        world[j.child]=child_T
    return world


def binary_stl_sample(path,max_triangles=4800):
    """Read triangles from a binary STL without installing trimesh."""
    raw=Path(path).read_bytes()
    if len(raw)<84:raise ValueError(f"Invalid STL {path}")
    count=struct.unpack_from("<I",raw,80)[0]
    if count<=0 or len(raw)!=84+count*50:
        raise ValueError(f"Unsupported/non-binary STL: {path}, n={count} bytes={len(raw)}")
    dtype=np.dtype([("normal","<f4",(3,)),("vertices","<f4",(3,3)),("attr","<u2")])
    triangles=np.frombuffer(raw,offset=84,count=count,dtype=dtype)["vertices"]
    if count>max_triangles:
        sel=np.linspace(0,count-1,max_triangles,dtype=np.int64)
        triangles=triangles[sel]
    return triangles.reshape((-1,3)).copy()


if __name__=="__main__":
    root,links,joints=load_robot_tree()
    print("G1_29_FK_ROBOT",root,"links",len(links),
          "joints",len(joints),"actuated",sum(j.kind!="fixed" for j in joints))
    f=np.load(DATA/"walk1_subject1_selected8s.npz")["frames"]
    T=fk_g1_pose(f[0,:3],f[0,3:7],f[0,7:],root,joints)
    for side in ("left","right"):
        end=side+"_ankle_roll_link"
        print("G1_29_FK_FOOT",end,np.round(T[end][:3,3],4))
    example=URDF.parent/"meshes/left_knee_link.STL"
    xyz=binary_stl_sample(example)
    print("G1_29_FK_MESH",example.name,"vertices",len(xyz))

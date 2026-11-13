#!/usr/bin/env python3
"""v156: authenticated original G1 URDF (NOT PhysX) offline 35D dynamics.

This model is a *real-source mathematical rigid-body model*: it is NOT measured
same-step PhysX M/h/GRF/CoP, a controller, or elite human football training.
Never use these offline matrices to send real-world or simulator commands.
"""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from wholebody_safety_interface_v055 import G1_JOINT_ORDER

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = "29809c5a8e7156363880808fa73afdc452258cec"
PROTOCOL_PATH = "configs/v156_frozen_urdf_dynamics_provenance.json"
FALLBACK_URDF = Path("/home/you/projects/kinetickick/configs/paid_g1_main.urdf")
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
NONFOOT = ("left_wrist_yaw_link", "left_hip_roll_link")

def frozen():
    expected = subprocess.check_output(["git", "-C", str(ROOT), "show", PROTOCOL_SHA + ":" + PROTOCOL_PATH])
    if (ROOT / PROTOCOL_PATH).read_bytes() != expected:
        raise ValueError("V156_PROTOCOL_CHANGED_AFTER_FREEZE")
    cfg = json.loads(expected)
    if (cfg["parent_sha"] != "343d20d94ad40e32e585d0f3648f371a3ddd6960"
            or cfg["model_checks"] != 48 or cfg["independent_kinematic_checks"] != 12
            or cfg["history_states"] != 6628):
        raise ValueError("V156_PROTOCOL_DENOMINATOR_DRIFT")
    return cfg

def pin_dep(cfg=None):
    cfg = cfg or frozen()
    if int(np.__version__.split(".")[0]) != cfg["required_numpy_major"]:
        raise RuntimeError("V156_PINOCCHIO_NUMPY_ABI_REQUIRES_NUMPY1_USE_PYTHONNOUSERSITE")
    import pinocchio as pin
    if pin.__version__ != cfg["required_pinocchio_version"]:
        raise RuntimeError("V156_PINOCCHIO_VERSION_UNFROZEN")
    return pin

def urdf_path(cfg=None):
    cfg = cfg or frozen()
    path = Path(os.environ.get(cfg["source_urdf_location_env"], str(FALLBACK_URDF))).resolve(strict=True)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != cfg["source_urdf_sha256"]:
        raise ValueError("V156_ORIGINAL_G1_URDF_SHA_MISMATCH")
    return path

def vec(v, n, tag):
    z = np.asarray(v, dtype=np.float64)
    if z.shape != (n,) or not np.all(np.isfinite(z)):
        raise ValueError("V156_INVALID_" + tag)
    return z

class OriginalUrdfModel:
    """Pin internal joint order differs from IsaacLab original joint order."""
    def __init__(self, source=None):
        self.cfg = frozen()
        self.pin = pin_dep(self.cfg)
        self.source = Path(source) if source is not None else urdf_path(self.cfg)
        if hashlib.sha256(self.source.read_bytes()).hexdigest() != self.cfg["source_urdf_sha256"]:
            raise ValueError("V156_ORIGINAL_G1_URDF_SHA_MISMATCH")
        p = self.pin
        self.model = m = p.buildModelFromUrdf(str(self.source), p.JointModelFreeFlyer())
        if (m.nq, m.nv, m.njoints-2) != (36,35,29):
            raise ValueError("V156_NOT_ORIGINAL29_FREEFLOAT")
        allnames = list(m.names)[2:]
        if len(set(allnames)) != 29 or set(allnames) != set(G1_JOINT_ORDER):
            raise ValueError("V156_URDF_ORIGINAL29_NAME_SET_CHANGED")
        self.idxq = np.asarray([m.joints[m.getJointId(name)].idx_q for name in G1_JOINT_ORDER], dtype=int)
        self.idxv = np.asarray([m.joints[m.getJointId(name)].idx_v for name in G1_JOINT_ORDER], dtype=int)
        if (set(self.idxq) != set(range(7,36)) or set(self.idxv) != set(range(6,35))):
            raise ValueError("V156_UNEXPECTED_JOINT_DOF_OR_ORDER")
        self.frames = {s:m.getFrameId(s) for s in FEET+NONFOOT}
        if any(fid >= m.nframes for fid in self.frames.values()):
            raise ValueError("V156_MISSING_REAL_URDF_FRAME")
        self.total_mass_urdf_kg = float(p.computeTotalMass(m))
        if not 20 < self.total_mass_urdf_kg < 100:
            raise ValueError("V156_UNEXPECTED_ORIGINAL_MASS")
        self.sha = self.cfg["source_urdf_sha256"]

    def pose(self, joint29, *, xyz=(0.,0.,.82), xyzw=(0.,0.,0.,1.)):
        q = self.pin.neutral(self.model)
        q[:3] = vec(xyz,3,"BASE_POS")
        quat = vec(xyzw,4,"BASE_QUAT_XYZW")
        if not np.isclose(np.linalg.norm(quat),1.0,atol=1e-8):
            raise ValueError("V156_NONUNIT_BASE_QUATERNION")
        q[3:7] = quat
        q[self.idxq] = vec(joint29,29,"Q29")
        return q

    def speed(self, joint29, *, base_body6=(0.,0.,0.,0.,0.,0.)):
        v = np.zeros(35)
        v[:6] = vec(base_body6,6,"BASE_BODY_TWIST6")
        v[self.idxv] = vec(joint29,29,"QDOT29")
        return v

    def evaluate(self, q, v):
        """Pin LOCAL_WORLD_ALIGNED twist ordering is [linear3, angular3]."""
        p=self.pin; m=self.model; d=m.createData()
        q=vec(q,36,"PIN_Q36"); v=vec(v,35,"PIN_V35")
        if not np.isclose(np.linalg.norm(q[3:7]),1.0,atol=1e-8):
            raise ValueError("V156_BAD_ROOT_UNIT_QUATERNION")
        M=p.crba(m,d,q)
        M=np.triu(M)+np.triu(M,1).T
        h=p.nonLinearEffects(m,d,q,v)
        p.forwardKinematics(m,d,q,v,np.zeros(35))
        p.computeJointJacobians(m,d,q)
        p.updateFramePlacements(m,d)
        j={}; jd={}; origin={}
        for key,fid in self.frames.items():
            j[key]=np.asarray(p.getFrameJacobian(m,d,fid,p.LOCAL_WORLD_ALIGNED)).copy()
            a=p.getFrameClassicalAcceleration(m,d,fid,p.LOCAL_WORLD_ALIGNED)
            jd[key]=np.asarray(a.vector).copy()
            origin[key]=np.asarray(d.oMf[fid].translation).copy()
        com=np.asarray(p.centerOfMass(m,d,q)).copy()
        return dict(M=M,h=np.asarray(h).copy(),J6=j,Jdotv6=jd,
                    link_origin_w=origin,com_model_w=com,
                    source_kind="original_g1_urdf_offline",
                    can_issue_control=False,can_certify_physx=False,
                    can_certify_grf_cop=False,can_certify_wrist_hip_manifold=False)

    def fixture(self,phase,mode,seed):
        c=self.cfg
        if phase not in c["phases"] or mode not in c["modes"] or seed not in c["seeds"]:
            raise ValueError("V156_NOT_PREDECLARED_FIXTURE")
        rng=np.random.default_rng(20261114+1000*c["phases"].index(phase)+47*seed+
                                  13*c["modes"].index(mode))
        q29=rng.uniform(-.12,.12,size=29)
        v29=rng.uniform(-.08,.08,size=29)
        # Fixed, source-declared offline fixture, not observed human/G1 pose.
        q=self.pose(q29)
        v=self.speed(v29,base_body6=(.02,-.01,.005,.003,-.004,.007))
        return q,v

def independent_invariants(model,q,v,out):
    """Dynamics identity independent of CRBA/NonLinearEffects values."""
    p=model.pin; m=model.model
    M=out["M"]; h=out["h"]
    if M.shape!=(35,35) or h.shape!=(35,) or not np.isfinite(M).all():
        raise AssertionError("V156_MASS_DIM_OR_FINITE")
    symmetry=float(np.max(np.abs(M-M.T)))
    min_eig=float(np.linalg.eigvalsh(M)[0])
    if symmetry>1e-10 or min_eig<1e-7:
        raise AssertionError("V156_INERTIA_NOT_SPD")
    rng=np.random.default_rng(131)
    acc=rng.uniform(-.2,.2,35)
    rnea=p.rnea(m,m.createData(),q,v,acc)
    err_inv=float(np.max(np.abs(rnea-(M@acc+h))))
    if err_inv>1e-7:raise AssertionError("V156_RNEA_CRBA_NOT_CONSISTENT")
    aba=p.aba(m,m.createData(),q,v,rnea)
    err_fwd=float(np.max(np.abs(aba-acc)))
    if err_fwd>1e-7:raise AssertionError("V156_ABA_RNEA_NOT_CONSISTENT")
    if tuple(out["J6"])!=FEET+NONFOOT:
        raise AssertionError("V156_NAMED_FRAME_ORDER")
    if any(x.shape!=(6,35) for x in out["J6"].values()):
        raise AssertionError("V156_FOOT_J6_NOT_FULL_FLOATING")
    origin=out["link_origin_w"]
    wrist_hip_origin_distance=float(np.linalg.norm(origin[NONFOOT[0]]-origin[NONFOOT[1]]))
    return dict(max_M_symmetry=symmetry,min_M_eigenvalue=min_eig,
                max_rnea_consistency_error=err_inv,max_aba_error=err_fwd,
                wrist_hip_link_origin_distance_m=wrist_hip_origin_distance,
                origin_distance_is_NOT_collider_gap=True,
                net_force_is_NOT_grf=True)

def independent_time_derivative(model,q,v,out,eps=1e-6):
    """Finite difference dJ(q)/dt @ v; checks 6D bias acceleration, independently."""
    p=model.pin;m=model.model
    qp=p.integrate(m,q,eps*v)
    qm=p.integrate(m,q,-eps*v)
    dplus=m.createData();dminus=m.createData()
    for qq,dd in ((qp,dplus),(qm,dminus)):
        p.computeJointJacobians(m,dd,qq);p.updateFramePlacements(m,dd)
    result={}
    for name,fid in model.frames.items():
        jp=p.getFrameJacobian(m,dplus,fid,p.LOCAL_WORLD_ALIGNED).copy()
        jm=p.getFrameJacobian(m,dminus,fid,p.LOCAL_WORLD_ALIGNED).copy()
        fd=(jp-jm)@v/(2*eps)
        error=float(np.max(np.abs(fd-out["Jdotv6"][name])))
        result[name]=error
        if error>2e-5:
            raise AssertionError("V156_BIAS_ACCELERATION_FD_FAIL:"+name+":"+str(error))
    return result

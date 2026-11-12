#!/usr/bin/env python3
"""CPU-only v127 geometry and provenance checks; synthetic math is not PhysX truth."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import v127_collider_proximity as a


@pytest.mark.parametrize("d,expected", [
    (1.0,1.0), (0.0,0.0), (4.0,4.0)])
def test_parallel_segment_separations(d,expected):
    p=np.array([[0.,0.,0.]])
    x=a.segment_distance(p,p+[[0,0,1]],p+[[d,0,0]],p+[[d,0,1]])
    assert x.item()==pytest.approx(expected)


def test_intersecting_perpendicular_segments():
    p=np.array([[-1.,0.,0.]])
    q=np.array([[0.,-1.,0.]])
    assert a.segment_distance(p,p+[[2,0,0]],q,q+[[0,2,0]]).item()<1e-10


def test_clamped_endpoints_segment_separation():
    p=np.array([[0.,0.,0.]])
    q=np.array([[2.,1.,0.]])
    assert a.segment_distance(p,p+[[1,0,0]],q,q+[[0,1,0]]).item()==pytest.approx(math.sqrt(2))


def test_many_simultaneous_segments():
    x=np.linspace(0.1,2.,500)
    p=np.zeros((500,3))
    q=np.zeros((500,3));q[:,0]=x
    v=a.segment_distance(p,p+[[0,0,1]],q,q+[[0,0,1]])
    assert np.allclose(v,x)


@pytest.mark.parametrize("bad", [np.nan,np.inf,-np.inf])
def test_nonfinite_endpoint_rejected(bad):
    p=np.array([[0.,0.,0.]])
    q=np.array([[bad,1.,0.]])
    with pytest.raises(ValueError, match="NONFINITE"):
        a.segment_distance(p,p+[[0,0,1]],q,q+[[0,0,1]])


def test_zero_length_segment_rejected():
    p=np.array([[0.,0.,0.]])
    with pytest.raises(ValueError,match="DEGENERATE"):
        a.segment_distance(p,p,p+[[1,0,0]],p+[[1,0,1]])


def test_bad_segment_dimensions_rejected():
    with pytest.raises(ValueError,match="BAD_SEGMENTS"):
        a.segment_distance(np.zeros((3,2)),np.zeros((3,2)),
                           np.zeros((3,2)),np.zeros((3,2)))


def test_joint_axis_rotation_and_nonnormalized_axis():
    r=a.axis_rotation([0.,0.,5.],np.array([0.,np.pi/2,np.pi]))
    assert np.allclose(r[0],np.eye(3))
    assert np.allclose(r[1]@np.array([1,0,0]),[0,1,0],atol=1e-12)
    assert np.allclose(r[2]@np.array([1,0,0]),[-1,0,0],atol=1e-12)


def test_bad_axis_rejected():
    with pytest.raises(ValueError,match="ZERO_JOINT_AXIS"):
        a.axis_rotation([0,0,0],np.array([0.]))


def test_rpy_rotation_order():
    M=a.rpy_matrix([0,0,np.pi/2])
    assert np.allclose(M @ np.array([1,0,0]),[0,1,0],atol=1e-12)
    M=a.rpy_matrix([0,np.pi/2,0])
    assert np.allclose(M @ np.array([0,0,1]),[1,0,0],atol=1e-12)


@pytest.mark.parametrize("txt", ["1 2","nan 2 3","a b c","1 2 3 4"])
def test_bad_urdf_vector_rejected(txt):
    with pytest.raises(ValueError,match="BAD_URDF_VECTOR"):
        a.numbers(txt,"0 0 0")


def test_synthethic_rpy_in_colliders_and_link_translation():
    shape={"type":"cylinder","radius":.05,"length":.2,
           "xyz":np.array([.1,0,0]),"R":a.rpy_matrix([0,np.pi/2,0])}
    tree={"links":{"target":[shape]}}
    identity=np.eye(3)[None,:,:]
    pose={"target":(identity,np.array([[1.,2.,3.]]))}
    x,y,r=a.cylinder_axis_endpoints(tree,pose,"target")
    assert np.allclose((x+y)/2,[1.1,2.,3.])
    assert np.allclose(y-x,[.2,0,0],atol=1e-12)
    assert r==.05


def test_missing_cylinder_refused():
    tree={"links":{"target":[]}}
    with pytest.raises(ValueError,match="CYLINDER"):
        a.cylinder_axis_endpoints(tree,{"target":(None,None)},"target")


def test_frozen_contract_and_real_29dof_urdf():
    cfg=a.frozen()
    assert cfg["native_physx_pair_certification_permitted"] is False
    src=Path("/home/you/projects/kinetickick-wt-v116/configs/paid_original/paid_main_resolved.urdf")
    a.require_source(src,cfg["source_urdf_sha256"])
    tree=a.parse_tree(src)
    assert tree["root"]=="pelvis"
    assert len(tree["joints"])==39
    assert len(tree["links"])==40
    poses=a.fk_targets(tree,np.zeros((2,29)),
                       {p for pair in cfg["focus_collision_pairs"] for p in pair})
    clear=a.clearance_for_episode(tree,np.zeros((2,29)),cfg["focus_collision_pairs"])
    assert len(clear)==4
    assert all(len(x)==2 and np.isfinite(x).all() for x in clear.values())
    assert all(v[0].shape==(2,3,3) and v[1].shape==(2,3)
               for v in poses.values())


def test_original_29_link_pose_changes_when_wrist_joint_rotates():
    cfg=a.frozen()
    src=Path("/home/you/projects/kinetickick-wt-v116/configs/paid_original/paid_main_resolved.urdf")
    tree=a.parse_tree(src)
    qs=np.zeros((2,29))
    qs[1,a.G1_JOINT_ORDER.index("left_wrist_yaw_joint")]=.6
    X=a.fk_targets(tree,qs,{"left_wrist_yaw_link"})["left_wrist_yaw_link"][0]
    assert not np.allclose(X[0],X[1])


def test_invalid_29dof_trajectory():
    cfg=a.frozen()
    src=Path("/home/you/projects/kinetickick-wt-v116/configs/paid_original/paid_main_resolved.urdf")
    tree=a.parse_tree(src)
    for q in (np.zeros((2,28)),np.full((2,29),np.nan)):
        with pytest.raises(ValueError,match="ORIGINAL_POSES"):
            a.fk_targets(tree,q,{"left_wrist_yaw_link"})


def test_three_phase_exact_contact_windows():
    cfg=a.frozen()
    t=np.arange(1,2801)*.005
    w=a.windows(t,1.58,cfg)
    assert sum(np.sum(m) for m in w.values())<2800
    assert [int(w[k].sum()) for k in
            ("precontact_400ms","postcontact_1250ms")]==[80,250]
    assert int(w["late_recovery"].sum())>2000


def test_truncated_contact_window_rejected():
    cfg=a.frozen()
    with pytest.raises(ValueError,match="INCOMPLETE"):
        a.windows(np.arange(30)*.005,1.58,cfg)


def test_source_sha_refuses_tampering_and_symlink(tmp_path):
    p=tmp_path/"foo.json"
    p.write_text('{"x":1}')
    digest=a.sha(p)
    a.require_source(p,digest)
    p.write_text('{"x":2}')
    with pytest.raises(ValueError,match="SHA256_DRIFT"):
        a.require_source(p,digest)
    s=tmp_path/"link"
    s.symlink_to(p)
    with pytest.raises(ValueError,match="SHA256_DRIFT"):
        a.require_source(s,a.sha(p))


def test_observer_negative_controls_must_not_certify_native_contact():
    cfg=a.frozen()
    assert not cfg["native_physx_pair_certification_permitted"]
    assert cfg["urdf_capsule_overlap_not_a_collision_proof"]
    assert cfg["crossed_pair_controls_are_not_true_collision_negatives"]
    assert cfg["expected_observer_repeat_count"]==5


def test_phase_metrics_keep_original_qdot_and_force_proxy():
    cfg=a.frozen()
    t=np.arange(1,2801)*.005
    clearance={"L__L":np.full(2800,.01)}
    dq=np.zeros((2800,29));dq[250,0]=2.
    nonfoot=np.zeros(2800)
    com=np.zeros((2800,3))
    support=np.ones((2800,2))*50.
    out=a.metrics_per_phase(clearance,t,1.58,dq,np.ones(29),
                            nonfoot,com,support,cfg)
    before=out["precontact_400ms"]
    assert before["original29_qdot_peak_limit_ratio"]==2.
    assert before["pair_geometry"]["L__L"]["under_20mm_alert_frame_fraction"]==1.
    assert before["pair_geometry"]["L__L"]["possible_overlap_frame_fraction"]==0.
    assert before["left_ground_net_force_proxy_integral_Ns"]==20.


def test_capsule_lower_bound_is_not_physics_truth():
    cfg=a.frozen()
    assert cfg["method"].startswith("per_frame_original_URDF")
    assert cfg["offline_read_only_no_actuation"]
    assert cfg["no_parameter_fitting"]


def test_scipy_independent_forward_kinematics_5_states():
    # Independent reference composition, does NOT call this implementation's
    # axis_rotation or rpy_matrix helpers; only XML tree data are shared.
    from scipy.spatial.transform import Rotation
    import xml.etree.ElementTree as ET
    src=Path("/home/you/projects/kinetickick-wt-v116/configs/paid_original/paid_main_resolved.urdf")
    root=ET.parse(src).getroot()
    tree=a.parse_tree(src)
    joint_by_child={}
    for j in root.findall("joint"):
        joint_by_child[j.find("child").get("link")]=j
    target={"left_wrist_yaw_link","left_hip_roll_link",
            "right_wrist_yaw_link","right_hip_roll_link"}
    rng=np.random.default_rng(127)
    qs=rng.uniform(-.3,.3,size=(5,29))
    fast=a.fk_targets(tree,qs,target)
    for i in range(5):
        for link in target:
            chain=[]; node=link
            while node!="pelvis":
                j=joint_by_child[node];chain.append(j)
                node=j.find("parent").get("link")
            pose=np.eye(4)
            for j in reversed(chain):
                o=j.find("origin")
                xyz=[float(v) for v in o.get("xyz","0 0 0").split()] if o is not None else [0,0,0]
                rpy=[float(v) for v in o.get("rpy","0 0 0").split()] if o is not None else [0,0,0]
                fixed=np.eye(4);fixed[:3,3]=xyz
                fixed[:3,:3]=Rotation.from_euler("xyz",rpy).as_matrix()
                pose=pose@fixed
                if j.get("type")=="revolute":
                    axis=j.find("axis")
                    v=np.array([float(w) for w in axis.get("xyz","1 0 0").split()]) if axis is not None else np.array([1.,0,0])
                    q=qs[i,a.G1_JOINT_ORDER.index(j.get("name"))]
                    move=np.eye(4)
                    move[:3,:3]=Rotation.from_rotvec(q*v/np.linalg.norm(v)).as_matrix()
                    pose=pose@move
            gotR,gotp=fast[link]
            assert np.allclose(gotR[i],pose[:3,:3],atol=1e-11)
            assert np.allclose(gotp[i],pose[:3,3],atol=1e-11)


def test_segment_separation_independent_scipy_minimization():
    from scipy.optimize import minimize
    rng=np.random.default_rng(127)
    for _ in range(8):
        p0=rng.uniform(-1,1,size=3)
        p1=p0+rng.uniform(-1,1,size=3)
        q0=rng.uniform(-1,1,size=3)
        q1=q0+rng.uniform(-1,1,size=3)
        got=a.segment_distance(p0[None,:],p1[None,:],q0[None,:],q1[None,:])[0]
        def objective(u):
            return float(np.linalg.norm((p0+u[0]*(p1-p0))-
                                        (q0+u[1]*(q1-q0)))**2)
        opt=minimize(objective,[.5,.5],method="L-BFGS-B",bounds=[(0,1),(0,1)],
                     options={"ftol":1e-14})
        assert opt.success
        assert got==pytest.approx(np.sqrt(max(0.,opt.fun)),abs=2e-6)

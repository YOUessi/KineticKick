"""v132: source integrity, topology accounting, convex outer-bound negatives."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize import minimize

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import v132_fullbody_broadphase as m
import v127_collider_proximity as kin
import v128_v54_event_geometry as old


def cap(a,b,r):
    return np.asarray(a,float).reshape(-1,3),np.asarray(b,float).reshape(-1,3),r


@pytest.mark.parametrize("distance,want",[
    (-.1,"OVERAPPROXIMATION_OVERLAP_UNKNOWN_NOT_NATIVE_CONTACT"),
    (0.,"OVERAPPROXIMATION_OVERLAP_UNKNOWN_NOT_NATIVE_CONTACT"),
    (.000011,"POSITIVE_GAP_WITHIN_20MM_WARNING"),
    (.020,"POSITIVE_GAP_WITHIN_20MM_WARNING"),
    (.02001,"POSITIVE_GAP_WITHIN_50MM_WATCH"),
    (.050,"POSITIVE_GAP_WITHIN_50MM_WATCH"),
    (.05001,"URDF_PRIMITIVES_DISJOINT_AT_OBSERVED_SAMPLE_ONLY"),
])
def test_gap_label_not_a_contact_claim(distance,want):
    assert m.risk_label(distance)==want


@pytest.mark.parametrize("invalid",[float("nan"),float("inf"),-float("inf")])
def test_nonfinite_label_refused(invalid):
    with pytest.raises(ValueError,match="NONFINITE"):
        m.risk_label(invalid)


@pytest.mark.parametrize("r",[-1,0,float("nan"),float("inf")])
def test_bad_primitive_radius_refused(r):
    with pytest.raises(ValueError,match="RADIUS"):
        m.conservative_gap(cap([0,0,0],[1,0,0],r),
                           cap([0,1,0],[1,1,0],.1))


def test_nonfinite_endpoints_rejected():
    with pytest.raises(ValueError,match="ENDPOINT"):
        m.conservative_gap(cap([math.nan,0,0],[1,0,0],.1),
                           cap([1,1,1],[2,1,1],.1))


def test_sphere_sphere_exact_positive_and_overlap_is_only_bound():
    a=cap([0,0,0],[0,0,0],.2)
    b=cap([1,0,0],[1,0,0],.3)
    assert m.conservative_gap(a,b)[0]==pytest.approx(.5)
    c=cap([.1,0,0],[.1,0,0],.3)
    assert m.conservative_gap(a,c)[0]==pytest.approx(-.4)
    assert "UNKNOWN" in m.risk_label(-.4)


def test_sphere_capsule_exact_point_to_segment():
    a=cap([0,0,0],[0,0,0],.1)
    b=cap([1,1,0],[3,1,0],.2)
    assert m.conservative_gap(a,b)[0]==pytest.approx(math.sqrt(2)-.3)


def test_capsule_sphere_argument_symmetry():
    a=cap([1,1,0],[3,1,0],.15)
    b=cap([0,0,0],[0,0,0],.25)
    assert m.conservative_gap(a,b)[0]==pytest.approx(
           m.conservative_gap(b,a)[0])


@pytest.mark.parametrize("pair", [
    (cap([0,0,0],[0,0,1],.1),cap([0,2,0],[0,2,1],.1),1.8),
    (cap([0,0,0],[0,0,1],.1),cap([1,0,.5],[1,0,1.5],.1),.8),
    (cap([0,0,0],[0,0,1],.1),cap([-1,0,.5],[1,0,.5],.1),-.2),
])
def test_parallel_offset_and_crossed_capsule_bounds(pair):
    a,b,expected=pair
    assert m.conservative_gap(a,b)[0]==pytest.approx(expected,abs=1e-10)


def test_invariance_under_common_translation():
    a=cap([0,0,0],[1,1,1],.15)
    b=cap([2,3,4],[4,3,2],.2)
    shift=np.array([[4.,-7.,3.]])
    x=m.conservative_gap(a,b)
    trans=lambda c:(c[0]+shift,c[1]+shift,c[2])
    np.testing.assert_allclose(m.conservative_gap(trans(a),trans(b)),x)


def test_vectorized_3d_analytic_against_independent_scipy():
    rng=np.random.default_rng(132)
    for _ in range(12):
        a,b,c,d=rng.normal(size=(4,3))
        a1,b1=cap(a,b,.03),cap(c,d,.02)
        direct=m.conservative_gap(a1,b1)[0]
        def objective(st):
            point1=a+st[0]*(b-a)
            point2=c+st[1]*(d-c)
            return float(np.sum((point1-point2)**2))
        r=min((minimize(objective,np.asarray(x),bounds=((0,1),(0,1)),
                        method="L-BFGS-B",options={"ftol":1e-13,"gtol":1e-11})
                for x in ((0,0),(.5,.5),(1,1))),key=lambda z:z.fun)
        assert direct==pytest.approx(math.sqrt(max(0,r.fun))-.05,abs=1e-6)


def test_graph_all_unique_and_real_topology():
    config=old.frozen_protocol()
    tree=kin.parse_tree(config["original_G1_urdf"])
    pairs,excluded=m.eligible_pairs(tree,3)
    assert len(pairs)==83
    assert excluded==8
    assert len(pairs)+excluded==91
    assert len({tuple(p) for p in pairs})==len(pairs)
    graph=m.adjacency(tree)
    for a,b in pairs:
        assert a!=b
        assert m.distances(graph,a)[b]>=3
    assert ("left_hip_roll_link","left_wrist_yaw_link") in pairs


def test_excluded_adjacent_not_false_safe():
    tree=kin.parse_tree(old.frozen_protocol()["original_G1_urdf"])
    graph=m.adjacency(tree)
    with pytest.raises(ValueError,match="INVALID_TOPOLOGY"):
        m.eligible_pairs(tree,0)
    assert graph["left_hip_roll_link"]


def test_source_original_urdf_29_shapes_include_multiple_per_link():
    cfg=old.frozen_protocol()
    tree=kin.parse_tree(cfg["original_G1_urdf"])
    shape_counts={}
    for shapes in tree["links"].values():
        for s in shapes:
            shape_counts[s["type"]]=shape_counts.get(s["type"],0)+1
    assert shape_counts=={"sphere":1,"cylinder":28}
    assert len(tree["links"]["left_ankle_roll_link"])==7
    assert len(tree["links"]["right_ankle_roll_link"])==7
    assert len(tree["links"]["torso_link"])==4


def test_all_link_shape_geometry_are_not_silently_dropped():
    urdf=Path(old.frozen_protocol()["original_G1_urdf"])
    tree=kin.parse_tree(urdf)
    q=np.zeros((1,29),float)
    poses=kin.fk_targets(tree,q,{x for x,s in tree["links"].items() if s})
    caps=m.primitive_capsules(tree,poses,m.sphere_radii(urdf))
    assert sum(map(len,caps.values()))==29
    assert len(caps["left_ankle_roll_link"])==7
    assert len(caps["right_ankle_roll_link"])==7
    assert len(caps["torso_link"])==4


@pytest.mark.parametrize("bad", [0,-.5,float("nan")])
def test_sphere_rejects_invalid_radius(bad):
    urdf=Path(old.frozen_protocol()["original_G1_urdf"])
    tree=kin.parse_tree(urdf)
    poses=kin.fk_targets(tree,np.zeros((1,29)),{x for x,s in tree["links"].items() if s})
    with pytest.raises(ValueError,match="SPHERE"):
        m.primitive_capsules(tree,poses,{"pelvis":bad})


def test_provenance_sha256_rejects_corrupt_original(tmp_path):
    p=tmp_path/"source.json"
    p.write_text("modified")
    with pytest.raises(ValueError,match="SOURCE_SHA256_DRIFT"):
        old.require_sha(p,"0"*64)


def test_rejects_swapped_joint_order_and_dropped_frame():
    cfg=old.frozen_protocol()
    p=Path(cfg["data_root"])/"v054_cmu_10_05_contact_diagnostics.json"
    real=json.loads(p.read_text())
    bad=copy.deepcopy(real)
    bad["joint_names"][0],bad["joint_names"][1]=bad["joint_names"][1],bad["joint_names"][0]
    with pytest.raises(ValueError,match="NOT_ORIGINAL_G1_29DOF"):
        old.validate_rows(bad,1029,.005)
    bad=copy.deepcopy(real)
    bad["traces"][0].pop(30)
    with pytest.raises(ValueError,match="MISSING_OR_TRUNCATED_REAL_PHYSICS"):
        old.validate_rows(bad,1029,.005)


def test_missing_event_step_not_invented():
    cfg=old.frozen_protocol()
    t=np.arange(1,1029)*.005
    witness=old.event_observability(t,cfg["true_nonfoot_event_t_s"],.005)
    assert not witness["event_has_directly_sampled_joint_state"]
    assert not witness["collision_at_unsampled_event_certified"]


def test_frozen_protocol_bytes_and_fail_closed():
    cfg=m.frozen()
    assert cfg["observation_dt_s"]==.005
    assert cfg["reference_steps_per_case"]["cmu_10_05"]==1028
    assert cfg["never_infer_collision_from_nonpositive_capsule_gap"]
    assert not cfg.get("actuator_authorized",False)


def test_real_audit_file_is_not_native_truth():
    path=Path(__file__).resolve().parents[1]/"reports/v132_fullbody_broadphase_original_G1_v3.audit.json"
    if not path.is_file():
        pytest.skip("ignored read-only real audit is local-only")
    audit=json.loads(path.read_text())
    assert audit["real_200hz_states_revalidated"]==6628
    assert audit["new_physical_episodes"]==0
    assert audit["exact_runtime_USD_shape_and_contact_manifold_unknown"]
    for name,result in audit["cases"].items():
        assert result["all_evaluated_pair_count"]==83
        assert result["entire_horizon_pairs_with_nonpositive_capsule_bound"]==0
        assert len(result["sampled_event_window"]["time_s"])==13
        assert result["no_native_PhysX_pair_witness"]
    failed=audit["cases"]["cmu_10_05"]
    assert failed["wrist_hip_near_event_rank"]==1
    assert failed["near_event_pairs_20mm_or_less"]==2
    assert failed["no_real_joint_sample_at_sensor_5_145"]

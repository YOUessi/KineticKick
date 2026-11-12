"""Pre-registered v129 separation-certificate properties and negative controls."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import v129_exact_cylinder_duality as m
import v127_collider_proximity as old
import v128_v54_event_geometry as historical

def C(x=0.,y=0.,z=0.,axis=(0,0,1),r=.05,half=.10):
    return m.Cylinder(np.array([x,y,z],float),np.asarray(axis,float),r,half)

@pytest.mark.parametrize("gap", [.001,.01,.04,.10,.30])
def test_exact_radial_disjoint_certificate_and_primal_matches(gap):
    a,b=C(),C(x=.1+gap)
    ans=m.separating_certificate(a,b,optimize=True)
    ub=m.primal_upper_bound(a,b)
    assert ans["strictly_separated"]
    assert abs(ans["lower_bound_m"]-gap)<1e-7
    assert abs(ub-gap)<1e-4
    assert ans["lower_bound_m"]<=ub+1e-6

@pytest.mark.parametrize("penetration", [.001,.02,.049,.08])
def test_overlapping_cylinders_never_certified_separated(penetration):
    a,b=C(),C(x=.10-penetration)
    ans=m.separating_certificate(a,b,optimize=True)
    assert not ans["strictly_separated"]
    assert ans["lower_bound_m"]==0
    assert m.primal_upper_bound(a,b)<1e-4

@pytest.mark.parametrize("s",[-.25,-.05,.0,.05,.25])
def test_dual_is_always_bound_on_simple_radial_witness(s):
    a,b=C(),C(x=.18)
    n=np.array([np.cos(s),np.sin(s),0])
    assert m.projection_gap(a,b,n)<=.08+1e-12

def test_axis_aligned_axial_separation():
    a,b=C(),C(z=.4)
    ans=m.separating_certificate(a,b,optimize=True)
    assert ans["strictly_separated"]
    assert abs(ans["lower_bound_m"]-.2)<1e-7
    assert abs(m.primal_upper_bound(a,b)-.2)<1e-4

def test_tangent_not_certified_at_epsilon():
    a,b=C(),C(x=.1)
    ans=m.separating_certificate(a,b,optimize=True)
    assert not ans["strictly_separated"] and ans["lower_bound_m"]<1e-6

def test_orthogonal_axes_separation():
    a,b=C(),C(y=.3,axis=(1,0,0))
    ans=m.separating_certificate(a,b,optimize=True)
    ub=m.primal_upper_bound(a,b)
    assert ans["strictly_separated"]
    assert abs(ans["lower_bound_m"]-.2)<1e-5
    assert abs(ub-.2)<1e-4

@pytest.mark.parametrize("bad",[0,float("nan"),float("inf"),-2])
def test_invalid_cylinder_radius_fail_closed(bad):
    with pytest.raises(ValueError):
        C(r=bad)

@pytest.mark.parametrize("bad",[np.array([0,0,0]),np.array([float('nan'),0,0]),np.array([1,1])])
def test_invalid_direction_fail_closed(bad):
    if len(bad)==3 and np.isfinite(bad).all():
        assert m.projection_gap(C(),C(x=.2),bad)==0
    else:
        with pytest.raises(ValueError):
            m.projection_gap(C(),C(x=.2),bad)

def test_disjoint_certificate_is_not_native_pair_claim():
    ans=m.separating_certificate(C(),C(x=.3))
    assert ans["strictly_separated"]
    assert ans["native_physx_contact_pair"] is None

def test_synthetic_urdf_capsule_can_be_looser_than_exact():
    # Two parallel finite cylinders offset along their axes: their bounding
    # capsules extend beyond the cylinder caps by r.
    a,b=C(),C(z=.23)
    exact=m.separating_certificate(a,b,optimize=True)["lower_bound_m"]
    capsule=(.23-.2)-.1
    assert capsule<0 and abs(exact-.03)<1e-7

def test_v129_parent_sha_and_freeze_rejection(tmp_path,monkeypatch):
    cfg=m.frozen()
    assert cfg["version"]=="v0.1.129"
    monkeypatch.setattr(m,"CONFIG",tmp_path/"bad_protocol.json")
    m.CONFIG.write_text("{}")
    with pytest.raises(ValueError,match="FROZEN_PROTOCOL_DRIFT"):
        m.frozen()

def test_real_v54_29dof_complete_and_no_invented_event_state():
    cfg=historical.frozen_protocol()
    root=Path(cfg["data_root"])
    rawpath=root/"v054_cmu_10_05_contact_diagnostics.json"
    historical.require_sha(rawpath,cfg["input_sha256"][rawpath.name])
    raw=json.loads(rawpath.read_text())
    data=historical.validate_rows(raw,1029,cfg["dt_s"])
    assert len(data["t"])==1028
    assert abs(data["t"][-1]-5.140)<2e-5
    assert not np.any(abs(data["t"]-5.145)<1e-4)
    assert data["q"].shape==(1028,29)
    tree=old.parse_tree(Path(cfg["original_G1_urdf"]))
    pairs=m.cylinders_from_q(tree,data["q"][-3:],*m.frozen()["primary_pair"])
    assert len(pairs)==2 and len(pairs[0])==3
    for a,b in zip(*pairs):
        dual=m.separating_certificate(a,b,optimize=True)
        primal=m.primal_upper_bound(a,b)
        assert dual["lower_bound_m"]<=primal+1e-6
        assert dual["native_physx_contact_pair"] is None

def test_pre_registration_excludes_motor_activation():
    cfg=m.frozen()
    assert cfg["no_gpu_or_actuation_during_gpu_mutex"]
    assert cfg["no_extrapolation_to_unsampled_sensor_event"]
    assert not cfg["native_physx_pair_certified"]
    assert cfg["no_tuning_on_holdout"]

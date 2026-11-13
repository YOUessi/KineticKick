"""v138 immutable protocol, C2 temporal 29DoF, solver hard constraints and hard veto."""
import dataclasses
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import v138_temporal_full29 as v


@pytest.fixture(scope="module")
def setup():
    c=v.frozen()
    p=v.lp.frozen()
    tree, limits=v.lp.original_kinematics(v.task.URDF,p)
    masses=v.task.mass_inertials(v.task.URDF,tree)
    q0=(limits["lo"]+limits["hi"])/2
    return c,p,tree,limits,masses,q0


@pytest.fixture(scope="module")
def baseline(setup):
    c,p,tree,limits,masses,q0=setup
    synth=v.chain.generate(v.chain.frozen(),q0,4.)
    ci=round(c["contact_proxy_time_s"]/c["sample_period_s"])
    model=v.lp.problem_at_pose(tree,masses,synth.q[ci],p)
    sol=v.min_change_witness(model,synth.dq[ci],limits,c,p,3.)
    assert sol["success"]
    return synth,model,sol


@pytest.fixture(scope="module")
def audit():
    return v.run()


def test_frozen_is_independent_previous_commit(setup):
    c,*_=setup
    assert subprocess.check_output(["git","-C",str(ROOT),"rev-parse",v.FROZEN_SHA+"^"],
                                   text=True).strip()==c["parent_sha"]
    assert c["mandatory_fail_closed"] is True
    assert c["actual_new_physx_rollouts"]==0
    assert c["strict_real_physics_gates"]["instep_impulse_fraction_min"]==.95


def test_freeze_drift_on_shadow_config(tmp_path,monkeypatch):
    shadow=tmp_path/"bad.json"
    shadow.write_bytes(v.CONFIG.read_bytes()+b" ")
    monkeypatch.setattr(v,"CONFIG",shadow)
    with pytest.raises(ValueError,match="PATH_DRIFT"):
        v.frozen()


def test_freeze_drift_at_same_path(tmp_path,monkeypatch):
    # Same path resolution can be simulated by replacing read_bytes rather than altering repository.
    c=v.frozen()
    monkeypatch.setattr(Path,"read_bytes",lambda path: b"{}")
    with pytest.raises(ValueError,match="CONFIG_BYTE_DRIFT"):
        v.frozen()


def test_wrong_urdf_source_cannot_pass(setup,tmp_path):
    c,p,*_=setup
    bad=tmp_path/"fake.urdf"
    bad.write_bytes(v.task.URDF.read_bytes()+b"\n<!-- modified -->")
    with pytest.raises(ValueError,match="SOURCE_SHA256_DRIFT"):
        v.lp.original_kinematics(bad,p)


@pytest.mark.parametrize("x",[-1.,0.,1.])
def test_bump_value_and_two_derivatives_zero_at_edges(x):
    h,d,d2=v.basis(np.array([x]))
    assert h[0]==pytest.approx(0,abs=1e-14) if abs(x)==1 else True
    if abs(x)==1:
        assert d[0]==d2[0]==0
    else:
        assert h[0]==0 and d[0]==1 and d2[0]==0


def test_bump_is_odd_first_derivative_even_and_second_odd():
    xs=np.array([-.91,-.37,-.1,.1,.37,.91])
    h,d,d2=v.basis(xs)
    np.testing.assert_allclose(h,-h[::-1],atol=1e-14)
    np.testing.assert_allclose(d,d[::-1],atol=1e-14)
    np.testing.assert_allclose(d2,-d2[::-1],atol=1e-14)


@pytest.mark.parametrize("x",[-.7,-.23,0.,.37,.83])
def test_analytic_derivatives_match_independent_centered_fd(x):
    eps=1e-6
    h,d,d2=v.basis(np.array([x]))
    hp,dp,_=v.basis(np.array([x+eps]))
    hm,dm,_=v.basis(np.array([x-eps]))
    np.testing.assert_allclose(d,(hp-hm)/(2*eps),rtol=2e-8,atol=1e-9)
    np.testing.assert_allclose(d2,(dp-dm)/(2*eps),rtol=2e-8,atol=1e-9)


@pytest.mark.parametrize("bad",[[np.nan],[np.inf],[-np.inf]])
def test_nonfinite_bump_rejected(bad):
    with pytest.raises(ValueError,match="NONFINITE_LOCAL_TIME"):
        v.basis(np.array(bad))


def test_original_g1_not_truncated(setup):
    c,p,tree,lim,masses,q0=setup
    assert q0.shape==(29,) and len(masses)==37
    assert len(v.lp.ARM)==14
    assert sum(m for _,m,_ in masses)==pytest.approx(33.341142022)


@pytest.mark.parametrize("bad",[-1.,3.5,False,float("nan")])
def test_target_grid_strict_reject(setup,baseline,bad):
    c,p,tree,lim,masses,q0=setup
    synth,model,sol=baseline
    with pytest.raises(ValueError,match="UNREGISTERED_VELOCITY_TARGET"):
        v.min_change_witness(model,synth.dq[590],lim,c,p,bad)


@pytest.mark.parametrize("size",[28,30])
def test_invalid_original29_input_to_min_norm_rejected(setup,baseline,size):
    c,p,tree,lim,masses,q0=setup
    synth,model,sol=baseline
    with pytest.raises(ValueError,match="BAD_29DOF"):
        v.min_change_witness(model,np.zeros(size),lim,c,p,1.)


def test_converged_min_norm_obeys_29velocity_com_arm_hard_constraints(setup,baseline):
    c,p,tree,lim,masses,q0=setup
    synth,model,sol=baseline
    x=sol["qdot"]
    assert sol["equality_error_mps"]<2e-5
    np.testing.assert_allclose(model["Jrelative"]@x,[3,0,0],atol=2e-5)
    assert np.max(abs(x)/lim["vmax"])<=1+1e-8
    assert np.max(abs(x[v.lp.ARM]-synth.dq[590,v.lp.ARM]))<=.25+1e-7
    assert np.max(abs(model["Jcom_relative"]@x))<=.1+1e-7


def test_min_norm_reduces_weighted_lp_vertex_cost(setup,baseline):
    c,p,tree,lim,masses,q0=setup
    synth,model,sol=baseline
    ref=synth.dq[590]
    bounds,A,b=v.lp.constraints(ref,lim,dict(model,com_cap=.1,arm_cap=.25),p)
    vertex=v.lp._lp(np.zeros(29),model["Jrelative"],np.array([3,0,0]),bounds,A,b)
    scale=np.maximum(lim["vmax"],1e-6)
    naive_obj=.5*np.sum(((vertex-ref)/scale)**2)
    assert sol["weighted_departure_obj"]<=naive_obj+1e-7


@pytest.mark.parametrize("width",[.1,.2,.4])
def test_exact_center_stitched_full_14s_200hz(setup,baseline,width):
    c,p,tree,lim,masses,q0=setup
    synth,model,sol=baseline
    q,dq,ddq=v.temporal_bridge(synth,c,width,sol["qdot"])
    assert q.shape==dq.shape==ddq.shape==(2800,29)
    ci=round(2.95/.005)
    np.testing.assert_allclose(q[ci],synth.q[ci],atol=1e-13)
    np.testing.assert_allclose(dq[ci],sol["qdot"],atol=1e-12)
    mask=np.abs(synth.t-2.95)>=width
    assert np.array_equal(q[mask],synth.q[mask])
    assert np.array_equal(dq[mask],synth.dq[mask])
    assert np.array_equal(ddq[mask],synth.ddq[mask])
    assert np.isfinite(ddq).all()


def test_invalid_temporal_halfwidth_fails(setup,baseline):
    c,p,tree,lim,masses,q0=setup
    synth,model,sol=baseline
    with pytest.raises(ValueError,match="UNFROZEN_C2_HALFWIDTH"):
        v.temporal_bridge(synth,c,.15,sol["qdot"])


def test_bad_temporal_velocity_shape_or_nan_rejected(setup,baseline):
    c,p,tree,lim,masses,q0=setup
    synth,model,sol=baseline
    for wrong in (np.zeros(28),np.r_[np.zeros(28),np.nan]):
        with pytest.raises(ValueError,match="BAD_TARGET_ORIGINAL29_VELOCITY"):
            v.temporal_bridge(synth,c,.2,wrong)


def test_broken_14s_input_fails_closed(setup,baseline):
    c,p,tree,lim,masses,q0=setup
    synth,model,sol=baseline
    bad=dataclasses.replace(synth,q=synth.q[:200])
    with pytest.raises(ValueError,match="NOT_14S_200HZ"):
        v.temporal_bridge(bad,c,.2,sol["qdot"])


def test_sampled_derivative_consistency_under_fast_bump(setup,baseline):
    c,p,tree,lim,masses,q0=setup
    synth,model,sol=baseline
    q,dq,ddq=v.temporal_bridge(synth,c,.2,sol["qdot"])
    fd=(q[2:]-q[:-2])/(.01)
    mask=np.abs(synth.t[1:-1]-2.95)<.2
    assert float(np.max(abs(fd[mask]-dq[1:-1][mask])))<.15


def test_kinematic_window_331_frames_and_4_pairs(setup,baseline):
    c,p,tree,lim,masses,q0=setup
    synth,model,sol=baseline
    z=v.kinematic_window(tree,masses,synth.q,c,v.chain.frozen()["geometry_pairs"])
    assert z["samples"]==331
    assert z["com_minus_ankle_m"].shape==(331,3)
    assert z["right_minus_left_ankle_m"].shape==(331,3)
    assert len(z["per_pair_minimum_capsule_separation_m"])==4


def test_sampled_model_and_real_ground_truth_cannot_be_confused(setup,baseline):
    c,p,tree,lim,masses,q0=setup
    synth,model,sol=baseline
    q,dq,ddq=v.temporal_bridge(synth,c,.1,sol["qdot"])
    w=v.kinematic_window(tree,masses,synth.q,c,v.chain.frozen()["geometry_pairs"])
    row=v.evaluate(synth,sol["qdot"],tree,masses,lim,c,.1,w)
    assert row["window_samples"]==331
    assert row["real_g1_physx_rollouts"]==0
    assert row["measured_grf_or_stance_patch"] is False
    assert row["native_contact_pair_certified"] is False
    assert row["motor_execution_authorized"] is False


def test_all_predeclared_45_attempts_and_15_seeds_in_audit(audit):
    assert audit["original29_dof"]==29
    assert len(audit["cases"])==15
    assert audit["summary_counts"]["temporal_attempts"]==45
    assert sum(len(row["temporal"]) for row in audit["cases"])==45
    assert audit["real_physx_rollouts_new"]==0
    assert audit["motor_execution_authorized"] is False


def test_frozen_grid_target_and_width_completeness(audit):
    keys={(c["multiplier"],c["target_right_ankle_link_origin_x_mps"],t["halfwidth_s"])
          for c in audit["cases"] for t in c["temporal"]}
    assert len(keys)==45
    assert keys=={(float(m),float(t),float(w))
                 for m in [0,1,2,4,-1] for t in [0,1,3] for w in [.1,.2,.4]}


def test_explicit_failure_metrics_not_censored(audit):
    for c in audit["cases"]:
        assert isinstance(c["witness"]["success"],bool)
        for t in c["temporal"]:
            if c["witness"]["success"]:
                for metric in ("position_limit_violating_200hz_frames",
                               "velocity_limit_violating_200hz_frames",
                               "descriptive_accel_violating_200hz_frames",
                               "min_urdf_four_pair_capsule_margin_m"):
                    assert metric in t
            assert not t["strict_physx_success"]
            assert not t["motor_execution_authorized"]


def test_frozen_45_grid_does_not_hide_failed_candidates(audit):
    counts=audit["summary_counts"]
    assert counts["min_change_solutions"]+counts["min_change_rejections"]==15
    assert counts["q_limit_bad_attempts"]>0
    assert counts["accel_screen_bad_attempts"]>0
    assert counts["cpu_screen_pass"]<45


def test_synthetic_virtual_anchor_never_mislabelled_true_instep(audit):
    assert audit["is_real_instep_contact_time"] is False
    assert audit["is_human_motion_or_ground_force_learned"] is False
    assert audit["is_contact_physics_authenticated"] is False
    assert audit["strict_real_physx_success_rate"].startswith("N/A")

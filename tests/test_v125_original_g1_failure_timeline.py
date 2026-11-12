"""Frozen SHA original 29DoF true-physics chronology and conjunctive hard gate."""
from __future__ import annotations
import copy,json,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import v125_original_g1_failure_timeline as m
CFG=json.loads(m.CFG_PATH.read_text())

def sample(impact=1.5):
    n=2800;t=.005*np.arange(1,n+1)
    vel=np.zeros((n,29),float)
    torque=np.zeros_like(vel)
    limits=np.ones(29)
    force=np.full((n,2),50.)
    com=np.stack((.3*t,np.zeros(n),np.ones(n)),axis=1)
    return dict(t=t,vel=vel,tau=torque,q=np.zeros_like(vel),limits=limits,
                support=force,com=com,nonfoot=np.zeros(n),impact=impact)

def gates(pass_now=True):
    out=dict(original_native_wrapper_source_certified=True,original_goal_entered=True,
             recovered=True,whole_14s_observed=True,measured_surface="instep",
             measured_instep_fraction=.97,ball_speed_peak_mps_post_5_to_110ms_proxy=4.5,
             original_29_qdot_max_ratio=.8,measured_nonfoot_max_net_N=0.,
             original_physical_nonfoot_terminated=False,hard_gate_pass=True,failed_gates=[])
    if not pass_now:
        out["original_29_qdot_max_ratio"]=1.1
        out["hard_gate_pass"]=False
        out["failed_gates"]=["original_29_qdot_within_limit"]
    return out

def test_frozen_sha_original_source_version_and_no_motor():
    m.assert_frozen(CFG)
    assert len(CFG["robot_sources"])==3
    assert CFG["no_motor_command"] and CFG["no_new_real_physics_if_gpu_busy"]

def test_nonoverlap_and_expected_80_250_windows():
    p=sample()
    s=m.windows(p["t"],p["impact"],CFG)
    assert int(s["precontact_400ms"].sum())==80
    assert int(s["follow_through_first1250ms"].sum())==250
    assert int(s["late_recovery"].sum())>2000
    assert np.all(np.sum(np.stack(list(s.values()),axis=0),axis=0)<=1)

def test_missing_true_postcontact_measured_horizon_fails_closed():
    p=sample()
    with pytest.raises(ValueError,match="CENSORED_POSTCONTACT"):
        m.windows(p["t"][:330],p["impact"],CFG)

def test_full29_joint_includes_last_joint_not_leg_only():
    p=sample();p["vel"][270,28]=1.35
    names=[f"joint_{i}" for i in range(29)]
    row=m.stage_metrics(p,m.windows(p["t"],p["impact"],CFG)["precontact_400ms"],names)
    assert row["maximum_original29_qdot_ratio"]==pytest.approx(1.35)
    assert row["max_ratio_joint"]=="joint_28"

@pytest.mark.parametrize("time_s,stage",[(.4,"precontact"),(1.5,"follow_through_first1250ms"),(4.0,"late_recovery")])
def test_first_qdot_violation_chronology_is_time_and_phase_qualified(time_s,stage):
    p=sample()
    i=round(time_s/.005)-1
    p["vel"][i,10]=1.0002
    out=m.locate_first_violation(p,[f"joint_{j}" for j in range(29)],CFG)
    assert out["found"] and out["stage"]==stage and out["joint"]=="joint_10"
    assert out["time_s"]==pytest.approx(time_s)

def test_first_violation_near_time_zero_is_not_boolean_false():
    p=sample();p["vel"][0,0]=1.1
    out=m.locate_first_violation(p,[f"j{i}" for i in range(29)],CFG)
    assert out["time_s"]==pytest.approx(.005)

def test_force_proxy_integral_and_no_human_grf_claim():
    p=sample()
    s=m.windows(p["t"],p["impact"],CFG)["precontact_400ms"]
    row=m.stage_metrics(p,s,[f"j{i}" for i in range(29)])
    assert row["left_foot_net_force_proxy_integral_Ns"]==pytest.approx(20.)
    assert row["right_foot_net_force_proxy_integral_Ns"]==pytest.approx(20.)
    assert row["proxy_left_minus_right_over_sum"]==0

def test_original_conjunction_pass_and_fail_on_nonfoot_or_speed():
    d=gates()
    assert m.independently_score(d,CFG)
    e=gates(False)
    assert m.independently_score(e,CFG) is False
    e=copy.deepcopy(d);e["measured_nonfoot_max_net_N"]=20.0;e["hard_gate_pass"]=False
    assert m.independently_score(e,CFG) is False
    e=copy.deepcopy(d);e["ball_speed_peak_mps_post_5_to_110ms_proxy"]=3.499;e["hard_gate_pass"]=False
    assert m.independently_score(e,CFG) is False

def test_original_gate_disagreement_and_missing_data_rejected():
    a=gates(False);a["hard_gate_pass"]=True
    with pytest.raises(ValueError,match="DISAGREEMENT"):
        m.independently_score(a,CFG)
    a=gates();a.pop("measured_surface")
    with pytest.raises(ValueError,match="MISSING_ORIGINAL"):
        m.independently_score(a,CFG)

def test_full_real_one_14second_episode_verified_by_original_sha():
    src=Path("/home/you/projects/kinetickick-wt-v116")
    fp=src/"reports/v116_AG_v32_off.json"
    if not fp.is_file():pytest.skip("local ignored real PhysX evidence unavailable")
    manifest=json.loads((src/"reports/v116_original_g1_batch.audit.json").read_text())
    row=next(x for x in manifest["runs"] if x["name"]=="AG_v32_off")
    m.prior.verify_file(fp,row["raw_sha256"])
    result=m.audit_episode(json.loads(fp.read_text()),row,CFG)
    assert result["physical_steps"]==2800
    assert result["first_qdot_violation"]["found"]
    assert result["failed_gates"]==["original_29_qdot_within_limit"]
    assert result["strict_hard_gate_pass"] is False
    assert result["native_collider_pair_certified"] is False

@pytest.mark.parametrize("bad",[None,float("nan"),True,"4.5"])
def test_missing_nonfinite_or_wrongly_typed_raw_numeric_evidence_fails_closed(bad):
    a=gates()
    a["ball_speed_peak_mps_post_5_to_110ms_proxy"]=bad
    with pytest.raises(ValueError,match="UNQUALIFIED_NUMERIC_HARD_GATE_EVIDENCE"):
        m.independently_score(a,CFG)

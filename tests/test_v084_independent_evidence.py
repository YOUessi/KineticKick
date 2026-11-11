"""v84 frozen new scenario / nonfoot / independent full-state audit regressions."""
import json, sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from v084_evidence import audit_episode
from v084_batch import parity
from v080_contact_poll import witness
def fixture_raw(n=5):
    return {"seed":1219,"root_fixed":False,"joint_names":[f"joint_{i}" for i in range(29)],
      "joint_velocity_limits":[[2.]*29],
      "records":[{"contact_time_s":None,"requested_surface_impulse_fraction":0.76,"goal":True,
                  "recovered":False}],
      "traces":[[{"time_s":round((i+1)*.005,3),"physics_dt_s":.005,
        "joint_vel":[0.1]*29,"support_foot_force_N":{"left":0.,"right":100.},
        "whole_body_com_xyz":[0.,0.,.5],"root_xyz":[0.,0.,.75],
        "root_tilt_deg":4.,"nonfoot_contact_N":0.} for i in range(n)]],
      "nonfoot_contact_diagnostics":[]}
def test_matrix_and_freeze_limits():
    cfg=json.loads((ROOT/"configs/v084_frozen_generalization.json").read_text())
    assert len(cfg["cases"])==7 and cfg["actual_reset_perturbation"] is False
    a,b=cfg["cases"][:2]
    assert a["control"]==b["control"]=="v32"
    assert (a["seed"],a["ball_offset_xy"],a["velocity_delta"])==(
             b["seed"],b["ball_offset_xy"],b["velocity_delta"])
    assert a["native_poll"] is False and b["native_poll"] is True
    assert len(set(x["name"] for x in cfg["cases"]))==7
    assert cfg["gates"]["instep_fraction_min"]>=.95
def test_true_29_joint_200hz_audit_support_and_no_imputed_window():
    s=audit_episode(fixture_raw(),{"seed":1219})
    assert s["frames"]==5 and s["peak_qdot_over_original_limit"]==.05
    assert s["support_fraction"]=={"R":1.}
    assert not s["measured_true_contact_window"] and not s["complete_14s"]
def test_true_qdot_all_indices_hard_limit_negative():
    obj=fixture_raw()
    obj["traces"][0][3]["joint_vel"][28]=4.
    assert audit_episode(obj,{"seed":1219})["peak_qdot_over_original_limit"]==2.
def test_reject_fabricated_29doF_and_time_gap_and_seed():
    obj=fixture_raw()
    obj["traces"][0][0]["joint_vel"].pop()
    with pytest.raises(ValueError,match="200HZ"):audit_episode(obj,{"seed":1219})
    obj=fixture_raw();obj["traces"][0][3]["time_s"]+=.01
    with pytest.raises(ValueError,match="NONCONTIGUOUS"):audit_episode(obj,{"seed":1219})
    with pytest.raises(ValueError,match="SEED"):audit_episode(fixture_raw(),{"seed":1223})
def test_reject_native_colliders_without_pair_impulse_and_contact_point():
    event={"actor_paths":["/robot/left_wrist_yaw_link","/robot/left_hip_roll_link"],
           "collider_paths":["/wrist/shape","/hip/shape"],
           "contacts":[{"point_w":[0,0,0],"normal_w":[0,1,0],"impulse_scalar":0.0}]}
    assert not witness(event)
    event["contacts"][0]["impulse_scalar"]=0.2
    assert witness(event)
    event["contacts"][0]["point_w"]=None
    assert not witness(event)
def test_exact_matching_original_state_pair(tmp_path):
    (tmp_path/"reports").mkdir()
    def make(poll):
        return {"traces":[[{"time_s":.005,"joint_vel":[0.]*29,"physics_dt_s":.005}]],
                "records":[{"goal":True}],"joint_names":[str(i) for i in range(29)],
                "termination_reasons":[[]],"sequence_records":[],"contacts":[],
                **({"physx_native_poll_v080":{"wrist_hip_direct_contact_certified":False,
                "native_headers":0}} if poll else {})}
    for k,poll in (("C_v32_off",False),("C_v32_on",True)):
        (tmp_path/"reports"/f"v084_{k}.json").write_text(json.dumps(make(poll)))
    res=parity(tmp_path)
    assert res["pass"] and res["frames_compared"]==1
    assert res["zero_native_headers_means"]=="UNKNOWN_NOT_ABSENCE"
    p=tmp_path/"reports"/"v084_C_v32_on.json"
    d=json.loads(p.read_text());d["traces"][0][0]["joint_vel"][28]=.001
    p.write_text(json.dumps(d))
    assert not parity(tmp_path)["pass"]
def test_native_missing_result_does_not_pass(tmp_path):
    (tmp_path/"reports").mkdir()
    d={"traces":[[{"a":1}]],"records":[],"contacts":[],"joint_names":[],
       "sequence_records":[],"termination_reasons":[]}
    for name in ("C_v32_off","C_v32_on"):
        (tmp_path/"reports"/f"v084_{name}.json").write_text(json.dumps(d))
    out=parity(tmp_path)
    assert not out["pass"] and not out["native_wrist_hip_contact_certified"]

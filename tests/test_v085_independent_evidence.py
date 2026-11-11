"""v85 frozen original-G1 contact-truth controls and independent hard-gate regressions."""
import json
import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from v085_contact_truth import pair_in_paths, valid_positive
from v085_batch import parity
from v085_evidence import audit_episode

def native_event(impulse=0.0, sides=("left_wrist_yaw_link","left_hip_roll_link")):
    return {"actor_paths": ["/env/Robot/"+sides[0],"/env/Robot/"+sides[1]],
        "collider_paths": ["/env/Robot/"+sides[0]+"/collision",
                           "/env/Robot/"+sides[1]+"/collision"],
        "contacts":[{"point_w":[0.0,0.0,0.1],"normal_w":[0.0,1.0,0.0],
                     "impulse_w":[0.0,impulse,0.0],"impulse_scalar":None}]}

def test_native_pair_requires_real_positive_geometry_and_matching_roles():
    event=native_event()
    assert pair_in_paths(event,("left_wrist_yaw_link","left_hip_roll_link"))
    assert not valid_positive(event)
    event=native_event(0.25)
    assert valid_positive(event)
    event["contacts"][0]["normal_w"]=[0.0,0.0,0.0]
    assert not valid_positive(event)

def test_cross_header_body_names_not_evidence():
    e=native_event(1.0)
    e["actor_paths"]=["/env/Robot/left_wrist_yaw_link","/env/Robot/ball"]
    e["collider_paths"]=["/env/Robot/hip_shape","/env/Robot/left_hip_roll_link"]
    assert not pair_in_paths(e,("left_wrist_yaw_link","left_hip_roll_link"))

def test_control_and_readonly_matrix_identical_reset():
    cfg=json.loads((ROOT/"configs/v085_frozen_native_contact.json").read_text())
    assert cfg["version"]=="v0.1.85"
    assert len(cfg["cases"])==6
    assert cfg["settings"]["min_free_gib"]>=20
    assert cfg["gates"]["instep_fraction_min"]>=0.95
    assert not cfg["actual_reset_perturbation"]
    assert not cfg["truth_limits"]["zero_native_headers_excludes_collision"]
    for i in (0,2,4):
        a,b=cfg["cases"][i:i+2]
        assert a["control"]==b["control"]
        assert (a["seed"],a["ball_offset_xy"],a["velocity_delta"]) == (
            b["seed"],b["ball_offset_xy"],b["velocity_delta"])
        assert not a["native_poll"] and b["native_poll"]

def dummy_raw():
    return {"root_fixed":False,"seed":1201,
      "joint_names":[f"j{k}" for k in range(29)],
      "joint_velocity_limits":[[2.0]*29],
      "records":[{"contact_time_s":None}],
      "traces":[[{"physics_dt_s":0.005,"time_s":round(0.005*(k+1),4),
          "joint_vel":[0.0]*29,"support_foot_force_N":{"left":60.0,"right":0.0},
          "whole_body_com_xyz":[0,0,0.7],"root_xyz":[0,0,0.75],
          "root_tilt_deg":2.0,"nonfoot_contact_N":0.0} for k in range(5)]]}

def test_qdot_requires_29_and_finite():
    x=dummy_raw()
    assert audit_episode(x,{"seed":1201})["frames"]==5
    x["traces"][0][2]["joint_vel"][28]=10
    assert audit_episode(x,{"seed":1201})["peak_qdot_over_original_limit"]==5
    x["traces"][0][2]["joint_vel"][28]=float("nan")
    with pytest.raises(ValueError,match="NONFINITE"):
        audit_episode(x,{"seed":1201})

def test_early_termination_never_imputes_14s():
    x=dummy_raw()
    evidence=audit_episode(x,{"seed":1201})
    assert not evidence["complete_14s"]
    assert not evidence["measured_true_contact_window"]

def test_pair_exhaustive_frame_and_global_equality(tmp_path):
    reports=tmp_path/"reports";reports.mkdir()
    d={"traces":[[{"physics_dt_s":.005,"joint_vel":[0]*29,"root_xyz":[0,0,.7]}]],
       "records":[{}],"contacts":[],"joint_names":[str(x) for x in range(29)],
       "termination_reasons":[[]],"sequence_records":[]}
    for name,poll in [("center_v32_off",False),("center_v32_on",True)]:
        datum=dict(d)
        if poll:
            datum["physx_native_report_audit_v085"]={
                "wrist_hip_direct_contact_certified":False,"native_headers":0}
        (reports/f"v085_{name}.json").write_text(json.dumps(datum))
    assert parity(tmp_path)["pass"]
    d2=json.loads((reports/"v085_center_v32_on.json").read_text())
    d2["traces"][0][0]["joint_vel"][28]=.001
    (reports/"v085_center_v32_on.json").write_text(json.dumps(d2))
    assert not parity(tmp_path)["pass"]

def test_missing_native_observer_fails_parity(tmp_path):
    reports=tmp_path/"reports";reports.mkdir()
    d={"traces":[[{"physics_dt_s":.005}]],"records":[],"contacts":[],
       "joint_names":[],"termination_reasons":[],"sequence_records":[]}
    for name in ("center_v32_off","center_v32_on"):
        (reports/f"v085_{name}.json").write_text(json.dumps(d))
    assert not parity(tmp_path)["pass"]

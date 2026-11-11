"""Postregistered regression for nonphysical termination annotation in 200Hz original G1 trace."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from v085_evidence_termination_shim import terminal_only_extra_step,audit_episode

def fixture():
    raw={"seed":1201,"root_fixed":False,
        "joint_names":[f"j_{i}" for i in range(29)],
        "joint_velocity_limits":[[3.]*29],
        "records":[{"contact_time_s":None}],
        "traces":[[]]}
    for i in range(3):
        raw["traces"][0].append({
            "time_s":(i+1)*.005,"physics_dt_s":.005,"joint_vel":[0.]*29,
            "support_foot_force_N":{"left":100.0,"right":0.0},
            "whole_body_com_xyz":[0.,0.,.7],"root_xyz":[0.,0.,.8],
            "root_tilt_deg":2.,"nonfoot_contact_N":0., "terminated":False})
    return raw

def test_exact_duplicate_termination_marker_is_not_extra_200hz_step():
    d=fixture()
    marker=dict(d["traces"][0][-1],time_s=.015001,terminated=True)
    d["traces"][0].append(marker)
    assert terminal_only_extra_step(d["traces"][0])
    result=audit_episode(d,{"seed":1201})
    assert result["raw_recorder_rows"]==4
    assert result["pure_termination_marker_rows"]==1
    assert result["frames"]==3
    assert not result["complete_14s"]

def test_modified_terminal_physics_is_not_silently_discarded():
    d=fixture()
    marker=dict(d["traces"][0][-1],time_s=.015001,terminated=True)
    marker["nonfoot_contact_N"]=25.
    d["traces"][0].append(marker)
    assert not terminal_only_extra_step(d["traces"][0])

def test_true_new_physics_step_never_discarded():
    d=fixture()
    marker=dict(d["traces"][0][-1],time_s=.020,terminated=True)
    d["traces"][0].append(marker)
    assert not terminal_only_extra_step(d["traces"][0])

"""CPU-only invariants. No stub result counted as true G1 physics."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from massfree_body_observer_v062 import SafeBodyObserver, BODIES, MODES
def fake():
    import torch
    names=list(BODIES)+["right_hip_roll_link"]
    robot=SimpleNamespace(body_names=names,data=SimpleNamespace(
        body_com_pos_w=torch.ones(1,len(names),3),
        body_com_lin_vel_w=torch.zeros(1,len(names),3),
        body_pos_w=torch.arange(len(names)*3.).reshape(1,len(names),3),
        body_lin_vel_w=torch.zeros(1,len(names),3),
        joint_pos=torch.zeros(1,29),joint_vel=torch.zeros(1,29)))
    sensor=SimpleNamespace(body_names=names,
                           data=SimpleNamespace(net_forces_w=torch.zeros(1,len(names),3)))
    return SimpleNamespace(num_envs=1,scene={"robot":robot,"contact_forces":sensor},
                           sim=SimpleNamespace(current_time=.005))
def test_config_has_eight_frozen_real_G1_cases():
    c=json.loads((ROOT/"configs/g1_massfree_probe_v062.json").read_text())
    assert c["version"]=="v0.1.62" and len(c["cases"])==8
    assert set(x["probe"] for x in c["cases"])==set(MODES)
    assert all(x["seed"]==1201 for x in c["cases"])
    assert c["control_tuning"] is False and c["original_29DoF"] is True
    assert len(set(x["output"] for x in c["cases"]))==8
def test_safe_full_emits_ten_body_and_29dof_read_only():
    x=SafeBodyObserver(fake(),"safe_full")
    for _ in range(30):x.capture(.005)
    s=x.result()
    assert s["actual_200hz_rows"]==len(s["rows"])==30
    assert len(s["rows"][0]["bodies_w"])==10
    assert len(s["rows"][0]["joint_vel_rad_s"])==29
    assert "weighted_com_w" not in s["rows"][0]
    assert s["body_com_is_not_mass_weighted_in_safe_full"]
    assert not s["physical_writes"] and not s["actuator_writes"]
def test_invalid_dt_and_mode_fail_closed():
    x=SafeBodyObserver(fake(),"safe_full")
    with pytest.raises(ValueError,match="200Hz"):x.capture(.02)
    with pytest.raises(ValueError,match="Insufficient"):x.result()
    with pytest.raises(ValueError,match="Unknown"):SafeBodyObserver(fake(),"made_up")
def test_no_actuation_contact_mutations_in_probe():
    source=(ROOT/"tools/massfree_body_observer_v062.py").read_text()
    for symbol in ["set_joint_position_target(","ApplyAPI(","set_world_poses("]:
        assert symbol not in source

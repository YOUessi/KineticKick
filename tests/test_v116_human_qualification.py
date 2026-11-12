"""Fail closed: motion capture markers do not imply real human impact forces."""
import pytest
from tools.v116_human_qualification import inspect, eligibility

def test_invalid_duration_fails():
    with pytest.raises(ValueError,match="DIMENSIONS"):
        eligibility(frames=0,rate=120,markers=30,analogs=0)

def test_no_false_force_or_elite_labels():
    s=eligibility(frames=400,rate=120,markers=50,analogs=0)
    assert s["kinematics_only"]
    assert not s["real_force_plate_grf_cop"]
    assert not s["verified_instep_impulse"]
    assert not s["elite_skill_verified"]

def test_real_unique_cmu_source_not_elite_kinetics():
    p=inspect()
    assert p["unique_trials"]==6 and p["unique_subjects"]==2
    assert p["marker_frames"]>3000
    assert p["analog_channels"]==0
    assert p["qualified_true_ball_impulse_trials"]==0
    assert all(x["rate_hz"]==120 for x in p["trials"])

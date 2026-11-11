"""Prevent ball-in-goal or legacy wrapper PASS hiding illegal instep fraction."""
import numpy as np
from tools.audit_v067_hard_gates import physically_scored

def raw_fraction(frac):
    return dict(records=[dict(contact_time_s=1.57,goal=True,
             requested_surface_impulse_fraction=frac,measured_surface='instep',recovered=True)],
        traces=[[dict(time_s=float(i*.005),joint_pos=[0.]*29,joint_vel=[.1]*29,
                      joint_target=[0.]*29, ball_velocity=[4.,0.,0.],
                      nonfoot_contact_N=0.,physics_dt_s=.005) for i in range(2801)]],
        joint_velocity_limits=[[5.]*29],termination_reasons=[[]],stride_shot_success=True)
def test_only_95percent_legal_instep_can_pass_all_other_gates():
    # Original structured report minimal fixture, NOT a real PhysX result.
    from unittest.mock import patch
    with patch('tools.audit_v067_hard_gates.assert_complete_physics_fields',lambda _:None):
        a=physically_scored(raw_fraction(.7298347),True)
        assert a['hard_gate_pass'] is False
        assert 'original_measured_instep_impulse_ge_95pct' in a['failed_gates']
        good=physically_scored(raw_fraction(1.),True)
        assert good['hard_gate_pass'] is True
        missing=physically_scored(raw_fraction(1.),False)
        assert missing['hard_gate_pass'] is False

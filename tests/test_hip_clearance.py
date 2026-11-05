"""Recorded collision geometry and post-contact-only braking contracts."""
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from hip_clearance import HipClearance, braking_correction
from running_reference_geometry import ReferenceGeometry, segment_distance
ROOT=Path(__file__).resolve().parents[1]


def test_recorded_self_collision_and_roll_dependent_clearance_match_original_urdf():
    geometry=HipClearance(ROOT/'configs/paid_original/paid_main_resolved.urdf','cpu')
    original=ReferenceGeometry(ROOT)
    # Failed combined episode at1.600,1.640,1.655s, plus a roll angle whose
    # closest capsule point is internal: an endpoint-only shortcut is wrong.
    pitch=torch.tensor([-1.27660203,-1.44901609,-1.49774158,-1.45])
    roll=torch.tensor([.25232443,.23441742,.22311985,.5])
    actual=geometry.gap(pitch,roll)
    expected=[]
    for a,b in zip(pitch,roll):
        q=np.zeros(29);q[original.joint_names.index('right_hip_pitch_joint')]=a
        q[original.joint_names.index('right_hip_roll_joint')]=b
        primitives=original.primitives(q,[0,0,0],[1,0,0,0])
        pelvis=next(p for p in primitives if p['name']=='pelvis')
        thigh=next(p for p in primitives if p['name']=='right_hip_roll_link')
        expected.append(segment_distance(pelvis['start'],pelvis['end'],thigh['start'],thigh['end'])-pelvis['radius']-thigh['radius'])
    np.testing.assert_allclose(actual,expected,atol=1e-7)
    assert actual[0]>0 and actual[1]>0 and actual[2]<0
    floor,feasible=geometry.pitch_floor(roll,.0005)
    assert bool(feasible.all())
    torch.testing.assert_close(geometry.gap(floor,roll),torch.full_like(floor,.0005),atol=1e-7,rtol=0)
    assert float(floor[3]-floor[0])>.1


def test_brake_anticipates_recorded_overshoot_but_preserves_contact_and_safe_motion():
    geometry=HipClearance(ROOT/'configs/paid_original/paid_main_resolved.urdf','cpu')
    floor,_=geometry.pitch_floor(torch.tensor([.25232443]),.0005)
    q=torch.tensor([-1.27660203]*5);v=torch.tensor([-5.12580633]*4+[2.])
    # Active separated swing, still touching, nominal protected, safe angle,
    # and the foot already returning away from the pelvis.
    q[3]=-.8
    active=torch.tensor([True,False,False,True,True])
    correction,predicted=braking_correction(q,v,floor.expand(5),active,
                                         deceleration=60.,gain=2.,maximum=.25)
    assert q[0]>floor[0] and predicted[0]<floor[0]
    assert 0<float(correction[0])<=.25
    assert bool((correction[1:]==0).all())


def test_support_gated_zero_velocity_is_not_pulse_completion():
    from hip_clearance import pulse_has_finished
    frame=torch.tensor([77.,79.,80.]);velocity=torch.tensor([0.,0.,-2.])
    # At frame77 the left support can be below80N, suppressing the pulse
    # temporarily. It may resume; zero commanded velocity is not completion.
    assert pulse_has_finished(frame,velocity,79.).tolist()==[False,True,False]

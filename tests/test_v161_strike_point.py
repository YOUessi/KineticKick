"""Verify task-space surface solving and contact-gated return logic on original Tang assets."""
import json
from pathlib import Path

import numpy as np
from cartesian_strike import LegKinematics
from v161_strike_point import solve_point,make_controller

ROOT=Path(__file__).resolve().parents[1]


def test_point_and_normal_ik_recovers_a_reachable_surface_target():
    cfg=json.loads((ROOT/'configs/v161_strike_r.json').read_text())
    leg=LegKinematics(cfg['urdf'])
    root=np.eye(4)
    intended=np.array([-.55,.10,.25,1.2,.30,-.10])
    pose,_=leg.forward(intended,root)
    local=np.array([.110,.01,-.015]);normal=np.array([.35,.05,.936]);normal/=np.linalg.norm(normal)
    world=pose[:3,3]+pose[:3,:3]@local;direction=pose[:3,:3]@normal
    q=solve_point(leg,root,intended+.04,local,normal,world,direction)
    actual,_=leg.forward(q,root)
    np.testing.assert_allclose(actual[:3,3]+actual[:3,:3]@local,world,atol=.001)
    np.testing.assert_allclose(actual[:3,:3]@normal,direction,atol=.015)
    assert np.all((q>=leg.limits[:,0]) & (q<=leg.limits[:,1]))


def test_landing_control_preserves_the_entire_unstarted_phase_bit_for_bit():
    cfg=json.loads((ROOT/'configs/v161_strike_r.json').read_text())
    names=json.loads((Path('/home/you/projects/kinetickick/reports/paid_physics_parity_clean_paid.json')).read_text())
    control=make_controller({},names,cfg)
    value=np.arange(29,dtype=float)/31.
    result=control({'t':1.5,'body_pos':np.zeros((30,3))},value)
    np.testing.assert_array_equal(result,value)

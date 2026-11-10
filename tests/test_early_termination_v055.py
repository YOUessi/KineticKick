"""CPU-only regression: valid original physical stops are scored negative, not discarded.

The generated fixture is synthetic *test data*, never treated as a PhysX result.
"""
from __future__ import annotations
import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from resume_terminated_physics_v055 import admit, sha
from audit_pair_dynamics_v055 import measure_200hz


class TestRealPhysicsTerminationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c=json.loads((ROOT/"configs/physx_pair_dynamics_v055.json").read_text())
        cls.case=c["cases"][9]
        cls.policy_sha=sha(ROOT/c["policies"][cls.case["policy"]])

    def fixture(self):
        case=self.case
        # A fake 200Hz 29DoF sequence ending with original 1us sentinel.
        times=list(np.arange(.005,1.645,.005))
        if times[-1] < 1.640:times.append(1.640)
        times.append(times[-1]+.000001)
        names=["left_hip_pitch_joint"]*12+["left_shoulder_pitch_joint"]*17
        rows=[]
        for t in times:
            rows.append(dict(time_s=float(t),joint_pos=[0.]*29,
                joint_vel=[0.]*29,joint_target=[0.]*29,
                whole_body_com_xyz=[0.,0.,.65],
                whole_body_com_velocity=[0.,0.,0.],
                support_foot_force_N={"left":130.,"right":0.},
                nonfoot_contact_N=25.,
                root_xyz=[0.,0.,.7],root_tilt_deg=2.))
        return dict(root_fixed=False,post_initialization_state_writes=False,
            seed=case["seed"],joint_strike=dict(ball_offset_xy=case["ball_offset_xy_m"],
                 velocity_y_delta=case["velocity_y_delta_mps"]),
            policy_sha256=self.policy_sha,traces=[rows],
            records=[dict(contact_time_s=1.57,recovered=False)],
            joint_names=names,joint_velocity_limits=[[10.]*29],
            termination_reasons=[["physical_nonfoot"]],
            nonfoot_contact_diagnostics=[dict(time_s=times[-1],body="pelvis")])

    def test_original_early_stop_is_valid_negative_not_excluded(self):
        r=self.fixture()
        n,complete=admit(r,self.case)
        self.assertEqual(n,len(r["traces"][0]))
        self.assertFalse(complete)
        m=measure_200hz(r)
        self.assertEqual(m["cadence_Hz"],200)
        self.assertFalse(m["postimpact_1250ms_complete"])
        self.assertTrue(m["preimpact_400ms_complete"])
        self.assertTrue(m["early_physical_stop_recovery_window_censored"])
        self.assertFalse(m["full_original_14s_end"])
        self.assertEqual(m["actual_200Hz_integrated_steps"],n-1)

    def test_does_not_silently_invent_recovery(self):
        r=self.fixture()
        r["records"][0]["recovered"]=True
        with self.assertRaises(ValueError):admit(r,self.case)

    def test_wrong_ball_rejected_even_on_early_stop(self):
        r=self.fixture()
        r["joint_strike"]["ball_offset_xy"]=[.5,.5]
        with self.assertRaises(ValueError):admit(r,self.case)

    def test_unphysical_terminal_jump_rejected(self):
        r=self.fixture()
        r["traces"][0][-1]["time_s"]+=.02
        with self.assertRaises(ValueError):admit(r,self.case)
        with self.assertRaises(ValueError):measure_200hz(r)

    def test_missing_sensor_proof_not_silently_accepted(self):
        r=self.fixture()
        r["nonfoot_contact_diagnostics"]=[]
        with self.assertRaises(ValueError):admit(r,self.case)

    def test_short_trajectory_without_real_termination_rejected(self):
        r=self.fixture()
        r["termination_reasons"]=[[]]
        with self.assertRaises(ValueError):admit(r,self.case)

if __name__=="__main__":
    unittest.main()

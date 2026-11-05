import copy
import hashlib
import math
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
from strike_baseline_metrics import first_force_episode, preparation_metrics, yaw_excursion
from strike_baseline_protocol import case_plan, verify_files


def row(t,force=0,vel=2,torque=3):
    return dict(time_s=t,physics_dt_s=.005,contact_force_N=force,
        ball_velocity=[0,-4 if t>=.025 else 0,0],joint_vel=[vel],joint_torque_estimate=[torque],
        right_foot_point_velocity=[0,-5,0])


class StrikeMeasurements(unittest.TestCase):
    def test_impulse_and_outgoing_use_first_contiguous_measured_episode(self):
        rows=[row(.005),row(.010,100),row(.015,200),row(.020),row(.025,500)]
        m=first_force_episode(rows)
        self.assertAlmostEqual(m['normal_impulse_Ns'],1.5)
        self.assertEqual(m['outgoing_time_s'],.020)
        self.assertEqual(m['outgoing_speed_xy_mps'],0.)
        self.assertEqual(m['force_duration_s'],.010)

    def test_absent_or_truncated_contact_does_not_invent_outgoing_velocity(self):
        self.assertIsNone(first_force_episode([row(.005),row(.010)]))
        self.assertIsNone(first_force_episode([row(.005),row(.010,100)])['outgoing_time_s'])

    def test_signed_work_excludes_collision_bin_and_keeps_negative_work(self):
        rows=[row(.005,vel=2,torque=3),row(.010,vel=-4,torque=3),row(.015,100,vel=100,torque=100)]
        m=preparation_metrics(rows,['right_knee_joint'],.015,window_s=.015)
        j=m['joints']['right_knee_joint']
        self.assertAlmostEqual(j['positive_work_J'],.03)
        self.assertAlmostEqual(j['negative_work_J'],-.06)
        self.assertEqual(j['peak_abs_velocity_rad_s'],4.)
        self.assertAlmostEqual(j['peak_velocity_time_to_contact_s'],-.005)

    def test_missing_joint_evidence_is_rejected_instead_of_zero_filled(self):
        rows=[row(.005),row(.010)];del rows[0]['joint_torque_estimate']
        with self.assertRaises(ValueError):preparation_metrics(rows,['right_knee_joint'],.015,.015)

    def test_nonfinite_or_gapped_physics_samples_are_rejected(self):
        with self.assertRaises(ValueError):first_force_episode([row(.005),row(.020,100)])
        with self.assertRaises(ValueError):first_force_episode([row(.005,float('nan'))])

    def test_copied_terminal_marker_is_not_a_second_force_sample(self):
        rows=[row(.005),row(.010,100)]
        terminal=copy.deepcopy(rows[-1]);terminal.update(time_s=.010001,terminated=True)
        rows.append(terminal)
        m=first_force_episode(rows)
        self.assertAlmostEqual(m['normal_impulse_Ns'],.5)
        self.assertIsNone(m['outgoing_time_s'])

    def test_yaw_unwrap_does_not_treat_wraparound_as_a_full_rotation(self):
        q=[[math.cos(a/2),0,0,math.sin(a/2)] for a in np.radians([179,-179,-178])]
        m=yaw_excursion(q)
        self.assertAlmostEqual(m['net_deg'],3.)
        self.assertAlmostEqual(m['range_deg'],3.)


class FrozenProtocol(unittest.TestCase):
    def test_ball_shift_cannot_change_goal_or_control_parameters(self):
        base={'ball_xy':[-.12,-4.02],'goal_xy':[.7,-6.52],'strike':{'peak':12}}
        before=copy.deepcopy(base)
        out=case_plan(base,{'ball_offset_xy_m':[.05,-.10],'root_velocity_y_delta_mps':.2})
        self.assertEqual(base,before)
        self.assertAlmostEqual(out['ball_xy'][0],-.07)
        self.assertAlmostEqual(out['ball_xy'][1],-4.12)
        self.assertEqual(out['goal_xy'],[.7,-6.52]);self.assertEqual(out['strike'],{'peak':12})

    def test_changed_frozen_file_is_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model';p.write_bytes(b'original');manifest={'model':hashlib.sha256(b'original').hexdigest()}
            verify_files(Path(d),manifest)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_files(Path(d),manifest)

    def test_nonfinite_offsets_are_rejected(self):
        with self.assertRaises(ValueError):case_plan({'ball_xy':[0,0]},{'ball_offset_xy_m':[float('nan'),0],'root_velocity_y_delta_mps':0})


class TaskOutcomeTests(unittest.TestCase):
    def test_a_faster_ball_cannot_hide_a_failed_goal_or_recovery(self):
        from strike_baseline_metrics import task_outcome
        r=dict(stride_shot_success=False,root_fixed=False,post_initialization_state_writes=False,
            records=[dict(contact_time_s=1.,surface_match=True,goal=False,recovered=False,failed=False)],
            running_evidence={'running_success':True},stride_shot_evidence={'continuous_running_contact':True},
            sequence_records=[{'illegal_body_ball_contact':False}])
        m=task_outcome(r)
        self.assertFalse(m['success']);self.assertEqual(m['failure_reasons'],['missed_goal','recovery_failed'])


class EvidenceBoundaryTests(unittest.TestCase):
    def test_prefix_removal_is_not_complete_physical_evidence(self):
        with self.assertRaises(ValueError):first_force_episode([row(.010),row(.015,100)])

    def test_early_truncation_cannot_retain_a_successful_episode(self):
        from strike_baseline_metrics import validated_episode
        data={'traces':[[row(.005),row(.010)]],'records':[{'success':True,'failed':False,'contact_time_s':.010}],'termination_reasons':[[]]}
        with self.assertRaises(ValueError):validated_episode(data,.015)

    def test_saved_contact_must_be_present_in_the_measured_trace(self):
        from strike_baseline_metrics import validated_episode
        data={'traces':[[row(.005),row(.010),row(.015)]],'records':[{'success':True,'failed':False,'contact_time_s':.010}],'termination_reasons':[[]]}
        with self.assertRaises(ValueError):validated_episode(data,.015)

    def test_nonfinite_measurement_window_is_rejected(self):
        with self.assertRaises(ValueError):preparation_metrics([row(.005),row(.010)],['knee'],.015,float('nan'))

    def test_analysis_rejects_changed_evaluator_and_unrequested_reset(self):
        from strike_baseline_evidence import verify_case_evidence
        case={'id':'probe','ball_offset_xy_m':[.05,0.],'root_velocity_y_delta_mps':.2}
        plan={'ball_xy':[.05,0.],'goal_xy':[0.,-2.]}
        evidence={'case':case,'matrix_sha256':'matrix','evaluator_sha256':'evaluator','protocol_sha256':'protocol',
                  'resolved_plan':plan,'initial_reset_observed':{'root_linear_velocity_w':[[0.,-.8,0.]]}}
        baseline={'strike_baseline':{'initial_reset_observed':{'root_linear_velocity_w':[[0.,-1.,0.]]}},'initial_robot_state':{'q':[1]},'controller_sources_sha256':{'a':'frozen'}}
        data={'strike_baseline':evidence,'points':[[.05,0.]],'goals':[[0.,-2.]],'initial_robot_state':{'q':[1]},'controller_sources_sha256':{'a':'frozen'}}
        sources={'evaluator_sha256':'evaluator','protocol_sha256':'protocol'}
        verify_case_evidence(data,case,'matrix',sources,plan,baseline)
        changed=copy.deepcopy(data);changed['strike_baseline']['evaluator_sha256']='changed'
        with self.assertRaises(ValueError):verify_case_evidence(changed,case,'matrix',sources,plan,baseline)
        changed=copy.deepcopy(data);changed['strike_baseline']['initial_reset_observed']['root_linear_velocity_w'][0][1]=-.9
        with self.assertRaises(ValueError):verify_case_evidence(changed,case,'matrix',sources,plan,baseline)

if __name__=='__main__':unittest.main()

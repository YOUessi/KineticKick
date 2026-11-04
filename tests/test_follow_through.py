import copy
import unittest

import numpy as np

from test_shooting_posture import run_fixture
from follow_through import audit_follow_through, check_follow_through_acceptance, FOLLOW_THROUGH_CRITERIA
from test_posture_acceptance import matrix_fixture


def follow_fixture(mode='smooth', body_only=False):
    run = run_fixture()
    template = run['kinematics'][0][0]
    quaternion = run['traces'][0][1]['contacts'][0]['foot_quat_w']
    samples = []
    for t in np.arange(1.94, 2.801, .02):
        dt = t-2.
        if dt<0 or dt<=.20:
            forward = 1.5*dt
        elif mode=='fast_return':
            forward = .30-3.*(dt-.20)
        elif dt<=.28:
            u=dt-.20
            forward=.30+1.5*u-1.5*u*u/(2*.08)
        else:
            forward=.36-.6*(dt-.28)
        if mode=='stop' and dt>0:
            forward = min(forward,.05)
        if mode=='delayed_fast_return' and dt>.5:
            forward=.36-.6*(.5-.28)-5.*(dt-.5)
        body = forward if body_only else .2*dt
        row = copy.deepcopy(template)
        row.update(time_s=float(t), feet={'right':{'position':[0.,-1.-forward,.15], 'quaternion':quaternion}})
        row['root_pose'][1] = -.5-body
        samples.append(row)
    run['kinematics']=[samples]
    return run


def follow_matrix_fixture():
    config, runs = matrix_fixture()
    config['follow_through_criteria'] = FOLLOW_THROUGH_CRITERIA.copy()
    provenance = dict(controller_sources_sha256={'tools/shooting_control.py': 'controller'},
                      motion_file='motion', motion_sha256='reference')
    config.update(provenance)
    for run in runs:
        run.update(copy.deepcopy(provenance))
        run['motion_files'] = ['motion']
        run['kinematics']=[copy.deepcopy(follow_fixture()['kinematics'][0]) for _ in range(3)]
    return config, runs


class FollowThroughTests(unittest.TestCase):
    def test_smooth_follow_through_passes_but_stop_and_fast_return_fail(self):
        good = audit_follow_through(follow_fixture(),0)
        self.assertTrue(good['follow_through_and_full_chain'])
        self.assertGreater(good['forward_advance_m'], .34)
        self.assertLess(good['peak_return_speed_mps'], .61)
        for mode in ['stop','fast_return']:
            self.assertFalse(audit_follow_through(follow_fixture(mode),0)['follow_through_and_full_chain'])

    def test_delaying_abrupt_retraction_beyond_half_a_second_cannot_pass(self):
        result=audit_follow_through(follow_fixture('delayed_fast_return'),0)
        self.assertFalse(result['follow_through_and_full_chain'])
        self.assertIn('fast_recovery_retraction',result['follow_through_failures'])

    def test_body_translation_does_not_count_as_leg_follow_through(self):
        result = audit_follow_through(follow_fixture(body_only=True),0)
        self.assertFalse(result['follow_through_and_full_chain'])
        self.assertAlmostEqual(result['pelvis_relative_advance_m'],0.)

    def test_late_ball_contact_cannot_replace_follow_through(self):
        run = follow_fixture()
        row = run['traces'][0][2]
        row['contacts'] = copy.deepcopy(run['traces'][0][1]['contacts'])
        row.update(contact_force_N=10.,contact_normal_foot=[0,0,1],contact_point_foot=[.075,0,-.015])
        result = audit_follow_through(run,0)
        self.assertTrue(result['late_ball_contact'])
        self.assertFalse(result['follow_through_and_full_chain'])

    def test_invalid_or_incomplete_kinematics_cannot_certify_motion(self):
        for mutation in ['nan','gap','short']:
            run = follow_fixture()
            if mutation=='nan':run['kinematics'][0][6]['time_s']=float('nan')
            if mutation=='gap':del run['kinematics'][0][6]
            if mutation=='short':run['kinematics'][0]=run['kinematics'][0][:-4]
            result = audit_follow_through(run,0)
            self.assertFalse(result['follow_through_and_full_chain'])
            self.assertIn('missing_motion_coverage',result['follow_through_failures'])

    def test_environment_translation_does_not_change_motion_measurements(self):
        run = follow_fixture()
        original = audit_follow_through(run,0)
        for sample in run['kinematics'][0]:
            sample['root_pose'][0] += 100.
            sample['root_pose'][1] -= 50.
            sample['feet']['right']['position'][0] -= 30.
            sample['feet']['right']['position'][1] += 20.
        shifted = audit_follow_through(run,0)
        for key in ['forward_advance_m','pelvis_relative_advance_m','forward_speed_at_120ms_mps','peak_return_speed_mps']:
            self.assertAlmostEqual(original[key],shifted[key])

    def test_acceptance_counts_motion_with_physics_and_rejects_changed_sources(self):
        config, runs = follow_matrix_fixture()
        self.assertTrue(check_follow_through_acceptance(config,runs,'frozen')['passed'])
        for i in [0,1]:
            runs[0]['kinematics'][i]=follow_fixture('stop')['kinematics'][0]
            result=check_follow_through_acceptance(config,runs,'frozen')
            self.assertEqual(result['passed'],i==0)
        self.assertEqual(result['styles']['instep']['posture_and_full_chain'],6)
        self.assertEqual(result['follow_through_successes']['instep'],4)
        for field in ['controller_sources_sha256','motion_sha256','motion_files']:
            config,runs=follow_matrix_fixture()
            runs[0][field]=['different/reference.npz'] if field=='motion_files' else 'changed'
            with self.assertRaises(ValueError):check_follow_through_acceptance(config,runs,'frozen')

    def test_continuation_and_reference_are_part_of_the_frozen_manifest(self):
        for key in ['continuation','reference']:
            config,runs=follow_matrix_fixture()
            if key=='continuation':
                runs[0]['trials'][0]['strike']={'continuation':{'window':[120,130,150], 'targets':{'right_knee_joint':.8}}}
            else:
                runs[0]['trials'][0]['reference']={'window':[120,130,150],'offsets':{'right_hip_pitch_joint':-1.3}}
            with self.assertRaises(ValueError):check_follow_through_acceptance(config,runs,'frozen')

    def test_unchanged_inside_is_a_physical_regression_not_a_new_follow_through_claim(self):
        config,runs=follow_matrix_fixture()
        config['styles']={'inside':{'strike':{}}}
        for run in runs:
            run['policy_goals']=run['goals']
            for trial in run['trials']:
                trial['style']='inside';trial.pop('policy_goal')
        # An all-inside matrix cannot certify the requested instep/outside change.
        with self.assertRaises(ValueError):check_follow_through_acceptance(config,runs,'frozen')


if __name__=='__main__': unittest.main()

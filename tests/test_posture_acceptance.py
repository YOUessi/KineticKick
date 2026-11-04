import copy
import unittest

import numpy as np

from test_shooting_posture import run_fixture, wxyz, Rotation
from shooting_posture import check_posture_acceptance


def matrix_fixture():
    config = dict(styles={'instep': {'strike': {}, 'policy_goal': [1., -6.]}},
                  acceptance_seeds=[903, 904], trial_steps=200, policy_dt_s=.02,
                  acceptance_conditions=[{'ball': [x, -1.], 'goal': [0., -3.]} for x in [0., -.02, .02]],
                  policy_sha256='model', asset_sha256='asset')
    reports = []
    for seed in config['acceptance_seeds']:
        run = run_fixture()
        coarse = run['traces'][0]
        dense = []
        for step in range(1, 801):
            time = step * .005
            row = copy.deepcopy(coarse[1 if step == 400 else 2 if step > 400 else 0])
            row['time_s'] = time
            row['ball_xyz'][1] = float(np.interp(time, [r['time_s'] for r in coarse], [r['ball_xyz'][1] for r in coarse]))
            dense.append(row)
        run.update(seed=seed, policy_sha256='model', asset_sha256='asset', config_sha256='frozen',
                   root_fixed=False, post_initialization_state_writes=False, perturb_reset=True,
                   contact_evidence_version=2, automatic_resets_suppressed=[], goal_width_m=1.6, goal_height_m=1.1)
        run['trials'] = [dict(style='instep', **c, **config['styles']['instep']) for c in config['acceptance_conditions']]
        run['records'] = [dict(success=True) for _ in range(3)]
        run['points'] = [t['ball'] for t in run['trials']]
        run['goals'] = [t['goal'] for t in run['trials']]
        run['policy_goals'] = [t['policy_goal'] for t in run['trials']]
        run['initial_robot_state']['root_position'] *= 3
        run['kinematics'] *= 3
        run['traces'] = [copy.deepcopy(dense) for _ in range(3)]
        for i, trace in enumerate(run['traces']):
            for row in trace:
                row['ball_xyz'][0] = run['points'][i][0]
        reports.append(run)
    return config, reports


class PostureAcceptanceTests(unittest.TestCase):
    def test_replay_cannot_be_overruled_by_success_labels(self):
        config, runs = matrix_fixture()
        self.assertTrue(check_posture_acceptance(config, runs, 'frozen')['passed'])
        for index in [0, 1]:
            runs[0]['traces'][index][399]['contacts'][0]['foot_quat_w'] = wxyz(
                Rotation.from_euler('z', -90, degrees=True) * Rotation.from_euler('y', 27, degrees=True))
            result = check_posture_acceptance(config, runs, 'frozen')
            self.assertEqual(result['passed'], index == 0)
        self.assertEqual(result['styles']['instep']['posture_and_full_chain'], 4)

    def test_heading_control_and_configuration_hash_are_part_of_the_frozen_candidate(self):
        for mutation in ['goal', 'hash', 'cartesian']:
            config, runs = matrix_fixture()
            if mutation == 'goal': runs[1]['trials'][0]['policy_goal'] = [2., -6.]
            if mutation == 'hash': runs[1]['config_sha256'] = 'changed'
            if mutation == 'cartesian': runs[1]['trials'][0]['cartesian'] = {'window': [0, 1, 2]}
            with self.assertRaises(ValueError): check_posture_acceptance(config, runs, 'frozen')

    def test_missing_physics_samples_cannot_certify_recovery(self):
        config, runs = matrix_fixture()
        del runs[0]['traces'][0][500]
        with self.assertRaises(ValueError): check_posture_acceptance(config, runs, 'frozen')

    def test_a_fall_fails_even_when_five_other_trials_pass(self):
        config, runs = matrix_fixture()
        runs[0]['traces'][0][-1]['root_xyz'][2] = .4
        self.assertFalse(check_posture_acceptance(config, runs, 'frozen')['passed'])


if __name__ == '__main__': unittest.main()

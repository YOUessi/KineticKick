import copy
import unittest
from test_targeted_shooting import config_fixture
from targeted_shooting import resolve_targeted_shot, STYLES, POWERS
from check_targeted_shooting_acceptance import validate_manifest, replay_trial, aggregate_results


def runs_fixture():
    config = config_fixture()
    result = []
    for seed in config['acceptance_seeds']:
        for style in STYLES:
            trials = [resolve_targeted_shot(config, style, x, power=power)
                      for power in POWERS for x in config['target_x_m']]
            result.append(dict(seed=seed, trials=trials, traces=[[] for _ in trials],
                points=[t['ball'] for t in trials], goals=[t['goal'] for t in trials],
                policy_goals=[t['policy_goal'] for t in trials],
                initial_robot_state=dict(root_position=[[0, 0, .8] for _ in trials]),
                targeted_config_sha256='f'*64, policy_sha256=config['policy_sha256'],
                asset_sha256=config['asset_sha256'], root_fixed=False, post_initialization_state_writes=False,
                perturb_reset=True, contact_evidence_version=2, goal_width_m=1.6, goal_height_m=1.1))
    return config, result


def trace_fixture():
    result = []
    for step in range(1, 1641):
        now = step*.005
        contact = step == 100
        y = -.93 - max(0., now-.5)*2
        event = dict(force_N=10., point_foot=[.075, .036, -.025], normal_foot=[0, 1, 0], foot='right', time_s=now)
        result.append(dict(time_s=now, ball_xyz=[.14, y, .11], ball_velocity=[0, -2, 0],
            root_xyz=[0, -.4 if now >= .5 else 0., .8], root_speed_xy=0., root_tilt_deg=0.,
            contact_force_N=10 if contact else 0, contact_point_foot=event['point_foot'] if contact else None,
            contact_normal_foot=event['normal_foot'] if contact else [0, 0, 0], foot='right',
            contacts=[event] if contact else [], physics_dt_s=.005,
            feet_support_N=300., nonfoot_contact_N=0., terminated=False))
    return result


class AcceptanceTests(unittest.TestCase):
    def test_requires_unique_complete_matrix_and_pinned_controller(self):
        config, runs = runs_fixture()
        self.assertEqual(len(validate_manifest(config, runs, 'f'*64)), 54)
        for change in ('missing', 'duplicate', 'controller', 'hash', 'geometry', 'old_contact'):
            broken = copy.deepcopy(runs)
            if change == 'missing': broken.pop()
            if change == 'duplicate': broken[-1] = broken[0]
            if change == 'controller': broken[0]['trials'][0]['policy_goal'][0] += .1
            if change == 'hash': broken[0]['targeted_config_sha256'] = '0'*64
            if change == 'geometry': broken[0]['goals'][0][0] += .1
            if change == 'old_contact': broken[0]['contact_evidence_version'] = 1
            with self.assertRaises(ValueError, msg=change):
                validate_manifest(config, broken, 'f'*64)

    def test_replay_scores_actual_trajectory_and_rejects_incomplete_trace(self):
        trial = dict(style='inside', ball=[.14, -.93], goal=[0, -3.7], goal_forward=[0, -1],
                     target=dict(target_x=.14, speed_mps=2.))
        trace = trace_fixture()
        self.assertTrue(replay_trial(trial, trace, [0, 0, .8])['targeted_success'])
        miss = copy.deepcopy(trial)
        miss['target']['target_x'] = -.3
        self.assertFalse(replay_trial(miss, trace, [0, 0, .8])['targeted_success'])
        with self.assertRaises(ValueError): replay_trial(trial, trace[:-1], [0, 0, .8])
        with self.assertRaises(ValueError): replay_trial(trial, trace[:200]+trace[201:], [0, 0, .8])
        terminated = copy.deepcopy(trace[:200])
        terminated[-1]['terminated'] = True
        self.assertFalse(replay_trial(trial, terminated, [0, 0, .8])['targeted_success'])

    def test_identical_power_outcomes_cannot_be_reported_as_strength_control(self):
        rows = [dict(style=style, power=power, target_x_m=x, targeted_success=True,
                     crossing=dict(position=[x, -3.7, .11]), outgoing_speed_mps=2.)
                for style in STYLES for power in POWERS for x in [-.3, 0, .3] for _ in range(2)]
        result = aggregate_results(rows)
        self.assertFalse(result['passed'])
        self.assertFalse(result['styles']['inside']['strength_separated'])
        for row in rows: row['outgoing_speed_mps'] += POWERS.index(row['power'])*.5
        self.assertTrue(aggregate_results(rows)['passed'])


if __name__ == '__main__':
    unittest.main()

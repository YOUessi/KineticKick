"""Aim and strength must be measured independently of the controller inputs."""
from pathlib import Path
import sys
import unittest
import copy
import hashlib
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from targeted_shooting import (TargetShotTracker, resolve_targeted_shot, validate_target_config,
                              MOTION_FILE, CONTROLLER_SOURCES, UPSTREAM_SOURCE)


def sample(t, position, velocity=(0, -2, 0), force=0):
    return dict(time_s=t, ball_xyz=position, ball_velocity=velocity,
                contact_force_N=force, terminated=False)


def config_fixture():
    return dict(schema_version=1, version='0.1.2', policy_sha256='489711f8654faf8aba49584a39266a33a9068a8a6bc8122e88f2bc7f06739ab1',
        asset_sha256='34ef7f147b91af2615ca5c88f42ad8fd5d1a00fa63ca66d8e01ec9245a2c7761',
        ball=[.14, -.93], goal=[0, -3.7], goal_forward=[0, -1], trial_steps=410,
        target_x_m=[-.3, 0., .3], acceptance_seeds=[811, 812], interpolation=False,
        motion_file=MOTION_FILE, motion_sha256='a'*64,
        controller_sources_sha256={p: 'b'*64 for p in (*CONTROLLER_SOURCES, UPSTREAM_SOURCE)},
        styles={style: dict(speeds_mps=dict(low=1., medium=1.4, high=1.8), nodes=[
            dict(power=power, target_x_m=x, control=dict(policy_goal=[x+i, -3.7], strike={}))
            for i, power in enumerate(['low', 'medium', 'high']) for x in [-.3, 0., .3]])
            for style in ['inside', 'instep', 'outside']})


class TargetCommandTests(unittest.TestCase):
    def test_target_and_power_resolve_without_moving_the_physical_goal(self):
        config = config_fixture()
        a = resolve_targeted_shot(config, 'inside', -.3, power='high')
        b = resolve_targeted_shot(config, 'inside', .3, speed_mps=1.)
        self.assertEqual(a['goal'], b['goal'])
        self.assertEqual(a['goal_forward'], [0, -1])
        self.assertEqual(a['target'], dict(target_x=-.3, speed_mps=1.8))
        self.assertEqual(b['target'], dict(target_x=.3, speed_mps=1.))
        self.assertEqual(a['policy_goal'], [1.7, -3.7])
        a['strike']['targets'] = {'bad': 3}
        self.assertEqual(config['styles']['inside']['nodes'][6]['control']['strike'], {})

    def test_unsupported_numeric_commands_are_rejected_without_clamping(self):
        for x, v in [(.1, 1.4), (.9, 1.4), (0., 1.2), (float('nan'), 1.4), (0., float('inf'))]:
            with self.assertRaisesRegex(ValueError, 'supported|finite'):
                resolve_targeted_shot(config_fixture(), 'inside', x, speed_mps=v)

    def test_ambiguous_strength_and_unknown_style_are_rejected(self):
        for args in [dict(power='low', speed_mps=1.), dict(), dict(power='huge')]:
            with self.assertRaises(ValueError):
                resolve_targeted_shot(config_fixture(), 'inside', 0, **args)
        with self.assertRaises(ValueError):
            resolve_targeted_shot(config_fixture(), 'toe', 0, power='low')

    def test_changed_model_geometry_duplicate_nodes_and_unsafe_controls_fail_validation(self):
        for change in ['model', 'goal', 'duplicate', 'nan', 'servo', 'interpolation']:
            config = copy.deepcopy(config_fixture())
            if change == 'model': config['policy_sha256'] = '0'*64
            if change == 'goal': config['goal'][0] = .2
            if change == 'duplicate': config['styles']['inside']['nodes'][0] = config['styles']['inside']['nodes'][1]
            if change == 'nan': config['styles']['inside']['nodes'][0]['control']['policy_goal'][0] = float('nan')
            if change == 'servo': config['styles']['inside']['nodes'][0]['control']['cartesian'] = {'max_joint_correction_rad': 2.}
            if change == 'interpolation': config['interpolation'] = True
            with self.assertRaises(ValueError, msg=change):
                validate_target_config(config)


class TargetMetricTests(unittest.TestCase):
    def test_first_crossing_above_goal_cannot_be_rescued_by_later_goal(self):
        tracker = TargetShotTracker(target_x=0, speed_mps=2.)
        for row in [sample(1, (0, -.93, .11), force=10), sample(1.1, (0, -1.13, .11)),
                    sample(2, (0, -3.6, 2.)), sample(2.2, (0, -3.9, 2.)),
                    sample(2.4, (0, -3.6, .11)), sample(2.6, (0, -3.9, .11))]:
            tracker.update(row)
        self.assertEqual(tracker.summary()['crossing']['position'][2], 2.)
        self.assertFalse(tracker.summary()['target_success'])

    def test_first_partial_crossing_cannot_borrow_a_later_full_sphere_goal(self):
        tracker = TargetShotTracker(target_x=0, speed_mps=2.)
        for row in [sample(1, (0, -.93, .11), force=10), sample(1.1, (0, -1.13, .11)),
                    sample(2, (0, -3.6, .11)), sample(2.1, (0, -3.71, .11)),
                    sample(2.2, (.8, -3.72, .11)), sample(2.3, (.8, -3.9, .11)),
                    sample(2.4, (0, -3.6, .11)), sample(2.6, (0, -3.9, .11))]:
            tracker.update(row)
        self.assertFalse(tracker.summary()['target_success'])

    def test_reference_and_source_identity_are_content_fingerprints(self):
        import targeted_shooting as module
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            motion = root / module.MOTION_FILE
            motion.parent.mkdir(parents=True)
            motion.write_bytes(b'reference')
            for name in module.CONTROLLER_SOURCES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('source')
            upstream = root / module.UPSTREAM_SOURCE
            upstream.mkdir(parents=True)
            (upstream / 'command.py').write_text('command')
            original = module.controller_provenance(root)
            self.assertEqual(original['motion_sha256'], hashlib.sha256(b'reference').hexdigest())
            module.verify_target_provenance(original, root)
            (root / module.CONTROLLER_SOURCES[0]).write_text('changed source')
            with self.assertRaises(ValueError): module.verify_target_provenance(original, root)
            (root / module.CONTROLLER_SOURCES[0]).write_text('source')
            motion.write_bytes(b'changed reference')
            with self.assertRaises(ValueError): module.verify_target_provenance(original, root)

    def test_interpolates_fixed_plane_crossing_and_requested_point_error(self):
        tracker = TargetShotTracker(target_x=.3, speed_mps=2.)
        tracker.update(sample(1., (.1, -.93, .11), force=5))
        tracker.update(sample(1.1, (.1, -1.13, .11)))
        tracker.update(sample(2., (.2, -3.5, .11)))
        tracker.update(sample(2.2, (.4, -3.9, .11)))
        result = tracker.summary()
        self.assertAlmostEqual(result['crossing']['time_s'], 2.1)
        self.assertAlmostEqual(result['crossing']['position'][0], .3)
        self.assertAlmostEqual(result['crossing']['position'][1], -3.7)
        self.assertAlmostEqual(result['aim_error_m'], 0)
        self.assertTrue(result['target_hit'])
        self.assertTrue(result['speed_match'])

    def test_first_crossing_outside_goal_is_recorded_and_not_replaced(self):
        tracker = TargetShotTracker(target_x=0, speed_mps=2.)
        tracker.update(sample(0., (1., -.93, .11), force=5))
        tracker.update(sample(.1, (1., -1.13, .11)))
        tracker.update(sample(1., (1., -3.6, .11)))
        tracker.update(sample(1.2, (1., -3.8, .11)))
        tracker.update(sample(1.4, (0., -3.6, .11)))
        tracker.update(sample(1.6, (0., -3.8, .11)))
        result = tracker.summary()
        self.assertAlmostEqual(result['aim_error_m'], 1.)
        self.assertFalse(result['target_hit'])

    def test_reverse_crossing_does_not_supply_target_evidence(self):
        tracker = TargetShotTracker(target_x=0, speed_mps=2.)
        tracker.update(sample(1., (0., -3.8, .11), force=5))
        tracker.update(sample(1.2, (0., -3.6, .11), velocity=(0, 2, 0)))
        self.assertIsNone(tracker.summary()['crossing'])
        self.assertFalse(tracker.summary()['target_success'])

    def test_speed_is_interpolated_at_contact_plus_point_one_not_peak(self):
        tracker = TargetShotTracker(target_x=0, speed_mps=3.)
        tracker.update(sample(1., (0, -.93, .11), velocity=(0, -8, 0), force=100))
        tracker.update(sample(1.08, (0, -1.4, .11), velocity=(0, -4, 0)))
        tracker.update(sample(1.12, (0, -1.5, .11), velocity=(0, -2, 0)))
        self.assertAlmostEqual(tracker.summary()['outgoing_speed_mps'], 3.)
        self.assertTrue(tracker.summary()['speed_match'])

    def test_short_trace_cannot_supply_speed(self):
        tracker = TargetShotTracker(target_x=0, speed_mps=2.)
        tracker.update(sample(1., (0, -.93, .11), force=100))
        tracker.update(sample(1.09, (0, -1.1, .11)))
        self.assertIsNone(tracker.summary()['outgoing_speed_mps'])
        self.assertFalse(tracker.summary()['target_success'])

    def test_missing_contact_nonfinite_input_and_termination_cannot_pass(self):
        for bad in ('contact', 'nan', 'terminated'):
            tracker = TargetShotTracker(target_x=0, speed_mps=2.)
            tracker.update(sample(1., (0, -.93, .11), force=0 if bad == 'contact' else 5))
            row = sample(1.1, (0, -1.13, .11))
            if bad == 'nan':
                row['ball_velocity'] = (float('nan'), -2, 0)
            if bad == 'terminated':
                row['terminated'] = True
            tracker.update(row)
            tracker.update(sample(2., (0, -3.6, .11)))
            tracker.update(sample(2.2, (0, -3.8, .11)))
            self.assertFalse(tracker.summary()['target_success'], bad)

    def test_duplicate_or_reversed_timestamps_are_rejected(self):
        tracker = TargetShotTracker(target_x=0, speed_mps=2.)
        tracker.update(sample(1., (0, -.93, .11)))
        with self.assertRaises(ValueError):
            tracker.update(sample(1., (0, -.93, .11)))


if __name__ == '__main__':
    unittest.main()

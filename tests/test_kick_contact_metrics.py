import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from kick_contact_metrics import contact_diagnostics, balance_proxies


def trace(forces):
    rows = []
    for i, force in enumerate(forces):
        event = dict(foot='right', force_N=force, normal_foot=[0, -1, 0],
                     foot_quat_w=[1, 0, 0, 0])
        rows.append(dict(time_s=(i+1)*.005, physics_dt_s=.005, terminated=False,
            contacts=[event], ball_velocity=[0, -1, 0], root_xyz=[0, 0, .8],
            root_speed_xy=.1, root_tilt_deg=5., feet_support_N=300., nonfoot_contact_N=0.))
    return rows


class ContactDiagnosticsTests(unittest.TestCase):
    def test_sustained_contact_is_one_episode_even_after_eighty_ms(self):
        result = contact_diagnostics(trace([0]+[10]*24+[0]), [0, -1, 0])
        self.assertEqual(len(result['positive_force_episodes']), 1)
        episode = result['positive_force_episodes'][0]
        self.assertAlmostEqual(episode['sampled_force_duration_s'], .12)
        self.assertAlmostEqual(episode['normal_impulse_magnitude_sum_Ns'], 1.2)
        self.assertAlmostEqual(episode['normal_impulse_toward_goal_Ns'], 1.2)
        self.assertTrue(episode['ended_with_zero_force_sample'])

    def test_early_separate_force_episode_is_not_hidden_by_eighty_ms_rule(self):
        result = contact_diagnostics(trace([0, 20, 20, 0, 0, 5, 0]), [0, -1, 0])
        self.assertEqual(len(result['positive_force_episodes']), 2)
        self.assertAlmostEqual(result['positive_force_episodes'][1]['gap_from_previous_s'], .01)

    def test_zero_force_manifold_and_terminal_duplicate_do_not_add_impulse(self):
        rows = trace([0, 20, 0])
        terminal = copy.deepcopy(rows[-1])
        terminal.update(terminated=True, time_s=rows[-1]['time_s']+1e-6)
        result = contact_diagnostics(rows+[terminal], [0, -1, 0])
        self.assertAlmostEqual(result['normal_impulse_magnitude_sum_Ns'], .1)
        self.assertEqual(len(result['positive_force_episodes']), 1)

    def test_impulse_direction_does_not_count_opposing_force_as_useful(self):
        rows = trace([0, 10, 10, 0])
        rows[2]['contacts'][0]['normal_foot'] = [0, 1, 0]
        result = contact_diagnostics(rows, [0, -1, 0])
        self.assertAlmostEqual(result['normal_impulse_magnitude_sum_Ns'], .1)
        self.assertAlmostEqual(result['normal_impulse_toward_goal_Ns'], 0.)

    def test_gaps_nan_and_unfinished_episode_are_not_success_evidence(self):
        for field in ['time_s', 'force_N', 'gap']:
            rows = trace([0, 20, 20, 0])
            if field=='time_s': rows[1]['time_s']=float('nan')
            if field=='force_N': rows[1]['contacts'][0]['force_N']=float('nan')
            if field=='gap': del rows[1]
            with self.assertRaises(ValueError): contact_diagnostics(rows, [0, -1, 0])
        unfinished = contact_diagnostics(trace([0, 20, 20]), [0, -1, 0])
        self.assertFalse(unfinished['positive_force_episodes'][0]['ended_with_zero_force_sample'])
        self.assertIsNone(unfinished['positive_force_episodes'][0]['first_zero_force_ball_velocity_mps'])
        self.assertEqual(unfinished['max_ball_speed_xy_before_first_foot_force_mps'], 1.)

    def test_vertical_settling_does_not_look_like_a_precontact_shot(self):
        rows=trace([0, 20, 0])
        rows[0]['ball_velocity']=[0, 0, -.2]
        result=contact_diagnostics(rows,[0,-1,0])
        self.assertEqual(result['max_ball_speed_xy_before_first_foot_force_mps'],0.)
        terminal=copy.deepcopy(rows[-1])
        terminal.update(terminated=True,time_s=float('nan'))
        with self.assertRaises(ValueError):contact_diagnostics(rows+[terminal],[0,-1,0])

    def test_roundoff_overlap_cannot_pollute_balance_extrema(self):
        rows=trace([0]*16)
        rows[14].update(root_xyz=[0,0,.1],root_speed_xy=9.,root_tilt_deg=80.,nonfoot_contact_N=1000.)
        result=balance_proxies(rows,.005,.070)
        self.assertTrue(result['coverage_complete'])
        self.assertAlmostEqual(result['observed_duration_s'],.065)
        self.assertEqual(result['root_height_min_m'],.8)
        self.assertEqual(result['root_speed_xy_max_mps'],.1)
        self.assertEqual(result['root_tilt_max_deg'],5.)
        self.assertEqual(result['nonfoot_net_force_peak_N'],0.)

    def test_balance_proxies_retain_transient_degradation_and_partial_coverage(self):
        rows = trace([0]*160)
        rows[20].update(root_xyz=[0, 0, .6], root_speed_xy=1.5, root_tilt_deg=40.,
                        feet_support_N=0., nonfoot_contact_N=100.)
        result = balance_proxies(rows, .01, .75)
        self.assertTrue(result['coverage_complete'])
        self.assertEqual(result['root_height_min_m'], .6)
        self.assertAlmostEqual(result['time_root_tilt_above_25deg_s'], .005)
        self.assertAlmostEqual(result['time_foot_net_vertical_force_below_80N_s'], .005)
        self.assertFalse(balance_proxies(rows[:40], .01, .75)['coverage_complete'])


if __name__ == '__main__': unittest.main()

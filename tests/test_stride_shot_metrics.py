"""Prevent a run-stop-kick preview from being accepted as a running shot."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from stride_shot_metrics import summarize_stride_shot


class StrideShotAcceptance(unittest.TestCase):
    def scenario(self, *, sideways=False, stop=False):
        rows, progress = [], 0.
        for step in range(1, 321):
            time = step * .005
            speed = 0. if stop and 1.1 <= time < 1.3 else 2.
            progress += speed * .005
            rows.append(dict(time_s=time, root_xyz=[progress if sideways else 0.,
                                                   0. if sideways else -progress, .8]))
        running = dict(flight_intervals=[
            dict(start_s=.6, end_s=.65, launch_side='right'),
            dict(start_s=.9, end_s=.95, launch_side='left'),
            dict(start_s=1.35, end_s=1.4, launch_side='right')])
        return rows, running

    def evaluate(self, **kwargs):
        rows, running = self.scenario(**kwargs)
        return summarize_stride_shot(rows, contact_time=1.6, running=running)

    def test_continuous_forward_strides(self):
        result = self.evaluate()
        self.assertTrue(result['continuous_running_contact'])
        self.assertAlmostEqual(result['forward_contact_speed_m_s'], 2.)

    def test_stop_then_fast_contact_is_rejected(self):
        self.assertFalse(self.evaluate(stop=True)['continuous_running_contact'])

    def test_sideways_speed_does_not_count_as_forward(self):
        self.assertFalse(self.evaluate(sideways=True)['continuous_running_contact'])

    def test_running_without_contact_is_rejected(self):
        rows, running = self.scenario()
        self.assertFalse(summarize_stride_shot(rows, contact_time=None, running=running)
                         ['continuous_running_contact'])


if __name__ == '__main__':
    unittest.main()

"""A brief loss of contact must not restore the old foot position into the ball."""
from pathlib import Path
import sys, unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from balanced_receive_motion import BalancedReceiveMotion
import test_inside_receive_motion as fixtures


class ImpactRetreatTests(unittest.TestCase):
    def test_retreat_survives_zero_contact_force_and_is_bounded(self):
        plan = fixtures.InsideReceiveMotionTests().motion().plan
        plan['cushion'].update(retain_retreat_after_contact=True, absorption_limit_m=.12)
        motion = BalancedReceiveMotion(plan)
        target = motion.initial_foot.copy()
        actual = target[:3, 3] - .07 * motion.normal
        moved = motion.yield_to_contact(target, actual, 0.)
        np.testing.assert_allclose(moved[:3, 3], actual)
        held, _, _, _ = motion.targets(1.)
        np.testing.assert_allclose(held[:3, 3], actual)
        unchanged = motion.yield_to_contact(held, actual + .02 * motion.normal, 0.)
        np.testing.assert_allclose(unchanged, held)
        motion.yield_to_contact(held, actual - motion.normal, 0.)
        self.assertAlmostEqual(motion.absorbed_displacement, .12)


if __name__ == '__main__':
    unittest.main()

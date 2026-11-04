import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from show_free_shooting_live import live_arguments


class LiveProfileTests(unittest.TestCase):
    def test_standard_preview_uses_the_corrected_pose(self):
        args = live_arguments(['--style', 'instep'])
        self.assertTrue(args[args.index('--config')+1].endswith('configs/free_shooting_posture_v1.json'))
        self.assertIn('--gui', args)
        self.assertIn('--hold', args)

    def test_targeted_experiments_keep_their_own_calibration(self):
        for option in ['--target-x=0.3', '--power', '--speed-mps=2', '--targeted-acceptance']:
            self.assertNotIn('--config', live_arguments([option]))

    def test_explicit_profile_and_headless_are_preserved(self):
        args = live_arguments(['--config=old.json', '--headless', '--output=run.json'])
        self.assertNotIn('--config', args)
        self.assertNotIn('--gui', args)
        self.assertNotIn('--hold', args)
        self.assertNotIn('--output', args)


if __name__ == '__main__': unittest.main()

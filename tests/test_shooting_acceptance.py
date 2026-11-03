import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from check_free_shooting_acceptance import validate_matrix


class AcceptanceMatrixTest(unittest.TestCase):
    def setUp(self):
        self.config = dict(acceptance_seeds=[1, 2], styles={'inside': {'strike': {}}},
                           acceptance_conditions=[{'ball': [i*.01, -1.], 'goal': [0., -3.]} for i in range(5)],
                           policy_sha256='model', asset_sha256='robot')
        self.reports = []
        for seed in (1, 2):
            self.reports.append(dict(seed=seed, policy_sha256='model', asset_sha256='robot',
                config_sha256='config', root_fixed=False, post_initialization_state_writes=False,
                perturb_reset=True, trials=[dict(style='inside', strike={}, **c) for c in self.config['acceptance_conditions']],
                records=[dict(success=True) for _ in range(5)]))

    def test_gate_is_per_style_and_seed_and_preserves_failures(self):
        self.reports[0]['records'][0]['success'] = False
        self.assertTrue(validate_matrix(self.config, self.reports)['passed'])
        self.reports[0]['records'][1]['success'] = False
        self.assertFalse(validate_matrix(self.config, self.reports)['passed'])

    def test_duplicate_seed_is_not_independent_evidence(self):
        with self.assertRaises(ValueError):
            validate_matrix(self.config, [self.reports[0], self.reports[0]])

    def test_duplicate_condition_cannot_replace_a_failure(self):
        self.reports[0]['trials'][1] = copy.deepcopy(self.reports[0]['trials'][0])
        with self.assertRaises(ValueError):
            validate_matrix(self.config, self.reports)

    def test_a_different_controller_cannot_be_called_frozen(self):
        self.reports[1]['trials'][0]['strike'] = {'offsets': {'hip': .1}}
        with self.assertRaises(ValueError):
            validate_matrix(self.config, self.reports)


if __name__ == '__main__':
    unittest.main()

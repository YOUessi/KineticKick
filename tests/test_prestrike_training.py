"""CPU guards for frozen recovery provenance and the declared development set."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / 'tools/train_prestrike_feedback.py'
    assert path.is_file(), 'The separate prestrike trainer is not implemented'
    spec = importlib.util.spec_from_file_location('train_prestrike_feedback', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_new_policy_starts_empty_and_the_v32_recovery_model_is_required():
    m = module()
    cfg = dict(prestrike_feedback=dict(checkpoint=None, checkpoint_sha256=None),
        post_shot_recovery=dict(checkpoint='frozen_v32.pt', checkpoint_sha256=m.FROZEN_RECOVERY_SHA256))
    before = copy.deepcopy(cfg)
    m.validate_checkpoint_config(cfg)
    assert cfg == before
    for group, key, value in [
        ('prestrike_feedback', 'checkpoint', 'new_existing.pt'),
        ('prestrike_feedback', 'checkpoint_sha256', 'unpaired_digest'),
        ('post_shot_recovery', 'checkpoint', None),
        ('post_shot_recovery', 'checkpoint_sha256', 'different_model'),
    ]:
        bad = copy.deepcopy(cfg); bad[group][key] = value
        with pytest.raises(ValueError): m.validate_checkpoint_config(bad)


def test_development_cases_reject_empty_duplicate_fresh_and_changed_conditions():
    m = module()
    declared = json.loads((ROOT/'configs/post_shot_recovery_study_v1.json').read_text())['existing_cases']
    cases = {x['id']:copy.deepcopy(x) for x in declared}
    names = list(cases)
    cfg = dict(prestrike_training=dict(case_ids=names))
    assert m.validate_development_cases(cfg, cases, declared) == names
    for selected in [[], names[:-1]+[names[0]], names[:-1]+['landing_eval_left_near']]:
        bad = copy.deepcopy(cfg); bad['prestrike_training']['case_ids'] = selected
        with pytest.raises(ValueError): m.validate_development_cases(bad, cases, declared)
    changed = copy.deepcopy(cases); changed['center']['root_velocity_y_delta_mps'] = .01
    with pytest.raises(ValueError): m.validate_development_cases(cfg, changed, declared)

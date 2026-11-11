"""Human source eligibility cannot be inferred from weak mocap or robot traces."""
import json
from tools.qualify_soccer_sources_v067 import CONFIG, qualify

def test_six_sources_no_verified_elite_instep_contact_kinetics():
    x=qualify(json.loads(CONFIG.read_text()))
    assert x['sources_reviewed']==6 and x['eligible']==0
    assert all(not s['eligible_for_professional_instep_contact_learning'] for s in x['entries'])

def test_missing_and_unknown_fail_closed():
    src=json.loads(CONFIG.read_text())
    item=src['sources'][0]
    src['sources'][0]=dict(item,measured_synchronized_GRF=True,
                          professional_human_provenance_verified=None)
    assert qualify(src)['eligible']==0
    src['sources'].append(dict(item))
    try:qualify(src)
    except ValueError:pass
    else:raise AssertionError('duplicate provenance accepted')
    src=json.loads(CONFIG.read_text())
    del src['sources'][0]['verified_synchronized_foot_ball_contact_truth']
    try:qualify(src)
    except ValueError:pass
    else:raise AssertionError('missing ground truth field accepted')

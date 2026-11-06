"""A partial or empty matrix cannot become a complete success through all([])."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from audit_recovery_matrix import summarize


def test_empty_and_omitted_cases_remain_unexecuted_and_goal_miss_is_not_dynamic_failure():
    planned = [{'id': 'old_'+str(i), 'group': 'development'} for i in range(18)]
    planned += [{'id': 'fresh_'+str(i), 'group': 'fresh'} for i in range(3)]
    empty = summarize(planned, [])
    assert not empty['complete'] and not empty['matrix_acceptance_pass']
    assert empty['groups']['development']['executed'] == 0
    assert empty['groups']['fresh']['unexecuted'] == 3
    assert len(empty['cases']) == 21
    assert all(c['status'] == 'unexecuted' for c in empty['cases'])
    partial = summarize(planned, [dict(case_id='old_0', status='executed', original_task=False,
        goal=False, legal_instep=True, complete_14s=True, stable=True, dynamic_quality_pass=True,
        candidate_acceptance_pass=False)])
    assert partial['groups']['development']['executed'] == 1
    assert partial['groups']['development']['goal_misses'] == 1
    assert partial['groups']['development']['dynamic_quality_failures'] == 0
    assert not partial['complete'] and not partial['matrix_acceptance_pass']
    assert sum(c['status'] == 'unexecuted' for c in partial['cases']) == 20

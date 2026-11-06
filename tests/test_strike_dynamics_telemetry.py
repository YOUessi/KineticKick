from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from strike_dynamics_telemetry import aggregate_buffer, certify_force_probe


def test_sparse_contact_buffer_handles_empty_pairs_and_rejects_overflow():
    values=np.array([[100.,0,0],[1.,2,3],[4.,5,6],[200.,0,0]])
    assert np.allclose(aggregate_buffer(values,[[2,0]],[[1,999]]),[5,7,9])
    with pytest.raises(ValueError,match='buffer'):
        aggregate_buffer(values,[[3]],[[2]])


def test_probe_distinguishes_normal_only_matrix_and_friction_sign():
    rows=[dict(dt_s=.01,force_matrix_w_N=[0,0,9.81],normal_w_N=[0,0,9.81],friction_raw_w_N=[-1.,0,0],velocity_w=[.99,0,0],capacity_exhausted=False)]
    result=certify_force_probe(rows,1.,[1,0,0],[0,0,-9.81],tolerance_Ns=1e-6)
    assert result['verified'] and result['friction_sign']==1 and result['matrix_semantics']=='normal_only'
    rows[0]['velocity_w']=[1.,0,0]
    assert not certify_force_probe(rows,1.,[1,0,0],[0,0,-9.81],tolerance_Ns=1e-6)['verified']


def test_probe_rejects_saturated_contact_buffers():
    rows=[dict(dt_s=.01,force_matrix_w_N=[0,0,9.81],normal_w_N=[0,0,9.81],friction_raw_w_N=[-1.,0,0],velocity_w=[.99,0,0],capacity_exhausted=True)]
    assert not certify_force_probe(rows,1.,[1,0,0],[0,0,-9.81],tolerance_Ns=1e-6)['verified']


def test_probe_loader_recomputes_certificate_and_rejects_tampering():
    import copy
    from strike_dynamics_telemetry import validated_force_probe
    rows=[dict(dt_s=.01,force_matrix_w_N=[0,0,9.81],normal_w_N=[0,0,9.81],friction_raw_w_N=[-1.,0,0],velocity_w=[.99,0,0],capacity_exhausted=False)]
    probe=dict(samples=rows,mass_kg=1.,initial_velocity_w=[1,0,0],gravity_w=[0,0,-9.81])
    probe['certificate']=certify_force_probe(rows,1.,[1,0,0],[0,0,-9.81],tolerance_Ns=1e-6)
    assert validated_force_probe(probe)['verified']
    for sign in [0,-1,2]:
        wrong=copy.deepcopy(probe);wrong['certificate']['friction_sign']=sign
        with pytest.raises(ValueError,match='certificate'):
            validated_force_probe(wrong)
    wrong=copy.deepcopy(probe);wrong['samples'][0]['velocity_w']=[1.,0,0]
    with pytest.raises(ValueError,match='certificate'):
        validated_force_probe(wrong)

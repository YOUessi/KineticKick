import numpy as np
import pytest
from ground_contact_observer import indexed_entries, measured_wrench

def test_sparse_normal_and_friction_pairs_are_independent():
    values=np.array([[99.,99.,99.],[1.,2.,3.],[4.,5.,6.]])
    rows=indexed_entries({'point_w':values},np.array([[1,0,1]]),np.array([[1,999,2]]))
    assert [r['pair_index'] for r in rows]==[[0,0],[0,2]]
    assert [r['point_w'] for r in rows]==[[1.,2.,3.],[4.,5.,6.]]
    assert [r['buffer_index'] for r in rows]==[1,2]
    with pytest.raises(ValueError):indexed_entries({'point_w':values},[[2]],[[2]])

def test_wrench_uses_actual_separate_anchors_and_declared_origin():
    normal=[dict(point_w=[1.,0.,0.],force_w_N=[0.,0.,2.])]
    friction=[dict(point_w=[0.,1.,0.],force_w_N=[3.,0.,0.])]
    result=measured_wrench(normal,friction,[0.,0.,0.])
    assert result['force_w_N']==[3.,0.,2.]
    assert result['moment_w_Nm']==[0.,-2.,-3.]
    shifted=measured_wrench(normal,friction,[0.,0.,1.])
    np.testing.assert_allclose(shifted['moment_w_Nm'],[0.,-5.,-3.])

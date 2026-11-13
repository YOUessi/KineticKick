import numpy as np
from v160_runtime_contact import step_metrics


def test_ground_normal_impulse_and_centroid_do_not_depend_on_native_pair_order():
    foot = 'right_ankle_roll_link'
    points = [dict(bodies=['ground', foot], normal=[0,0,-1], impulse_on_first_Ns=[0,0,-.03], position=[-.1,0,0]),
              dict(bodies=[foot,'ground'], normal=[0,0,1], impulse_on_first_Ns=[0,0,.01], position=[.1,0,0])]
    initial = np.zeros(13); final = initial.copy(); final[9] = .04-.04905
    result = step_metrics(points,foot,initial,final,1.,np.array([0.,0.,-9.81]),.005)
    np.testing.assert_allclose(result['sole_normal_centroid_world_m'],[-.05,0,0],atol=1e-10)
    np.testing.assert_allclose(result['closure_residual_Ns'],0,atol=1e-10)


def test_unloaded_sample_has_no_pressure_location():
    state=np.zeros(13)
    assert step_metrics([], 'soccer_ball', state, state, 1., np.zeros(3), .005)['sole_normal_centroid_world_m'] is None

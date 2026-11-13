"""Sentry checks for deployment frame conventions and 50/200 Hz alignment."""
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from v160_tracking_cpu import interpolate, observation


def test_observation_keeps_reference_angular_velocity_in_world_but_sensor_and_ball_in_root():
    # A +90° yaw takes body X to world Y. World +Y velocity and target must appear as +X in body.
    root = SimpleNamespace(q=np.array([2**-.5,0,0,2**-.5]),p=np.array([1.,2.,.8]))
    ref = {'joint_pos':np.zeros((1,29)), 'joint_vel':np.zeros((1,29)),
           'body_ang_vel_w':np.array([[[0.,4.,0.]]])}
    obs = observation(np.zeros(29),np.zeros(29),root,[0.,2.,0.],np.array([1.,3.,.8]),
                      np.array([0.,2.,.8]),np.zeros(29),ref,0,0,np.zeros(29))
    assert obs.shape==(160,)
    np.testing.assert_allclose(obs[58:67], [0,0,-1,0,4,0,2,0,0], atol=1e-6)
    np.testing.assert_allclose(obs[-6:], [1,0,0,0,1,0], atol=1e-6)


def test_interp_reaches_reference_on_exact_policy_step_and_holds_end_after_clip():
    values = np.array([[0.,4.],[8.,-4.],[12.,4.]])
    # First physical sample after reset, first complete control step, last sample, and held tail.
    clock = np.array([.25,1.,2.,17.])
    np.testing.assert_allclose(interpolate(values,clock),[[2.,2.],[8.,-4.],[12.,4.],[12.,4.]])

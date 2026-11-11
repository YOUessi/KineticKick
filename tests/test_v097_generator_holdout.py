"""29DoF causal generators must be constrained and never claim motor authority."""
import json,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import v097_trajectory_generator as g
def test_train_and_fresh_test_scenes_are_disjoint():
 assert len(g.TRAIN)==4 and all(x[0] in 'CD' for x in g.TRAIN)
 assert len(g.TEST)==6 and all(x[0] in 'GH' for x in g.TEST)
 assert set(g.TRAIN).isdisjoint(g.TEST)
def test_group_partition_all_original_29():
 from wholebody_safety_interface_v055 import G1_JOINT_ORDER
 ix=g.group_indices(list(G1_JOINT_ORDER))
 assert sorted(i for a in ix.values() for i in a)==list(range(29))
def test_generation_is_29dof_and_phase_ablation_measurable():
 from wholebody_safety_interface_v055 import G1_JOINT_ORDER
 ix=g.group_indices(list(G1_JOINT_ORDER))
 v=np.arange(2*20*29,dtype=float).reshape(2,20,29)/20000
 assert np.array_equal(g.apply_ablation(v,'time_reversed',ix),v[:,::-1])
 frozen=g.apply_ablation(v,'freeze_arms',ix)
 assert np.count_nonzero(frozen[:,:,ix['left_counterarm']])==0
 assert np.count_nonzero(frozen[:,:,ix['right_arm']])==0
 assert np.any(v[:,:,ix['right_strike_leg']]!=0)
 only=g.apply_ablation(v,'strike_leg_only',ix)
 assert np.array_equal(only[:,:,ix['right_strike_leg']],v[:,:,ix['right_strike_leg']])
 for group,idx in ix.items():
  if group!='right_strike_leg':assert not only[:,:,idx].any()
 pos=g.integrate_measured_q0_29dof(v,np.zeros((2,29)),np.ones(29))
 assert pos.shape==(2,20,29)
 with pytest.raises(ValueError):
  g.integrate_measured_q0_29dof(v,np.zeros((2,28)),np.ones(29))
def test_candidate_does_not_include_true_future_as_input():
 assert g.ALPHA==10. and g.H==20
 # These are robot-only OFFLINE candidate models, not human or native-collision certificates.
 assert g.obs.V84.name=='kinetickick-wt-v084'

import json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from physx_pair_dynamics_v055 import valid_pair_record,PAIR
from wholebody_safety_interface_v055 import adapt,phase_coupled_intent,DOF,G1_JOINT_ORDER
class TestPairDynamicsV055(unittest.TestCase):
 def test_direct_pair_needs_manifold(self):
  row=dict(time_s=5.145,pair=list(PAIR),manifold=[dict(
      point_w=[0,0,0],normal_w=[1,0,0],normal_force_scalar_raw=25.)])
  self.assertTrue(valid_pair_record(row))
  with self.assertRaises(ValueError):valid_pair_record({**row,'pair':['left_wrist','ground']})
  with self.assertRaises(ValueError):valid_pair_record({**row,'manifold':[dict(point_w=[0]) ]})
 def test_immutable_protocol_matrix(self):
  c=json.loads((ROOT/'configs/physx_pair_dynamics_v055.json').read_text())
  self.assertEqual(len(c['cases']),15)
  self.assertEqual(len({x['id'] for x in c['cases']}),15)
  self.assertEqual(len([x for x in c['cases'] if x['id'].startswith('diag_center')]),3)
  self.assertEqual({x['seed'] for x in c['cases']},{1201,1509})
  self.assertEqual(len([x for x in c['cases'] if x['id'].startswith('holdout')]),12)
 def test_generated_29d_not_1mrad_and_coupled(self):
  d=phase_coupled_intent(.5)
  self.assertEqual(d.shape,(DOF,))
  self.assertGreater(np.count_nonzero(d),20)
  self.assertGreater(abs(d[11]),.04)
  self.assertLessEqual(np.max(np.abs(d)),.075)
  with self.assertRaises(ValueError):phase_coupled_intent(1.2)
 def args(self):
  z=np.zeros(DOF);one=np.ones(DOF)
  return dict(base_action=z,desired_delta_rad=phase_coupled_intent(.5),
    previous_delta_rad=z,actual_q_rad=z,actual_dq_rad_s=z,
    joint_q_lower_rad=-one,joint_q_upper_rad=one,joint_qdot_limit_rad_s=10*one,
    action_scale=.25*one,left_ground_force_N=150.,
    right_ground_force_N=0.,com_to_left_ankle_xy_m=.03,
    left_wrist_to_hip_origin_m=.3,physical_pair_contact=False,time_s=1.1,
    joint_names=G1_JOINT_ORDER,pair_evidence_available=True)
 def test_live_action_transform_and_original_limits(self):
  args=self.args();cmd,delta,e=adapt(**args)
  self.assertTrue(e['eligible'])
  self.assertGreater(np.linalg.norm(cmd),.01)
  self.assertLessEqual(np.max(np.abs(delta)),.006000001)
  self.assertTrue(np.allclose(cmd,delta/.25))
  args['previous_delta_rad']=delta
  cmd,delta2,e=adapt(**args)
  self.assertLessEqual(np.max(np.abs(delta2-delta)),.006000001)
 def test_unsafe_pair_and_velocity_original_base_untouched(self):
  for name,value in [('physical_pair_contact',True),
                     ('left_wrist_to_hip_origin_m',.11),
                     ('left_ground_force_N',35.),
                     ('com_to_left_ankle_xy_m',.30),
                     ('time_s',2.3),
                     ('already_hit_ball',True),
                     ('actual_dq_rad_s',np.ones(DOF)*9.)]:
   args=self.args();args[name]=value
   cmd,delta,e=adapt(**args)
   self.assertFalse(e['eligible'],name)
   self.assertTrue(np.array_equal(cmd,args['base_action']))
   self.assertTrue(np.array_equal(delta,np.zeros(DOF)))
 def test_fail_closed_without_actual_pair_proof_or_joint_mapping(self):
  for name,value in [('pair_evidence_available',False),('joint_names',None),
                     ('joint_names',list(reversed(G1_JOINT_ORDER))),
                     ('physical_pair_contact',None)]:
   args=self.args();args[name]=value
   cmd,delta,e=adapt(**args)
   self.assertFalse(e['eligible'],name)
   self.assertTrue(np.array_equal(cmd,args['base_action']))
   self.assertTrue(np.array_equal(delta,np.zeros(DOF)))
 def test_reject_missing_and_bad_joint_limit(self):
  args=self.args();args['desired_delta_rad']=np.ones(28)
  with self.assertRaises(ValueError):adapt(**args)
  args=self.args();args['joint_qdot_limit_rad_s']=np.zeros(DOF)
  with self.assertRaises(ValueError):adapt(**args)
if __name__=='__main__':unittest.main()

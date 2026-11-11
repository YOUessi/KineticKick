from pathlib import Path
import ast, json, unittest
ROOT=Path(__file__).resolve().parents[1]
class TestV94Contract(unittest.TestCase):
 def test_matrix_freeze(self):
  cfg=json.loads((ROOT/'configs/v094_frozen_matrix.json').read_text())
  self.assertEqual(len(cfg['matrix']),6)
  self.assertEqual(len(set(x['case'] for x in cfg['matrix'])),6)
  self.assertEqual(cfg['new_original_G1_experiments'],0)
  self.assertEqual([x['case'] for x in cfg['matrix']],
   ['cpu_pose_off','cpu_tensor_on','gpu_pose_off','gpu_tensor_on','gpu_native_on','gpu_separated_on'])
 def test_api_mismatch_repaired(self):
  s=(ROOT/'tools/v094_box_pair_probe.py').read_text()
  self.assertIn('view.sensor_count',s)
  self.assertNotIn('view.count',s)
  self.assertIn('view.sensor_paths',s)
  self.assertIn('view.filter_paths',s)
  self.assertIn('get_contact_data(.005)',s)
  ast.parse(s)
 def test_frozen_runner(self):
  s=(ROOT/'tools/v094_box_batch.py').read_text()
  for t in ['FROZEN','WRITE_ONCE','GLOBAL_GPU_MUTEX_BUSY','protocol_sha256','v0.1.94']:
   self.assertIn(t,s)
  ast.parse(s)
 def test_native_positive_requires_positive_real_contact(self):
  import importlib.util
  spec=importlib.util.spec_from_file_location('p',ROOT/'tools/v094_box_pair_probe.py')
  m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  base=dict(actors=['/World/ProbeBall','/World/ProbeBox'],
   colliders=['/World/ProbeBall/geometry/mesh','/World/ProbeBox/geometry/mesh'],
   points=[dict(position=[0,0,.38],normal=[0,0,1],impulse=[0,0,.1])])
  self.assertTrue(m.native_positive(base))
  self.assertFalse(m.native_positive({**base,'points':[dict(position=[0,0,.38],normal=[0,0,1],impulse=[0,0,0])]}))
  self.assertFalse(m.native_positive({**base,'colliders':['/World/ProbeBall/mesh','/World/Other/mesh']}))
  self.assertFalse(m.native_positive({**base,'points':[dict(position=None,normal=[0,0,1],impulse=[0,0,1])]}))
if __name__=='__main__':unittest.main()

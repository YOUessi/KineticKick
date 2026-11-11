"""Independent CPU-only guards for KineticKick v94 evidence and G1 censoring."""
from __future__ import annotations
import importlib.util, json, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("v094_audit",ROOT/"tools/v094_independent_audit.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class TestIndependentV094(unittest.TestCase):
 def test_180_exact(self):
  x=[[i+1,.38,0] for i in range(180)]
  self.assertTrue(m.exact_parity({"runtime_z_vz":x},{"runtime_z_vz":[p[:] for p in x]})["exact_at_9_digits"])
 def test_nonidentical_is_rejected(self):
  x=[[i+1,.38,0] for i in range(180)]
  y=[p[:] for p in x];y[44][1]+=.00001
  self.assertGreater(m.exact_parity({"runtime_z_vz":x},{"runtime_z_vz":y})["max_abs_z_vz"],0)
 def test_missing_frames_not_equivalence(self):
  self.assertFalse(m.exact_parity({"runtime_z_vz":[]},{"runtime_z_vz":[]})["valid"])
 def test_duplicate_physics_step_rejected(self):
  x=[[i+1,.38,0] for i in range(180)]
  y=[p[:] for p in x];y[0][0]=2
  with self.assertRaises(ValueError):m.exact_parity({"runtime_z_vz":x},{"runtime_z_vz":y})
 def test_source_tampering_detected(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/"a").write_text("not committed")
   with self.assertRaises(Exception):m.source_sha_check(root,["a"],"nonexistent")
 def test_full_six_cases_from_actual_verified_sources(self):
  p=ROOT/"reports/v094_frozen_contact_batch.audit.json"
  if not p.exists():self.skipTest("physical batch not executed")
  r=m.evaluate(ROOT)
  self.assertEqual(r["new_isolated_real_PhysX_probe_runs"],6)
  self.assertEqual(r["true_200hz_probe_steps"],1080)
  self.assertFalse(r["original_G1_wrist_hip_native_manifold_certified"])
  self.assertEqual(r["original_G1_new_shots"],0)
  self.assertFalse(r["gpu_separated_false_positive"])
  self.assertTrue(all(x["exact_at_9_digits"] for x in r["read_only_parity"].values()))
 def test_original_g1_reference_not_promoted_to_new_model(self):
  p=ROOT/"reports/v094_independent_evidence.audit.json"
  if not p.exists():self.skipTest("historical source not mounted")
  r=json.loads(p.read_text())
  self.assertEqual(r["historical_original_G1_strict_pass_count"],0)
  self.assertEqual(len(r["historical_original_G1_heldout"]),4)
  self.assertEqual(r["new_learned_human_coordination_actuation_episodes"],0)
  self.assertTrue(any(not x["post_1p25s_complete"] for x in r["historical_original_G1_heldout"].values()))
if __name__=="__main__":unittest.main()

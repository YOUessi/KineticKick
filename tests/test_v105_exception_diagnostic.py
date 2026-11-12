import sys,unittest,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import v105_trace_original_exception as t
import v105_g1_root_cause_batch as b

class TestFailureOnlyDiagnostic(unittest.TestCase):
 def test_trace_gated_to_original_main(self):
  self.assertEqual(t.NAME,"v076_eval_guarded.py")
  self.assertEqual(t.FINAL_LINES,(183,192))
 def test_immutable_frozen_control(self):
  c=json.loads(b.CFG.read_text())
  self.assertEqual(c["gpu_attempt_limit"],1)
  self.assertEqual(c["gpu_child_timeout_s"],75)
  self.assertEqual(c["gpu_ball_offset_xy"],[-.095,-.035])
  self.assertEqual(c["gpu_seed"],1601)
  self.assertFalse(c["safe_policy_actuation_allowed"])
 def test_preflight_source_checks_exist(self):
  self.assertIn("tools/v076_eval_guarded.py",b.FROZEN)
  self.assertIn("tools/v080_run_poll_strike.py",b.FROZEN)
 def test_evidence_write_once(self):
  from unittest.mock import patch
  with patch.object(b,"MASTER",Path("/dev/null")):
   with self.assertRaises((FileExistsError,RuntimeError)):b.preflight()
if __name__=="__main__":unittest.main()

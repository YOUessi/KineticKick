import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import v104_startup_preflight as p

class TestForensic(unittest.TestCase):
    def test_failed_log_does_not_fabricate_physics(self):
        row=p.describe_log('Actor MLP:\nFile "simulation_app.py" in close\n_app_control_on_stop_handle_fn')
        self.assertFalse(row["physics_ready"])
        self.assertFalse(row["original_physics_raw_trace_mark"])
        self.assertTrue(row["shutdown_stop_callback_witness"])
        self.assertTrue(row["true_root_exception_unknown"])
    def test_ready_log_classification_is_distinct(self):
        z=p.describe_log("FREE_SHOOTING_READY\nCLOSED_LOOP_STRIKE_RESULT")
        self.assertTrue(z["physics_ready"])
        self.assertFalse(z["shutdown_stop_callback_witness"])
    def test_original_exception_not_invented(self):
        self.assertFalse(p.describe_log("start")["pre_close_python_exception_captured"])
    def test_source_replay_log_sha_and_local_assets(self):
        row=p.inspect()
        self.assertTrue(row["confirmed_pre_step_failure"])
        self.assertTrue(row["true_root_exception_unknown"])
        self.assertEqual(len(row["local_ignored_evidence"]),4)
        self.assertFalse(row["not_a_native_actor_collider_pair_proof"] is False)

if __name__=="__main__":unittest.main()

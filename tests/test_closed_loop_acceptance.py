import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from closed_loop_acceptance import full_task_success

class OriginalAcceptance(unittest.TestCase):
    def test_standing_without_required_expert_handoff_is_not_full_success(self):
        running={'running_success':True};stride={'continuous_running_contact':True};sequence={'sequence_success':True}
        self.assertFalse(full_task_success(running,stride,sequence,[]))
        handoff=[dict(kind='post_shot_standing_recovery',time_s=4.94,physical_state_write=False)]
        self.assertTrue(full_task_success(running,stride,sequence,handoff))
        self.assertFalse(full_task_success(running,stride,{'sequence_success':False},handoff))

if __name__=='__main__':unittest.main()

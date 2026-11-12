import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from v100_startup_forensics import inspect_log,audit

def test_pre_stepping_stall_from_realistic_log():
 x=inspect_log('[INFO]: Completed setting up the environment...\nActor RNN: Memory\nOWN_CHILD_ONLY_TIMEOUT')
 assert x['milestones']['first_actor_recurrent_model_initialized']
 assert not x['milestones']['first_physics_frame_ready']
 assert x['has_watchdog_termination']
def test_success_positive_control():
 x=inspect_log('Actor RNN: Memory\nActor MLP: Sequential(\n  (0): Linear(in_features=154\nSEQUENCE_GROUND_COLLIDERS\nFREE_SHOOTING_READY\nFREE_SHOOTING_STEP 50\nCLOSED_LOOP_STRIKE_RESULT')
 assert x['last_completed_milestone']=='shot_result_computed'
def test_write_once_and_uncertified_fail_closed(tmp_path):
 good=tmp_path/'ok.log';bad=tmp_path/'bad.log';bad2=tmp_path/'bad2.log'
 good.write_text('CLOSED_LOOP_STRIKE_RESULT')
 bad.write_text('Actor RNN: Memory\nOWN_CHILD_ONLY_TIMEOUT')
 bad2.write_text('Actor RNN: Memory\nOWN_CHILD_ONLY_TIMEOUT')
 o=tmp_path/'old.json';n=tmp_path/'new.json';out=tmp_path/'audit.json'
 o.write_text(json.dumps({'completed_certified_original_G1_cuda':0,'runs':[{'timeout':True}]}))
 n.write_text(o.read_text())
 r=audit(good,bad,bad2,o,n,out)
 assert r['certified_new_original_G1_physics']==0
 with pytest.raises(FileExistsError):audit(good,bad,bad2,o,n,out)
 n.write_text(json.dumps({'completed_certified_original_G1_cuda':1,'runs':[{'timeout':False}]}))
 with pytest.raises(ValueError):audit(good,bad,bad2,o,n,tmp_path/'other.json')

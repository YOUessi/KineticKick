"""CPU-only preregistered fail-closed shutdown process classification."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("v075",ROOT/"tools/v075_guarded_batch.py")
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
GOOD="EIGHT_REAL_SIM_STEPS_DONE_NOT_G1\nAPP_CLOSE_ENTER\nAPP_CLOSE_RETURN\nPYTHON_MAIN_RETURN"

def test_no_child_pid_g1_claim(): assert mod.classify(GOOD,0,False)=="EIGHT_STEPS_CLEAN_NORMAL_EXIT"
def test_nonzero_exit_disallowed(): assert mod.classify(GOOD,1,False)=="INCOMPLETE_OR_NONZERO"
def test_incomplete_8steps_disallowed(): assert mod.classify("APP_CLOSE_RETURN\nPYTHON_MAIN_RETURN",0,False)=="INCOMPLETE_OR_NONZERO"
def test_close_hang_not_clean(): assert mod.classify("EIGHT_REAL_SIM_STEPS_DONE_NOT_G1\nAPP_CLOSE_ENTER",143,True)=="APP_CLOSE_TIMEOUT"
def test_stop_hang_not_clean(): assert mod.classify("SIM_STOP_ENTER",143,True)=="STOP_TIMEOUT"
def test_after_close_hang_not_clean(): assert mod.classify(GOOD,143,True)=="EXIT_AFTER_CLOSE_TIMEOUT"
def test_init_hang_not_clean(): assert mod.classify("APP_LAUNCH_ENTER",143,True)=="INIT_OR_STEPS_TIMEOUT"
def test_failed_exit_unconditionally_closed(): assert mod.classify("APP_CLOSE_RETURN\nPYTHON_MAIN_RETURN",1,True)!="EIGHT_STEPS_CLEAN_NORMAL_EXIT"
def test_protocol_contains_both_variants():
 import json
 c=json.loads(mod.PROTO.read_text())
 assert len(c["variants"])==2 and c["minimum_disk_gib"]>=20
def test_sha_chunk_streaming():
 import hashlib,tempfile
 with tempfile.TemporaryDirectory() as d:
  p=Path(d)/"source";p.write_bytes(b"abc")
  assert mod.digest(p)==hashlib.sha256(b"abc").hexdigest()

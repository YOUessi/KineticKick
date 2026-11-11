from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("v76_runner", ROOT/"tools/v076_g1_batch.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_identical_fields_pass():
 a={"seed":1201,"traces":[[{"joint_vel":[1]*29}]],"contacts":[1]}
 assert m.parity(a,dict(a))==(True,[])
def test_mutated_joint_speed_fails():
 a={"seed":1201,"traces":[[{"joint_vel":[1]*29}]]}
 b={"seed":1201,"traces":[[{"joint_vel":[2]*29}]]}
 assert m.parity(a,b)==(False,["traces"])
def test_modified_seed_fails():
 assert m.parity({"seed":1201},{"seed":1202})==(False,["seed"])
def test_missing_nonfoot_field_fails():
 a={"seed":1201,"termination_reasons":["physical_nonfoot"]}
 b={"seed":1201}
 assert "termination_reasons" in m.parity(a,b)[1]
def test_step_parity_no_float_tolerance():
 a={"seed":1,"traces":[[{"root_xyz":[1,2,3.0]}]]}
 b={"seed":1,"traces":[[{"root_xyz":[1,2,3.000000001]}]]}
 assert m.parity(a,b)[0] is False

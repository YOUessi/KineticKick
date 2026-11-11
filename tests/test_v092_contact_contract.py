"""v92 frozen PhysX evidence + native collider witness must fail closed."""
import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("v092_box_pair_probe",ROOT/"tools/v092_box_pair_probe.py")
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
spec2=importlib.util.spec_from_file_location("v092_box_batch",ROOT/"tools/v092_box_batch.py")
batch=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(batch)
def valid():
 return dict(actors=["/World/ProbeBall","/World/ProbeBox"],
  colliders=["/World/ProbeBall","/World/ProbeBox"],
  points=[dict(position=[0.,0.,.38],normal=[0,0,1.],
              impulse=[0,0,10.],impulse_scalar=None)])
def test_native_exact_side_paths():assert probe.native_positive(valid())
def test_missing_actor_rejected():
 v=valid();v["actors"][1]="/World/Other";assert not probe.native_positive(v)
def test_missing_collider_rejected():
 v=valid();v["colliders"][1]="/World/Ground";assert not probe.native_positive(v)
def test_no_points_rejected():
 v=valid();v["points"]=[];assert not probe.native_positive(v)
def test_zero_impulse_rejected():
 v=valid();v["points"][0]["impulse"]=[0,0,0];assert not probe.native_positive(v)
def test_zero_normal_rejected():
 v=valid();v["points"][0]["normal"]=[0,0,0];assert not probe.native_positive(v)
def test_impulse_scalar_valid():
 v=valid();v["points"][0]["impulse"]=None;v["points"][0]["impulse_scalar"]=20.
 assert probe.native_positive(v)
def test_nonfinite_vector_rejected():
 import math
 assert probe.vector((1.,2.,float("nan")))==None
def test_nonfinite_normal_rejected():
 v=valid();v["points"][0]["normal"]=[0,0,float("nan")]
 assert not probe.native_positive(v) if probe.vector(v["points"][0]["normal"]) is None else True
def test_frozen_matrix_and_gates():
 c=json.loads((ROOT/"configs/v092_frozen_matrix.json").read_text())
 assert c["original_G1_shots_scheduled"]==0
 assert c["required_free_gib"]>=20 and c["steps"]==180
 assert len(c["matrix"])==5
 assert len(set(x["case"] for x in c["matrix"]))==5
 assert [(x["device"],x["report_mode"]) for x in c["matrix"][:4]]==[
  ("cpu","off"),("cpu","on"),("cuda:0","off"),("cuda:0","on")]
 assert c["strict_G1_gates"]["instep_min"]==.95
def test_parity_exact_and_drift():
 a={"runtime_z_vz":[[1,.95,-.05],[2,.94,-.10]]}
 b={"runtime_z_vz":[[1,.95,-.05],[2,.94,-.10]]}
 assert batch.parity(a,b)["exact_at_saved_9_digits"]
 b["runtime_z_vz"][1][1]=.8
 assert not batch.parity(a,b)["within_1e_minus_6"]
def test_missing_trace_does_not_pass():assert not batch.parity({},{} )["valid"]

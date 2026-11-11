"""v96 frozen contract and deliberately negative contact witness acceptance tests."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from v096_contact_probe import native_positive
def test_frozen_matrix_protects_source():
 c=json.loads((ROOT/"configs/v096_frozen_matrix.json").read_text())
 assert len(c["matrix"])==9
 assert len({x["case"] for x in c["matrix"]})==9
 assert c["physical_G1_actuation_new"] is False and c["original_G1_shots_scheduled"]==0
 assert c["steps"]==180 and c["required_free_gib"]==20
 assert all(x["box_mode"] in ("kinematic","dynamic") and x["sensor_mode"] in ("manual","spawn","collider","both") for x in c["matrix"])
 assert len(c["parity_pairs"])==5
def test_native_actor_collider_impulse_and_normal_are_all_mandatory():
 e=dict(actors=["/World/ProbeBall","/World/ProbeBox"],colliders=["/World/ProbeBall/geometry/mesh","/World/ProbeBox/geometry/mesh"],points=[dict(position=[0.,0.,.2],normal=[0.,0.,1.],impulse=[0.,0.,.2])])
 assert native_positive(e)
 for alter in ({"actors":["/World/ProbeBall"]*2},{"colliders":["/World/ProbeBox"]*2},{"points":[dict(position=[0,0,0],normal=[0,0,0],impulse=[0,0,.2])]},{"points":[dict(position=[0,0,0],normal=[0,0,1],impulse=[0,0,0])]}):
  assert not native_positive({**e,**alter})

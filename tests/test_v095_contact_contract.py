"""Frozen v95 experiments and positive-contact acceptance tests."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from v095_contact_probe import native_positive
def test_positive_requires_paths_normal_impulse():
 e=dict(actors=["/World/ProbeBall","/World/ProbeBox"],colliders=["/World/ProbeBall/geometry/mesh","/World/ProbeBox/geometry/mesh"],points=[dict(position=[0,0,.2],normal=[0,0,1],impulse=[0,0,.03])])
 assert native_positive(e)
 assert not native_positive({**e,"colliders":["/World/ProbeBall/geometry/mesh"]*2})
 assert not native_positive({**e,"points":[dict(position=[0,0,.2],normal=[0,0,1],impulse=[0,0,0])]})
 assert not native_positive({**e,"points":[dict(position=[0,0,.2],normal=None,impulse=[0,0,.1])]})
def test_matrix_pre_registered_and_nonactuating():
 cfg=json.loads((ROOT/"configs/v095_frozen_matrix.json").read_text())
 assert cfg["original_G1_shots_scheduled"]==0 and cfg["physical_G1_actuation_new"] is False
 assert cfg["steps"]==180 and len(cfg["matrix"])==9
 assert len(set(r["case"] for r in cfg["matrix"]))==9
 assert len([r for r in cfg["matrix"] if r["device"]=="cuda:0"])==6
 assert any(r["subscribe_phase"]=="post_reset" for r in cfg["matrix"])
 assert all(r["capture"]=="pose" if r["report_mode"]=="off" else r["capture"]!="pose" for r in cfg["matrix"])

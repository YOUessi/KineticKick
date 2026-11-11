"""Offline fail-closed v88 contact witness checks (does not launch Isaac)."""
import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("v088_physx_positive",ROOT/"tools/v088_physx_positive.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def good():
    return {"actors":["/World/KickProbeBall","/World/Ground/CollisionPlane"],
            "colliders":["/World/KickProbeBall","/World/Ground/CollisionPlane"],
            "points":[{"position":[0,0,.2],"normal":[0,0,1],
                       "impulse_w":[0,0,12.0],"impulse_scalar":None}]}
def test_valid_positive():
    assert m.pair_has_paths(good()) and m.validated_point(good())
def test_reject_missing_collision_side():
    obj=good();obj["colliders"][1]="/World/Other"
    assert not m.pair_has_paths(obj) and not m.validated_point(obj)
def test_reject_zero_impulse():
    obj=good();obj["points"][0]["impulse_w"]=[0,0,0]
    assert not m.validated_point(obj)
def test_reject_invalid_normal():
    obj=good();obj["points"][0]["normal"]=[0,0,0]
    assert not m.validated_point(obj)
def test_reject_no_point():
    obj=good();obj["points"]=[]
    assert not m.validated_point(obj)
def test_json_protocol_frozen_matrix():
    c=json.loads((ROOT/"configs/v088_frozen_matrix.json").read_text())
    assert c["original_G1_gpu_shots_scheduled"]==0
    assert len(c["matrix"])==4 and len({x["case"] for x in c["matrix"]})==4
    assert c["min_free_gib"]>=20 and c["steps"]==180
def test_xyz_rejects_broken():
    assert m.xyz(None) is None and m.xyz([1,float("nan"),2]) is None
def test_xyz_good():
    assert m.xyz((1,2,3))==[1.,2.,3.]

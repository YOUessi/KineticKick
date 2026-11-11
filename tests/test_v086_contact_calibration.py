"""Pure CPU regression for physical witness parser and frozen v86 eligibility."""
import importlib.util
from pathlib import Path
from unittest.mock import Mock
import json
BASE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("v086_probe",BASE/"tools/v086_isolated_physx_contact.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def event(pair=True,impulse=(1.,0.,0.)):
    return {"actors": ["/World/KickProbeBall","/World/KickProbeFloor"] if pair else
            ["/World/KickProbeBall","/World/Unknown"],
            "colliders":["/World/KickProbeFloor","/World/KickProbeBall"],
            "points":[{"position":[0.,0.,0.], "normal":[0.,0.,1.],
                      "impulse":list(impulse) if impulse is not None else None}]}
def test_strict_native_pair_positive():
    assert module.positive(event()) is True
def test_wrong_actor_is_not_pair():
    assert not module.valid_pair(event(False))
    assert not module.positive(event(False))
def test_zero_impulse_does_not_certify():
    assert not module.positive(event(impulse=(0.,0.,0.)))
def test_missing_impulse_does_not_certify():
    assert not module.positive(event(impulse=None))
def test_no_contact_point_does_not_certify():
    e=event();e["points"][0]["position"]=None
    assert not module.positive(e)
def test_no_normal_does_not_certify():
    e=event();e["points"][0]["normal"]=None
    assert not module.positive(e)
def test_single_actor_missing_is_false():
    e=event();e["actors"]=e["actors"][:1]
    assert not module.valid_pair(e)
def test_no_new_g1_experiment_in_config():
    c=json.loads((BASE/"configs/v086_physx_contact_calibration.json").read_text())
    assert c["steps"]==180 and c["min_free_gib"]>=20
    assert [x["name"] for x in c["cases"]]==[
        "cpu_wildcard","gpu_wildcard","cpu_authored_empty","gpu_authored_empty"]
    assert all(x["mode"] in ("wildcard","authored_empty") for x in c["cases"])
    assert all(x["device"] in ("cpu","cuda:0") for x in c["cases"])

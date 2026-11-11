"""No mocked 3D G1 is counted as physics; only instrumentation protocol unit tests."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("stage",ROOT/"tools/staged_body_probe_v061.py")
stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)

class Stub:
    def __init__(self):
        self.sim=SimpleNamespace(current_time=.005)
        self.scene={"robot":SimpleNamespace(body_names=["left_wrist_yaw_link","left_hip_roll_link"],data=SimpleNamespace())}

def test_stages_unique_and_locked():
    assert len(stage.MODES)==9 and len(set(stage.MODES))==9
    c=json.loads((ROOT/"configs/g1_staged_probe_v061.json").read_text())
    assert c["version"]=="v0.1.61"
    assert c["original_29DoF"] and not c["control_tuning"]
    assert len(c["cases"])==12
    assert set(x["probe"] for x in c["cases"])==set(stage.MODES)
    assert all(x["seed"]==1201 for x in c["cases"])
    assert len({x["output"] for x in c["cases"]})==12

def test_empty_stage_keeps_actual_200hz_time_and_never_writes():
    p=stage.StageProbe(Stub(),"empty")
    p.capture(.005);p.capture(.005)
    o=p.result()
    assert o["capture_count"]==2
    assert o["last_successful_stage"]=="capture_complete"
    assert o["rows"]==[{"t":.005},{"t":.005}]
    assert not o["controller_writes"] and not o["physx_writes"]

def test_non_200hz_denied():
    p=stage.StageProbe(Stub(),"empty")
    with pytest.raises(ValueError,match="200Hz"):p.capture(.02)
    with pytest.raises(ValueError,match="No real physics"):p.result()

def test_unknown_stages_denied():
    with pytest.raises(ValueError,match="Unregistered"):stage.StageProbe(Stub(),"unsafe_write")

def test_source_does_not_write_to_actuators_or_colliders():
    s=(ROOT/"tools/staged_body_probe_v061.py").read_text()
    assert "set_joint_position_target(" not in s
    assert "set_world_poses(" not in s
    assert "ApplyAPI(" not in s
    assert "get_contact_report()" in s

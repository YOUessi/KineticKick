"""CPU contracts only: no fake USD stage becomes real PhysX contact truth."""
from pathlib import Path
import sys
from dataclasses import dataclass
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import v133_runtime_stage_contract as m


@dataclass
class Prim:
    path: str
    parent: object = None
    collider: bool = False
    body: bool = False
    report: bool = False


def inspect(prims):
    return m.ancestry_owner_map(
        prims,path_of=lambda p:p.path,parent_of=lambda p:p.parent,
        collision_enabled=lambda p:p.collider,rigid_body_enabled=lambda p:p.body,
        report_enabled=lambda p:p.report)


def fixture():
    world=Prim("/World")
    robot=Prim("/World/G1",world)
    shoulder=Prim("/World/G1/left_shoulder_yaw_link",robot,body=True,report=True)
    torso=Prim("/World/G1/torso_link",robot,body=True,report=True)
    wrist=Prim("/World/G1/left_wrist_yaw_link",robot,body=True,report=False)
    cylinders=[
        Prim("/World/G1/torso_link/col_0",torso,collider=True),
        Prim("/World/G1/torso_link/col_1",torso,collider=True),
        Prim("/World/G1/left_shoulder_yaw_link/col_0",shoulder,collider=True),
        Prim("/World/G1/left_wrist_yaw_link/col_0",wrist,collider=True)]
    return [world,robot,shoulder,torso,wrist,*cylinders]


def test_multiple_colliders_and_exact_parent_ownership():
    r=inspect(fixture())
    owners=r["collider_to_nearest_rigidbody_owner"]
    assert len(owners)==4
    assert owners["/World/G1/torso_link/col_1"]=="/World/G1/torso_link"
    assert r["rigid_body_owners_without_report_API"]==[
        "/World/G1/left_wrist_yaw_link"]
    assert not r["all_owners_have_report_API"]
    assert not r["native_pair_detected"]
    assert not r["actuation_authorized"]


def test_extra_unowned_collider_is_missing_not_safe():
    pr=fixture()
    pr.append(Prim("/World/stray/collider",None,collider=True))
    r=inspect(pr)
    assert r["missing_owner_is_not_no_collision"]
    assert r["colliders_with_no_rigidbody_owner"]==["/World/stray/collider"]


def test_collision_report_all_present_still_not_truth():
    pr=fixture()
    for p in pr:
        if p.body:
            p.report=True
    r=inspect(pr)
    assert r["all_owners_have_report_API"]
    assert not r["native_pair_detected"]


def test_ancestor_rigidbody_not_collider_parent_and_inner_wins():
    pr=fixture()
    pr[1].body=True
    pr[1].report=True
    r=inspect(pr)
    assert r["collider_to_nearest_rigidbody_owner"]["/World/G1/torso_link/col_0"]=="/World/G1/torso_link"


def test_cycle_rejected():
    a=Prim("/World/A",collider=True)
    b=Prim("/World/B",parent=a)
    a.parent=b
    with pytest.raises(ValueError,match="CYCLE"):
        inspect([a,b])


def test_duplicate_stage_path_rejected():
    p=fixture()
    p.append(Prim("/World/G1/torso_link"))
    with pytest.raises(ValueError,match="DUPLICATE"):
        inspect(p)


def test_reject_unsafe_actuation_claim(tmp_path):
    with pytest.raises(ValueError,match="UNSAFE"):
        m.save_preflight({"actuation_authorized":True},tmp_path/"p.json")


def test_safe_report_written_exclusive(tmp_path):
    dest=tmp_path/"preflight.json"
    m.save_preflight(inspect(fixture()),dest)
    assert dest.exists()
    with pytest.raises(FileExistsError):
        m.save_preflight(inspect(fixture()),dest)

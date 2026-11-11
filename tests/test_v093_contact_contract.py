"""v93 independent matrix/reporting contract: safe-before-physics, no G1 writes."""
import importlib.util,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def load(name):
 s=importlib.util.spec_from_file_location(name,R/"tools"/(name+".py"))
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
P=load("v093_box_pair_probe");B=load("v093_box_batch")
def event():
 return dict(actors=["/World/ProbeBall","/World/ProbeBox"],
  colliders=["/World/ProbeBall","/World/ProbeBox"],
  points=[{"position":[0,0,.38],"normal":[0,0,1],"impulse":[0,0,4],"impulse_scalar":None}])
def test_real_native_actor_and_collider_requirements():
 assert P.native_positive(event())
 for typ in ("actors","colliders"):
  e=event();e[typ][1]="/World/Other";assert not P.native_positive(e)
def test_event_fails_without_impulse_or_valid_contact():
 e=event();e["points"][0]["impulse"]=[0,0,0];assert not P.native_positive(e)
 e=event();e["points"][0]["impulse_scalar"]=2.;assert P.native_positive(e)
 e["points"][0]["normal"]=[0,0,0];assert not P.native_positive(e)
def test_malformed_no_positive():
 for e in ({},{"actors":["/World/ProbeBall"],"colliders":[],"points":[]},
           {"actors":[],"colliders":[],"points":[]}):
  assert not P.native_positive(e)
def test_vector_checks():
 assert P.vector([1,2,3])==[1.,2.,3.]
 assert P.vector([1,float("nan"),3]) is None
 assert P.vector(None) is None
def test_frozen_matrix_and_no_g1_motor_writes():
 c=json.loads((R/"configs/v093_frozen_matrix.json").read_text())
 assert c["version"]=="v0.1.93" and c["steps"]==180 and c["timeout_s"]>=60
 assert c["required_free_gib"]>=20 and c["original_G1_shots_scheduled"]==0
 assert not c["physical_G1_actuation_new"]
 assert len(c["matrix"])==7 and len(set(t["case"] for t in c["matrix"]))==7
 assert all(t["report_mode"]=="off" and t["capture"]=="pose"
            or t["report_mode"]=="on" and t["capture"] in ("native","tensor")
            for t in c["matrix"])
 assert c["off_requires_absent_PhysxContactReportAPI"]
 assert c["no_kinematic_velocity_write"]
def test_recovered_bug_not_in_new_source():
 p=(R/"tools/v093_box_pair_probe.py").read_text()
 assert p.count("activate_contact_sensors=False")==2
 assert "box.write_root_state_to_sim" not in p
 assert "V093_EXCEPTION" in p and "traceback.format_exc()" in p
 assert "all_colliders" in p
 assert "V093_VIEW_READY" in p
def test_strict_physical_instrument_parity():
 a={"runtime_z_vz":[[1,.95,-.01],[2,.94,-.02]]}
 b={"runtime_z_vz":[[1,.95,-.01],[2,.94,-.02]]}
 x=B.parity(a,b)
 assert x["valid"] and x["exact_at_saved_9_digits"]
 b["runtime_z_vz"][1][1]=.93
 assert not B.parity(a,b)["within_1e_minus_6"]
 assert not B.parity({},b)["valid"]

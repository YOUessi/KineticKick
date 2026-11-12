import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from v120_contact_batch import CONFIG, parity, diagnose
from v120_contact_phase_probe import native_positive,vector

def test_protocol_frozen_matrix():
    c=json.loads(CONFIG.read_text())
    assert c["version"]=="v0.1.120"
    assert c["steps"]==180 and c["dt_seconds"]==.005
    assert c["min_free_gib"]==20 and c["timeout_seconds"]==150
    assert c["original_G1_motor_changed"] is False
    assert [r["name"] for r in c["matrix"]]==[
        "cuda_positive_off","cuda_positive_on","cuda_spawner_on",
        "cuda_rootshape_on","cuda_postreset_on","cuda_render_on","cuda_separated_on"]

def test_scientific_instrument_axes_are_orthogonal():
    c=json.loads(CONFIG.read_text())["matrix"]
    assert {r["authoring"] for r in c}=={"root","spawner","root_shape"}
    assert {r["subscribe_phase"] for r in c}=={"before_reset","after_reset"}
    assert {r["step_mode"] for r in c}=={"plain","render"}
    assert all(r["device"]=="cuda:0" for r in c)

def test_readonly_replay_parity_requires_identical_records():
    x={"device":"cuda:0","runtime_z_vz":[[1,.9,0.1]],"box_position_w":[0,0,0]}
    assert parity(x,dict(x))["recorded_pose_velocity_parity"]
    assert not parity(x,dict(x,runtime_z_vz=[[1,.91,.1]]))["recorded_pose_velocity_parity"]

def test_native_incomplete_evidence_fails_closed():
    full={"actors":["/World/ProbeBall","/World/ProbeBox"],
          "colliders":["/World/ProbeBall/geometry/mesh","/World/ProbeBox/geometry/mesh"],
          "points":[{"position":[0,0,.3],"normal":[0,0,1],"impulse":[0,0,.12]}]}
    assert native_positive(full)
    for broken in (dict(full,points=[]),
                   dict(full,points=[dict(full["points"][0],impulse=[0,0,0])]),
                   dict(full,colliders=["/World/ProbeBall/geometry/mesh","/World/Floor"])):
        assert not native_positive(broken)

def test_tensor_forces_cannot_certify_manifold():
    x=diagnose({"actual_runtime_frames":180,"tensor_contact_counts":113,"tensor_peak_force_N":667.08})
    assert x["runtime_steps"]==180
    assert not x["native_pair_witness"]

def test_vector_rejects_nonfinite():
    assert vector([1.,2.,float("nan")]) is None
    assert vector([1.,2.,3.]) == [1.,2.,3.]

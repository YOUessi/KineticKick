"""Stage-2 immutable center OFF/ON G1 filtered tensor diagnostic contract."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def cfg():return json.loads((ROOT/"configs/v098_pair_protocol.json").read_text())
def test_center_original_three_controls_paired_off_on():
 c=cfg()
 assert c["version"]=="v0.1.98" and c["seed"]==1201
 assert c["parent_frozen_physics_sha"]=="99c84dfc82d2bea78744a8eb9081f008119eb6c2"
 assert [(z["control"],z["observer"]) for z in c["pair_matrix"]]==[
   (p,v) for p in ("v32","cmu02","cmu05") for v in (False,True)]
 assert len(set(z["name"] for z in c["pair_matrix"]))==6
 assert c["center_ball_offset_xy"]==[0.,0.]
def test_contact_not_manifold_and_no_actuation_change():
 c=cfg()
 assert c["safety"]["no_change_to_original_controller"]
 assert c["safety"]["no_change_to_original_physx_scene"]
 assert c["safety"]["no_native_manifold_claim"]
 assert c["safety"]["failed_tensor_instrumentation_is_unknown"]
 assert c["safety"]["paired_state_parity_mandatory"]
 assert c["safety"]["stop_on_drift"]
 assert c["resources"]["min_free_gib"]>=20
 assert c["resources"]["per_child_timeout_s"]<=180
def test_runner_has_only_optional_tensor_observer():
 original=(ROOT/"tools/v080_run_poll_strike.py").read_text()
 copied=(ROOT/"tools/v098_run_pair_strike.py").read_text()
 assert "G1PairDiagnostic" in copied and "tensor-pair-v098" in copied
 assert "G1PairDiagnostic" not in original
 assert "if a.tensor_pair_v098 and dt>0:" in copied
 assert "self.tensor_pair_v098.capture(dt)" in copied
def test_exact_parity_comparator_is_recursive_not_summary():
 import sys
 sys.path.insert(0,str(ROOT/"tools"))
 from v098_pair_batch import _max_abs
 assert _max_abs({"frame":[1.,{"q":[2,3,4]}]},{"frame":[1.,{"q":[2,3,4]}]})==0.
 assert _max_abs({"frame":[1.,{"q":[2,3,4]}]},{"frame":[1.,{"q":[2,3,4.01]}]})>.009
 assert _max_abs({"bodies":[1,2]},{"bodies":[1,2,3]})==float("inf")

"""v91 frozen matrix, no source mutation, no arbitrary success promotion."""
import json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from v091_gpu_batch import CFG,FROZEN
def test_frozen_scene_matrix():
 c=json.loads(CFG.read_text())
 assert c["version"]=="v0.1.91"
 assert c["parent_commit"]=="c87672181f0a8224aeaece6e11dae2a06a6f5ab8"
 assert len(c["cases"])==4
 assert [z["name"] for z in c["cases"]]==["E_v32","E_cmu02","F_v32","F_cmu02"]
 assert {z["scene"] for z in c["cases"]}=={"E","F"}
 assert {z["control"] for z in c["cases"]}=={"v32","cmu02"}
 assert set(z["seed"] for z in c["cases"])=={1319,1327}
 assert len(FROZEN)==4
def test_gates_require_real_recovery_and_actual_instep():
 c=json.loads(CFG.read_text())
 assert c["gates"]["instep_impulse_min"]==0.95
 assert c["gates"]["full_recovery_s"]==14
 assert c["gates"]["max_nonfoot_contact_N_exclusive"]==20
 assert c["gates"]["ball_speed_min_mps"]==3.5
def test_no_future_contact_or_test_leak():
 c=json.loads(CFG.read_text())
 assert c["causal_contract"]["training_uses_test_samples"] is False
 assert c["causal_contract"]["oracle_future_impact_as_input"] is False
 assert c["causal_contract"]["causal_support_obs_only"] is True
 assert set(c["causal_contract"]["heldout_scenes"]).isdisjoint({"C","D"})
def test_no_unqualified_biomechanics_labels():
 c=json.loads(CFG.read_text())
 assert any("professional" in s for s in c["false_claim_prohibitions"])
 assert any("native" in s for s in c["false_claim_prohibitions"])

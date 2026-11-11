"""Frozen v98 independent I/J/K 9-shot original G1 source and gates."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def cfg():return json.loads((ROOT/"configs/v098_frozen_matrix.json").read_text())
def test_exact_nine_episodes():
 c=cfg()
 assert c["version"]=="v0.1.98" and c["parent_commit"]=="ff6dfd194141f5089285c8ec1babe991eaa36829"
 assert [z["scene"] for z in c["cases"]]==["I"]*3+["J"]*3+["K"]*3
 assert [z["control"] for z in c["cases"]]==["v32","cmu02","cmu05"]*3
 assert len(set(z["name"] for z in c["cases"]))==9
 assert {z["seed"] for z in c["cases"]}=={1471,1487,1493}
 assert all(max(abs(v) for v in z["ball_offset_xy"])<=.15 for z in c["cases"])
def test_frozen_safety_and_integrity():
 c=cfg()
 assert c["settings"]["min_free_gib"]>=20 and c["settings"]["max_new_G1_episodes"]==9
 assert c["settings"]["gpu_lock"].endswith("kinetickick_physics_gpu.lock")
 assert c["gate_protocol"]["min_instep_fraction"]==.95
 assert c["gate_protocol"]["min_ball_speed_mps"]==3.5
 assert c["gate_protocol"]["max_nonfoot_net_N_exclusive"]==20
 assert c["gate_protocol"]["zero_native_callback_means"]=="UNKNOWN"
 assert c["causal_contract"]["no_test_result_used_to_tune_actions"]
 assert not any(z["native_poll"] for z in c["cases"])
def test_same_immutable_upstream_config():
 import hashlib
 for name,p in cfg()["controls"].items():
  a=json.loads((ROOT/p).read_text())
  for field in ("template","reference","expert_plan"):
   source=ROOT/a[field]
   assert source.is_file()
   assert hashlib.sha256(source.read_bytes()).hexdigest()==a[field+"_sha256"]

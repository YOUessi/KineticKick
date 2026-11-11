#!/usr/bin/env python3
"""Idempotency-protected v098 new-worktree scaffolding before protocol freeze."""
from pathlib import Path
import json, shutil, hashlib
ROOT=Path(__file__).resolve().parents[1]
assert ROOT.name=="kinetickick-wt-v098"
def new(path,data):
 p=ROOT/path
 assert not p.exists(), str(p)
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(data)
 return p
old=json.loads((ROOT/"configs/v097_frozen_matrix.json").read_text())
old["version"]="v0.1.98"
old["parent_commit"]="ff6dfd194141f5089285c8ec1babe991eaa36829"
old["purpose"]="Preregistered independent I/J/K ball placement stress test; all three unchanged original G1 29DoF controllers, 200Hz CUDA PhysX, no new actuator authority"
old["settings"]["max_new_G1_episodes"]=9
design=[("I",1471,[0.005,-0.070],-0.02),
        ("J",1487,[-0.065,0.015],0.0),
        ("K",1493,[0.065,0.065],0.02)]
old["cases"]=[dict(name=f"{name}_{control}",scene=name,control=control,seed=seed,
                   ball_offset_xy=xy,velocity_delta=vel,native_poll=False)
              for name,seed,xy,vel in design for control in ("v32","cmu02","cmu05")]
old["causal_contract"]["previous_scenes_are_not_blind_training"]=["C","D","E","F","G","H"]
old["causal_contract"]["new_pre_registered_scenes"]=["I","J","K"]
old["causal_contract"]["scenes_holdout_only_within_this_version"]=True
old["comparison_labels"]["v32"]="unchanged v32 baseline"
new("configs/v098_frozen_matrix.json",json.dumps(old,indent=2)+"\n")
p=(ROOT/"tools/v097_gpu_batch.py").read_text().replace("v097","v098").replace("v0.1.97","v0.1.98")
p=p.replace('len(c["cases"])!=6','len(c["cases"])!=9')
p=p.replace('["G","H"]','["I","J","K"]')
p=p.replace('6 preregistered original G1 shots','9 preregistered original G1 shots')
new("tools/v098_gpu_batch.py",p)
p=(ROOT/"tools/v097_independent_dynamics.py").read_text().replace("v097","v098").replace("v0.1.97","v0.1.98")
p=p.replace('CASES=("G_v32","G_cmu02","G_cmu05","H_v32","H_cmu02","H_cmu05")',
            'CASES=("I_v32","I_cmu02","I_cmu05","J_v32","J_cmu02","J_cmu05","K_v32","K_cmu02","K_cmu05")')
new("tools/v098_independent_dynamics.py",p)
parent=old["parent_commit"]
test='''"""Frozen v98 independent I/J/K 9-shot original G1 source and gates."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def cfg():return json.loads((ROOT/"configs/v098_frozen_matrix.json").read_text())
def test_exact_nine_episodes():
 c=cfg()
 assert c["version"]=="v0.1.98" and c["parent_commit"]=="'''+parent+'''"
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
'''
new("tests/test_v098_protocol.py",test)
doc='''# v0.1.98 Frozen pre-experiment protocol (UNSEEN I/J/K)
**Parent**: `'''+parent+'''`. The exact config is `configs/v098_frozen_matrix.json`; the deterministic executor is `tools/v098_gpu_batch.py`.
Frozen before any new v98 G1 GPU shot. Do not modify control gains, collision, motor gains, stage, GRF/contact thresholds, human labels or policy. Worktree checked clean and source SHA locked. I=(+0.005,-0.070)m, delta vy=-0.020, seed1471; J=(-0.065,+0.015)m, delta vy=0, seed1487; K=(+0.065,+0.065)m, delta vy=+0.020, seed1493. Each has v32, CMU10_02, CMU10_05, exactly 9 shots. The seed may NOT perturb initial dynamics; count experimental units as ball positions, not seeds.
Acquire existing global `flock`; each CUDA PhysX episode serial, resource check before every, watchdog 180s own child only; >=20GiB free. Each raw source output `reports/v098_*.json`, local write-once. Independent source SHA audit, true 200Hz steps (exclude +1 microsecond terminal-only annotation), actual all29 qdot, >95% instep impulse, actual goal, >=3.5m/s proxy, non-foot strictly <20N, 14s full recovery. Never promote a ball crossing goal to full pass. Non-native contact readback uncalibrated means UNKNOWN, no collider path proof.
**STOP** when a physics certification fails; do not delete or overwrite. New candidate generators remain offline only. Historic C/D/E/F/G/H are not paper-level blind human demonstrations; six CMU sequences lack synchronized expert true strike impulse/GRF.
Hypothesis H1: fixed legacy controller is sensitive to ball placement, with identical policies able to pass or fail. H2: missing human-contact labels and native pair calibration block certified fullbody imitation control. Predeclared qualitative comparisons, NO hyperparameter tuning on I/J/K test.
'''
new("docs/V098_FROZEN_PROTOCOL.md",doc)
print("SCAFFOLD",len(old["cases"]),"new files OK")

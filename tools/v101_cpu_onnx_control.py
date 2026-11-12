#!/usr/bin/env python3
"""CPU ONNX-only positive control; no Isaac/PhysX and no claimed shot."""
import hashlib, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from soccer_onnx_policy import OnnxSoccerPolicy

def probe():
    p=ROOT/".pretrained_checkpoints/paid_official/policy_30000.onnx"
    if not p.is_file():raise FileNotFoundError(p)
    start=time.monotonic()
    model=OnnxSoccerPolicy(p,1)
    return dict(ready=True,elapsed_s=round(time.monotonic()-start,6),
                model_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                joint_count=len(model.joint_names),
                observation_count=len(model.observation_names))

if __name__=="__main__":
    result=probe()
    print("V101_CPU_ONNX_READY",json.dumps(result,sort_keys=True),flush=True)

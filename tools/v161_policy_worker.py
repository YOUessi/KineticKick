"""CPU policy sequence service with an explicit recurrent checkpoint for development search."""
import json
import sys
from pathlib import Path
import numpy as np

assets=Path(sys.argv[1])
sys.path.insert(0,str(assets/'experiments/runtime/onnxruntime'))
from soccer_onnx_policy import OnnxSoccerPolicy
actor=OnnxSoccerPolicy(assets/'.pretrained_checkpoints/paid_official/policy_30000.onnx',1)
print(json.dumps(actor.metadata),flush=True)
for line in sys.stdin:
    msg=json.loads(line)
    if msg=='save':
        h,c=actor.h.copy(),actor.c.copy()
        result=True
    elif msg=='restore':
        actor.h[:]=h
        actor.c[:]=c
        result=True
    else:
        frame,obs=msg
        result=actor.step(np.asarray([obs],dtype=np.float32),frame)[0].tolist()
    print(json.dumps(result),flush=True)

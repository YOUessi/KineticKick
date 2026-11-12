#!/usr/bin/env python3
"""Generate a v98-local copy of the original v80 launcher with only optional read-only tensor diagnostics.

This generator executes before contact matrix freeze; it neither overwrites nor
modifies the original v080 script or upstream controllers.
"""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
src=(root/"tools/v080_run_poll_strike.py").read_text()
dst=root/"tools/v098_run_pair_strike.py"
assert not dst.exists()
one="    p.add_argument('--contact-diagnostics',action='store_true',help='Record non-foot collision bodies without changing termination')"
assert src.count(one)==1
src=src.replace(one,one+"\n    p.add_argument('--tensor-pair-v098',action='store_true',help='Read-only filtered left/right wrist-hip tensor pairs after physics step')")
one="                self.pair_poll_v080=None"
assert src.count(one)==1
src=src.replace(one,one+"\n                self.tensor_pair_v098=None")
one="                    self.task.after_physics(dt)"
assert src.count(one)==1
src=src.replace(one,one+"""
                    if a.tensor_pair_v098 and dt>0:
                        if self.tensor_pair_v098 is None:
                            from v098_g1_tensor_pair import G1PairDiagnostic
                            self.tensor_pair_v098=G1PairDiagnostic(self.env)
                        self.tensor_pair_v098.capture(dt)""")
one="            if actor.pair_poll_v080 is not None:result['physx_native_poll_v080']=actor.pair_poll_v080.result()"
assert src.count(one)==1
src=src.replace(one,one+"\n            if actor.tensor_pair_v098 is not None:result['g1_tensor_pair_v098']=actor.tensor_pair_v098.result()")
dst.write_text(src)
print("LOCAL_COPIED_RUNNER_NO_ORIGINAL_MUTATION",len(src))

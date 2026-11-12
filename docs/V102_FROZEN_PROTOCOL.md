# KineticKick v0.1.102 — preregistered startup diagnosis, 2026-11-12

**Parent:** `fbfd3e35016538e6168af063b5d64f19451f7b9e`, clean version/v0.1.100. Do not overwrite any v99/v100 timeout, original CMU/C3D, checkpoints, or PhysX JSON. This experiment is not a blind generalization test. No model weights or policy gains may change after observing data.

## Dependency-gated chain

1. CPU-only official ONNX session metadata positive control, no simulated episode.
2. CUDA tensor isolated ball/box **actual 180-step PhysX** positive control using the exact frozen v094 probe. Failure/timeout blocks G1.
3. Original 29DoF G1 old controller L scene v32, CMU02, CMU05 each observer OFF/ON in that order, 6 runs with fixed seed1601, ball offset(-.095,-.035), zero velocity delta; same original wrappers as v100. CPU/GPU batch serialized under unchanged global GPU flock. **First failed/uncertified G1 ends batch**. Do not retry the same run; unstarted trials are not failures.
4. Every Python child runs inside read-only Python `runpy` trampoline with periodically printed all-thread stack trace (15s), without modifying model/control/PhysX settings. This changes execution context and must be treated as diagnostic, not as independent OFF/ON physical nondrift proof. Validate all shared physical fields and original shot records exactly for any completed OFF/ON pair.
5. Build auditable source-hashed report and tests. Raw JSON, logs, stack traces, tensor outputs and media remain local ignored. Published material only code/config/protocol/tests/report. All original hard gates conjunctive (instep impulse >=95%, goal, speed >=3.5m/s, all real 29 original qdot limits, 14s recovery, no nonfoot collision).

GPU safe minimum disk 20GiB and no other GPU compute, child-only watchdog max 120s, no globally killed process, no existing artifact overwritten, no branches force-pushed. An environment startup stall before a certified 200Hz G1 step is NOT a failed shot. Isolated box contact success is not G1 safety certification. Native actor-collider manifold remains unknown until positively calibrated.

## v101 CLI correction, independently refrozen

The v101 first CUDA box positive did NOT start PhysX: incorrectly used `--device cuda` (must be `cuda:0`) and `--report-mode off --capture tensor` (must use `on`). Both corrections apply exclusively to infrastructure positive control, **not** to old G1 controllers, ball geometry, seeds, ONNX, reward or physical gates. New write-once v102 outputs; v101 logs and master untouched. A contract test checks both exact accepted CUDA choice and reporter/capture match before GPU.

# KineticKick v0.1.119 — CPU/CUDA native PhysX contact-pair causality test (real physics, negative)

**2026-11-12, Tang, Ubuntu + RTX4090.** Branch `version/v0.1.119` is a *new clean branch* from independently verified v118 final source SHA `b2ff2fc4254d41f1e871928a27fc8200b40ebec4`. Frozen and remotely pushed `configs/v119_native_backend_frozen.json`, `tools/v119_native_backend_batch.py`, tests and protocol at SHA `4164d2f2c9b53cbba25a6ac58425fd66626008d7` **before executing any v119 experiments**. Protocol SHA256 `6ea97e30e60255e9b935bbfc2f9e912f494cd9ddcddd66f63c3b6e7ec4d16540`.

## Safety and limits

Tang online; approximately 31 GiB free >20 GiB threshold, no GPU compute jobs. All six Isaac Sim children were run sequentially inside the existing nonblocking shared GPU lock with 120s per-child watchdog, no unrelated process killed. Original 29DoF G1 source, URDF, actuator weights, CMU/C3D, and all previous raw PhysX evidence unchanged. All v119 raw JSONs, logs, audit manifests and a measured trajectory figure remain **gitignored locally**, individually SHA256 pinned and created write-once. Reused frozen, committed v118 contact probe without altering any physics controller.

## Six new real isolated 200 Hz PhysX runs

Two simulation backends (CPU / CUDA:0), same contact probe, each 180 actual physics steps: true ball-box colliding positive OFF/ON and deliberately separated negative ON. **Six actual physical rollouts = 1,080 true simulation steps; zero v119 original G1 shooting episodes**. Do not confuse this with v118's separate 8 G1 shots and 3 isolated calibrations. The purpose is debugging the *precondition* for G1 actor/collider truth, not to claim new shooting success.

| Backend | Positive OFF actual contact dynamics | Positive ON actual contact dynamics | Separated ON | ON Tensor-positive samples | Native regular/full callback/poll headers | Valid native actor/collider manifold |
|---|---|---|---|---:|---:|---|
| CPU | yes | yes | separated; no Tensor sample | **0** | **0 / 0 / 0 / 0** | **NO** |
| CUDA | yes | yes | separated; no Tensor sample | **113** | **0 / 0 / 0 / 0** | **NO** |

CUDA positive ON measured a real Tensor peak contact force **667.0803 N** and 113 recorded contact sample counts; CPU also showed colliding physical ball dynamics (z trajectory), but its Tensor view emitted zero positive samples. This is **an evidence-channel limitation**, not proof that CPU collision was absent. Both backends' USD scene evidence shows `RigidBodyAPI` and applied `PhysxContactReportAPI` on `/World/ProbeBall` and `/World/ProbeBox`, with the actual `CollisionAPI` collision shapes under `/World/ProbeBall/geometry/mesh` and `/World/ProbeBox/geometry/mesh`. Both backends nevertheless returned zero regular and full native contact headers, through callbacks and per-step polling. No error from native API was logged.

**OFF/ON causal read-only check:** CPU 180/180 and CUDA 180/180 measured ball z/vz samples and final box position were *exactly equal* within each backend (2/2 pairs). No cross-backend exact-physics equivalence is assumed. CUDA positive ON raw JSON SHA256 matches the prior v118 positive ON raw JSON exactly; that independently verifies repeatability of the previously negative native report experiment.

The independent analyzer SHA256-verified all six write-once raw files, checked every chronological 0.005 s frame, positive and separated dynamics, case order, and both parity pairs. It explicitly refuses to infer absent self-collision from missing reports and confirms **no native actor/collider witness**. The true experimental negative is that changing the runtime from CUDA to CPU **alone** does not restore native contact reporting in this *specific probe*. Broader PhysX/backend incapacity is *not* established.

## Scientific implications and negative control

Earlier v54 left wrist / left hip equal-and-opposite net contact forces, and newer G1 Tensor pair zeros, are **still insufficient** to prove a direct physical self-contact pair or its absence. This v119 backend study does not give the missing actor/collider paths, point, normal, and impulse. Hence the new KineticChain-29 safety admission remains **FAIL CLOSED**, and no unverified learned 29DoF reference or torque target is sent to the robot. No future human professional forceplate or boot-ball impact ground truth is fabricated. Existing K1-only CMU marker labels cannot establish elite instep dynamics.

The next genuinely necessary native-positive path is an isolated USD contact report *with authenticated rigid-body and collider shapes, correct authoring phase, simulation event delivery and native per-step buffer*; then freeze OFF/ON on the original G1 with joint-state hash parity, and only then probe wrist–hip. Keep the robot-side distinction between collision contact pairs and filtered net rigid-body forces.

This study builds on, but does not replace, v118's **8 original G1 real-shooting episodes** (4/8 inclusive strict pass, 3/6 independent strict pass), actual 200Hz 29DoF and ball-goal-plane negative evidence. No test-seen failed ball position was retuned or reused as a held-out performance claim.

## Reproduction and source location

Frozen executable matrix: `configs/v119_native_backend_frozen.json`; launcher: `tools/v119_native_backend_batch.py`; unchanged existing physical probe: `tools/v118_native_full_probe.py`; independent re-verification: `tools/v119_independent_backend_audit.py`; measured figure generator: `tools/v119_plot_native_backend.py`; targeted tests in `tests/test_v119_*.py`. Run the original simulator only with Tang authorized GPU, disk and lock preflight. All raw data are unique and write-once; never overwrite to rerun.

Full-suite exact count and final branch SHA are reported after the final GitHub commit and independent remote check.

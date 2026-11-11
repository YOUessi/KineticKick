# v0.1.86 — FROZEN positive-control qualification (before first GPU run)

**Source:** v0.1.85 `3822151c8b86850bce2ca40daef3b56c932c5e4e`. New clean `version/v0.1.86` worktree. Commit this protocol, config, probe, batch and tests **before** physics.

## Primary falsifiable question
Can the installed Isaac Sim/PhysX report a known two-body impact with (a) BOTH actor paths, (b) BOTH collider paths, (c) world contact point, (d) unit-ish finite normal, (e) positive contact impulse? Prior original free-root G1 v85 reports were blind even during independently observed foot-ball impacts. Equal opposite link net forces are *not* contact-manifold truth.

## Four preregistered isolated two-body conditions (serial)
1. `cpu_wildcard`: fresh dynamic sphere onto fresh static cuboid, PhysxContactReportAPI on sphere **before** physics reset, reportPairs relationship **not authored**, threshold=0.
2. `gpu_wildcard`: otherwise identical, CUDA dynamics.
3. `cpu_authored_empty`: deliberately author empty reportPairs before reset (semantics/negative control).
4. `gpu_authored_empty`: deliberately author empty reportPairs before reset (semantics/negative control).

Physics dt = 0.005s (200Hz), 180 steps, ball radius 0.2m, mass 1kg, z=0.85m, flat floor top z=0m and nominal -9.81 m/s² gravity. With no collision/contact stream, this does **not** certify noncollision: possible physics setup, USD reporting registration, lifecycle or DirectGPU readback failures. Empirically compare wildcard and authored-empty; official documentation versions disagree on empty reportPairs meaning.

## Execution and stop rules
- Use frozen source commit SHA, all files byte-equal HEAD, git tracked clean, global `fcntl` GPU lock, **GPU idle** before each child, and ≥20GiB available.
- Every child writes unique raw JSON/log/audit using exclusive-create, has 185s watchdog that only kills its **own process group**; never interrupt other experiments. If a run fails to initialize, times out, does not close, or produces invalid 180-step evidence, stop the matrix. Preserve failure logs.
- Do not modify original G1 USD, controller, 120Hz CMU C3D, checkpoints, evidence traces or existing v85 output. No original G1 football experiment is scheduled in this matrix.
- Native positive witness: both actor and collider paths match expected actors in either order; finite position and normal; impulse vector norm>0.0001. Callback and poststep poll both recorded; raw response retained only locally.
- A **positive isolated** result permits a *future* original G1 readback lifecycle experiment but does **not** yet prove the original CUDA G1 contact stream usable; a **negative isolated** result blocks direct-wrist-hip absence claims and G1 collision barrier activation.
- At end, independent empirical source/provenance and tests; output strict real G1 shot count **0**, no fake learning success. All human data currently kinematic-only lacking verified professional instep GRF; no policy training or privileged test tuning allowed.

## Companion CPU-only research work
Establish a typed fail-closed safety evidence contract; do not accept unknown geometry, stale contact-stream calibration, malformed full-29 state, or force proxies as a valid collision barrier. 200Hz v85 source evidence can be analyzed without modifying its files, and the CMU subject/session split remains unqualified for professional-impulse supervision.

## Sources
- NVIDIA Omni Physics Contact Reports: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physx/docs/dev_guide/contact_reports.html
- NVIDIA PhysX API population: https://nvidia-omniverse.github.io/PhysX/ovphysx/latest/population/PhysxContactReportAPI.html

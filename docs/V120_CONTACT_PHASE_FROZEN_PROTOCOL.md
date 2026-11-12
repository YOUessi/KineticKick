# KineticKick v0.1.120 — native contact-report lifecycle calibration (frozen before physics)

Parent: `version/v0.1.119`, SHA `b57b855433b3fa6f6f266ee565f7e959b333c288`. New untouched `version/v0.1.120` worktree; no existing branch reused. Date: 2026-11-12.

## Falsifiable hypothesis

The v119 report-zero result may originate from USD API authoring location, Isaac Lab spawner contact activation, pre/post reset subscription, or renderer/update event delivery, rather than an absence of genuine PhysX contact. To test these separately, execute EXACTLY the 7 matrix rows in `configs/v120_contact_frozen.json` on Tang, sequentially with the shared GPU lock, 180 commanded steps of 0.005 s, each in a separate Isaac process with write-once SHA256 data. Ball falling onto fixed box is the positive control; spatial separation is negative. One observer OFF/ON pair must have identical recorded ball position, velocity, and box final location.

The seven cases: same baseline positive OFF; root-API positive ON; spawner sensor ON; root+collider-API ON; after-reset subscription ON; rendered stepping ON; separated negative ON. CPU-vs-CUDA was already tested in v119 and is not repeated. All report ON variants use exactly the same physical ball and box except for expressly stated SDK contact instrumentation / event-loop interventions; **render=True might itself alter stepping and must not be silently represented as a no-perturbation observer**. Native witness requires actual two actor and collider paths and a finite nonzero contact impulse, position and normal from a PhysX native contact event. No Tensor-only reading substitutes.

Both observer ON and OFF mechanically valid but zero native headers -> still no native collider witness. Fail closed; do not equate net-force symmetry with a proven wrist–hip contact pair. V120 cannot certify G1 self-contact without isolated positive calibration and on-G1 read-only proof.

## Safety

Disk must stay ≥20 GiB; foreign GPU process and busy global `kinetickick_physics_gpu.lock` abort. Never kill unrelated jobs. Child-only 150 s watchdog. `reports/v120_native_*.json`, logs and audit manifests are create-exclusive and gitignored; no overwrite. Original data, checkpoints and historical PhysX are untouched. Frozen source compare to the committed SHA and require clean tracked working tree. No original G1 torque/position control changed.

Official implementation references: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physx/docs/dev_guide/contact_reports.html and https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physx/docs/api/python.html. Contacts in the native buffer are only valid for one simulation step; copy data during callbacks.

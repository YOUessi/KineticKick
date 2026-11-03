# Stage 0: Initial Natural-Motion Research Audit (2026-10-21)

## Research hypothesis, NOT an achieved result
Can explicit temporal phase and inter-segment coordination priors learned from skilled humans produce novel, human-like humanoid soccer motions with *better human-rated naturalness* than pose tracking/style-only baselines, when valid kicking and balance are held comparable?

## Executed and independently checked
- Tang environment: Ubuntu/Linux; RTX 4090 Laptop, nominal 16 GB GPU; installed Isaac Sim 4.5, Isaac Lab 2.1.0 lineage and Python 3.10 with CUDA-enabled PyTorch.
- BeyondMimic motion tracking source checked out at third_party/whole_body_tracking, commit 9db9158 (official repository archived).
- PAiD source and reference motions sparse-checked out at third_party/HumanoidSoccer, commit e72e470 (code is for Isaac Lab 2.1.1 and not yet installed/trained).
- Audited 13 PAiD reference clips: 10 standard / 3 stylized, 50 FPS, G1 29 joint DOFs and 30 rigid bodies.
- Isaac Lab built-in G1 robot spawned and stepped, reporting 37 joints and 44 bodies; that schema is NOT compatible with the 29/30 PAiD motion arrays. The simulator emitted success before external timeout during shutdown; do not call it a clean end-to-end simulator test.
- All five offline unit tests passed.
- Inventory: reports/reference_inventory.csv; exact reproducible command in repository README.

## Initial descriptive reference-motion results (not policy rollouts)
| Metric, per-clip median | Standard (n=10) | Stylized (n=3) |
| --- | ---: | ---: |
| Joint velocity RMS (rad/s) | 0.9001 | 1.3268 |
| Joint acceleration RMS (rad/s^2) | 12.5684 | 19.1016 |
| Joint jerk RMS (rad/s^3) | 271.8499 | 406.0226 |
| Position-to-velocity consistency MAE (rad/s) | 0.0881 | 0.1416 |

These groups differ in raw dynamic intensity. **These numbers do not establish a causal effect, superior naturalness, or a better policy.** Jerk is sensitive to FPS, filters, velocity scale, and choreography length. The 3 stylized clips are too few for robust group comparisons.

## What is missing
1. True **human** 3D motion capture and synchronized foot-ball contact labels: the NPZ clips are already retargeted to G1. They are not ground-truth human biomechanics.
2. Real policy rollouts and explicit per-body semantic maps. No claim about trained robot gait/beauty yet.
3. Blinded human ratings under matched functional success and matched visual presentation.
4. Isolated PAiD v2.1.1 environment. Do not modify current Isaac Lab v2.1.0 install.

## Next experiments — prioritized by evidence strength
1. Headless Isaac Lab smoke test -> rigid-body G1 simulation -> single-motion tracking rollout.
2. Record body names, joint names, events, ball state, ground-contact forces, timestamps and policy outputs.
3. Matched baselines: pose-tracking, style discriminators, phase/cross-segment coordination ablations.
4. Event-aligned biomechanical descriptors (support foot stabilization, hip-thigh-shank timing, coordinated counter-swing, follow-through) and independent blind paired naturalness ratings.
5. Only after failure-mode analysis, introduce morphology-aware, phase-conditioned cross-body coordination prior; test novel ball positions and held-out motion styles.

## Falsifiability and guardrails
- If coordination loss improves MPJPE but not blinded human naturalness, hypothesis H1 is not supported.
- If naturalness improves only by slowing the swing, it is not supported at comparable function.
- If comparison uses different robot models, camera, motion distribution, or success envelopes, it is confounded.
- If no observed phases/contact labels exist, do not fabricate impacts from a fixed frame.
- Avoid presenting experimental design as a paper already accepted or a trained controller as already finished.

# KineticKick v0.1.116 — recovered original G1 PhysX, frozen AG/AH/AI and phase-specific qdot falsification

**Executed on Tang, 2026-11-12.** This is an actual CUDA PhysX experiment, **not** a trained human-like KineticChain-29 demonstration. Pre-physics source/protocol committed and pushed at `c6b8085507349431fe749803d8877cf3166db489` from clean branch `version/v0.1.116` (parent `f822fce5e274cf802c2ed0c6bdced8884de63b35`). Raw evidence stays write-once, Git-ignored, on Tang. The experiment restores the **same v115 preregistered AG/AH/AI scenes/seeds** because v115 produced *zero original G1 physics frames* due to an absent local Git-ignored recovery checkpoint; no post-outcome ball/seed selection.

## Source integrity and the decisive operational repair

- Verified online RTX4090 with no other GPU compute process; nonblocking shared physical lock; 32–33GiB free (minimum 20GiB). No tracked source dirt, no pre-existing v116 branch or results. Older dirty worktrees and original datasets were not modified.
- Repaired the missing clean-worktree assets by **non-overwriting symlinks** to the existing original CMU/C3D/URDF/ONNX/recovery weights. The original v32 recovery checkpoint `experiments/post_shot_v032/ppo_s3201/last_model.pt` passed SHA256 `ee452b4ee4359727f5984f23628683d5c444bc2e3e7143ec512a5a1a3e96c326` for **all three frozen controller configs**. The new preflight explicitly rejects missing/mismatched recovery models before starting GPU. Original control policy and 200Hz physics, contact limits, model SHA and controller parameters are unchanged.
- The v116 frozen original-runner has 140-second child-only watchdog, serial global `flock`, fail-closed first-invalid logic, and write-once files. One observer ON repeat of the same AG/v32 physics is not counted as an independent seed/control case.

## D. Ten **new** real 29DoF G1 physical executions, 28,000 observed 200Hz steps

All ten rollouts finished real 14s (2,800×0.005s), passed startup/trajectory certification and had 100% *reported* instep impulse share, entered the goal, recorded ball speed proxy above the frozen 3.5m/s minimum, no measured nonfoot net-contact gate event, and observed recovery. **Six failed the true original 29-joint speed limit**, despite goal entry, valid instep contact and standing recovery. The baseline and weak CMU references were not tuned.

| Prespecified scene / seed | Frozen original controller | Instep share | Ball speed proxy | Peak actual max(|qdot|/source limit) | All old hard gates |
|---|---|---:|---:|---:|---|
| AG / 2731 | v32 OFF (+identical ON repeat) | 100% | 5.3345 m/s | **1.01496** | FAIL qdot |
| AG / 2731 | CMU10_02 OFF | 100% | 5.3482 m/s | **1.01496** | FAIL qdot |
| AG / 2731 | CMU10_05 OFF | 100% | 5.3348 m/s | **1.36636** | FAIL qdot |
| AH / 2777 | v32 OFF | 100% | 4.9123 m/s | **1.18047** | FAIL qdot |
| AH / 2777 | CMU10_02 OFF | 100% | 4.9029 m/s | 0.90852 | PASS |
| AH / 2777 | CMU10_05 OFF | 100% | 4.8972 m/s | **1.17485** | FAIL qdot |
| AI / 2819 | v32 OFF | 100% | 4.7012 m/s | 0.90865 | PASS |
| AI / 2819 | CMU10_02 OFF | 100% | 4.7109 m/s | 0.92164 | PASS |
| AI / 2819 | CMU10_05 OFF | 100% | 4.7126 m/s | 0.91866 | PASS |

**Strict scoring:** 4/10 actual repeat-inclusive executions, **4/9 statistically independent scene×controller cells**. AG/v32 OFF and ON are a paired instrumentation control, *not* extra independent success trials. These four successes are from **old original controllers** and minimum original task gates, not a new human-trained generator, native collision-manifold clearance, elite ball speed or human biomechanical force transfer.

## A. Contact observability and no-perturbation

The one preregistered AG/v32 OFF/ON pair has **2,800/2,800 identical actual 200Hz frames** and exact match across recorded physical states, shot records, contacts, nonfoot, termination and torque evidence. The maximum observed difference is **0**; this confirms read-only observation parity in the recorded fields for that pair. It does **not** certify that all unobserved states match, or prove any native wrist–hip actor/collider pair. The previously observed opposed wrist/hip forces are only body-level evidence. The actor0/actor1, collider0/collider1, true contact point/normal/impulse manifold requirement remains **CLOSED**. No arbitrary native-contact-zero-to-no-collision claim was made.

## B. Independent original 29DoF/200Hz causality and full-body mechanics

Original raw G1 JSON SHA256 is verified against the immutable batch master for each episode; real 200Hz timing, complete 29-joint order and original per-joint qdot limits are checked; terminal duplicate event markers are excluded only if physical content is identical. **10/10 have 14s** and full observed precontact 0.4s/postcontact 1.25s; none require fabricated recovery rows. Independent estimates of positive/negative motor work use *simulated estimated joint torque × actual qdot*, not biological muscle power; observed dual-foot support forces are *proxies*, not force-plate GRF/CoP.

- **AG:** first original joint speed violation at **1.390s**; first real CMU-v32 29DoF measurable difference at **1.405s** for both comparisons. Two paired contrasts prove this specific *pre-existing* 1.390s hazard cannot be fixed retroactively by a reference that first acts at 1.405s. CMU05 exacerbates AG peak qdot ratio to **1.36636**.
- **AH:** v32 and CMU05 show first original qdot hazard at **3.190s**, while CMU02 has no observed qdot violation and is the sole AH strict PASS. This is a valid **single-scenario favorable control outcome**, not a statistically demonstrated improvement in natural movement or robust safety across all three scenes.
- **AI:** three controls satisfy the original qdot limits. All six measured CMU-v32 pairs first differ at **1.405s**; full pre-impact RMS joint-speed differences are small and inconsistent. This is not a long-term learned approach and cannot establish causal human power transfer.
- **AG/v32, actual final 0.4s before impact:** left/right support-force proxy impulses ~**94.52/45.64 Ns**; right-leg estimated positive motor work ~**74.897J**, left-arm ~**4.436J**. AG/CMU02 right-leg positive work ~**74.908J**; this ~0.011J contrast does not demonstrate enhanced full-body drive.

Local-only generated raw source-audited figures: `reports/v116_AG_actual_29qdot.png`, `v116_AH_actual_29qdot.png`, `v116_AI_actual_29qdot.png`. No raw 30MB JSON, NPZ, media or weights are pushed to GitHub.

## C. Human training-data qualification and method boundary

Executed a new read-only actual ezc3d audit: **6 unique C3Ds, 2 subjects, 3,358 real 120Hz marker frames, zero native analog channels**. Marker streams have some missing/nonfinite positions; source quality is not assumed perfect. These are usable for kinematic phase hypotheses only; **0 professional-impact/GRF/CoP-supervised trials** and **0 verified real elite instep impulses**. [GroundLink](https://csr.bu.edu/groundlink/) and [SoccerKicks](https://github.com/nayariml/SoccerKicks) are candidates for limited biomechanical or visual priors, **not qualified synchronized elite kicking impulse truth**. Full-body generator **not trained or actuated** this batch. See `docs/V116_HUMAN_FORCE_AND_KINETICCHAIN29.md` for a seven-phase support–hip/trunk–leg–arm–landing hypothesis, eligibility and falsification/ablation contract.

## E. Repeatability and next dependent gate

Tracked deliverables: pre-physics SHA-locked original batch, source/checkpoint hash preflight and tests; independent 29DoF phase-work/contact-window and first-qdot-hazard comparisons; read-only six-trial CMU source qualification; measured original qdot plotting; full-body temporal research protocol and this report. All generated raw `reports/v116_*.json`, audit JSON, PNG and logs remain write-once local evidence. Original source logs and the v115 startup-negative result are preserved. **A strong result is the restored physics pipeline and true qdot safety falsification; not a trained professional kick.**

Next: native actor/collider positive/separated-negative contact truth from the installed PhysX physics scene, then original G1 wrist–hip read-only OFF/ON; obtain multi-subject real synchronized athlete instep + GRF/CoP/ball impulse sources; pre-freeze an unseen physical evaluation split; train and separately certify a time-coherent whole-body policy with 29DoF q/qdot/effort, support/slip, self-collision, speed, legality and 14s recovery. Never fit the new policy on AG/AH/AI measured failures.

## Final software and physical-source validation

- v116 targeted tests: **33 passed**, including source-asset SHA preflight, exact no-perturbation parity, independent physical trajectory guards, C3D native-channel negative qualification and SHA mismatch rejection.
- Full repository regression on Tang IsaacLab environment: **1,376 passed; 2 skipped; 69 subtests passed; 0 failed (35.98s)**. Original local logs: `reports/v116_full_pytest.log`; all raw PhysX data and diagnostic figures kept ignored/local-only.
- Final GitHub source SHA verified separately after final report/code commit. Frozen *pre-physics* protocol SHA remains `c6b8085507349431fe749803d8877cf3166db489`; its history is unchanged, and no post-result controller tuning was permitted.

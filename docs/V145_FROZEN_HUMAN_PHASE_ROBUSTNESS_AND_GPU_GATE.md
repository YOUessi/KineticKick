# v0.1.145 PRE-REGISTERED Human Phase-Order Robustness and Original PhysX Blocker
Date: 2026-11-13. Parent SHA `51c99a9a72816a981c47cc046f38ffebaf4c0809`. This preimplementation freeze follows v144's observed 2/2 right-side pelvis→thigh→shank *unsigned marker orientation rate* peak ordering. It is an explicit **post hoc sensitivity/falsification analysis**, not an independent hypothesis confirmation.

## Frozen, before analysis
Verify original 6 C3D/AMC source SHA and original v144 protocol. Only v050-prequalified weak uncalibrated ball marker launch anchors, 10_02 and 10_05, both subject 10. Preserve all six clips as original data denominator. At 120 Hz compute un-interpolated anatomically named 3D vector orientation-rate proxies. Sweep angle halfspan [1,2,3,4] frames; pre-onset windows [0.4,0.6,0.8] seconds; weak anchor shift [-12,-6,0,+6,+12] frames; rate median smoothing lengths [1,5,9] frames, masking invalid values (no fill across missing). Each event has 4×3×5×3=180 settings, across 2 events and 2 chains = 720 evaluations; no settings dropped from denominator.

Compute right pelvis/right thigh/right shank strict temporal peak ordering and >=2-frame adjacent margins; repeat left pelvis/left thigh/left shank as anatomical *specificity negative control*. For original v144 setting (span=2,window=.6,shift=0,median=1), right order should match source report. Missing values/invalid windows are separately counted and do not count as successes. Save lead differences in raw 120 Hz frames; do not rescale to claim impact timing.

Independent audit must reconstruct angular rates using atan2(cross,dot), compare standard setting and both-chain evidence and ensure source-level anchors and original SHA match. Recheck the frozen 12 original v53 physical rolls and v54 6,628 real states; NO GPU PhysX execution while original v120 pytest occupies GPU. Do not remove/kill/modify protected processes or source data.

## Interpretation policy
Grid fractions describe *analysis-choice sensitivity*, not statistical prevalence or p-values. Both strong anchors are one subject, weak launch timing (not true contact). Right-vs-left comparison is a specificity control, NOT a kinetic causality test. Any failure of ordering under reasonable smoothness/anchor variation weakens v144's 2/2 timing claim. Even robust ordering cannot certify elite biomechanics, hip energy transfer, legal instep, GRF, or dynamic power. All adverse and unresolved results preserved.

## Gate
Code, sampled plots, provenance and tests + report in new branch; write new local artifacts in append-only ignored reports and do not push raw C3D/AMC/JSON/PNG. Original G1 strict PhysX success remains 0/4 each v32, CMU10_02, CMU10_05; new shots 0 while GPU held. The native left wrist/hip manifold and true 200Hz post-failure recovery remain unknown.

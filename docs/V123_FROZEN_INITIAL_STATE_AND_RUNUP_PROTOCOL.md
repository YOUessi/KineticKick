# v0.1.123 | Frozen initial-state and run-up observability protocol

Date 2026-11-12. Parent v122 SHA: 5b1cddf6d090b52aff4dc0e21f63f7a3341a29f7. Pre-register **before** cohort-wide calculation. CPU-only because pre-existing v120 pytest holds GPU; do not kill or contend.

## Hypothesis and limits
v122 estimated robot motor-power onsets of 0.020–0.045 s across 24 frozen G1 conditions. A moving first actual 200Hz sample means natural run-up *initiation* is not observable; it does not mean the robot cannot move toward the ball, nor that it lacks human-like motion. No biological causal claims.

## Frozen cohort, provenance, non-independence
Only v116, v117, v118 original G1 200Hz traces: 29 actual runs (24 distinct scene-controller cells, 5 OFF/ON repeats). Verify exact source Git HEAD and master SHA against v122 source manifest and each raw JSON SHA against batch master. No imputation; no new physical runs. Observer repeats excluded from independent denominator; compare matched controls on identical ball/seed.

## Measures
Every run: measured first-frame XY root speed, root tilt, absolute maximum 29qdot and max original qdot limit ratio, bilateral >50N support proxies, precontact planar root path length and chord, pre-contact COM path chord, start-to-contact sampled time coverage, first sustained >50N support (3 true consecutive 5ms frames) with boundary censor flag, first actual 29DoF qdot violation time and joint, post-contact 1.25s observed/censored, optional 14s witnessed. First 0.2s fraction of frames with both support proxies below threshold and with both >= threshold. Frozen illustrative 'rest-like first frame' = root XY speed <= 0.05 m/s AND every abs 29qdot <= 0.1 rad/s. This sentinel is **not a human-quality metric**. If foot-force proxy zero, cannot infer no mechanical contact.

Same-scene/seed original v32 vs CMU02 and CMU05: earliest measured 200Hz row where max absolute 29-joint q difference OR 29-joint qdot difference > 1e-5 (numerical significance only, not controller efficacy); report common-prefix frames and timing relative to impact. For OFF/ON duplicate control pair, require identical joint q/qdot within threshold. No threshold selection after seeing cohort results.

## Falsification and tests
Synthetic tests: first-frame already moving; quiet root yet a single moving joint; all29 rest-like; support onset 3-frame requirement and first-sample censor; qdot violation first row; 200Hz time mismatch; shuffled/duplicate controls; mismatched source hash; write-once output. All source metadata immutable. Failure stays in evidence. Intended CPU-only script+unit tests+report, raw and large media local-only. No native PhysX manifold, collision-free safety, human elite data, or learned whole-body policy claims.
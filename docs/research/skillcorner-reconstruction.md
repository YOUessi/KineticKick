# SkillCorner professional-match reconstructed3D — v0.1.30

This is an executable **continuous professional-match shot source**, with an explicit MIT data-use basis and a working 309-frame adapter. It closes part of the participant/action-provenance gap. It does **not** establish instep technique, maximum effort, precise impact, optimal form or independently measured 3D accuracy, and does **not** complete P2.

The implementation is separate from the CMU physical-marker pipeline. Its inputs are original published vendor files containing **broadcast-video model estimates**, not raw optical motion capture. No CMU algorithm/output meaning, robot reference, policy or checkpoint is changed.

## Source and acquisition

The [official dataset card](https://huggingface.co/datasets/SkillCorner/opendata-bodypose/blob/a62e1ec1e3b82952042a7f7fa9dafd77c0e0822a/README.md) explicitly licenses the body-pose data under MIT, matching the [official repository license](https://github.com/SkillCorner/opendata/blob/4340d274572876239c154c90bc507a9b3250a656/LICENSE). Keep the copyright/permission notice and credit SkillCorner. This does not grant rights to underlying broadcast video; no video was downloaded.

The acquired [official sample](https://raw.githubusercontent.com/SkillCorner/opendata/4340d274572876239c154c90bc507a9b3250a656/data/bodypose/sample_1925299_phase406.jsonl.gz) is 1,926,895 bytes, SHA-256 `eb10f537c5c3b3c80018d5670f16cb788b14a3529d0be0b36a077bd1b41aeb4a`. It contains a 12.32-second interval from Brisbane Roar–Perth Glory on 2024-12-21. The eight acquired files total **7,145,321 bytes**: sample, full small event CSV, phases CSV, match JSON, manifest, body-pose README, license, and pinned Hugging Face card. They stay ignored under `data/research_human_kicks/skillcorner_probe_20261107/`; `acquisition_manifest.json` preserves URL, byte count, revision and SHA-256 for every file. The adapter verifies **tracked hash pins**, not only the mutable local manifest.

Revisions are GitHub `4340d274572876239c154c90bc507a9b3250a656` and Hugging Face `a62e1ec1e3b82952042a7f7fa9dafd77c0e0822a`. The two full-match pose archives are 623,408,231 and 644,327,811 bytes; neither was downloaded during the initial sample stage. The later one-match acquisition is documented below. No login, author message, payment or GPU was used.

## What is verified separately

| Dimension | Evidence and status |
|---|---|
| Professional participation | Jacob Brazete, Brisbane number 18, appears in the official [Perth Glory match report](https://perthglory.com.au/news/alm-match-wrap-williams-goal-seals-vital-victory-over-roar/) as a 67-minute substitute in this A-League Men match. Vendor match metadata maps him to `player_id=965698`. This verifies participation; the framewise identity assignment remains the vendor's estimate. |
| Continuous motion | Source pose frames **124487–124795**, 309 consecutive frames at 25 Hz, match timestamps **77:34.48–77:46.80**, period 2. All 29 shooter landmarks have finite coordinates, and shooter/ball detection flags are true on all 309 frames. |
| Shot occurrence | Official [event CSV](https://github.com/SkillCorner/opendata/blob/4340d274572876239c154c90bc507a9b3250a656/data/matches/1925299/1925299_dynamic_events.csv) row `8_871`: `player_possession`, player `965698`, `end_type=shot`, `is_header=False`, `fully_extrapolated=False`, start/end matched flags true. This is vendor event evidence, not an independent video adjudication. |
| Shot time anchor | 10 Hz `frame_end=49907`, match **77:45.7**. The official conversion gives pose frame **124767.5**, bracketed by frames 124767/124768 at 77:45.68/77:45.72. Event-grid resolution is 0.1 s; it is not a verified impact-time error bound. |
| Instep, kick side, force/intensity, optimal form | **Unknown.** A non-header shot is not enough to infer instep or maximum effort. |
| Actual contact, support force, ground elevation | **Unavailable.** No force, collision, reliable common world-z frame or foot-surface geometry is supplied. |
| Independent accuracy / body shape | No independent ground-truth pose, participant shape/mesh, inertia, calibrated foot geometry or anatomical joint rotation is acquired. |

Three observed confounders are retained. The sample filename says `phase406`, but the current official events/phases re-export renumbered this interval to **phase397**, frames 49793–49919. Always join by match, period and frame/time. The old pose sample's `possession.player_id` never names Brazete (238 frames are null), despite the newer event annotation; possession is not an alternative contact/identity truth. Vendor match metadata says Kayo Stadium, whereas both clubs' reports identify Suncorp Stadium; the raw metadata is preserved unchanged.

The [repository README](https://github.com/SkillCorner/opendata/blob/4340d274572876239c154c90bc507a9b3250a656/README.md) reports approximately 97% player-identity accuracy overall. That is not a per-frame confidence score for this player or an independent validation of this clip.

## Coordinate and sampling semantics

There are 29 named estimated joints: head/eyes/ears, neck, shoulders/elbows/wrists/fingers, midHip/hips/knees/ankles/heels/toes. The [official body-pose README](https://github.com/SkillCorner/opendata/blob/4340d274572876239c154c90bc507a9b3250a656/data/bodypose/README.md) specifies metres, pitch-centred x/y, and z reliable relative to the player's centroid rather than to the pitch. Negative z is allowed. We subtract estimated `midHip` and retain the source axes; this is **not an anatomical pelvis rotation frame** and creates no floor.

The 2D player x/y fields are native 10 Hz tracking resampled by the publisher to 25 Hz. Every fifth pose frame aligns with an even tracking frame. A `True` detection flag does not make the interpolated frames native observations. In this sample, all three ball-coordinate axes are also exactly piecewise linear on the 10-to-25 Hz grid within floating-point precision. Its z is published only with body-pose data, but has no per-point error, calibrated centre or collision annotation. Upstream joint smoothing/interpolation is unspecified. This adapter performs **zero new interpolation or smoothing** and computes no derivative or contact events.

## Output contract

[`tools/build_skillcorner_coordination.py`](../../tools/build_skillcorner_coordination.py) reads the fixed schema and eight source pins in [`configs/skillcorner_coordination_v1.json`](../../configs/skillcorner_coordination_v1.json). It writes NPZ loadable with `allow_pickle=False`, JSON metadata and a hash manifest under ignored `data/research_human_kicks/skillcorner_coordination_v1/`.

| Array group | Meaning |
|---|---|
| `source_frame_numbers`, `match_time_s`, `clip_time_s`, `period` | Native pose index and checked times; seconds, no half-time/frame-zero assumption. |
| `joints_reconstructed_m[T,29,3]`, `joint_predicted_p90_cm/m[T,29]` | Vendor coordinates preserved; predicted error retained in cm and explicitly divided by 100 to metres. |
| `joint_present`, `joint_geometry_valid`, `joint_error_valid`, `joint_quality_pass` | Separate presence, finite geometry, finite/nonnegative error, and quality screen. Missing values remain NaN. |
| `player_is_detected`, `player_detection_flag_available` | Upstream player-level screen detection and whether the flag exists; not joint visibility. |
| `pipeline_joint_interpolated` | False everywhere, describes only this adapter. |
| `player_xy_resampled_m`, `tracking_xy_resampled_10_to_25hz`, `exact_even_tracking_frame_alignment` | Explicit 10-to-25 Hz series-resampling status and aligned source anchors. |
| `ball_reconstructed_xyz_m`, `ball_geometry_valid`, `ball_is_detected`, flag availability | Unchanged estimates plus masks; do not infer calibrated ball centre or native 25 Hz accuracy. |
| `possession_player_id_available`, `possession_is_shooter` | Keep missing/other possession separate from vendor shot identity. |
| `joints_pelvis_relative_m`, corresponding validity/error budget | Translation relative to same-frame midHip; source axes retained. |
| `joints_normalized`, `length_scale_m`, scale eligibility/error and normalized validity | Median of eligible mean bilateral estimated hip-knee-ankle chains; a sample normalizer, not measured anthropometry. |
| 13 `segment_*` arrays | Trunk, pelvis/shoulder lines, bilateral upper arms/forearms/thighs/shanks/feet. Directed vectors, lengths in m, unit directions, availability, errors and quality. |
| 14 `relation_*` arrays | Directed-segment cosines across body parts, geometry availability, error budgets and quality; these are not anatomical angle truth. |

JSON separates `professional_participation`, `shot_event`, `non_header`, `instep`, `kick_strength`, `precise_contact`, `independent_metric_ground_truth` and `p2_complete`. It retains the original event fields and fractional frame/time association. Contact time, kick side, instep, force/intensity and support truth remain null.

## Error propagation and actual quality

Vendor `p90_mae_cm` is a **predicted** per-joint accuracy measure. The card describes p90 MAE while the README describes an error radius; neither supplies independent calibration or simultaneous coverage. For sensitivity analysis only, the adapter treats each value as a radial budget. It never multiplies probabilities or presents the resulting values as joint 90% confidence intervals.

Relative-position and segment-vector budgets add their endpoint budgets; the identical midHip variable cancels to exactly zero when subtracted from itself. For vector length L and budget e<L, the conditional unit-direction budget is `2*sin(asin(e/L)/2)`; if e≥L, arbitrary direction is possible and the budget is 2. The cosine budget is capped at 2 after adding its direction budgets. Normalizer uncertainty is propagated through median lower/upper chain envelopes with eligibility held fixed; normalized-position budgets are available only when the normalizer exceeds its budget. These are conditional geometric sensitivity calculations, not measured error distributions.

The provisional screen requires finite detected geometry, valid predicted errors ≤0.20 m, and relation cosine sensitivity budget ≤0.75. These are explicit exploratory thresholds, not externally validated acceptance criteria. Geometry is retained even when the screen rejects it.

| Actual sample check | Result |
|---|---|
| Geometry present | 8,961 / 8,961 shooter joint-frames; 29 × 309. |
| Joint screen | 6,922 / 8,961 pass the 20 cm screen. |
| Whole-clip predicted p90 | Median 14.19 cm; 95th percentile 31.64 cm; max 43.01 cm. |
| Shot ±1 s (50 frames) predicted p90 | Median 16.62 cm; 95th percentile 35.05 cm; max 43.01 cm. Ankle medians ≈19–20 cm, toe medians ≈22–26 cm. These are distributions of predicted errors, not independently measured residuals. |
| Relationship geometry / quality | 4,326 / 4,326 geometrically available; only **303 / 4,326** pass the screen. |
| Whole-clip passing relationships | Trunk–left-thigh 51 frames; trunk–right-thigh 35; trunk–shoulder-line 217. All other relationship groups have zero passing frames. |
| Shot ±1 s relationships | Only **13 trunk–shoulder-line frames** pass; all leg chains, foot/shank, shoulder/pelvis and cross-arm/leg relationships fail this screen. |
| Sample length normalizer | 0.787636 m from 204 eligible frames; propagated sensitivity budget 0.455368 m. Its large uncertainty must remain visible. |

Thus the file supports a provenance-preserving parsing/quality prototype and a small set of coarse torso features. It is **not yet a usable whole-body shooting coordination prior** under this screen. Do not drop the masks, loosen thresholds merely to make the sample pass, or infer knee/foot/contact precision from complete coverage.

## Reproduce and verify

```bash
/home/you/conda_env/env_isaaclab/bin/python -m unittest discover \
  -s tests -p test_skillcorner_coordination.py
/home/you/conda_env/env_isaaclab/bin/python tools/build_skillcorner_coordination.py
```

Eight targeted tests pass: metre/cm conversion and frame semantics, missing/error/detection separation, fractional event timing and unavailable impact truth, malformed frame/id/unit rejection, bounded-perturbation sensitivity coverage, rigid transform/scale behavior, source hash failure, and the real 309-frame NPZ roundtrip. Tests initially failed because the builder did not exist; one subsequent assertion-message case mismatch was corrected without changing behavior. The full Tang-local suite was not run.

The main executable route is this pinned SkillCorner adapter. A later **separately bounded** acquisition could scan non-header shot windows in the two full-match archives for lower extremity errors, preserving these same masks and event/identity checks. That would be a search for better reconstructed sequences, not a guarantee of instep labels or impact truth. The bounded one-match follow-up below now performs this scan; no model training has been performed.

## Bounded full-match follow-up: no qualifying coupled-motion candidate

A subsequent explicitly authorized experiment acquired **only match 1925299**, from the [same pinned official archive](https://huggingface.co/datasets/SkillCorner/opendata-bodypose/resolve/a62e1ec1e3b82952042a7f7fa9dafd77c0e0822a/raw/1925299.jsonl.zip). Preflight found 41.35 GB free, HTTP Content-Length **623,408,231 bytes**, and an unchanged MIT card hash. The archive hash is `0415665098d594fb5681815000f7a6312037788a3237910a9f41bb8de5c336a0`, matching the official manifest. The second match was **not acquired**.

[`tools/scan_skillcorner_shots.py`](../../tools/scan_skillcorner_shots.py) streams the ZIP member `1925299.jsonl`: **153,252 frames / 3,345,254,058 uncompressed bytes**, without writing the full JSONL to disk. It ignores the ancillary `__MACOSX` member. Each stored gzip window contains only the identified shooter plus the original shared frame/ball fields. Every retained numeric value remains an upstream estimate, and absent shooter joints remain missing.

The official event table has **30 shot-ending possessions: 24 non-headers and 6 headers**. The 24 non-header windows cover ±1.5 s of the vendor event anchor, using 75 or 76 frame records according to its integer/half-integer pose index. All **1,817 planned records** were retained, all 24 windows parsed, and 16 windows have full joint geometry coverage. The six header events (`8_67`, `8_270`, `8_688`, `8_790`, `8_888`, `8_1009`) are listed with their exclusion reason in the plan/report. Non-header `8_308` is also excluded from reliable candidates because the vendor marks it fully extrapolated and its possession end unmatched; it has no shooter pose. Its raw window and failed-quality summary are retained rather than omitted.

The **0.20 m joint-error and 0.75 cosine-budget thresholds are unchanged**. Ranking uses four coarse groups: shoulder–pelvis; either trunk–thigh; either thigh–shank chain; and either upper-arm–opposite-thigh relation. It first considers a continuous intersection of all four groups covering the two event-bracket frames, then their coverage near ±0.5 s, then geometry coverage and lower core-joint error. Core error summaries cover midHip/neck, bilateral shoulders/elbows/hips/knees/ankles. Earlier torso-only success cannot qualify the shot anchor.

**Negative result:** across all 24 windows, shoulder–pelvis, both leg chains, both arm chains, both cross-arm/leg pairs and foot/shank pairs have **zero quality-passing frames**. Therefore there is no simultaneous supported core-coordinate frame anywhere in these shot windows, and no qualifying continuous whole-body candidate. There are 439 passing trunk–left-thigh relation-frames, 390 trunk–right-thigh and 1,142 trunk–shoulder-line relation-frames. Those partial features do not fill the missing pelvis/leg/arm couplings. This is a negative result under the existing conservative sensitivity contract, **not proof that the physical actions or all vendor poses are intrinsically inaccurate**; the vendor errors are predicted, the propagation is conservative, and calibration remains unavailable. No thresholds were relaxed to make a candidate pass.

The best partial result by the predeclared ranking is `8_804`, Quinn MacNicol at 72:02.5: 74/76 frames with joints, core-joint predicted p90 median 8.32 cm / 95th percentile 16.48 cm, and 72/76 frames with a passing trunk–thigh relation. Its pelvis/leg-chain/cross-arm support remains zero. `8_746` (H. Van Der Saag) and `8_970` (J. O'Shea) have complete geometry and core-error 95th percentiles around 14.9 cm, yet still fail the required couplings. These are better **partial torso/leg geometry** options than the initial Brazete window, not a complete shooting prior.

| Rank | Event | Player | Event time | Frames | Joint coverage | Core predicted p90 p95, cm | Trunk–leg passing frames | All-core passing frames |
|---|---|---|---|---:|---:|---:|---:|---|
| 1 | `8_804` | Q. MacNicol | 72:02.5 (P2) | 76 | 97.4% | 16.48 | 72 | 0 |
| 2 | `8_628` | A. Lebib | 52:06.3 (P2) | 76 | 77.6% | 23.37 | 44 | 0 |
| 3 | `8_746` | H. Van Der Saag | 61:53.1 (P2) | 76 | 100.0% | 14.94 | 46 | 0 |
| 4 | `8_970` | J. O'Shea | 88:49.6 (P2) | 75 | 100.0% | 14.93 | 70 | 0 |
| 5 | `8_120` | A. Bugarija | 09:24.9 (P1) | 76 | 100.0% | 17.14 | 65 | 0 |
| 6 | `8_806` | Q. MacNicol | 72:04.7 (P2) | 76 | 100.0% | 19.41 | 70 | 0 |
| 7 | `8_227` | A. Faisal | 18:36.2 (P1) | 75 | 100.0% | 27.11 | 51 | 0 |
| 8 | `8_305` | A. Bugarija | 26:21.9 (P1) | 76 | 100.0% | 15.68 | 53 | 0 |
| 9 | `8_317` | K. Jelacic | 29:11.1 (P1) | 76 | 97.4% | 16.95 | 31 | 0 |
| 10 | `8_332` | H. Van Der Saag | 29:56.1 (P1) | 76 | 80.3% | 21.98 | 19 | 0 |
| 11 | `8_535` | A. Zimarino | 50:11.5 (P1) | 76 | 100.0% | 15.62 | 22 | 0 |
| 12 | `8_805` | K. Jelacic | 72:03.9 (P2) | 76 | 100.0% | 14.69 | 46 | 0 |
| 13 | `8_290` | A. Zimarino | 24:37.2 (P1) | 75 | 100.0% | 23.35 | 7 | 0 |
| 14 | `8_510` | A. Faisal | 46:59.6 (P1) | 75 | 100.0% | 29.27 | 3 | 0 |
| 15 | `8_979` | K. Ngo | 89:45.2 (P2) | 75 | 100.0% | 21.84 | 6 | 0 |
| 16 | `8_352` | A. Bugarija | 34:57.8 (P1) | 75 | 100.0% | 27.74 | 0 | 0 |
| 17 | `8_701` | A. Faisal | 59:29.5 (P2) | 76 | 100.0% | 29.21 | 3 | 0 |
| 18 | `8_636` | A. Zimarino | 52:57.3 (P2) | 76 | 100.0% | 29.27 | 5 | 0 |
| 19 | `8_699` | J. O'Shea | 58:33.7 (P2) | 76 | 100.0% | 29.27 | 0 | 0 |
| 20 | `8_973` | J. Brazete | 89:13.7 (P2) | 76 | 100.0% | 29.27 | 0 | 0 |
| 21 | `8_871` | J. Brazete | 77:45.7 (P2) | 76 | 97.4% | 24.11 | 0 | 0 |
| 22 | `8_961` | J. Brazete | 87:53.2 (P2) | 75 | 88.0% | 29.30 | 0 | 0 |
| 23 | `8_607` | J. O'Shea | 49:41.5 (P2) | 76 | 84.2% | 29.30 | 3 | 0 |
| 24 | `8_308` | D. Williams | 26:31.7 (P1) | 76 | 0.0% | — | 0 | 0; event fully extrapolated / end unmatched |

All events, players, source/event quality exclusions, mask counts, uncertainty distributions, continuous-run lengths and qualification unknowns are in ignored `data/research_human_kicks/skillcorner_match1925299_scan_v1/scan_report.json`. The directory preserves the original ZIP, acquisition manifest, predeclared shot plan, 24 gzip shooter windows and 24 NPZ/JSON coordination pairs, with hashes. Instep, kick side, maximal strength, exact impact and independent ground truth remain unknown for every window. P2 is still incomplete.

The entire experiment uses approximately **629.3 MB decimal**, including the original archive and all window/coordination/report files, below the hard **750,000,000-byte** bound. The scanner checks that bound before each output write. It makes no network requests, performs no training and uses no GPU. This one-match search is closed with a negative coupled-motion result; it does not trigger another download.

```bash
/home/you/conda_env/env_isaaclab/bin/python -m unittest discover \
  -s tests -p 'test_*skillcorner*.py'
/home/you/conda_env/env_isaaclab/bin/python tools/scan_skillcorner_shots.py
```

The 12 targeted SkillCorner tests pass (the original eight plus four scan tests): explicit header/event exclusions and 75/76-frame ranges; overlapping shooter windows streamed without extraction; complete geometry or earlier torso-only success failing to qualify the anchor; and total-byte-budget rejection before writing. No full Tang suite was run.

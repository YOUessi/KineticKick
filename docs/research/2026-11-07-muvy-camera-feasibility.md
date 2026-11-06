# MUVY camera feasibility: two rejected conditional estimates

Recorded on `version/v0.1.33`, 2026-11-07. **No metric camera or human 3D is accepted.** The existing cam_03/cam_04 recordings contain useful professional kicking video, but neither tested cam_04 camera model meets its reserved-point check. P2 remains incomplete. This note does not change the v32 release report or source qualifications.

All diagnostics are preserved locally under ignored `data/research_human_kicks/muvy_event08_probe_v1/diagnostics/camera_feasibility_v1/`. The 25 first-pass files were copied byte-for-byte from `/tmp`, including their original results, scripts, images and report; historical `/tmp` path strings are retained for provenance. `persistence_manifest.json` records their hashes. The two original videos and four locked manifest/qualification files remain unchanged.

## Existing-frame observations

The first pass used six decoded frames: cam_03 136/156/187 and cam_04 1139/1159/1190. Background-candidate displacement medians reached **29.01 px** and **37.56 px** over the kick window. Both views require a changing image mapping; local homography scale does not uniquely identify optical zoom. Image registration had small median held-out errors, but cam_04 also contained **425–852 px** erroneous matches on repeated advertising texture. All residuals were retained; image registration is not metric-camera or 3D validation.

Cam_04 exposes four goal-mouth corners and four candidate goal-area/penalty-area front corners. Cam_03 has weaker verified field-point coverage beyond the planar goal frame. The manual coordinates remain approximate. World coordinates use a nominal regulation-layout hypothesis, not a surveyed stadium: goal 7.32 × 2.44 m, goal-area offset 5.5 m and penalty-area offset 16.5 m. No body dimensions or learned pose prior supplies calibration points.

## One additional, preregistered condition

The parent authorized exactly one materially different CPU check. It retained the **same six fit points and two reserved penalty-area corners**, without changing coordinates. The model fixed principal point **(640, 360)** in the 1280×720 image, imposed **fx = fy**, zero skew and zero distortion, and estimated one focal length plus camera rotation/translation. These are camera assumptions, not measured facts.

Initialization used **f₀ = 1280 px**, chosen solely as image width, followed by one deterministic EPNP call on the six fit points. One `scipy.optimize.least_squares` TRF call minimized their unweighted pixel residuals. There was no focal sweep, multiple start, robust reweighting or retry. The numerical focal bounds were preregistered as 1–1,000,000 px. The fit was frozen to disk before numeric held-out coordinates were consumed for scoring. Both reserved points had to have error **≤3 px** to pass the conditional image check.

| Conditional model | Fit RMS | Reserved far-corner error | Reserved near-corner error | Result |
|---|---:|---:|---:|---|
| Original general DLT | 0.4067 px | 38.9749 px | 84.5919 px | Rejected; original failure preserved |
| Centered, square-pixel, zero-skew/distortion camera | 3.2820 px | 51.2745 px | 66.1639 px | **Rejected under the preregistered ≤3 px rule** |

The constrained optimizer reported numerical convergence (`ftol`, 485 reported function evaluations), but focal length reached **999,999.99999 px**, its numerical upper bound, with nominal translation depth about **29,354 m**. This is not a credible measured focal length or accepted pose. Numerical convergence does not validate the model. No further fit or label adjustment was attempted.

The two reserved points were excluded from each fit and the new initializer. They are reused across these two research conditions, so they are not a fresh blind test set for the overall adaptive investigation. Their world coordinates are also conditional on the same nominal field layout. Neither model supplies independent surveyed 3D accuracy. Current failures do not isolate whether approximate corner localization, layout/structure interpretation, camera assumptions or their combination dominates.

Exposure, crop/resize history, distortion and rolling shutter remain unknown. The existing coarse timing candidate (+33.466767 s, visually allowed anchor interval 33.333300–33.600233 s) remains unchanged and is not frame-accurate validation. No human reconstruction follows from these camera checks. The current annotation/layout/model chain needs independent geometric measurement and validation before further motion reconstruction; this bounded experiment is closed.

## Local evidence and verification

- First pass: [full report](../../data/research_human_kicks/muvy_event08_probe_v1/diagnostics/camera_feasibility_v1/first_pass_report.md), [registration diagnostics](../../data/research_human_kicks/muvy_event08_probe_v1/diagnostics/camera_feasibility_v1/feasibility_diagnostics.json), [original DLT result](../../data/research_human_kicks/muvy_event08_probe_v1/diagnostics/camera_feasibility_v1/conditional_cam04_fit.json).
- New condition: [preregistration](../../data/research_human_kicks/muvy_event08_probe_v1/diagnostics/camera_feasibility_v1/centered_camera_pre_registration.json), [six-point-only input](../../data/research_human_kicks/muvy_event08_probe_v1/diagnostics/camera_feasibility_v1/centered_camera_fit_only_input.json), [frozen optimization](../../data/research_human_kicks/muvy_event08_probe_v1/diagnostics/camera_feasibility_v1/centered_camera_optimization_frozen.json), [result](../../data/research_human_kicks/muvy_event08_probe_v1/diagnostics/camera_feasibility_v1/centered_camera_result.json), [held-out overlay](../../data/research_human_kicks/muvy_event08_probe_v1/diagnostics/camera_feasibility_v1/centered_camera_heldout_overlay.png).
- [Verification record](../../data/research_human_kicks/muvy_event08_probe_v1/diagnostics/camera_feasibility_v1/centered_camera_verification.json): all 25 historical copies match, all six protected source hashes match, six fit labels are identical, and only one optimizer call occurred. The new helper refuses a second attempt once its attempt record exists.

No network, new download, GPU, training, source-manifest mutation, old-adapter change or robot-source change occurred. Raw data and diagnostic media remain ignored. No controller or full-project test suite was run for this documentation/data diagnostic.

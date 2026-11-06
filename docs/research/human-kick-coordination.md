# Original-human marker coordination pipeline

Checked 2026-11-07 on `version/v0.1.29`. This pipeline now runs on **five acquired CMU optical soccer-kick captures**, producing actual numeric human-marker relations for a bounded P3 prototype. It does **not** close the skilled/professional instep evidence gap: every output has `sourcequalification=false`, with skill, kick subtype, kicking-side truth, impact time, ball center and ground force left unknown.

The source decision and restrictions remain in [human-kick-sources.md](human-kick-sources.md), with all URLs/hashes in [the source catalog](../../configs/human_kick_sources_v1.json). The [interface schema](../../configs/human_coordination_schema_v1.json), [builder](../../tools/build_human_kick_coordination.py) and [tests](../../tests/test_human_kick_coordination.py) are tracked; original data and generated arrays are ignored.

## Actual acquisition and reader validation

The [official CMU subject-10 index](http://mocap.cs.cmu.edu/search.php?subjectnumber=10) identifies soccer trials `10_01`, `10_02`, `10_03`, `10_05`, `10_06`. Their C3D and AMC files plus the shared `10.asf` are present. Walking `10_04` is excluded by the acquisition and by the builder CLI. The 11 original motion/skeleton files and four original source-page snapshots total **17,078,488 bytes**, recorded individually in `data/research_human_kicks/cmu_subject10/download_manifest.json`. The [CMU custom-use terms](http://mocap.cs.cmu.edu/) remain attached; data are not relabeled under a code license.

No independent C3D reader was installed in the existing environment. A **31,316-byte MIT-licensed [py-c3d 0.6.0 wheel](https://pypi.org/project/c3d/)** was therefore fetched into ignored `data/research_human_kicks/reader_validation/`, checked against its published SHA-256, and imported from that explicit path. No package was installed into the simulator or Python environment.

Independent point decoding agrees exactly with the bounded local reader on **all five clips**: every exported human/ball coordinate, every frame number and every validity mask; maximum coordinate difference **0 m**. This is a parser agreement check, not a certificate of measurement accuracy.

Four original files (`02/03/05/06`) have inconsistent **unused** analog metadata: zero analog channels, header analog samples/frame `1`, `ANALOG:RATE=600`, and point rate `120`. Stock py-c3d rejects that ratio. For the independent comparison only, a narrowly scoped subclass first receives this exact assertion, verifies both analog counts are zero, temporarily sets the reader's unused header ratio to `5`, and reruns all metadata checks. Its point-decoding code is unchanged; every original source byte is preserved. Trial `01` requires no normalization. This exception and the comparison are recorded in `reader_validation/crosscheck.py` and `crosscheck.json`; it is not reported as five successful unmodified generic-reader loads.

The production reader remains intentionally limited to Intel floating-point C3D with no analog samples. Unsupported layouts fail; it is not a general C3D implementation. Source zero-residual flags are retained separately because zero residual can indicate upstream modeled, filtered or filled samples. The exact upstream operation is unknown.

## Representation and validity contract

All landmarks are **surface-marker geometry proxies**. Shoulder, knee and ankle markers are not treated as anatomical joint centers; pelvis-side means are not hip joint centers. No anatomical joint angles or SMPL/body dynamics are reconstructed.

| Representation | Construction and dimensions |
|---|---|
| Source samples | Original `T × 41 × 3` positions in meters, source-valid mask and source zero-residual flags. |
| Optional gap handling | Internal gaps of at most two frames use time-linear interpolation only when valid observations bracket the gap. Raw arrays remain unchanged; new fills have separate flags. No edge extrapolation, long-gap fill or ball fill. |
| Twenty landmarks | Pelvis/thorax means; left/right pelvis sides, shoulders, elbows, wrist means, knees, ankles, heels, toes and MT5 markers. Each requires all declared contributors. |
| Pelvis frame | Origin is the four waist markers' mean. Left axis uses left-minus-right side means; forward uses front-minus-back, orthogonalized against left; third axis is their cross product. This is a right-handed marker frame, not a calibrated gravity/ground frame. Degenerate frames are unavailable. |
| Pelvis coordinates | Landmarks expressed in that moving marker frame. World and local coordinates retain independent validity. |
| Length scale | Median of both sides' valid pelvis-side–knee–ankle chain lengths across the source clip. A marker-chain scale, not measured anthropometric leg length. Missing scale invalidates normalized fields without deleting available unnormalized geometry. |
| Velocities | Central differences, one-sided at endpoints, only when the full stencil is valid. World velocities use m/s. Derivatives of normalized pelvis coordinates use 1/s and include pelvis-frame motion. Interpolation flags propagate across derivative stencils. |
| Thirteen segments | Trunk, pelvis line, shoulder line; bilateral upper arm, forearm, thigh, shank and heel–toe foot vectors/lengths/directions. |
| Fourteen relations | Dot products between valid unit segment directions, including bilateral, adjacent-leg, trunk–thigh, shoulder–pelvis and contralateral arm–thigh pairs. These cosines are not joint-angle truth or a learned professional-player prior. |

For each numeric feature, its corresponding mask in the schema defines availability. Invalid values remain NaN. `*_source_valid` excludes this pipeline's new interpolation, while `*_interpolated` identifies features affected by a new fill. Source-present does not promise that CMU's upstream processing never filled a gap; `source_zero_residual_flag` preserves that separate evidence.

## Event candidates, not impact labels

The JSON stores each supported candidate's method, sample time/frame, evidence and qualitative low confidence. Confidence is not a calibrated probability. The conservative rules are engineering heuristics; they have no manual phase-label validation here.

- **Approach:** pelvis translation exceeds 0.15 m/s for at least three source-valid frames before the toe-speed peak. It is not verified running or a measured approach phase.
- **Support:** contralateral heel/toe midpoint stays below 0.2 m/s near the peak for at least five source-valid frames. Low motion does not establish floor contact or force.
- **Backswing/forward swing:** an interior rearward extremum in pelvis-forward normalized toe coordinates, then sustained forward coordinate rate. Moving pelvis axes are a confounder.
- **Toe-speed peak:** separately retained, with a candidate kicking side. **Never used as contact truth.**
- **Ball motion:** a fixed two-marker subset is chosen by source-valid completeness. Its centroid must have a stationary initial baseline and sustained later speed/displacement. A changing marker set is never averaged into an artificial motion jump. This is a marker-centroid motion-onset candidate, not a ball center, collision or exact impact time. Missing observations expand the onset interval or remove the candidate.
- **Follow-through:** sustained toe-speed decay after the toe peak is a low-confidence recovery candidate, not a verified phase aligned to impact.
- **Strike/contact:** always `null` for this dataset. Unsupported phases also remain `null` with a reason.

All candidate calculations exclude new pipeline interpolation. The original capture clock aligns human and ball markers, but this does not resolve marker-to-ball-center calibration or physical contact geometry.

## Five actual outputs

Generated under ignored `data/research_human_kicks/coordination_v1/`, one NPZ and one JSON per clip plus `manifest.json`. The NPZ files total **10,803,533 bytes** and contain **2,761 frames**, 20 landmarks, 13 segment directions and 14 relation values per frame. These are five trials from the same source subject/session, not five independently recruited skilled players.

| Clip | Frames / original range | Marker-chain scale (m) | Source missing human samples | Toe-speed peak candidate (s) | Ball-marker motion candidate (s) |
|---|---|---:|---|---:|---:|
| `10_01` | 801 / 8–808 | 0.963399 | None | 4.975000 | 5.025000 |
| `10_02` | 590 / 8–597 | 0.966569 | None | 2.916667 | 2.941667 |
| `10_03` | 362 / 268–629 | 0.967304 | None | 1.133333 | 1.166667 |
| `10_05` | 436 / 287–722 | 0.968121 | LHEE at final source frame 722 | 2.483333 | 2.525000 |
| `10_06` | 572 / 19–590 | 0.969689 | LBHD at final source frame 590 | 3.416667 | unavailable |

Times are relative to each clip's first acquired sample, **not impact-normalized**. The two missing human samples are at boundaries and remain missing, so the default processing inserted **zero** new human samples in these real captures. Source zero-residual human samples number 74, 42, 53, 129 and 205 respectively and remain explicitly flagged. Ball tracks have further missing samples; they are never filled.

Every real clip satisfies the heuristic approach/support/backswing/forward-swing/recovery rules; this records rule availability only. All five retain `phases.strike=null` and `truth_labels.contact_time_s=null`. The fifth clip's absent ball-motion candidate is not replaced with the toe-speed peak.

JSON files contain original C3D hash, schema hash, builder hash, output hash, parameters, feature-validity fractions, normalization and the unknown qualification fields. The aggregate manifest has per-artifact hashes and deliberately supplies no professional population statistics.

## Reproduction and checks

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/build_human_kick_coordination.py \
  --input-dir data/research_human_kicks/cmu_subject10 \
  --output-dir data/research_human_kicks/coordination_v1

/home/you/conda_env/env_isaaclab/bin/python -m unittest discover \
  -s tests -p test_human_kick_coordination.py -v

/home/you/conda_env/env_isaaclab/bin/python \
  data/research_human_kicks/reader_validation/crosscheck.py
```

The CLI checks original C3D hashes against the download manifest, accepts only the five verified soccer-trial IDs, and writes no robot files. NPZs load with `allow_pickle=False`. Read normalized feature arrays together with their validity and interpolation masks, and consult the JSON before using any candidate event as an alignment anchor.

**Ten targeted tests passed**, covering known surface geometry, rigid-transform/scale invariance, flagged short gaps, unfilled long/boundary gaps, degenerate pelvis frames, missing landmarks, unavailable scale without loss of other geometry, rejected robot sources/time/name errors, source-versus-new fill flags, independent ball-motion candidates and absent contact truth. One integration test runs all five genuine local captures; it is explicitly skipped on clones lacking the ignored data. The full Tang-local project suite was not run by this P2 agent. No GPU, model training, author contact, payment, robot-source modification or Git commit was performed.

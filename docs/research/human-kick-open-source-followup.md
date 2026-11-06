# New open-source qualification audit after v0.1.30

Checked 2026-11-07; recorded on `version/v0.1.32`. This bounded pass considered exactly three new candidates after reading [the earlier source audit](human-kick-sources.md). CMU, SkillCorner, KTH, SportsPose, WorldPose and the other previously audited sources were not counted again. KTH remains closed. Existing adapters and source meanings are unchanged.

No candidate is yet a qualified continuous high-skill instep coordination prior. **Unknown reconstruction accuracy is different from demonstrated poor accuracy or unavailable data.** The promising next route is a visibility and synchronization audit of one MUVY professional shot event; no new 3D has been produced.

| New source | Player/action evidence | Public use and direct access | Continuous whole-body evidence | Decision at discovery stage |
|---|---|---|---|---|
| MUVY | Official event index names elite professional matches and shooters, including Neymar in the 2015 Champions League final and 2016 Olympic final. A goal/free-kick title does not establish instep contact, force or an unobscured full kick cycle. | Zenodo record explicitly declares CC BY 4.0; paper says source recordings were selected under YouTube Creative Commons. One public ZIP, 7,758,643,869 bytes. Individual source licenses were not independently re-audited. | Multiple spectator videos of the same event; no supplied joint trajectories, camera calibration or frame-accurate synchronization verified. Actual full-body visibility and reconstructability remain unknown. | Only candidate worth a bounded selective video audit. Dataset rights, event identity, visibility, contact type and 3D precision remain separate dimensions. |
| xGHub | Paper documents 704 matches from five major European leagues in 2015–16 and 2021–22. Shots are genuine professional match events. Its body-part/technique categories do not label instep or maximum effort. | Dataset card labels annotations MIT but expressly separates restricted video. Institutional email, access request and NDA required for video; no request or acceptance performed. | Published release schema gives per-shot metrics, including 3D orientation, but does not establish downloadable per-frame whole-body trajectories. The paper's pose-estimation pipeline is not a released accuracy certificate. | Access/schema gap; do not count a public card as acquired videos or classify unknown pose accuracy as a failed test. |
| EEG–kicking study / OSF `4jdqg` | 24 U17 state-level players, winners of São Paulo Interior League 2020; explicitly an instep task. This supports trained competitive youth, not senior professional status. | Seven public files can be downloaded without an account. OSF node has `node_license: null`; no explicit data reuse license was verified. | Public kinematic workbook contains trial-level scalars, not the time series acquired in the experiment. No original video or full-body trajectory file is listed. | Useful protocol/statistical reference; cannot supply continuous whole-body coordination. Do not equate the word “raw” with raw marker trajectories. |

## MUVY: exact public route and limits

Primary sources: [official project](https://swperfi-project.github.io/Pages-dev/MuvyDataset-page/), [event/video index](https://swperfi-project.github.io/Pages-dev/MuvyDataset-page/event_index.html), [dataset record](https://zenodo.org/records/13883315), [record API](https://zenodo.org/api/records/13883315), [dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13333307/), and [author scripts](https://github.com/swperfi-project/muvy-dataset-scripts).

The record API reports open access, `license.id = cc-by-4.0`, version `1.0.0`, and file `MUVY_v01.zip` with **7,758,643,869 bytes**, MD5 `607051346013728c1e14dcdd94f319c1`. The [official archive endpoint](https://zenodo.org/api/records/13883315/files/MUVY_v01.zip/content) is a direct download, not evidence that HTTP Range or small per-event extraction works. The dataset license, rather than the article license alone, is the research-use basis. Preserve attribution and the archived terms. The authors state that the original YouTube recordings were selected as Creative Commons; this audit has not independently established every uploader's rights.

The project statistics and event index include **8 soccer events / 35 videos**. The older record/project introductory text lists fewer sports and omits soccer, so it is insufficient as a manifest. The index resolves that discrepancy at the metadata level. Its “Available” labels are author-maintained oEmbed checks, not successful video acquisition by this project.

| Soccer event | Index-described action | Views | Selection implication |
|---|---|---:|---|
| 01 | Neymar, Brazil–Germany Olympic final penalty shootout | 5 | Skilled action, but penalty is not automatically the requested powerful instep kick. |
| 02 | Messi free kick, Barcelona–Liverpool 3–0, 1 May 2019 | 3 | Free-kick candidate; foot contact and complete visibility unknown. |
| 03 | Cristiano Ronaldo free kick, Real Madrid–Wolfsburg 3–0 | 5 | Free-kick candidate; same unknowns. |
| 04 | Neymar against Villarreal | 3 | Event subtype cannot be assigned from title alone. |
| 05 | Cristiano Ronaldo bicycle kick against Juventus, 3 April 2018 | 3 | Aerial bicycle action; poor match for a planted-foot instep target. |
| 06 | Neymar, Juventus–Barcelona 1–3, 2015 Champions League final | 6 | Open-play candidate with strong match provenance. |
| 07 | Messi's 2–0 goal against Bayern | 4 | Do not relabel this goal as a power shot. |
| 08 | Neymar free kick, Brazil–Germany Olympic final, Rio 2016 | 6 | Preferred next bounded audit under the parent task's free-kick priority. |

For event 06, [UEFA's match report](https://www.uefa.com/newsfiles/UCL/2015/2015227_FR.pdf) independently records Neymar's goal at 90 minutes plus 6 minutes 56 seconds. That corroborates the named professional match event, not the visibility or exact contact type of each spectator recording.

Event 08 source IDs are `2eeeoMwh4og`, `7oGe8f1IQxQ`, `WiWXQnEjQTs`, `X7NYdrEmU88`, `_fkPwtg7ozU`, `nSlzNETOri0`; for example, [the explicitly titled free-kick recording](https://www.youtube.com/watch?v=X7NYdrEmU88). Event 06 IDs are `8MF99Pz28eM`, `CuSaKVBYdOE`, `IEsm4j6e_60`, `b95P6Un4_Qs`, `gGZHpEc-ZoY`, `u6IZ9NmB5d0`. These are references, not inspected action labels.

Released formats described by the paper/scripts are MP4, extracted JPG, recording metadata (`video_info.txt`) and object detections (`detections_info.txt`). Native recording rates are retained. Camera viewpoint classes describe coarse spectator positions; they are not extrinsics. Player/goalkeeper/referee/ball detections are not body joints. Handheld movement, zoom, blur and occlusion require explicit handling; no metric scale or synchronization guarantee was verified. A prospective reconstruction must estimate and validate these rather than inherit the professional identity as an accuracy claim.

The next authorized step is limited to HTTP Range directory inspection and necessary files from one event, at most 100 MB newly transferred. Stop if Range is unsupported, terms do not cover the archive videos, or the chosen files exceed the remaining budget. Do not download the 7.76 GB archive or use a mirror to bypass a restriction. Assess complete action visibility and whether at least two views can be aligned before considering reconstruction; do not create instep/strong/contact labels from the event name.

## xGHub: professional source, restricted video

Primary sources: [author release](https://github.com/mguti97/xGHub), [dataset card](https://huggingface.co/datasets/mguti97/xGHub), [public dataset API](https://huggingface.co/api/datasets/mguti97/xGHub), and [author-hosted paper, §3.1](https://www.iri.upc.edu/people/aagudo/Papers/JCR/CVIU/CVIU2026/mgutierrezperez_etal_cviu26.pdf).

The checked Hugging Face revision was `8025d6eaba3b3858bee968f2a348479ffd70ba6e`; its API reports `private: false`, **`gated: auto`**. The public file list contains `labels.zip`, `videos.zip`, `clip_split.json`, and `xGHub NDA.pdf`, but listing filenames does not grant video access. The card explicitly requires institutional-email/NDA access for video and prohibits video redistribution/commercial exploitation. Annotation licensing must not be copied onto the video corpus.

The documented layout is `labels/<shot_id>/tabular_data.json` and `videos/<shot_id>.mp4`. Published fields include match code, outcome, body part, technique, pressure, occlusion, distances and orientation angles. No frame/time axis or full joint sequence is established by this schema. The paper derives pose using estimated detections/meshes and field calibration; these remain estimates. No archive, video, NDA acceptance or access request was performed. This source stops at the current access/representation boundary.

## OSF kicking study: inspect the actual public files

Primary sources: [paper, DOI 10.1007/s11571-022-09786-2](https://pmc.ncbi.nlm.nih.gov/articles/PMC9666621/), [author's OSF project](https://osf.io/4jdqg/), [node API](https://api.osf.io/v2/nodes/4jdqg/), and [complete public file listing](https://api.osf.io/v2/nodes/4jdqg/files/osfstorage/).

The experiment used an instep target task at 18 m, eight lower-limb/pelvis markers, four 240 Hz cameras and DLT reconstruction. It did not instrument both arms and the thorax. The published cohort mean ball speed is 95.13 ± 7.45 km/h; that is a paper statistic, not evidence of a maximum-effort instruction or a new measurement. Participants were competitive U17 players (15.9 ± 0.8 years), not a documented senior-professional cohort. The reported reconstruction uncertainty concerns that experiment and cannot certify unavailable full-body trajectories.

The API returned seven files and no next page or subdirectory:

| Public file | Bytes | Direct author-provided link |
|---|---:|---|
| `raw_ball_velocity.xlsx` | 20,075 | [download](https://osf.io/download/hqk4m/) |
| `lower_limb_factors.xlsx` | 168,057 | [download](https://osf.io/download/qnvhc/) |
| `EEG_SpectralPower_raw_data.xlsx` | 508,603 | [download](https://osf.io/download/7r34w/) |
| `participants.xlsx` | 10,868 | [download](https://osf.io/download/2qpwt/) |
| `EEG_ERS_ERD_raw_data.xlsx` | 478,794 | [download](https://osf.io/download/pzgqx/) |
| `raw_data_accuracy.xlsx` | 20,606 | [download](https://osf.io/download/wv86g/) |
| `study_data_EEG.study` | 124,512 | [download](https://osf.io/download/uwtq5/) |

Only `lower_limb_factors.xlsx` was fetched, read in memory for schema inspection, then discarded; SHA-256 `680a014aa70d849a13017afa97c5a41667e1f84868eb701ed9c359f1364b0fa1`. Its 18 sheets have range `A2:U32`: participant/trial scalar tables such as foot velocity, joint peak angular velocity, range of motion and joint angles at impact. They do not expose frame-by-frame XYZ. Blank/extra spreadsheet rows must not be reinterpreted as additional study participants. No motion arrays were imported into the pipeline. Public availability and a data-availability sentence are not a substitute for an explicit reuse license; `node_license: null` is retained as an unresolved rights field.

## Scope and next decision

Discovery transferred small public metadata plus the 168,057-byte in-memory workbook, and no corpus/video archive. No author contact, access application, payment, external AI service, GPU, training or reconstruction was used. There is no newly qualified motion sequence. MUVY alone proceeds to the separately bounded selection/visibility audit; xGHub and OSF do not warrant more acquisition under the current task. Any results of the authorized selective MUVY audit must be appended with actual bytes, hashes, observed action coverage and limitations, leaving the discovery claims above distinguishable from later visual evidence.

## Authorized selective MUVY acquisition: actual Event 08 observations

The one-event follow-up is complete and acquisition is closed. **Two acquired views visibly contain the real professional free-kick approach, leg swing and follow-through. No metric 3D, calibrated cameras or exact contact truth has been obtained.** This is a useful public RGB reconstruction candidate, with unresolved precision and instep/effort qualifications.

The [FIFA match report](https://inside.fifa.com/news/neymar-s-golden-penalty-sees-brazil-to-victory-2823069) independently corroborates Neymar's free-kick goal against Germany in the Rio 2016 final; it distinguishes that goal from the later penalty shootout. [His club's contemporary report](https://www.fcbarcelona.com/en/news/1064052/neymar-jr-and-rafinha-win-olympic-gold) supports professional participation. This event identity agrees with the archived recording titles and observed match action. Neither report specifies a measured maximum-effort kick or verifies the foot surface visible in these clips.

### Acquisition and integrity

All data are under ignored `data/research_human_kicks/muvy_event08_probe_v1/`. The official archive returned exact HTTP **206** ranges. Its ZIP64 end records list 333,858 entries and a 57,249,225-byte central directory; reading only the final **4,000,000 bytes** of that directory located all six Event 08 cameras. No complete archive or directory was downloaded. Preserved provenance includes the Zenodo record, official event index, CC BY 4.0 legal text, ZIP tail/directory ranges, response URLs/offsets, source-member metadata and hashes.

Acquisition transferred **55,266,584 body bytes**: four videos, six tiny recording-metadata files, four original first-frame JPEG previews, directory/provenance/terms bytes, and required local headers. Extracted member size/CRC checks passed; SHA-256 values below identify the original decompressed MP4 bytes. Large compressed member payloads were removed after successful extraction, with their range and hash retained in `download_manifest.json`. The archive-wide MD5 is publisher-supplied and was **not** locally verified, because the archive was not acquired. No second event, alternate host, paid service or external AI was used.

| Acquired camera / MP4 | Original bytes | SHA-256 |
|---|---:|---|
| `cam_01_S08/vid_nSlzNETOri0.mp4` | 14,905,825 | `64ee149d9f77966d93f4289841bcba2de74a1bc93ccdef3baf41f28777c996f4` |
| `cam_02_S08/vid_X7NYdrEmU88.mp4` | 26,305,985 | `46a2cdc58eb38b1995a7357cc3b9b2a1578ebc0d0749459e64bf755dd7b4f312` |
| `cam_03_S08/vid__fkPwtg7ozU.mp4` | 1,866,442 | `96e09e72789c60ad88bd89be3f942e325e026cb5eef05d5e55ddd14687b48d84` |
| `cam_04_S08/vid_2eeeoMwh4og.mp4` | 7,775,324 | `fef4d8bb29791b15402536bbb3f8e711c6d1dd0be00ff9dca9be742b251ceff8` |

The archive-relative prefix is `MUVY_v01/sport_events/soccer/soccer_event_08/`; extracted originals retain that hierarchy. All four streams are H.264, 1280×720, **30000/1001 Hz**, whereas their text metadata rounds FPS to 30. FFprobe frame timestamps and independent OpenCV decoding agree on 1,628 / 2,663 / 241 / 1,279 frames respectively. PTS are local recording times, not match time or synchronized clocks; encoded regular timestamps do not prove absence of duplicated images or rolling-shutter effects.

### Observed coverage and exclusions

| View | Actual visual evidence | Limitation / role |
|---|---|---|
| 03, source label LEFT | Clear approach and leg swing around 4–6 s; native-frame detail 144–164 covers PTS 4.771–5.439 s. Foot-to-ball transition and subsequent ball flight are visible. | Roughly 90-pixel player height in a reviewed frame; fast foot/ball blur prevents reliable shoe-surface classification. Useful first candidate view. |
| 04, source label BEHIND_RIGHT | Native-frame sample 1111–1171 covers PTS 37.037–39.039 s, including approach, ground-level kick and follow-through with no defensive-wall occlusion of the critical leg. | Roughly 60-pixel player height in a reviewed frame. Useful complementary candidate view, but limited spatial detail; camera jolts after the event. |
| 02, source label RIGHT | Closer preparation, approach and upper-body swing; source-frame detail 520–546 covers PTS 17.317–18.185 s. | Defensive wall hides the ball/foot and parts of the lower limb near the strike. May supplement torso/arm/event evidence, not replace missing foot observations. |
| 01, source label BEHIND_RIGHT | Same free kick and aftermath, around 33–36 s. | Shooter only roughly 30 pixels high in a reviewed frame; retained for provenance/event context, excluded as a preferred fine-motion view. |
| 05 and 06 | Only official first-frame previews and recording metadata inspected. | Wide/distant framing in the previews; videos not acquired. No claim about their uninspected full sequences. |

The pixel sizes above are approximate manual visual measurements of specific frames, not detector statistics or precision thresholds. Complete action coverage means the source includes the approach-to-follow-through interval; it does not mean every anatomical joint or foot surface is visible in every frame.

`action_qualification.json` records separate dimensions for verified professional event provenance, visible ground-level kicking, candidate multi-view coverage, unknown instep/maximum effort, absent metric 3D/calibration, and unknown reconstruction accuracy. There is no blanket source-qualified flag. Visual kick-transition candidates are bounded around view 03 PTS 5.138–5.205 s and view 04 PTS 38.538–38.739 s. These are coarse visual evidence intervals, **not measured physical impact times**. The wall jump, run-up, ball departure and goal sequence give common temporal landmarks, but synchronization has not been solved or validated; no offset is applied to any output.

Review the ignored local files `diagnostics/cam_03_S08_kick_detail.png` and `diagnostics/cam_04_S08_kick_detail.png` together, and `diagnostics/cam_02_S08_kick_detail.png` for the explicit occlusion. Corresponding JSON records source frame numbers, PTS and crop bounds. Overview sheets preserve broad sequence context. These are native-pixel scientific inspection crops, not generated pose images or reconstructed motion.

The bounded result closes part of the public high-skill/real-action/video-access gap. **P2's qualified instep coordination target remains unmet.** A useful next decision would be a small manual 2D landmark/visibility and temporal-alignment audit on views 03/04, using view 02 only where observable, before committing to any reconstruction. Moving cameras, lack of supplied calibration, small/blurred feet and no independent 3D truth must remain explicit. This round performed no pose estimation, camera fitting, smoothing, 3D reconstruction, training or GPU work.

### Existing-frame follow-up: five phase pairs and coarse timing

At the parent's request, the already acquired views 03/04 were inspected further without network access. `phase_pair_observability.json` and `diagnostics/phase_pair_candidates.png` contain five **candidate** pairs: approach, support, backswing, strike neighborhood and follow-through. Representative source frames (view 03 / view 04, one-based) are **145/1148, 154/1157, 152/1155, 156/1159, 164/1167**. Each carries its local PTS, broader candidate interval, visibility/blur/self-occlusion notes and a rough native-pixel body-height range. Support and backswing windows overlap; these are not disjoint force-defined phases, joint-center labels or inferred ground reactions. The native crops are padded, not enlarged, so the spatial limitations remain visible.

A frozen manual timing candidate is **`t_cam04 ≈ t_cam03 + 33.466767 s`**, based on the same ball-departure/leg-swing neighborhood. Differences between the manually inspected anchor intervals permit offsets **33.333300–33.600233 s**; these are conservative candidate bounds, not a measured synchronization-error interval. `diagnostics/sync_candidate_before_heldout.json` was saved before inspecting the separate goalkeeper check; its hash is recorded in the final JSON. No drift was fitted or correction applied to the recordings.

The unused goalkeeper dive/upward-arm progression was then inspected at view 03 frames **181–187 (6.006–6.206 s)** and view 04 frames **1184–1190 (39.473–39.673 s)**. Both move from lower/crouched hands toward upward diving extension. Their coarse windows are consistent with the frozen offset; no offset was adjusted after viewing them. The planned exact first-airborne check is **unresolved**: another player partly hides the goalkeeper's feet in view 03, and the post/net obstruct them in view 04. Thus this is a distinct visual consistency check, **not frame-accurate synchronization validation**. Clock drift, a calibrated temporal error bound and high-rate impact timing remain unknown.

The final lock files record verified SHA-256 for all **14 extracted original members**, four complete CPU decodes agreeing with FFprobe on **5,811 frames**, strictly increasing PTS, and the closed **55,266,584-byte** transfer ledger. All retained original data, metadata and derived diagnostic images occupy approximately **93 MB**, also below the 100 MB cap. No whole-archive checksum, independent 3D precision, instep subtype or maximum-effort claim is made.


## Additional lower-body leads screened without acquisition

Two later metadata checks do not close the whole-body requirement. [Wilmes et al. 2020](https://www.mdpi.com/1424-8220/20/9/2527) explicitly includes maximal instep kicks, but its 11 participants are amateur players and its five IMUs / 20 optical markers cover pelvis, thighs and shanks, not the trunk, arms and complete feet. Its [public Zenodo record](https://zenodo.org/records/3732988) lists a 987.4 MB processed-data archive; no archive was downloaded and no per-file license/schema was inferred from the article. This may later support lower-body method validation, not the requested skilled whole-body prior.

[Vieira et al. 2022](https://www.mdpi.com/1660-4601/19/3/1179) describes six U17 academy players and 240 Hz multiview recording, but its measured landmark protocol is lower-limb only. The linked OSF `jer9u` resource was not acquired or reclassified as continuous whole-body capture. Neither lead warrants downloading a large archive to fill body segments the protocol did not measure.

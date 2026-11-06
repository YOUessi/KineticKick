# Human kick sources — P2 acquisition and provenance

Checked **2026-11-07**, branch `version/v0.1.29`. Machine-readable decisions, source URLs and hashes are in [`configs/human_kick_sources_v1.json`](../../configs/human_kick_sources_v1.json).

## Decision

Use the acquired **five CMU subject-10 original optical soccer captures** to exercise the human-marker input pipeline; the follow-up pipeline and its actual outputs are documented in [human-kick-coordination.md](human-kick-coordination.md). Keep its player skill, instep subtype, foot–ball contact time and ball-center calibration **unknown**. This is a usable ungraded human soccer-kick reference, not evidence that the requested high-skill/professional instep target has been obtained.

No source verified in this bounded audit jointly supplies open access, original metric human 3D, documented skilled/professional soccer participants, and adequate instep/event metadata. P2's high-skill requirement remains **unmet**. Published biomechanics informs feature definitions; it cannot supply unobserved trajectories or replacement measurements.

## Existing local material

| Material inspected | Actual content | Appropriate role |
|---|---|---|
| `third_party/HumanoidSoccer/motions` | 13 NPZs: ten standard and three stylized references. All contain 29 G1 joint trajectories and 30 robot-body trajectories at 50 Hz; `kick_leg` is a robot-reference label. | Existing robot reference/controller baseline. Original human skeleton, anthropometry and measured ball impact are absent from these arrays. |
| `data/soccer-standard-001_right.npz` | Same robot-reference schema, 260 frames. | Local robot asset; not an additional human participant. |
| `data/human_motion/lafan1_g1_29dof` | Four walking and one running selected clips, each `240 × 36`: root XYZ/quaternion plus 29 robot joints, 30 Hz. | Retargeted locomotion comparison. No selected soccer kick. |
| `data/human_motion/straight_reference29` and `g1_29dof_reference_bank_v0.npz` | Derived G1 references and inferred stance proxies. | Generated locomotion references, not measured human force/contact data. |
| `tools/extract_human_coordination_priors.py` | Reads the above G1 arrays and computes robot-joint/FK gait statistics; stance is inferred from low/slow foot links. | Existing gait diagnostic only. It does not extract original human kicking coordination. |

PAiD upstream revision is `e72e470230047dedaf66df0983f1d0ab746faeb5`; its README declares CC BY-NC 4.0 for the codebase. The LAFAN1 retargeting README explicitly distinguishes motion data (original LAFAN1 CC BY-NC-ND 4.0), code (MIT), and robot assets (local Unitree BSD-3-Clause notice). An asset/code license must not be used to relabel motion rights. The retargeting README also says the optimization enforces kinematics, not dynamics or actuator limits. [PAiD release](https://github.com/TeleHuman/HumanoidSoccer/tree/e72e470230047dedaf66df0983f1d0ab746faeb5), [LAFAN1 retargeting source](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset), [original LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset).

## Acquired original human capture

The [CMU subject-10 index](http://mocap.cs.cmu.edu/search.php?subjectnumber=10) lists five soccer-kick trials: `10_01`, `10_02`, `10_03`, `10_05`, `10_06`; **`10_04` is walking**. The initial acquisition was `10_01`; a subsequent authorized collection added `10_02`, `10_03`, `10_05`, and `10_06` without acquiring walking `10_04`. The expanded catalog and manifest record every URL/hash. Initial sample links:

- [10_01.c3d — optical capture](http://mocap.cs.cmu.edu/subjects/10/10_01.c3d), 4,292,816 bytes, SHA-256 `ae454c3e8c621076d01a00b510c48e4daf8efdf5b4e19e8d80d76d062ef415f1`.
- [10.asf — calibrated subject skeleton](http://mocap.cs.cmu.edu/subjects/10/10.asf), 7,260 bytes, SHA-256 `d4e604fd582403e9f4bca3382e74dd0cc01e15cd712255e7bd34863b922b6200`.
- [10_01.amc — fitted joint motion](http://mocap.cs.cmu.edu/subjects/10/10_01.amc), 638,446 bytes, SHA-256 `4277e748947e8703825f340f94582f48997ca01b308f2447075eab4c42029e84`.

The initial three originals and four official provenance-page snapshots totaled **4,974,797 bytes**; the completed five-trial collection now contains 11 original motion/skeleton files plus four snapshots, **17,078,488 bytes**, all under ignored `data/research_human_kicks/cmu_subject10/`. `download_manifest.json` records URLs, UTC retrieval time, sizes and hashes. Official HTTPS certificate validation failed in the acquisition environment; the original site's documented HTTP endpoints returned the data. No corpus crawl, registration, author message or payment was performed.

Direct inspection of the acquired binary establishes:

| Field | Observed value and interpretation |
|---|---|
| Sampling | 801 frames, source indices 8–808, 120 Hz. Relative sample timestamps 0–6.666667 s. |
| Metric scale | `POINT:UNITS=mm`; export divides coordinates by 1000. Laboratory orientation/translation are preserved. |
| Human evidence | 41 physical surface markers: head, trunk, pelvis, bilateral shoulders/arms, hips/thighs, knees/shanks, ankles, heels, toes and fifth-metatarsal markers. These are not anatomical joint centers. |
| Extra points | 333 stored points total; 289 model-output points are excluded from the physical-marker export. |
| Body shape | Subject-specific ASF bone lengths are available; no measured mass/inertia, SMPL shape, footwear geometry or foot-surface mesh was verified. |
| Ball | `Soccerball:WEP1/2/3` marker tracks exist. They are not a calibrated ball-center trajectory. |
| Ball visibility | WEP1 valid on source frames 8–607; WEP2 8–632; WEP3 8–607 and 610–631. Valid fractions 74.9%, 78.0%, 77.7%. Missing observations remain NaN with masks. |
| Contact/force labels | `EVENT:USED=0`, `FORCE_PLATFORM:USED=0`, `ANALOG:USED=0`. No measured impact label, force or impulse. |
| Human marker presence | All 41 have nonnegative stored validity flags for all 801 frames. This does not prove no upstream gap filling or independently certify accuracy. |
| Skill and kick subtype | Not documented by the inspected files/index. Neither professional skill nor instep contact may be inferred from the filename or visual appearance. |

CMU's [FAQ](http://mocap.cs.cmu.edu/faqs.php) explains that optical marker data and fitted ASF/AMC motion have different meanings, and permits copying/modification/redistribution. Its [home page](http://mocap.cs.cmu.edu/) permits research use, prohibits direct resale even after conversion, requests attribution, warns that early low-number sessions can be lower quality, and flags noisy toe/hand and synthetic finger/thumb joints. Preserve those custom terms; do not mark the data CC0 or MIT. The [format explanation](http://mocap.cs.cmu.edu/info.php) gives the ASF/AMC length conversion `(1/0.45) × 2.54/100` to meters; it must not be applied again to C3D millimeters.

## Other primary-source candidates and evidence gaps

| Source | What is actually available | Decision |
|---|---|---|
| [SportsPose](https://christianingwersen.github.io/SportsPose/) / [paper](https://arxiv.org/html/2304.01865v1) | Seven calibrated, hardware-synchronized cameras at 90 Hz; 17 COCO-style joint estimates, triangulation and temporal smoothing. Twenty-four participants perform five activities. Separate optical validation reports 32.0 mm soccer mean error. Player skill is not established; participants were allowed creative freedom. | Useful multiview candidate, separate from optical-marker truth. The [access/license page](https://christianingwersen.github.io/SportsPose/download.html) requires an academic email request and limits use to academic, nontransferable access. No request sent or dataset acquired. Raw optical validation files are not verified available. |
| [Zhou et al., elite/amateur instep coordination](https://onlinelibrary.wiley.com/doi/full/10.1002/ejsc.70041) | Study documents 15 elite and 15 amateur players. [Data statement](https://pmc.ncbi.nlm.nih.gov/articles/PMC12358685/) explicitly says the raw data are not public. CRP analysis code is linked separately. | Best direct skill-level evidence among audited biomechanics sources, but not an actionable raw-data download. Paper/analysis-code access does not grant motion access. |
| [Experienced/novice vector-coding study](https://www.mdpi.com/2306-5354/12/11/1151) | Fourteen soccer-major college students and 32 novices; 200 Hz optical capture, 2000 Hz EMG and 60 Hz ball video; a two-step 45° instep protocol. Data statement offers request-based access. | Relevant measurement/phase specification. Experienced college participants are not automatically professional. No raw public archive or license verified, and no author contacted. |
| [AutoSoccerPose / 3DSP](https://github.com/calvinyeungck/3D-Shot-Posture-Dataset) | Professional match shot images; manually annotated 2D poses lifted using MotionAGFormer; repository contains [3dsp.zip](https://github.com/calvinyeungck/3D-Shot-Posture-Dataset/blob/main/3dsp.zip). | Professional context is supported, metric original 3D is not. Qualitative/2D evidence only. Repository Apache-2.0 does not establish unrestricted rights to underlying SoccerNet/broadcast video. Not acquired. |
| [SoccerKicks](https://github.com/nayariml/SoccerKicks) | Thirty-eight video clips processed with AlphaPose/OpenPose then HMMR; predicted 25-joint poses, SMPL rotations/shape, JSON/pickle. Repository declares dataset CC0 and code MIT. | Model estimates, not measured human 3D; per-file skill and underlying-video rights were not verified. Not acquired. |
| [SkillCorner body pose](https://huggingface.co/datasets/SkillCorner/opendata-bodypose) | Dataset card identifies 3D body pose derived from broadcast video. | Discovery lead for video comparison only. Not original mocap; detailed metadata/license not inspected and no acquisition. |

A targeted additional search covered Zenodo, Figshare, OSF and primary paper supplements using soccer/instep/kinematics/motion-capture/elite terms. No qualifying open skilled-player raw archive was verified. Two apparently relevant open resources do **not** close the gap: [elite Bundesliga event/position data](https://www.nature.com/articles/s41597-025-04505-y) describe player/ball locations, not body landmarks; [183-athlete motion screening](https://www.nature.com/articles/s41597-023-02082-6) contains mobility/stability tests, not soccer kicks. A [12-elite-player long-pass study](https://www.mdpi.com/1424-8220/26/12/3654) uses Xsens but restricts data access to the research team. These exclusions are recorded so future work does not rediscover them as supposed solutions.

The actionable source is therefore the **five acquired CMU trials for pipeline development**, with a clearly unresolved skilled-instep target. SportsPose is a promising future multiview option only after valid access and inspection; its published cohort information still does not satisfy the skill requirement.

## Input contract and checks

[`tools/inspect_human_kick_assets.py`](../../tools/inspect_human_kick_assets.py) is a bounded Intel floating-point/no-analog C3D reader. It rejects unsupported layouts rather than guessing units or byte order. It exports only physical human markers and ball markers; it does not smooth, interpolate, fit joint centers, detect impact, classify skill or retarget to G1.

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/inspect_human_kick_assets.py \
  --cmu-c3d data/research_human_kicks/cmu_subject10/10_01.c3d \
  --report data/research_human_kicks/cmu_subject10/inspection.json \
  --export data/research_human_kicks/cmu_subject10/10_01_measured_markers.npz
```

The NPZ contains `human_markers_m[T,41,3]`, `human_marker_names[41]`, `human_marker_valid[T,41]`, `ball_markers_m[T,3,3]`, `ball_marker_valid[T,3]`, `ball_marker_names[3]`, `time_s[T]`, `source_frame_numbers[T]`, `fps`, source class and C3D hash. Load with `allow_pickle=False`. Its initial SHA-256 is `b88e2eb13ff0ea49ee652b11b570bc0e5e21b2a278035f1dbc57d17cc596daf2`.

Any downstream anatomy or event estimator must add provenance instead of changing raw meaning: name its marker-to-segment rule, make body/ball validity explicit, retain per-feature availability, attach inferred-event uncertainty, and keep source skill/subtype null. Shoulder/hip surface markers are not automatically joint centers; ball-marker centroid is not automatically ball center; kinematic foot stationarity is not a force measurement. Do not fill unavailable professional population statistics with values from robot references.

A known-capture end-to-end acceptance check passed: frame/marker counts, independently decoded first-marker coordinates and unit conversion, exclusion of model-output points, missing-ball masks, and absent force/event labels. The check first failed when the utility did not exist; a later fixture rounding discrepancy was resolved by independently decoding the original binary rather than altering the export. The full Tang-local project suite was not run by this source-audit agent. Originals and generated artifacts remain ignored; no existing asset or checkpoint was replaced.

## P3/P4 release availability, checked 2026-11-07

[BeyondMimic's official training repository](https://github.com/HybridRobotics/whole_body_tracking) is archived from 2026-10-16. Its README marks it retired and points to [mjlab's tracking task](https://github.com/mujocolab/mjlab/tree/main/src/mjlab/tasks/tracking). The official [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller) supplies C++ ONNX CPU inference with ROS 2 deployment. Training expects retargeted robot generalized coordinates. Code availability and its license are separate from source motion rights, a skilled soccer dataset, or evidence that the guided-diffusion portion/checkpoints are released.

The [official SuperMimic page](https://supermimic.github.io/) and [paper](https://arxiv.org/abs/2610.28480) provide method/results and a browser demo. This audit did not verify an author-linked training repository or downloadable training checkpoint with license. Its BONES-SEED tracking benchmark is not evidence of professional soccer capture. Community reproductions must retain their own identity; neither a demo nor a reported compute result establishes a ready replacement for the current controller.

## v0.1.30 follow-up: continuous professional reconstructed3D

The earlier SkillCorner discovery-only row below is superseded by the [acquired sample and working independent adapter](skillcorner-reconstruction.md). Under the explicitly expanded reconstructed3D scope, the official MIT dataset supplies a **309-frame, 25 Hz continuous A-League sequence** for Jacob Brazete and a synchronized official non-header shot event (`8_871`, 77:45.7). Official club match reporting corroborates his professional match participation. This is model-estimated body pose with predicted per-joint errors, not original optical mocap.

Eight source files totaling **7,145,321 bytes** are preserved with URLs and hashes under ignored `data/research_human_kicks/skillcorner_probe_20261107/`. The separate v30 adapter preserves 29 estimated joints, uncertainty, detection/missing/resampling semantics, fractional event timing and 14 coarse geometric relationships. The earlier CMU data and interpretation are unchanged.

The skill/continuous-action provenance gap is partly closed; **P2 remains incomplete**. Instep, maximum effort and actual contact remain unknown. Shot-window predicted errors have median 16.62 cm and maximum 43.01 cm. Under the explicit exploratory sensitivity screen, only 13 trunk–shoulder-line frames pass within shot ±1 s; no knee/foot or cross-arm/leg relation passes. Complete joint coverage is not evidence of adequate fine biomechanics. Source qualification is therefore recorded in separate dimensions, with no blanket claim that this is a qualified whole-body shooting prior.

### One-match bounded scan result

A subsequent authorized acquisition added only the 623,408,231-byte official archive for match 1925299. Its full 153,252-frame stream yielded all 24 non-header shot windows (±1.5 s, 1,817 shooter frame records); six header shots and one fully extrapolated/unmatched non-header event have explicit exclusion records. Under the unchanged 20 cm joint-error / 0.75 relation-budget screen, no window supports simultaneous pelvis–leg–trunk–upper-arm coordination. `8_804` Quinn MacNicol is a better partial trunk–leg example (72/76 passing frames), but no leg chain or cross-arm/leg relation passes anywhere in the scan. The [complete report and reproducible scanner](skillcorner-reconstruction.md#bounded-full-match-follow-up-no-qualifying-coupled-motion-candidate) document this negative result. Archive plus outputs remain below 750 MB; no second match was downloaded, and P2 remains incomplete.

## Three-source follow-up: KTH selected for a bounded prototype

This round inspected exactly three previously unaudited sources. The actionable result is **KTH Multiview Football II, Player 1 Sequence 2**: a small, acquired continuous reconstruction from a documented professional match. Independent 3D accuracy and instep/contact labels remain unknown. It is a source-conditioned P3 input prototype, not completion of P2.

| New source | Primary evidence and access | Decision |
|---|---|---|
| [KTH Multiview Football II](https://www.csc.kth.se/cvap/cvg/?page=footballdataset2) | Official page identifies professional Allsvenskan players and labels P1S2 as a goal sequence. Academic research only; commercial use excluded, BMVC 2013 attribution requested. | Numerical 2D/3D/camera files acquired by HTTP Range; qualified professional context, continuous 3D, unknown independent precision and instep. |
| [AthletePose3D](https://github.com/calvinyeungck/AthletePose3D) / [corrected paper v3](https://arxiv.org/html/2503.07499v3) | Eight university/national representatives perform running, five throwing motions, and six figure-skating jumps. Hardware-synchronized multiview capture at 60/120 Hz; 55/86 landmarks. | **Activity mismatch:** no football kick in the listed 12 motions. No raw files acquired. This is not a failed accuracy test. The claimed sub-millimeter camera calibration is not a bound on joint accuracy. |
| [WorldPose](https://eth-ait.github.io/WorldPoseDataset/) / [official access terms](https://worldpose.ait.ethz.ch/) | Continuous 2022 FIFA World Cup reconstructions, SMPL poses and cameras. An institutional-email application and review are required; FIFA footage needs an additional agreement. | **Access not obtained:** no application submitted and no pose/video downloaded. The research license is nontransferable, prohibits redistribution and re-identification. Accuracy remains unassessed locally, not rejected. |

AthletePose3D's [data license](https://github.com/calvinyeungck/AthletePose3D/blob/main/license/README.md) permits noncommercial scientific research and restricts reproduction, modification and third-party provision without written permission. Its paper's CC BY-NC-SA license does not replace those data terms. The author-linked download exists behind that terms page; no archive was needed to establish the activity mismatch. The repository also flags corrected running-camera preprocessing; any future use should pin the corrected release.

### KTH acquisition and exact source schema

Original bytes and provenance are local under `data/research_human_kicks/kth_player1_sequence2_probe_v1/`. The [official archive](https://www.csc.kth.se/cvap/cvg/MultiViewFootballData/player1sequence2.zip) has HEAD Content-Length **214,590,675 bytes**. HTTP Range was honored with status 206. A 65,536-byte suffix contains the central directory; a separate range `bytes=214466611-214531658` contains all three numerical members. No image/video or complete archive was downloaded. The complete archive hash is deliberately null.

Together with the [7,952-byte official loader](https://www.csc.kth.se/cvap/cvg/MultiViewFootballData/code3d.zip), landing-page license snapshot and [3,065,992-byte methods paper](https://www.csc.kth.se/cvap/cvg/papers/Burenius4DMOD2011.pdf), total new downloaded bodies are **3,210,408 bytes**, below the authorized 15,000,000-byte ceiling. `acquisition_manifest.json` preserves response ranges, source URLs, byte counts and SHA-256 values; decompressed numeric members were checked against their ZIP CRCs.

| Original member | Bytes | SHA-256 | Python array order |
|---|---:|---|---|
| `positions3d.txt` | 132,300 | `f98feba689ce5bda4ad592e16e963e7a0bdfab36649db6d0db0a82a0f2bff83e` | `loadtxt(...).reshape(175,14,3)` = frame, joint, xyz |
| `positions2d.txt` | 88,200 | `38dbb657eb95746a1e94e14e5d53bd1c9aaa0fc04445d211ec89f8b187f3e9ce` | `reshape(175,3,14,2)` = frame, camera, joint, xy |
| `cameras.txt` | 75,600 | `9f815dfa57f78d85ba9c7d65e8c1003a5b96c369de585adcfdc853c9ad06fada` | `reshape(175,3,4,2).transpose(0,1,3,2)` = frame, camera, row, column |

Joint order is `R_ANKLE, R_KNEE, R_HIP, L_HIP, L_KNEE, L_ANKLE, R_WRIST, R_ELBOW, R_SHOULDER, L_SHOULDER, L_ELBOW, L_WRIST, BOTTOM_HEAD, TOP_HEAD`. This covers bilateral hip/knee/ankle and shoulder/elbow/wrist, but supplies no heel, toe, ball, thorax-surface landmarks, force or body shape. Each of the three camera folders contains image member indices 1–175 without gaps. The methods paper §4 documents three 25 Hz cameras and a seven-second goal sequence; sample times are therefore 0–6.96 s, with no match-clock or impact anchor.

The official `Reconstruct3d.m` reconstructs points and orthographic cameras from manually labeled 2D observations. Its Euclidean rectification is followed by rotation into the last-camera convention and a **sequence-wide unit-cube normalization**. The release supplies no meters-per-unit. The algorithm removes each frame's joint mean; observed maximum absolute 3D frame mean is `2.22e-9`. It supplies neither measured ground height nor absolute player trajectory. Its camera matrices are fitted estimates, not independently measured perspective intrinsics/extrinsics. The 2×3 camera component is exactly constant across these 175 frames; image translations vary.

### KTH CPU adapter and diagnostic result

The [adapter](../../tools/build_kth_coordination.py), [contract](../../configs/kth_coordination_v1.json) and [five focused tests](../../tests/test_kth_coordination.py) preserve the original numerical arrays and generate scale-free geometry:

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/build_kth_coordination.py \
  --source-dir data/research_human_kicks/kth_player1_sequence2_probe_v1 \
  --output-dir data/research_human_kicks/kth_coordination_v1 --plot
```

The resulting NPZ loads with `allow_pickle=False`; the adjacent JSON documents source hashes, definitions, qualification dimensions, array shapes and limitations. `manifest.json` records all output hashes. Main arrays are:

| Output family | Shape | Meaning |
|---|---|---|
| `joints_reconstructed_au` / `joint_geometry_valid` | 175×14×3 / 175×14 | Unchanged arbitrary-unit reconstruction and finite-value mask. |
| `manual_positions2d_px` / `orthographic_cameras` | 175×3×14×2 / 175×3×2×4 | Unchanged manual image labels and fitted orthographic matrices. |
| `joints_pelvis_relative_au` / `joints_normalized` | 175×14×3 | Subtract estimated bilateral hip midpoint, retain source axes; normalize by median bilateral hip–knee–ankle chain length. No anatomical pelvis orientation or metric calibration. |
| `landmarks_source_axes_au` / `landmark_geometry_valid` | 175×21×3 / 175×21 | Direct joints, declared midpoint proxies and explicit unavailable landmarks. Thorax center, heel, toe and MT5 masks remain false. |
| `segment_directions_source_axes` / `segment_geometry_valid` | 175×13×3 / 175×13 | Eleven computable segment directions; both foot directions unavailable. Trunk uses a shoulder-midpoint proxy, distinct from CMU's thorax-marker construction. |
| `relation_cosines` / `relation_geometry_valid` | 175×14 | Same pair names/order as the CMU contract. Eleven relations have finite geometry; three foot-dependent relations remain NaN/false. Values are directed-vector relationships, not anatomical joint-angle truth. |
| `reprojection_residual_norm_px` / `joint_reprojection_rms_px` | 175×3×14 / 175×14 | Per-camera/joint pixel residual and same-frame camera RMS. These are fitting-observation diagnostics, not independent 3D error. |
| `segment_length_relative_deviation` / `segment_length_cv` | 175×13 / 13 | Length divided by its sequence median minus one; sequence std/mean. No bone-length correction or rejection. |

All 2,450 joint frames contain finite source 3D; 1,925 of 2,450 relation slots have computable geometry. `joint_visibility_known` and `relation_precision_known` are false throughout, because the source has no corresponding labels. **Unknown precision does not mean failure.** The unavailable relations are `left_foot__right_foot`, `left_shank__left_foot`, and `right_shank__right_foot`. No interpolation, smoothing, new reconstruction, quality filtering, velocity/acceleration fitting or event estimation is performed.

The measured reprojection residual is RMS **10.48 px**, median **5.70 px**, p95 **17.79 px**, maximum **101.02 px**. Left-shank length spans 0.45–1.40 times its sequence median (CV 14.0%); left forearm CV is 17.6%, pelvis-line CV 14.5%, shoulder-line CV 7.3%. These observed instabilities warrant caution for detailed coordination, but they neither quantify independent 3D error nor justify applying SkillCorner's centimeter gate. All original values remain available.

The local `data/research_human_kicks/kth_coordination_v1/projection_diagnostic.png` shows manual 2D labels and reprojections for frames 1, 83 (largest frame RMS), and 175 in all three cameras. It contains plotted numerical coordinates, not downloaded match imagery. The source label verifies a goal sequence, while exact shot frame, named player, instep, effort, ball center and true contact remain null. **P2 remains incomplete.**

Validation: five targeted tests passed for source hashes/shape order, known synthetic projection, scale/rotation/translation invariance, separate 2D-observation/3D-geometry masks, degenerate and missing segments, preservation of high residuals, source qualification, exact 175-frame roundtrip and JSON/NPZ types. A real CPU run generated the NPZ/JSON/PNG, and the figure was visually inspected. This source investigation ended after the three candidates; no second SkillCorner match, GPU job, dependency installation, author contact or paid access was used.


### KTH后续有界资格检查

固定相机逐点重三角化只复现原rank-3列空间的最小二乘结果，整体RMS仍10.4751px，3D最大差4.08e-8任意单位；不支持“直接重新三角化即可恢复快速踢腿”的假设。两视图预测另一视图的诊断仍使用全数据拟合的相机，不能称为独立验证。结果在ignored`kth_retriangulation_probe_v1/`。

在总下载10,545,932B内检查原75–91帧多视图及1–175帧的12个均匀关键帧。61–91帧为空中争抢；91–121帧是被遮挡的低置信可能动作区间；后段有庆祝样奔跑。依然不能确认脚部射门、正脚背或实际触球。此源的本轮资格搜集已关闭，不因goal标签或庆祝推断instep。记录/接触表在ignored`kth_action_qualification_v1/`，已有adapter原始数值和未知标记未改。

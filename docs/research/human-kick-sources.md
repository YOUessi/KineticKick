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

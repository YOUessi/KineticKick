# KineticKick v0.1.117 frozen original G1 200Hz protocol
Parent source SHA: 402b87f019dd7cd31f9d5af980ad38a2b63a7e3f. Clean new version/v0.1.117 branch and distinct worktree. Freeze and commit the batch code, matrix, tests, and this protocol BEFORE any GPU physical run. No prior original controller, raw dataset, model, or previous evidence overwritten. All generated raw JSON, NPZ and media remain ignored and local; each new output is write-once.

## New independent physics matrix
AJ seed 2843: offset [-0.018,+0.041]m, velocity delta -0.013; AK seed 2897: [+0.035,-0.029], +0.012; AL seed 2939: [-0.059,-0.022], -0.017. Original v32, CMU10_02 and CMU10_05 in all scenes (9 independent cells), plus AJ/v32 and AL/v32 ON repeated observer paired to OFF (11 physical executions). This deliberate new spatial and sign coverage is not selected for favorable performance. No tuning after observing outcomes. Historical AG/AH/AI remain development evidence.

## Unchanged physical and safety gates
Original free-root 29DoF G1, original SHA-checked checkpoint and controllers, 200Hz CUDA PhysX, 14s recovery and original hard-gate evaluator. PASS iff real goal AND legitimate instep share >=0.95 AND ball speed proxy >=3.5m/s AND all original 29 individual qdot limits AND absence of nonfoot >=20N AND actual observed 14s standing recovery. Speed floor is not elite human speed. Never reclassify over-speed as PASS just for goal entry.
Global nonblocking GPU mutex, no other GPU compute job, >=20GiB free disk, 140-second bounded child watchdog, write-once logs/JSON/audits and fail-closed on uncertified launch or OFF/ON physical mismatch. OFF/ON is not a statistically independent shot, nor is a startup timeout a valid shot result.

## Preregistered independent analyses
SHA-check every actual raw 200Hz rollout against the master, 29-joint order and qdot limits. Compare first measured per-joint qdot hazard vs first CMU-v32 actuation difference. Measure 0.4s precontact, >=1.25s postcontact or explicitly censor, 14s observed recovery, COM, bilateral stance force proxies, hip/torso/leg/opposite-arm group kinetics and estimated torque-work; distinguish measured robot forces from athlete GRF.
Native actor/collider contact identity is unqualified until positive native pair, separated negative, original G1 wrist-hip witness and OFF/ON parity. Tensor force zero alone cannot establish absence of self-collision.

## Human biomechanical eligibility and motor actuation
Existing six C3Ds from only two subjects have zero native analog channels, do not certify elite human instep impacts, force plate GRF/CoP or multi-athlete kinetics; no synthetic truth. New human-phase or safety designs may run only offline/CPU. No learned human policy may drive G1 before native collision witness, stance force/slip, 29DoF joint position/velocity/acceleration/torque and physical recovery validation.

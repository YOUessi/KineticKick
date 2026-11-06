# Late landing contact diagnosis — 2026-11-07

The observed sequence is **rearfoot contact followed by loading of the forefoot and the positive-y sole row**, with substantial tangential braking and an asymmetric friction-anchor moment. The positive normal-force points lie on the lower capsule surfaces. They do not support a dorsum strike or toe-first landing explanation. The subsequent momentum residual remains consistent with the previously documented articulation-velocity scaling hypothesis; its upstream contact dynamics and the solver correction are separate observations.

This is the second right-foot landing after the shot, beginning in the bin ending 2.245 s. Both feet report zero force for the eight bins ending 2.205–2.240 s. The previous support foot has therefore unloaded before this landing. The original v28 already has the same late overspeed at 2.250 s: 20.988663 rad/s versus the new 21.143692 rad/s, against the unchanged 20 rad/s knee limit. The original task still passes, but this all-episode velocity-quality failure remains. Final standing is 8.925 s in v28 and 8.795 s in v30.

## Observation integrity and scope

The incremental report contains 17 contact-detail bins, with end times 2.200–2.280 s. A direct old-key comparison verifies all 2,800 original trace rows, original records, kinematics and sequence records are exactly unchanged from the preceding v30 report. Added `ground_contact_detail` keys do not constitute changed physical output. No controller, physics or state-write operation is performed by the offline diagnostic.

Normal points and friction anchors are read using their own counts and starts. They are never paired by buffer index. All recorded buffers are unsaturated. A float64 sum of the serialized normal forces differs from the original normal-force matrix by at most 0.00006044 N over the 17 bins; friction reconstruction is exact in these saved values. The original independent force-sign probe is revalidated. This force check does not independently calibrate contact torque or certify SDK point timing.

The raw SDK point stage remains unknown. The diagnostic transforms every positive-force point with both bracket poses and compares it with the original capsule surfaces. The digest-matched URDF specifies seven foot cylinders; the original importer enables `replace_cylinders_with_capsules=True`. The closest geometric primitive index below is not a reported SDK shape ID.

For the right foot's 34 positive normal points, maximum pre-pose capsule surface residual is **0.02108 mm**, versus post-pose median **0.10350 mm** and maximum **2.96505 mm**. For the 78 nonzero friction anchors, maximum pre-pose residual is **0.03539 mm**, versus post-pose median **0.29069 mm** and maximum **11.18137 mm**. This strongly favors a pre-step geometric correspondence; it is not proof of exact solver staging or material-point identity. No sphere/ball contact-point correction is applied to these ground points.

## Which part of the foot carries force?

Foot coordinates use x toward the toe and z upward. Positive y is the right foot's medial row in the neutral link convention; its world direction depends on foot yaw. Capsule centers are at z = −0.025 m, with the relevant lower surfaces near z = −0.035 m. A point at y ≈ 0.026 m, z ≈ −0.035 m is on the bottom of the positive-y capsule row, not its vertical sidewall.

| Bin end s | Normal force N | Pre-foot local point [x,y,z] m | Interpretation |
|---:|---:|---|---|
| 2.245 | 208.537 | [−0.052841, 0.010151, −0.034964] | Only positive normal point: rearfoot/heel-side bottom |
| 2.250 | 1709.607 | [0.049350, 0.026204, −0.034977] | Positive-y short forefoot capsule row, lower surface; 57.12% of normal load |
| 2.250 | 908.308 | [−0.052650, 0.010204, −0.034977] | Rearfoot bottom; 30.35% |
| 2.250 | 231.855 | [0.129350, 0.010204, −0.034977] | Toe-side bottom; 7.75% |
| 2.250 | 97.594 | [0.131350, 0.000204, −0.034977] | Toe-side bottom; 3.26% |
| 2.250 | 45.557 | [0.099350, 0.026204, −0.034977] | Anterior end of positive-y short capsule row; 1.52% |

The normal-force-weighted pre-foot centroid at 2.250 s is [0.028027, 0.019261, −0.034977] m. Load has moved forward and toward the positive-y sole row, while appreciable heel load persists. Heel contact itself is not a failure criterion.

The 2.245 s friction anchors already span approximately 0.182 m from rearfoot to toe-side capsule positions. One anterior anchor is at raw world height 17.260 mm and carries about 104.27 N friction despite the only positive normal point being at the heel. An anchor is a solver friction reference, not proof of a separate geometric ground-touch point. At 2.250 s, the two positive-y forefoot anchors each carry about 877.58 N of friction. Their spatial distribution is important to the reconstructed yaw moment.

Under the conditional interpretation that each normal point is a pre-step material point, advecting it using the observed post pose gives near-plane positions: the first heel point goes from 1.951 mm to −0.006 mm world height; the dominant forefoot point from 6.294 mm to −0.219 mm; one toe-side point from 11.819 mm to 0.225 mm. Positive raw separation at the beginning of a bin therefore should not be described as measured penetration, or as proof that the surface never approaches the plane during that step.

## Approach velocities and foot rotation

The reconstruction uses the explicitly recorded link-origin velocity:

`v_point = v_link_origin + omega × (point_world − link_origin_world)`.

Normal velocity is positive away from the reported ground normal. These are velocities at the raw world point using the bracketed state. The JSON separately retains the conditional advected material point; it does not mix that point with a fixed world-point calculation.

| Bin end s / point | Pre-state normal velocity m/s | Pre-state tangential speed m/s |
|---|---:|---:|
| 2.245 heel | −0.46373 | 2.55297 |
| 2.250 dominant positive-y forefoot point | −0.95115 | 1.47774 |
| 2.250 toe-side points | −1.48872 to −1.49418 | 1.56389 to 1.58290 |
| 2.250 heel | −0.23547 | 1.44849 |
| 2.250 normal-force-weighted values | −0.79854 | 1.47950 |

Foot angular-speed norm grows from **1.64345 rad/s at 2.240 s**, to **7.14000 at 2.245 s**, to **25.83595 at 2.250 s**, then falls to **4.70944 at 2.255 s**. Across the main impact bin, world angular velocity changes from [6.80430, −1.41292, 1.63849] to [0.30318, −21.07737, −14.93818] rad/s. The foot reference-point velocity reverses sharply in world y, and knee speed crosses its limit in the same bin. This is an articulated contact response; the external contact moment is not equal to a knee actuator or joint-reaction torque.

## Force and moment reconstruction

At 2.250 s the original certified observer records right-foot total force [−87.03164, 2464.19551, 2992.92105] N, norm **3877.810 N**. Normal z is 2992.92090 N; the friction vector is [−87.03204, 2464.19551, 0.00015324] N. Left-foot force is zero during both impact bins.

The following moments sum `r × F` using raw contact/anchor points and the **pre-step ankle-roll-link origin**, in world axes. Forces are step averages, but their substep point history is absent: these are snapshot-based estimates of mean contact moment, not independently measured joint torque or an exact angular-momentum closure.

| Bin end s | Normal moment [x,y,z] N·m | Friction moment [x,y,z] N·m | Sum N·m |
|---:|---|---|---|
| 2.245 | [9.16273, −5.24291, ≈0] | [6.62433, −1.68624, 0.38561] | [15.78706, −6.92914, 0.38561] |
| 2.250 | [−103.52174, −23.83237, ≈0] | [74.53751, −2.74780, −106.38015] | [−28.98424, −26.58017, −106.38014] |

At 2.250 s the same sum using the post-step origin is [−25.32752, −30.24009, −103.26044] N·m. The large negative yaw component persists under this frame choice; the two estimates are sensitivity checks, not rigorous bounds on the true time-varying wrench.

Over 2.240–2.250 s, right normal and friction impulses are [0.00000215, 0, 16.00729] and [−0.180338, 13.365949, 0.000000756] N·s. The pre-origin snapshot moment integral is [−0.065986, −0.167547, −0.529973] N·m·s; the post-origin result is [−0.039785, −0.187866, −0.514285]. The origin moves, so these integrals must not be presented as a fixed-origin foot angular-momentum balance or an energy-transfer partition.

The COM balance again separates the impact from the subsequent correction: residual norm is **0.00000602 N·s in 2.245–2.250 s**, then **1.110305 N·s in 2.250–2.255 s**. The unchanged limit gives gamma = 20/21.143692 = 0.945908595. The existing all-DOF scaling hypothesis predicts the next-bin residual to **0.00002745 N·s**. This is strong supporting inference, with the installed solver binary and URDF-inferred local COM still unverified; the public mechanism is described in [PhysX PR #534](https://github.com/NVIDIA-Omniverse/PhysX/pull/534). Nothing is subtracted from the measured residual.

## What this resolves and the smallest next intervention

The data reject a toe-first or dorsum-contact account for the first loaded bin. They support rearfoot contact followed by rapid forefoot/positive-y-sole loading, appreciable ground-relative slip, and a large asymmetric friction-anchor moment. They do not uniquely identify why the upstream running policy arrives at that pose and velocity, nor prove that the contact patch implementation alone causes the knee overspeed. Original v28 lacks this point/wrench telemetry, although its same-time overspeed and foot rebound establish that the late failure class predates v30.

The smallest useful next control experiment is a **shared post-shot support-handover rule**: coordinate the outgoing support duration, incoming foot lowering/retraction, and pelvis horizontal trajectory using measured predicted touchdown time, foot normal/tangential approach velocity and foot angular velocity. Test whether the incoming foot begins carrying load at lower slip and rotation while the previous support is still useful. This targets the observed late handover, rather than the already inactive strike envelopes. No case-specific gain or controller change is made here.

A falsifiable prediction is that a successful handover change lowers the friction-anchor yaw moment and late knee-speed peak while preserving the protected ball strike and full task. If slip/rotation are reduced but the moment/overspeed remain, the contact-patch and articulation-constraint explanation needs further examination. Existing evidence is sufficient to motivate that bounded trajectory/support experiment; another identical observation replay is not needed merely to establish the loaded foot region.

## Reproduction and provenance

```sh
python3 reports/late_landing_contact_diagnostic.py --overwrite
```

The ignored script writes `reports/late_landing_contact_diagnostic.json`; this report provides the interpretation. Four analytic checks cover capsule distance, quaternion/world-to-local direction, normal/tangent velocity signs, and the moment cross-product sign. All pass. The JSON contains every positive-force point/anchor, both pose interpretations, conditional material-point advection, independent component wrenches, interval integrals and dependency hashes.

| Input / artifact | SHA-256 |
|---|---|
| Incremental raw report | `4e52a3a81059a667b83e997b9f0fd30948908bdd770864be92f98bc77f16c264` |
| Previous v30 raw report | `5a302ea3a25443125559b05ba69aebf9c6e515866b05b6299217a1d50e07f484` |
| Original v28 raw report | `95b81b5cd1ddec19a0d959536db9d9b0631e66d9f00a472c87ff0ddfed3154ba` |
| Original resolved URDF | `34ef7f147b91af2615ca5c88f42ad8fd5d1a00fa63ca66d8e01ec9245a2c7761` |
| Force probe | `048cc33377338189d0940ad072bf72138f98064b4b3ae68726d79edfd818a316` |
| Contact-detail observer | `33fca0ebe615fb7a616574591bc2831ac59220a79d5f7c6879915c42a7b35cd4` |
| Ignored diagnostic script | `03979f87c43d442396179c8969221b4bd36fe75ea3b50334cfc122fb4b71495e` |
| Generated diagnostic JSON | `0204467e9f47ae2c344bec6011e8c83890553985e5bf1607118c5b90e0b9f8dc` |

Existing six-case/matrix and v1/v2 audits were not modified by this diagnosis. No GPU run, model training, simulator state write, or physics/control modification was performed.


The repository also includes `tools/analyze_late_landing_contact.py`, copied from the locked diagnostic with only its tracked-status description updated. Run it with a new `--output` path to preserve the original JSON. The original ignored script and evidence hashes above remain unchanged. This tracked copy does not retroactively change acquisition or analysis provenance.

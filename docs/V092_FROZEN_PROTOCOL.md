# KineticKick v0.1.92 — Frozen preregistration before any new PhysX

Parent released SHA: `4ea0b5d740976428abeec3acb740e19d5a6e4bb0` (v91).
Branch: `version/v0.1.92`; all new evidence is write-once locally on Tang.
The protocol and runner MUST be committed before trials. Every trial log embeds the frozen commit SHA and matrix SHA-256.

## Hypothesis A — Is the native/contact tensor instrumentation calibrated?

v86/v88 saw a falling sphere physically settle on a **GroundPlane**, but the filtered contact matrix and native contact headers stayed empty. IsaacLab documentation and an upstream April 2025 issue describe unsupported GPU **GroundPlane** contact filters in some PhysX versions. This batch uses an independently authored **kinematic rigid box**, explicit collision API on both bodies, and a separate falling rigid sphere. This is a novel instrument positive control, **not a G1 episode**.

Freeze five independent simulations, 180 steps at 200Hz: CPU/off, CPU/on, GPU/off, GPU/on, GPU/on with noncontact box shifted 5m laterally. Positive contact requires actual PhysX runtime ball z and velocity consistent with a supported landing; negative requires the ball to pass the displaced box. Native contact actor/collider manifold requires **both named paths, finite contact position+normal and positive impulse**. Tensor contact is a separate, weaker witness. Instrument off/on kinematic equality is audited without any controller changes. Empty callback/zero tensor **never proves absence of physical contact**.

## Research tasks B/C — No dishonest human learning claims

Independently revalidate v91's original real G1 E/F trials via SHA, all 29 actual qdot, support and 0.4s pre/1.25s post windows, censor recovery upon nonfoot termination. Read-only quality and per-label geometry of six authentic 120Hz CMU clips may supply anatomical phase candidates, **not measured support GRF, professional identity, ground-truth human football impulse, or qualified professional instep power**.

No new G1 actuation is authorized in this calibration batch: the original robot collider geometry and native self-collision pair have not been calibrated. Hence no new G1 shots are scheduled; distinguish physical probe executions from the score denominator. Failing this safety gate must not be bypassed in response to historical 0/4 heldout G1 scores.

## Safe execution and reproducibility

Before each child verify >=20GiB free, CUDA idle, and exclusive flock on the original project GPU mutex. Stop on process crash, timeout or missing runtime evidence. Never terminate another process or modify any old worktree. Preserve all raw G1 PhysX JSON, CMU C3D, model weights, losses and rejected hypotheses locally. **Do not adjust the frozen protocol after looking at outcomes.** New scripts, configs, independent tests and short textual report are the only intended GitHub artifacts. No force push.

Strict downstream real G1 gate retained: >=95% actual right instep share, goal, ball speed >=3.5m/s, all original 29 actual qdot within limits, nonfoot <20N, genuine 14s recovery. Natural motion requires independent heldout human coordination and signed impact evidence; neither ball entry nor a small velocity prediction gain is sufficient.

Prior facts: v91 four original 29DoF CUDA PhysX heldout attempts all failed hard gates. Their unchanged source SHA remains authoritative. No new contact or kicking result should be fabricated.

# v0.1.67 preregistration — falsifying temporal direction of a synthetic 29-DoF phase intervention

**Parent Git SHA:** `d80785015174884246daf6a73b4bb61f7492b236` (v0.1.64).
**Scope:** new independent branch; v0.1.65 and v0.1.66 worktrees remain untouched.
This study asks whether **pre-impact temporal direction** matters at all for physically
executed, safety-gated, wholebody actions. It cannot establish natural football motion,
professional instep impact, human contact/GRF, or a verified GPU wrist–hip manifold.

## Prior results and experimental limitation

v64 source-trained CMU geometric reference had 8/8 candidates with **zero applied steps**;
v63 synthetic fullbody phase was executable but lacked human kinematic contact truth.
This preregistered study is a physically measured **synthetic negative-control**:
original v32, original forward v63 29-DoF synthetic phase, and *same phase
samples reversed in time* during 0.86–1.56s. Reversal conserves the per-joint
absolute-amplitude multiset and all original 20ms support, qdot, ball-distance,
wrist–hip **link-origin proxy**, nonfoot and joint position rejection gates.
The original 29DoF actuator, PD, collision shapes, ball physics, nonfoot termination
and landing recovery are **not changed**.

## Frozen conditions (no outcome tuning)

- Three **new robot reset seeds**, distinct from prior 3107/3119/3137/6407/6419/6431:
  seed 6707, ball offset (+0.011,-0.012)m, delta -0.025m/s;
  seed 6719, offset (-0.014,+0.011)m, delta +0.032m/s;
  seed 6731, offset (+0.020,-0.007)m, delta +0.012m/s.
- One identical scene seed/physics initial state across three policies per context,
  nine original free-root G1 PhysX cases total. **Unit of independence is THREE
  environments, not nine independent subjects or nine independent seeds**.
- `configs/g1_phase_directionality_batch_v067.json` records every exact
  source SHA256, controller SHA256 and write-once relative evidence path.
- Frozen code/contract SHA committed before any new physical case; a 20GiB
  disk protective stop, GPU exclusive `flock`, GPU PID check and a 240s child
  per-case deadline (timeouts are UNSCORABLE, not physical failures).
- Strict original physical evaluation: actual 29-joint qdot limits, legal
  ≥95% measured instep impulse, ball speed ≥3.5m/s, complete goal,
  no nonfoot collision, physically safe recovery to 14s. Cannot score an
  invalid or missing trajectory as PASS/FAIL.

## Primary scientific decision rule

Only claim the *intervention materially executed* if the original 20ms
action audit measures ≥1 nonzero step, and same-seed pre-intervention
(0–0.5s) 25-field original physics is equal. Then report **all negative
strict gates and measured 200Hz coupling**. Compare forward vs reverse
only as *synthetic temporal-direction sensitivity*, not learned human
coordination. If candidate is inert, no efficacy comparison is identifiable.
No post hoc time shifting, safety threshold relaxation or test-case tuning.

## Contact and motion evidence

Audit original 5ms traces: all 29 actual q,dq, qdot limits, CoM, biped
ground-filtered support loads, named nonfoot net contact, link-origin arm–leg
distance and actual duration [contact-.4s,contact+1.25s], with censoring
at early termination. Body origin separation is NOT mesh signed distance,
sensor net force is NOT direct GPU collider pair. No GPU left-wrist/left-hip
contact-path or human GRF/true ball impact certification is attempted here.

## Follow-on hypothesis, deliberately not claimed as achieved

A professional-quality human-foot contact/stance annotated dataset and
multi-person split, a learned temporal whole-body target representation,
a support/mesh-clearance-verified model predictive adaptation over G1 29DoF,
and the same strict original PhysX gates plus blinded naturalness
measurement are needed before claiming natural coordinated shooting.
The v67 comparison is **one prerequisite falsification**, not this final policy.

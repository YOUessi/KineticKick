# KineticKick v0.1.153 — frozen collision identifiability and temporal provenance

Parent: `version/v0.1.151` at `cda3b6e62a9d99c05842d201574c83585b1f2584`. Independent branch; v152 untracked WIP is excluded.

## Research question and falsification

Historical v54 CMU10_05 shows equal and opposite 3D *per-body net contact forces* on the left wrist and left hip at an approximately 5.145s named diagnostic. Does that uniquely identify an actual wrist–hip collider pair? **No.** Distinct contact graphs can yield the exact same body net-force measurement. The study must construct the direct pair graph and at least one graph that has no wrist–hip edge, validate identical net forces via an independently implemented incidence matrix, and retain both hypotheses.

The historical 200Hz nonfoot force spike occurs at 5.135s, while the last measured state is at 5.140s and the named-body diagnostic is at about 5.145s. Never assert those are the same native PhysX tick; never fill missing states. Read original SHA-pinned PhysX JSON and v54 protocol only.

## Frozen 288-model adversarial matrix

24 deterministic nonzero 3D directions × 3 positive scales × 4 distinct physical contact graphs. A graph is *symbolic*, not a reproduced IsaacLab contact. It exists to falsify inference from per-body force equality to native contact identities. Reject malformed source, changed hashes, wrong 29DoF, nonfinite vectors, invalid timestamps, or false certainty. Independent auditor must recompute graph incidence and recorded body force without importing the graph builder.

## Complete pipeline and limits

Record raw historical 6,628 5ms rows for three v54 strategies, support/COM/qdot/torque summaries and last-state censoring for the failed trial; independently check historical evidence. Generate an explicitly labeled CPU sensitivity matrix and failure illustration. Execute CPU regression and negative tests. Keep >30MB raw JSON and PNG under Tang local ignored paths; push only executable code, configs, tests and report. Do not touch occupied GPU, original controls, old experiment JSON, CMU/C3D, or trained weights. This **does not certify** a wrist–hip pair, collision manifold, GRF, human elite instep transfer, or any successful new shot.

## Acceptance

Frozen protocol committed before implementation. 288 graph cases complete; each must reproduce the wrist/hip net force and at least 3 non-direct graph hypotheses. Both primary and independent audit must pass. All history checks SHA-verified, history 2,800+2,800+1,028=6,628, one extra terminal annotation excluded, timestamps/sidecar mismatches flagged rather than interpolated. Explicit GPU refusal if foreign compute context; no new PhysX runs. SHA-verified new `version/v0.1.153` remote push and clean worktree.

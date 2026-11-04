# Near-vertical instep at contact

The user clarified that the kicking toes should point down and the dorsal surface should be nearly vertical. Start with the instep shot, currently about55 degrees toe-down. Keep the physical free-root approach/contact/goal/recovery and existing original29DoF foot. Target a toe axis within15 degrees of downward vertical (at least75 degrees toe-down), with the dorsal normal facing the real goal.

Approaches considered: ankle-only changes are ruled out by the existing0.454rad soft limit; replacing geometry or fixing the root conflicts with the task; full Cartesian takeover already failed earlier. Chosen implementation: coordinate the existing hip/knee/ankle targets and their timing, use measured geometry to choose bounded candidates, and retain the balancing policy. This is an implementation decision within the user's continuing authorization, not an additional user-approved design.

Create version/v0.1.4. Preserve v0.1.3 and the old target/power experiments. No root/ball state writes, new shoe geometry, soft-limit expansion, hidden reset, external force or training. Two diagnostic batches of at most12 trials each; diagnose between them. After two ineffective variants, change mechanism or report the limitation instead of expanding a blind search.

Near-vertical acceptance supplements all original posture/full-chain checks: first-contact and impulse-weighted toe Z≤−cos15°, first and weighted dorsal-forward≥0.90. Estimate conservative foot-envelope clearance from measured foot poses, including bracketing samples near contact; report clearance and reject any sampled near-contact envelope below1cm. Freeze before two new seeds × nominal/±2cm ball X (6 trials/style), require5/6 combined successes and zero falls. Keep inside/outside controls unchanged and regression-check them.

Deliver a separately versioned profile only if physically supported, an audit that cannot call the current55-degree pose vertical, and a matched-camera comparison showing the complete physical sequence. The physical appearance is for user review; these engineering tests do not certify human likeness, general navigation or precise power/aim.

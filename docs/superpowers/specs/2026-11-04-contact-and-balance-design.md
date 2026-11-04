# Purpose of follow-through: ball output and whole-body balance

The user's correction supersedes the v0.1.5 proxy objective. Follow-through should support useful force transfer and controlled ball trajectory while allowing balanced motion through approach, strike, release and recovery. Forward travel is descriptive, not an objective to maximize. Longer contact alone does not establish greater useful impulse; once separated, further leg motion cannot add impulse to the departed ball.

Implement a small diagnostic layer first, using existing recorded evidence. Preserve v0.1.5's historical audit unchanged so earlier failures remain reproducible. Do not relabel failed controls as successful by changing thresholds.

Alternatives: more fixed-frame pose tuning repeats a failed mechanism; immediate contact-feedback control is premature because existing foot-only logs show ball motion before first recorded foot force. Chosen engineering step: separate force-bearing episodes, measure normal-only directional impulse and observed ball output, and expose transient balance proxies. Then investigate all robot-body/ball contacts and synchronization, with real COM and ground-contact telemetry, before altering controls.

This is a measurement and objective correction, not a new working controller. No GPU parameter search, no default motion change, no new source fingerprint changes to the simulator/controller in this revision.

Positive-force episodes are contiguous 5 ms force-bearing bins for the kicking foot. A zero-force sample separates episodes, but is not proof of geometric separation. Zero-force manifold entries do not add duration. Durations are sampled integration durations, not exact continuous contact times. Do not use a fixed 80 ms cutoff to classify continued contact as a second hit. Missing timestamps/invalid values reject measurement; truncated episodes stay open.

Normal impulse is computed both as sum of magnitudes and a world vector reconstructed from per-point normal and foot quaternion. It excludes tangential force; neither field is energy or the full cause of ball motion. Report ball motion before recorded foot force and velocity at the first subsequent zero-force sample; do not assert true release or causality from these alone.

Balance diagnostics expose root height/tilt/speed, net ankle vertical force and nonfoot force over a fixed observed window with coverage status. These are not true COM, ground-filtered support geometry, CoP/ZMP, or a stability certificate. Desired ball speed/direction, legitimate touch and balanced completion remain the final functional outcomes. Joint smoothness/swing length are secondary descriptors.

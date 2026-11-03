# Targeted free-root shooting, v0.1.2

## Outcome and scope

Give a stationary ball a requested horizontal aim position in the fixed goal and a requested shot strength. Retain the complete approach, physical foot contact and standing recovery of v0.1.1 for inside, instep and outside shots. This first targeted version controls horizontal placement of low shots, not arbitrary elevated corners or the final resting place of the ball. The user was asked the goal-point versus resting-place distinction; pending a correction, use the former.

The default ball remains (0.14, -0.93) m. The physical goal stays centered at (0, -3.7) m, with forward normal (0, -1), width 1.6 m and height 1.1 m. Changing an aim command must never move or rotate this goal. Target x is a world-frame horizontal coordinate on that fixed plane. Only a measured feasible range will be advertised. Start calibration with left/center/right targets within x=±0.35 m.

Strength is an outgoing horizontal ball speed in m/s, measured 0.10 s after first foot contact by interpolation of 5 ms samples. It is not an injected force and not the maximum speed observed anywhere in the episode. Offer low/medium/high presets plus numeric speed only where supported by calibration; explicitly show each style's feasible speed range. Probe whether a shared range is possible, but do not invent one. Require at least three meaningfully separated speed levels (adjacent levels ≥0.30 m/s) and distinct lateral placements (adjacent targets ≥0.25 m) before claiming both controls work.

## Control architecture

Reuse the pinned recurrent soccer policy and its normal joint position actuators. Separate the policy's virtual aim input from the physical goal. Calibrate virtual aim and bounded strike trajectory parameters against actual physical crossings and outgoing speeds. Prefer the smallest stable controller, with a finite validated command envelope; refuse unsupported requests rather than extrapolating or silently clamping. Continuous interpolation is permitted only with intermediate command validation. Do not use online future outcome information to choose a controller for a trial.

The existing StrikeResidual is the starting actuator interface. Calibration may introduce a smooth strike trajectory parameter (e.g. joint orientation or swing timing) after measuring its effect. The exact parameters, feasible range and lookup coefficients are implementation decisions determined by pilot evidence; freeze them in a versioned configuration before held-out acceptance. Keep the old configuration and old command defaults compatible.

## Evidence and success

Add an independent target metric that records the first forward crossing of the ball center at the fixed goal plane, interpolated position/time/velocity, absolute lateral aim error, and speed measured at first contact +0.10 s. Retain the existing full-sphere goal criterion, real surface/impulse classification, approach and continuous ≥1 s standing criteria. A targeted success requires all of them, target error ≤0.15 m and speed error ≤max(0.20 m/s, 12% of requested speed). A miss must keep its actual crossing even outside the aperture; missing/incomplete evidence is a failure.

Separate calibration data from acceptance. Freeze a 3 styles × 3 aim points × 3 speed levels matrix, then run it at two new seeds with the existing ±1 cm root xy, ±0.01 rad yaw and ±0.005 rad joint reset perturbations. Require ≥80% complete targeted successes per style and per strength level across the matrix, and report every failure. If the full range cannot meet this criterion, do not declare completion: record the measured limit and the remaining work. No selection of only successful reruns as acceptance.

## Boundaries and validation

- Work on `version/v0.1.2`, preserve previous branches and local-only assets.
- Root stays free. No physical root/ball state writes after initialization, no external kick impulse, no automatic reset hiding a fall.
- One simulator/GPU process at a time. Initial probes: at most two 24-trial batches to identify sensitivity. After two ineffective variants of one approach, inspect contacts/kinematics and change the hypothesis before another batch.
- No training job, cloud spending, dependency upgrades or weight upload.
- Test geometry, speed timing, invalid commands, stale/mismatched calibration and acceptance completeness. Replay physical traces rather than trusting summary flags.
- Provide CLI/live controls, measured report and a short actual simulation demonstration. Publish source and report on the version branch; generated media and raw evidence remain local.

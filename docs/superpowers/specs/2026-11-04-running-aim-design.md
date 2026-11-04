# v0.1.8 physical running and controlled shooting

User scope: continue repeated flight-phase running, outside-foot full sequence, and goal-point/low-medium-high control for the new motions. Branch: version/v0.1.8 from verified v0.1.7.

Keep the original free 29DoF G1, original actuator limits, physical 0.43 kg ball and fixed goal centre [0,-3.7]. No post-initialization robot/ball state writes, reference wrapping, fixed root or artificial impulses. Running means multiple alternating stance/flight cycles after initial settling, never initial free fall. Three post-settle flight intervals >=20 ms separated by left and right support, forward progress >=1 m and no fall/nonfoot ground contact form the first running gate; connecting the shot additionally requires the existing full-sequence gate.

Outside uses measured world-frame foot/ball geometry and support timing, not a copied joint pose from a different base orientation. All promoted modes require proper first foot contact, intended contact surface, actual goal crossing, no illegal contact, and final >=1 s continuously stable recovery. Mode dominance is kinematic hip/knee contribution, not work.

Aim is the ball centre crossing the fixed y=-3.7 plane within 0.15 m of the requested x. The physical goal plane is identical for shot and target trackers. Speed is measured 0.10 s after the first significant foot-ball contact. Low/medium/high must have >=0.30 m/s measured separation at each target and seed, and meet the declared speed tolerance. Calibration and holdout trials are separate, with immutable source fingerprints. An unavailable combination must be rejected, never silently substituted.

Three bounded investigations: (1) same-morphology LAFAN RUN reference with the existing 29DoF actor; (2) outside foot geometry on a short approach; (3) local sensitivity of goal observation and strike timing. At most two reasoned batches per unchanged failure class; diagnose before changing approach. Do not start open-ended training or parameter sweeps. Record failed attempts honestly and retain v0.1.7 as a usable baseline. Publish previews only from completed physical trials, labelled with actual accepted scope.

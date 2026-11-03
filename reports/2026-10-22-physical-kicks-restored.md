# KineticKick / Tang: Physical soccer kick demonstration (2026-10-22)

After the user reported no kicking, the running 29-DoF kinematic walking reference viewer was stopped. A single Isaac Sim GUI now runs tools/show_g1_soccer_real_kicks.py, title: KineticKick | PHYSICAL SOCCER KICK LAB.

## Actual physics verified
- One original G1 37-DoF with ROOT FIXED, no added football shoes, and one soccer ball (PhysX RigidObject, mass 0.43 kg, radius 0.11 m).
- Ball receives velocity ONLY from the foot/ball physical collision, never direct post-hit velocity injection.
- Original G1 foot shape: dorsal top, medial and lateral contact surfaces were inferred by simplified foot-cuboid geometry, not measured precise PhysX contact locations.
- The pitch is grass with line markings; training goal and net are present; the ball has one attached 12-pentagon, 20-hexagon black/white visual skin with no extra collider.
- Medium physics tests: instep_top (first force 203.81 N, ball peak horizontal speed 4.170 m/s, goal crossing inside), medial (92.64 N, 1.615 m/s, inside), lateral (96.40 N, 1.964 m/s, inside). Initial ball positions differ across techniques.
- There were over 30 logged successful shot/crossing pairs, however these repeat the same fixed configuration and are NOT independent random seeds.
- Goal crossing is determined by the simulated ball position passing the known target plane, not an image detector.

## Current GUI and controls
Default: AUTO: 3 KICKS cycles instep_top, medial, lateral. User can manually choose each kick, LOW/MED/HIGH power, WHOLE FIELD versus CLOSE FOOT camera, SLOW or NORMAL speed, and REPLAY PHYSICAL SHOT. WHOLE FIELD was clicked and real view inspected: G1 and blue goal visible, the UI indicates PHYSICAL impact and INSIDE goal.
No files have been copied to Mac. Tang paths:
- tools/show_g1_soccer_real_kicks.py
- reports/show_g1_soccer_real_kicks.log
- reports/live_soccer_physical_shots.csv
- reports/real_soccer_kick_live_audit.json
- media/g1_real_physical_kick_live.jpg
- media/g1_physical_kick_wide_goal.jpg

## Scientific limitations and research next step
This is a fixed-root scripted collision prototype. It is NOT a dynamically balanced, autonomously walking, human-motion learned soccer policy. Previous free-root attempts with raw scripted limb command caused early accidental contact or missed the ball, and the robot fell.
Next research gate: train free-root humanoid with controllable support-foot location, whole-body balance, coordination and follow-through, then connect it to goal-aware walking. Avoid claiming a natural human-like kick until real physics tests meet the criteria.

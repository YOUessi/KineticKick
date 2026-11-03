# KineticKick Stage 3A — Soccer Walk-up (Tang, 2026-10-22)

Executed directly on Tang via connected Desktop Commander. Previous claim that Tang could not be controlled from this chat was incorrect.

## Files and actual runtime

- tools/soccer_walkup_scene.py: a marked green practice soccer pitch; netted three-dimensional goal; one physical football.
- tools/show_g1_soccer_walkup_reference.py: existing validated 29DoF LAFAN1 straight walk reference shifted in world X to land at a fixed ready position.
- tests/test_soccer_walkup_stage3a.py: five new tests. Full project: 36 tests passed.
- GUI title: KineticKick | SOCCER WALK-UP A -> BALL
- Current GUI log: reports/show_g1_soccer_walkup_reference.log
- Captured on Tang: media/g1_soccer_walkup_stage3a_first.jpg
- Smoke-check log: reports/soccer_walkup_headless_smoke.log

## Scene / task

Football: one PhysX RigidObject at (4.8,0,0.11) m, radius 0.11m, mass 0.43kg. It has classic 12 black pentagon + 20 white hexagon noncolliding visual panels (no extra ball body).
Ready target: (4.2,0)m, 0.6m behind the stationary football.
Goal center: (8.8,0)m, 3.0m wide 1.7m high training goal, visible posts, net.
There are no additional marker spheres pretending to be soccer balls.

Default NORMAL walk begins at x=-0.4511m and finishes at x=4.2m. At this stop point, observed reference target error ~0.00002m, ball separation 0.6000m. Log event: SOCCER_WALKUP_READY_REACHED.
SLOW/NORMAL/BRISK/RUN reference motions each have a different source trajectory length; the per-mode initial A is adjusted to ensure all END at the same ready target and the ball and goal stay fixed. Joint and timing source data are not changed. At arrival the actor holds its last pose. There is NO automatic looping.

Available controls include SLOW WALK, NORMAL WALK, BRISK WALK, RUN, REPLAY TO BALL, PAUSE, CONTINUE, FOLLOW G1 and FULL PITCH.

## Research boundaries

This is human-derived 29DoF motion-kinematic playback, not a physics-driven, dynamically balanced or trained goal-conditioned G1 locomotion policy. The ball has Physics but the human reference G1 render actor has no collision or actuator. No foot-ball impact or shooting occurs at Stage 3A. Independent 29DoF PhysX articulation and imitation learning remain the next major research step.

All generated files are on Tang only; none copied to Mac.

## Live GUI interaction verification (Tang)
- GUI live process after final restart: 4019543; status tracked via reports/show_g1_soccer_walkup_reference.log.
- Tested SLOW WALK button by actual X11 click: logged SOCCER_WALKUP_UI_MODE_CLICK WALK SLOW and completed at ready (4.2,0), ball exactly 0.6m ahead.
- Tested REPLAY TO BALL button by actual X11 click: logged SOCCER_WALKUP_UI_REPLAY_CLICK WALK SLOW and restarted the clip from A.
- Switched back to NORMAL WALK through an actual click: logged SOCCER_WALKUP_UI_MODE_CLICK WALK NORMAL and a new clip start from x=-0.4511m.
- Full test suite after Stage 3A: 36 tests passed.

# KineticKick v0.1.120 — preregistered original G1 AO/AP controls

Parent SHA `b57b855433b3fa6f6f266ee565f7e959b333c288`; immutable config `configs/v120_g1_frozen.json`. Original 29DoF free-base G1, original v32 and CMU10_02 / CMU10_05 controllers, original on/off Tensor observer, simulator 200 Hz, intended 14 s recovery. No new learned actuation or data-driven scene tweaks.

Unseen scenes AO `ball_offset_xy=[0.041,-0.038]`, `velocity_delta=+0.004`, `seed=3119`; AP `ball_offset_xy=[-0.036,0.073]`, `velocity_delta=-0.004`, `seed=3167`. Exactly seven runs: AO v32 OFF/ON, CMU02 OFF, CMU05 OFF, AP v32 OFF, CMU02 OFF, CMU05 OFF. The AO OFF/ON duplicate is not an independent statistical sample. All seven are original locked, unchanged control, not a new learning policy.

Freeze protocol before any physical simulation. GPU serial under shared nonblocking global lock, never co-run other workloads; child-only 140 s watchdog, halt if uncertified G1, parity mismatch or disk less than 20 GiB. Use exclusive write-once raw JSON/log/audit. Verify all original ONNX/URDF/reference/expert file hashes and event records. Exact parity compares ALL common 200Hz physical states, shot metrics, contact and termination, not rounded summaries.

Strict pass requires legitimate ≥95% instep impulse fraction, actually crossing the goal mouth, ball-speed gate, **every original qdot** bound, contact safety, and genuinely witnessed recovery; score alone cannot convert a FAIL. Early termination means censored recovery, not demonstrated stand-up. Human CMU data are only unqualified kinematic phase proxies. No certified professional foot/ball/GRF/CoP truth is available.

Contact manifold remains unverified until independently established native collider-pair positive control. Do not infer wrist–hip self-contact solely from equal-and-opposite net forces or zero native headers. The new whole-body method must condition full pre-support, pelvis-trunk-leg-arm sequence AND post-impact recovery; an old 1mrad weak reference is not a claimed scientific solution.

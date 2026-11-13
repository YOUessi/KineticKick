# v0.1.144 pre-registered human temporal coordination & strict physical evidence separation

Parent is the clean immutable v142 SHA `a0fe4eb41ef026d89e233f4443621a7008b79bf5`, built as an independent parallel branch while v143 is under active development. **No files or branch of v143 are modified.**

### Hypothesis (weak, falsifiable)
Six actual 120 Hz CMU C3D/AMC soccer-like clips may expose body-segment orientation-rate *proxies* around an **uncalibrated ball-marker launch onset**, potentially useful for an exploratory human whole-body temporal prior. These are NOT calibrated angular velocity, measured ball-foot impact, elite-quality instep kicks, synchronized GRF, muscle power, joint torques or verified G1 successful shots. In particular the two strongest launch cues are **both from subject 10**, making subject-disjoint event-model evaluation impossible.

### Frozen analysis protocol
- Recheck SHA256 of original v50 quality audit, source manifest, and original v53 strict matrix; verify all six original C3D+AMC SHA256 bytes against manifest independently, C3D 120Hz and marker label uniqueness. No interpolation across absent marker samples.
- Use all six files as denominator; **only** the two preidentified strong continuous high-speed ball-marker clips (10_02, 10_05) may enter event-aligned pre-onset timing descriptors. Do NOT upgrade the short-burst clip to an impact event.
- Construct 12 named real human body-segment direction vectors from anatomically labeled C3D points. Compute unsigned segment orientation-change rate from normalized directions at t−2 and t+2 (angular displacement/4 frames); record valid coverage, median, p90, pre-onset maxima and their lag to the preselected ball-marker onset. This quantity is *not* anatomical joint angular speed.
- Window is exactly [anchor−0.6s, anchor) with >=50 valid samples; no post-onset requirement or inferred missing samples. Other four clips are unanchored; count them without fabricated event times.
- Run an independently constructed synthetic constant-vector / phase-shift / invalid-data unit-test suite. Report an explicit negative-control **reverse-pre-onset-frame-order sign reversal of peak temporal lag**; do NOT claim reverse ordering represents plausible human motion or that lag proves causal kinetic chain.
- Predeclare evidence label requirements: true foot-surface, impact impulse, force plate/CoP, actor/collider pair and 14s G1 recovery must remain UNKNOWN until actually measured; missing field fails closed.
- Independently SHA-confirm original v53 12 physical file outputs from its immutable summary, require all frozen 12 policy cells, 0/4 strict per policy and no accidental trial promotion. Recheck v54 6,628 actual history frames with v142 source lock. These are historical runs, **new PhysX count=0**.
- CPU only; if any other GPU process holds context or disk <20GiB, do not simulate. All raw files and large JSON/plots remain ignored on Tang; only code, protocol, tests and concise text reports can be pushed.
- Do not fit or tune human motion model on the two clips and then claim generalization. No professional power-chain learning until qualified multi-subject data and real impact/GRF supervision exists.

### Research-to-control next interface
A future continuous whole-body action policy must learn segment phase and arm/trunk/hip coupling from qualified humans and expose original29 q,dq, torque, contact, CoP/GRF, self-collider and 14s recovery witnesses for real physical admissions. Today's exploratory temporal proxies MUST NOT authorize actual torques or imply 95% instep impulse, goal or >=3.5m/s ball speed.

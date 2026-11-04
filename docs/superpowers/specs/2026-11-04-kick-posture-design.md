# Free-root kick posture correction (v0.1.3)

The user confirmed goal-plane targets and low/medium/high strength, then prioritized the visibly incorrect instep and outside-foot postures compared with the older fixed-root demonstration. Correct posture before further power tuning or training. The robot must still approach, physically strike and recover standing with a free root.

## Observed problem

At first contact in the verified v0.1.1 video, the instep foot's top axis is world (-0.280,-0.394,0.875): it faces mainly up, while the goal is approximately -Y. Its contact is on the medial upper edge (local y≈+0.023 m). The fixed-root dorsal experiment reported a forward top-axis component of 0.87–0.89.

The free outside shot has measured right hip yaw≈135°, pelvis yaw≈-154° against a goal heading≈-93°, and a contact near the front edge (local x≈0.124 m). The fixed-root outside preset used hip yaw1.5rad (86°). Its different37DoF cuboid foot is not the29DoF cylinder-foot asset, so copying joint angles is insufficient.

## Approach and boundaries

Keep the actual29DoF asset, pretrained balance policy, foot geometry and real contact evidence. First reconstruct desired contact orientation and leg configuration from measured poses, using the existing URDF kinematics. Test bounded changes to strike pose and preparation of the support leg/heading. Do not repeat broad aim/power lookup fitting, use root fixation, directly write ball speed, add shoes or weaken the contact definition to make a new pose pass.

Directly copying fixed-root motion is rejected because it lacks support/balance adaptation. Full Cartesian pose takeover was already unstable in v0.1.2. Full policy training is deferred until this diagnostic step identifies a usable posture target; no training/GPU spending beyond local bounded physical trials is assumed.

The geometric targets below are engineering criteria for this correction, not proof of human likeness. Final evidence includes matched-camera before/after actual simulation and the user's ability to judge appearance.

## Measured posture criteria

- Instep: first contact and majority impulse on the existing instep classification; foot top axis dot horizontal shot direction ≥0.75; toe-axis world Z≤-0.60. Report medial offset of contact; prefer |local y|≤0.018m and x≤0.11m so an inner/front edge does not stand in for central dorsal contact.
- Outside: existing outside classification; actual hip yaw magnitude ≤1.90rad at first contact (versus2.36rad before); pelvis heading error to the shot ≤45°; outer foot axis dot shot direction ≥0.75; prefer local x≤0.11m to avoid the front edge. Do not reinterpret a top/inside contact as outside.
- Both: keep the v0.1.1 full-chain criteria, no fall, actual first-contact timing, individual manifold points, and ≥1s contiguous final standing. Report any trade-off in goal success or recovery rather than hiding it.

Freeze a candidate before a small perturbation check: each changed style at three nearby ball positions (nominal, x−0.02m, x+0.02m), two new seeds, same existing root/joint reset jitter. Require at least5/6 complete physical successes and no falls per style, plus posture criteria. Keep inside-foot baseline available. Accurate targeted power remains unresolved until independently revalidated.

## Execution and evidence

Work on version/v0.1.3, preserve all previous branches/profiles and local evidence. Audit stored traces first; write tests that reject the current wrong poses and accept a geometric reference without depending on requested labels. Initial physical probe is at most24 trials; inspect actual pose/contact/root response before a second bounded probe. After two ineffective variants of a mechanism, stop that mechanism and diagnose, rather than enlarging random searches. One simulator at a time. Render only physically viable candidates. Do not claim the user's posture issue fixed if the desired posture remains unattained or unstable.

Motion references: the local fixed-root configs/reports and the actual free-root traces are primary comparison evidence. Coaching sources distinguish laces/top-foot and side-foot finishes; they do not prescribe robot joint angles. Relevant primary source: https://learn.englandfootball.com/articles-and-resources/coaching/resources/2022/Different-finishing-techniques-in-football .

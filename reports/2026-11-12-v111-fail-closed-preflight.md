# v0.1.111 fail-closed negative: missing local ignored human-motion source

Frozen experiment parent was GitHub v107 SHA 62f4d1f3715cc6e74352cdfdd6d40ce10e3b5584; protocol SHA 13f36495e0710c578298e45f9cfc31856a3c17f5. Tang online, GPU idle, 34GiB free.

Ten preregistered runs: W/X v32+CMU02 OFF/ON and Y v32+CMU02 OFF, across three new seeds. **One subprocess was attempted, zero certified original G1 PhysX episodes, zero 200Hz steps.** The first source-provenance read raised FileNotFoundError before physics started: third_party/HumanoidSoccer/motions/soccer-standard/soccer-standard-001_right.npz missing in new worktree.

Write-once raw evidence: reports/v111_W_v32_off.log, reports/v111_W_v32_off.audit.json and reports/v111_original_g1_batch.audit.json (all local, ignored). Following nine runs cancelled fail-closed, no overwrites. 15 focused pretests passed. This is an infrastructure negative, not 0/10 shooting success rate. No learned human-like policy or native collider manifold certified.

Next independent branch must restore ignored source assets via local read-only symlinks and verify all provenance BEFORE starting GPU jobs; never rerun/overwrite the v111 failed run.

# KineticKick: One-command CMU human soccer → G1 semantic bridge

This research workflow does not train or deploy a new robot policy. It
qualifies six real CMU 120 Hz soccer clips and compares the two clips with
observed outgoing ball-marker motion against two frozen, real original
G1/PhysX traces.

## Environment

Tang Ubuntu, IsaacLab Python 3.10, torch, scipy, numpy and ezc3d 1.7.2.
Additional dependency pin: requirements/cmu-mocap-v050.txt.

    cd /home/you/projects/kinetickick
    /home/you/conda_env/env_isaaclab/bin/python -m pip install -r requirements/cmu-mocap-v050.txt

The original 14 CMU source files (6 C3D, 6 AMC, 2 ASF) and SHA manifest
are kept on Tang under:

    data/research_human_kicks/cmu_soccer_120hz_v050/raw/{10,11}
    data/research_human_kicks/cmu_soccer_120hz_v050/original_source_manifest.json

These and the original G1/PhysX sources are intentionally NOT in Git. The two
required original G1 14-second physical sources are:

    reports/v048_preimpact_new01_original_v32.json
    reports/v048_preimpact_new01_learned_preimpact.json

Run the complete pinned source-qualification, cross-morphology and test chain:

    PYTHONPATH=tools /home/you/conda_env/env_isaaclab/bin/python tools/run_cmu_g1_research_batch_v051.py --run

Or verify all already-present pinned source files without recomputing:

    PYTHONPATH=tools /home/you/conda_env/env_isaaclab/bin/python tools/run_cmu_g1_research_batch_v051.py

This runner never overwrites original/derived evidence, retrains a controller,
relaxes G1 joint limits, silently re-downloads from an insecure source or
pretends that the inferred CMU ball marker motion onset is a real contact label.

## Batch outputs

| Stage | Source script | Tang output |
| --- | --- | --- |
| CMU original source download/SHA | tools/fetch_cmu_soccer_v050.py | raw + original_source_manifest.json |
| Human C3D/AMC, ball visibility and semantic pose | tools/audit_cmu_soccer_v050.py | processed_v050 / 6 NPZ + quality_audit.json |
| Human ↔ G1 6-cosine URDF-FK comparisons | tools/cmu_g1_semantic_bridge_v051.py | cmu_g1_cross_morph_v051 / 20x6 NPZ + cross_morphology_audit.json |
| Optical ball and pose quality figures | tools/plot_cmu_soccer_v050.py | media/v050_* |
| Cross-morphology phase curves | tools/plot_cmu_g1_bridge_v051.py | media/v051_* |
| Real original source & numerical-FK tests | tests/test_cmu_soccer_v050.py + tests/test_cmu_g1_bridge_v051.py | Passed in IsaacLab environment |
| Frozen protocol & text-only research results | configs/cmu_soccer_quality_v050.json, configs/cmu_g1_cross_morph_v051.json | reports/2026-11-10-v050-*, reports/2026-11-10-v051-* |

## Ground-truth limitations

Six clips (3,358 human 3D frames, two people) have measured or fitted 120Hz
body point positions; exactly two have continuous fast tracked ball-marker
movement. The three WEP markers do not define a certified physical soccer ball
center. Missing ball markers remain missing, with no invented positions.

The human t=0 is an INFERRED optical ball-marker motion onset, while G1 t=0 is
a PhysX force-based right-foot/ball contact. These events are not verified
synchronized human/robot impact, and the six feature cosines are not qualified
professional human muscle power, naturalness or anatomical joint angles.

The CMU website states research and product use is permitted but the original
mocap must not be resold directly. Publications should credit CMU and NSF
EIA-0196217. Original CMU HTTPS issuer-chain verification failed on Tang, so
the v50 fetch script records its insecure transport; local SHA256 checksums
are not publisher-signed authenticity guarantees.

Keep frozen v32/G1 controls untouched until genuinely calibrated human
high-speed instep motion, contact force, ball spin and support/GRF data become
available and a new independent PhysX policy benchmark is registered.

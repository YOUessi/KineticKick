# KineticKick v0.1.94 — Pre-registered PhysX contact-view API repair

Parent SHA d69904df2bd2c0f833eb384520a63f4867334623 (v93 committed protocol). Preserve v93 untracked evidence as-is. v93's RigidContactView.count AttributeError is **not** a physics failure.

## Frozen hypotheses
H1: Documented RigidContactView.sensor_count/filter_count and sensor_paths/filter_paths will fix v93's pre-step API mismatch. H2: A real 200Hz dynamic sphere settling onto a kinematic box produces a pair-filtered tensor force and optionally contact points/normals; a box shifted 5m must not. Filtered tensors are weaker than original native collider manifold truth. H3: OFF (no contact report API) and tensor/native ON (contact API present) must retain identical 180-step sphere z/vz per device. Any change is reported, never suppressed. An app-close marker is an independent lifecycle signal.

## Pre-registered independent 180-step x 5ms cases
1 CPU pose OFF, 2 CPU tensor ON, 3 GPU pose OFF, 4 GPU tensor ON, 5 GPU native ON, 6 GPU 5m-separated tensor ON. Fresh process, new write-once JSON and log per case, original global flock and CUDA exclusive, >=20GiB free. Stop only on true child crash, missing real dynamics or disk/GPU guard. The source, config, and this protocol are committed before first physical step; never change them based on observed results.

Native actor and collider paths, finite contact point+normal and positive impulse are mandatory for original G1 direct pair certification. Tensor sensor/filter paths do not certify a native original G1 collider pair. If native not certified, **no new original G1 action**: zero newly scheduled G1 shots. Do not claim real human explosive instep learning from CMU optical phase-only C3D without verified elite player/GRF/instep impulse.

Original G1 strict gates unchanged: right instep impulse >=95%, goal, ball speed >=3.5m/s, all 29 measured qdot, nonfoot <20N, true 14s recovery; natural movement needs independent expert motion validation. Original historical negative data retained without tuning. Post batch independently hash all sources, test complete suite, report true physics counts and G1 denominator, push clean independent version branch, no force.

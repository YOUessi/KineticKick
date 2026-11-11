# v0.1.81 frozen original-G1 PhysX filtered rigid-pair tensor observation
2026-11-11; parent d5e1bd944206dc7c10262dacaee39392fe94827d (version/v0.1.80).

## Objective and scientific status
v80's native get_contact_report and native event subscription returned zero headers despite 25.890153 N equal-opposite left wrist/left hip sensor forces. Zero native headers do not prove absence of self-collision. Test whether an existing PhysX tensor simulation view can provide a positive contact-point/normal/force witness filtered by two rigid-body names; independently check a known right-foot to ball positive contact. This is diagnosis, not learned human coordination.

## Frozen comparison
Three original unchanged controls v32 / CMU10_02 / CMU10_05, fixed original seed 1201, each run (off, tensor) with unchanged original 29DoF G1, free root, 0.005s physics. Six real runs, fixed order, unmodified original shot/recovery controls and hard gates. Tensor-enabled run uses create_rigid_contact_view at scene reset to request wrist->hip and hip->wrist filtered pair data; contact positions/normals/impulses are read after physical update. A separate known ball->right-foot contact is a positive control. Creation can theoretically change how PhysX creates reports; only exact off/on whole-state parity of original physical fields establishes tested noninterference. No USD contact report API authored, no collision mask or actuator modifications.

Tensor contact data by filter may prove direct rigid-body pair if complete, and stage collider inventory can identify plausible meshes, but must not be called native collider/manifold evidence absent actual actor0, actor1, collider0, collider1 native header and contact geometry.

## Safety and audit
Run only if Tang online, >20GiB, no other CUDA process, global nonblocking flock acquired; 180s child watchdog. SHA lock controller/config/source/old-v54 physical evidence. All JSON/log/NPZ/media local and write-once; only protocol, code, tests and text report to GitHub. Strict shooting gates: instep impulse >=95%, ball speed proxy >=3.5m/s, goal required, all original 29 real qdot, nonfoot sensor 20N, full 14s recovery; do not call early termination a PASS. Whole-body dynamics 200Hz pre 0.4s/post 1.25s and right censoring must be preserved. If view constructor fails or removes parity, stop, report and do not infer no contact. No tuning on previously seen failures; human C3D without force channels is geometry-only.

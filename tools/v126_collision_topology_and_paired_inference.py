#!/usr/bin/env python3
"""v126: offline G1 contact topology and exact *scene-paired* effect falsification.

Read-only: no IsaacLab import, motor commands, USD modifications, or new PhysX claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v126_collision_topology_and_cluster_inference.json"
PROTOCOL_COMMIT = "8e489a00506c06441cf30885ab6a71578e8da261"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def frozen_config():
    tracked = subprocess.check_output(
        ["git", "-C", str(ROOT), "show",
         PROTOCOL_COMMIT + ":configs/v126_collision_topology_and_cluster_inference.json"]
    )
    if CONFIG.read_bytes() != tracked:
        raise ValueError("FROZEN_PROTOCOL_CHANGED")
    cfg = json.loads(tracked)
    if (cfg["version"] != "v0.1.126" or cfg["no_physics_actuation"] is not True
            or cfg["no_native_manifold_inference_from_urdf"] is not True
            or cfg["no_new_physics_rerun"] is not True):
        raise ValueError("UNSAFE_FROZEN_CONFIGURATION")
    if cfg["source_report_sha256"] != "2a4348cd0c5bfe3878c255f23d181d1cf224a32d5a4df5cc83a2008eac9fc8a3":
        raise ValueError("SOURCE_REPORT_LOCK_CHANGED")
    return cfg


def parse_urdf(path: Path) -> dict:
    root = ET.parse(path).getroot()
    if root.tag != "robot":
        raise ValueError("NOT_A_ROBOT_URDF")
    links = {}
    for element in root.findall("link"):
        name = element.get("name")
        if not name or name in links:
            raise ValueError("MISSING_OR_DUPLICATED_LINK")
        collisions = []
        for collision in element.findall("collision"):
            geometry = collision.find("geometry")
            if geometry is None or len(list(geometry)) != 1:
                raise ValueError("MISSING_OR_AMBIGUOUS_COLLISION_GEOMETRY:" + name)
            shape = list(geometry)[0]
            collisions.append({
                "kind": shape.tag,
                "radius_m": shape.get("radius"),
                "length_m": shape.get("length"),
                "size_m": shape.get("size"),
                "mesh_name": Path(shape.get("filename", "")).name if shape.tag == "mesh" else None,
                "origin_xyz_m": collision.find("origin").get("xyz", "0 0 0")
                  if collision.find("origin") is not None else "0 0 0",
            })
        links[name] = collisions

    joints = []
    child_parent = {}
    edges = {link: [] for link in links}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError("JOINT_WITHOUT_LINKS")
        a, b = parent.get("link"), child.get("link")
        if a not in links or b not in links or a == b or b in child_parent:
            raise ValueError("INVALID_URDF_JOINT_TOPOLOGY")
        child_parent[b] = a
        edges[a].append((b, joint.get("name")))
        edges[b].append((a, joint.get("name")))
        joints.append({"name": joint.get("name"), "type": joint.get("type"),
                       "parent": a, "child": b})

    disabled = []
    for node in root.findall("disable_collisions"):
        disabled.append((node.get("link1"), node.get("link2")))
    return dict(links=links, joints=joints, edges=edges, disabled_pairs=disabled)


def shortest_joint_path(edges: dict, src: str, dst: str) -> dict:
    if src not in edges or dst not in edges or src == dst:
        raise ValueError("MISSING_OR_IDENTICAL_TARGET_LINKS")
    queue = deque([(src, [src], [])])
    seen = {src}
    while queue:
        node, link_path, joint_path = queue.popleft()
        for nxt, joint in edges[node]:
            if nxt in seen:
                continue
            if nxt == dst:
                return dict(distance_joints=len(joint_path) + 1,
                            links=link_path + [nxt], joints=joint_path + [joint])
            seen.add(nxt)
            queue.append((nxt, link_path + [nxt], joint_path + [joint]))
    raise ValueError("DISCONNECTED_TARGET_LINKS")


def static_collision_audit(cfg: dict, src: Path) -> dict:
    if git_head(src) != cfg["urdf_source_commit"]:
        raise ValueError("G1_SOURCE_HEAD_DRIFT")
    urdf = src / cfg["urdf_relative_path"]
    if sha256(urdf) != cfg["urdf_sha256"]:
        raise ValueError("G1_RESOLVED_URDF_SHA_DRIFT")
    robot = parse_urdf(urdf)
    n = sum(j["type"] not in ("fixed", "floating") for j in robot["joints"])
    if n != cfg["expect_29_actuated_dof"]:
        raise ValueError("G1_IS_NOT_ORIGINAL_29DOF")
    a, b = cfg["focus_links"]
    if not robot["links"].get(a) or not robot["links"].get(b):
        raise ValueError("MISSING_PHYSICAL_COLLISION_PRIMITIVES")
    path = shortest_joint_path(robot["edges"], a, b)
    if path["distance_joints"] <= 1:
        raise ValueError("TARGETS_ARE_ADJACENT_EXPECT_NONADJACENT")
    # Informational only; source Python declaration is not a USD-runtime witness.
    src_g1 = src / "third_party/HumanoidSoccer/source/whole_body_tracking/soccer/robots/g1.py"
    source = src_g1.read_text()
    declared = bool(re.search(r"enabled_self_collisions\s*=\s*True", source))
    return dict(
        urdf_sha256=sha256(urdf), robot_name="g1",
        link_count=len(robot["links"]), joint_count=len(robot["joints"]),
        actuated_dof=n,
        target_link_collision_geometry={name: robot["links"][name] for name in (a, b)},
        topological_connection=path,
        explicit_urdf_disabled_pairs=robot["disabled_pairs"],
        self_collision_enabled_in_readonly_python_source=declared,
        python_source_sha256=sha256(src_g1),
        urdf_pair_has_two_declared_collision_shapes=True,
        usd_collision_api_verified=False, usd_actor_collider_ownership_verified=False,
        effective_self_collision_filter_verified=False,
        native_physx_contact_pair_found=False,
        native_contact_manifold_certified=False,
        conclusion="STATIC_CANDIDATE_ONLY_NATIVE_CONTACT_UNKNOWN",
    )


def two_sided_discordant_exact(candidate_only: int, baseline_only: int) -> float:
    """Conditional exact McNemar p under symmetric paired discordance."""
    if (not isinstance(candidate_only, int) or isinstance(candidate_only, bool)
            or not isinstance(baseline_only, int) or isinstance(baseline_only, bool)
            or min(candidate_only, baseline_only) < 0):
        raise ValueError("NONNEGATIVE_INTEGER_DISCORDANCES_REQUIRED")
    n = candidate_only + baseline_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(candidate_only, baseline_only) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def exact_scene_paired_audit(episode_rows: list, controls: list, expected_scenes: int):
    """No observer repeats, per-frame pseudoreplication, or iid trial bootstrap."""
    if len(episode_rows) != expected_scenes * len(controls):
        raise ValueError("NOT_THE_FROZEN_24_INDEPENDENT_CONDITIONS")
    cells = {}
    for row in episode_rows:
        scene, ctl = row["scene"], row["control"]
        key = scene, ctl
        if key in cells:
            raise ValueError("DUPLICATE_SCENE_CONTROLLER")
        if ctl not in controls or type(row.get("strict_hard_gate_pass")) is not bool:
            raise ValueError("UNQUALIFIED_FROZEN_OUTCOME")
        if int(row.get("physical_steps", 0)) != 2800:
            raise ValueError("NOT_TRUE_14_SECOND_200HZ")
        if row.get("native_collider_pair_certified") is not False:
            raise ValueError("UNAUTHORIZED_NATIVE_CONTACT_CLAIM")
        cells[key] = row
    scenes = sorted({k[0] for k in cells})
    if len(scenes) != expected_scenes or set(scenes) != set("AG AH AI AJ AK AL AM AN".split()):
        raise ValueError("UNQUALIFIED_SCENE_COHORT")
    if any((scene, ctl) not in cells for scene in scenes for ctl in controls):
        raise ValueError("UNPAIRED_CONTROLLER_SCENE")
    if any(len({cells[scene, ctl]["seed"] for ctl in controls}) != 1 for scene in scenes):
        raise ValueError("SCENE_SEED_NOT_MATCHED")

    comparisons = []
    for ctl in controls[1:]:
        successes = {k: sum(cells[scene, k]["strict_hard_gate_pass"] for scene in scenes)
                     for k in (controls[0], ctl)}
        b = [cells[scene, controls[0]]["strict_hard_gate_pass"] for scene in scenes]
        c = [cells[scene, ctl]["strict_hard_gate_pass"] for scene in scenes]
        candidate_only = sum(y and not x for x, y in zip(b, c))
        baseline_only = sum(x and not y for x, y in zip(b, c))
        exact_p = two_sided_discordant_exact(candidate_only, baseline_only)
        effects = [int(y) - int(x) for x, y in zip(b, c)]
        loo = [sum(e for i, e in enumerate(effects) if i != omit) / (len(scenes) - 1)
               for omit in range(len(scenes))]
        comparisons.append(dict(
            candidate=ctl, baseline=controls[0], matched_scenes=len(scenes),
            baseline_successes=successes[controls[0]], candidate_successes=successes[ctl],
            candidate_only_wins=candidate_only, baseline_only_wins=baseline_only,
            both_pass=sum(x and y for x, y in zip(b, c)),
            both_fail=sum(not x and not y for x, y in zip(b, c)),
            scene_success_difference_fraction=sum(effects) / len(scenes),
            leave_one_scene_out_difference_range=[min(loo), max(loo)],
            exact_conditional_p_two_sided=exact_p,
            individual_scene_outcomes=[
                dict(scene=s, seed=cells[s, ctl]["seed"],
                     baseline_pass=cells[s, controls[0]]["strict_hard_gate_pass"],
                     candidate_pass=cells[s, ctl]["strict_hard_gate_pass"]) for s in scenes],
        ))
    # Holm step-down, monotonically nondecreasing in ascending raw-p order.
    ordered = sorted(range(len(comparisons)),
                     key=lambda i: comparisons[i]["exact_conditional_p_two_sided"])
    adjusted = 0.0
    for rank, i in enumerate(ordered):
        adjusted = max(adjusted, min(1.0, (len(comparisons) - rank)
                                     * comparisons[i]["exact_conditional_p_two_sided"]))
        comparisons[i]["holm_adjusted_p"] = adjusted
        comparisons[i]["passes_familywise_0_05"] = adjusted < 0.05
    return dict(independent_scenes=len(scenes), independent_cells=len(cells),
                old_strict_pass=sum(x["strict_hard_gate_pass"] for x in cells.values()),
                comparisons=comparisons,
                eight_frozen_scenes_not_random_population_sample=True,
                wholebody_human_coordination_learned_from_this=False)


def source_reconcile_v125(cfg: dict, source: Path):
    if git_head(source) != cfg["parent_sha"]:
        raise ValueError("V125_GIT_SHA_DRIFT")
    report = source / cfg["source_report"]
    if sha256(report) != cfg["source_report_sha256"]:
        raise ValueError("V125_REAL_PHYSICS_AUDIT_SHA_DRIFT")
    data = json.loads(report.read_text())
    if (data.get("true_original_G1_physx_episodes_reaudited") != 29
            or data.get("original_independent_scene_controller_cells") != 24
            or data.get("observer_repeats_not_independent") != 5
            or data.get("observer_off_on_exact_parity") is not True
            or data.get("human_force_or_original_native_contact_pair_certified") is not False
            or data.get("strict_pass") != 13):
        raise ValueError("ORIGINAL_PHYSICS_CERTIFICATION_DRIFT")
    return data


def run(cfg: dict, source125: Path, source116: Path):
    if cfg["no_gpu_sim_when_gpu_occupied"] is not True:
        raise ValueError("NO_GPU_INTERFERENCE_REQUIREMENT_MISSING")
    a = static_collision_audit(cfg, source116)
    v125 = source_reconcile_v125(cfg, source125)
    paired = exact_scene_paired_audit(
        v125["per_episode"], cfg["controller_order"], cfg["expect_8_matched_scenes"])
    if paired["old_strict_pass"] != v125["strict_pass"]:
        raise ValueError("STRICT_GATE_DISAGREEMENT")
    return dict(version="v0.1.126", frozen_protocol_commit=PROTOCOL_COMMIT,
                parent_version="v0.1.125", original_200hz_episodes_referenced=29,
                original_200hz_frames_referenced=81200, new_physx_episodes=0,
                original_motor_command_modified=False, source_urdf_contact_topology=a,
                original_scene_clustered_effect_test=paired,
                native_contact_pair_conclusion="UNKNOWN",
                human_elite_kick_force_ground_truth=False,
                real_new_learned_fullbody_policy=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source125", type=Path,
                        default=Path("/home/you/projects/kinetickick-wt-v125"))
    parser.add_argument("--source116", type=Path,
                        default=Path("/home/you/projects/kinetickick-wt-v116"))
    opts = parser.parse_args()
    if opts.output.exists() or opts.output.is_symlink():
        raise FileExistsError("WRITE_ONCE_EVIDENCE_ONLY")
    data = run(frozen_config(), opts.source125, opts.source116)
    with opts.output.open("x") as file:
        json.dump(data, file, indent=2, allow_nan=False)
        file.write("\n")
    print("V126_REPORT", json.dumps({
        "sha256": sha256(opts.output),
        "static_collision": data["source_urdf_contact_topology"]["conclusion"],
        "exact_paired": data["original_scene_clustered_effect_test"]["comparisons"],
        "new_physx_episodes": 0
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

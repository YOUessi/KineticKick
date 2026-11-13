#!/usr/bin/env python3
"""v149 read-only callback capture: preserve event evidence outside scene.update.

The scene update is NOT the PhysX step. Callback arrival phase and adjacent
candidate state indices are NOT a calibrated native tick; nothing in this
module authorizes control, proves contacts or declares contact absence.
"""
from __future__ import annotations

import copy
import json
import math
import subprocess
from pathlib import Path

import v134_native_contact_vector_gate as native
from v142_native_pair_ledger import safe_write_once, verify_original_state
from v146_native_original_bridge import (
    OriginalG1ReadOnlyCapture, OriginalG1StateSampler, gpu_preflight, json_sha,
    read_only_live_stage,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN_SHA = "8f6670c5942c330cc91679e798e1911e41e0606e"
CONFIG_PATH = "configs/v149_frozen_native_clock_and_reporting_eligibility.json"
TARGET_NAMES = frozenset(("left_wrist_yaw_link", "left_hip_roll_link"))


def frozen():
    source = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", FROZEN_SHA + ":" + CONFIG_PATH]
    )
    if source != (ROOT / CONFIG_PATH).read_bytes():
        raise ValueError("V149_FROZEN_PROTOCOL_DRIFT")
    cfg = json.loads(source)
    if (
        cfg.get("version") != "v0.1.149"
        or cfg.get("parent_sha") != "4505e095ec5e13e9226d35adcfb9ba368e053156"
        or cfg.get("physical_dt_s") != 0.005
        or cfg.get("original_g1_dof") != 29
        or cfg.get("original_historical_total_rows") != 6628
        or cfg.get("synthetic_event_matrix", {}).get("total") != 48
        or cfg.get("uncalibrated_native_pair_truth") != "UNKNOWN"
        or cfg.get("minimum_disk_free_gib") != 20
        or cfg.get("strict_kick_gates", {}).get("legal_instep_impulse_fraction_min") != .95
        or cfg.get("strict_kick_gates", {}).get("ball_speed_mps_min") != 3.5
        or not all(cfg.get(k) is True for k in (
            "required_native_report_api_existing",
            "native_callback_may_run_outside_scene_update",
            "no_callback_does_not_establish_no_contact",
            "scene_update_not_physx_step",
            "never_self_certify_event_step_pair",
            "never_modify_usd_collision_report_schema",
            "never_change_original_controller",
            "never_overwrite_old_physx_or_cmu_or_c3d",
            "stop_new_gpu_when_foreign_context",
            "no_cpu_fixture_proves_runtime_contact",
            "no_proxy_is_professional_human_force_truth",
            "source_only_positive_negative_controls_not_runtime",
            "outputs_local_gitignored_write_once",
        ))
    ):
        raise ValueError("V149_FROZEN_ASSUMPTIONS_DRIFT")
    return cfg


def _path(value):
    return native.canonical_usd_path(value)


def _target_map(owner_map):
    if not isinstance(owner_map, dict):
        raise ValueError("V149_USD_COLLIDER_OWNER_MAP_MISSING")
    selected = {}
    for collider, owner in owner_map.items():
        _path(collider)
        _path(owner)
        name = owner.rsplit("/", 1)[-1]
        if name in TARGET_NAMES:
            selected.setdefault(name, set()).add(owner)
    if set(selected) != set(TARGET_NAMES) or any(len(v) != 1 for v in selected.values()):
        return None
    return {k: next(iter(v)) for k, v in selected.items()}


def existing_schema_census(owner_map, *, has_owner_report, has_reporting_articulation_ancestor):
    """Read-only eligibility, NEVER confirmation of an emitted PhysX event.

    A ContactReportAPI on an arbitrary child collider alone must not
    masquerade as a reporting rigid actor. A reporting articulation root may
    enable link events. Callers provide *read-only* USD queries.
    """
    if not callable(has_owner_report) or not callable(has_reporting_articulation_ancestor):
        raise ValueError("V149_READONLY_SCHEMA_QUERY_REQUIRED")
    owners = _target_map(owner_map)
    if owners is None:
        return {
            "status": "UNKNOWN_TARGET_ACTOR_OWNERS",
            "target_owners": None,
            "actor_has_existing_reporting_schema": {},
            "pair_report_schema_eligible": False,
            "no_events_means_no_contact": False,
            "runtime_positive_negative_controls_passed": False,
        }
    flags = {}
    evidence = {}
    for name, owner in sorted(owners.items()):
        direct = has_owner_report(owner)
        ancestor = has_reporting_articulation_ancestor(owner)
        if type(direct) is not bool or type(ancestor) is not bool:
            raise ValueError("V149_UNTRUSTED_SCHEMA_BOOL")
        flags[name] = direct or ancestor
        evidence[name] = {
            "owner": owner,
            "direct_owner_physx_contact_report_api": direct,
            "ancestor_articulation_root_contact_report_api": ancestor,
        }
    return {
        "status": ("EXISTING_REPORTING_ACTOR_FOUND" if any(flags.values())
                   else "NO_EXISTING_REPORTING_ACTOR_FOUND"),
        "target_owners": owners,
        "actor_has_existing_reporting_schema": evidence,
        "pair_report_schema_eligible": any(flags.values()),
        "no_events_means_no_contact": False,
        "runtime_positive_negative_controls_passed": False,
    }


def live_stage_with_schema_census():
    """No USD authoring and no PhysxContactReportAPI.Apply call."""
    stage_id, owner_map, decode, stage_review = read_only_live_stage()
    from pxr import PhysxSchema, UsdPhysics
    import omni.usd

    stage = omni.usd.get_context().get_stage()

    def direct(path):
        prim = stage.GetPrimAtPath(path)
        return bool(prim and prim.IsValid() and
                    prim.HasAPI(PhysxSchema.PhysxContactReportAPI))

    def ancestor(path):
        parts = path.strip("/").split("/")
        for n in range(len(parts) - 1, 0, -1):
            prim = stage.GetPrimAtPath("/" + "/".join(parts[:n]))
            if (prim and prim.IsValid()
                    and prim.HasAPI(UsdPhysics.ArticulationRootAPI)
                    and prim.HasAPI(PhysxSchema.PhysxContactReportAPI)):
                return True
        return False

    census = existing_schema_census(
        owner_map, has_owner_report=direct,
        has_reporting_articulation_ancestor=ancestor,
    )
    record = dict(stage_review)
    record["v149_existing_report_schema_census"] = census
    return stage_id, owner_map, decode, record


class V149CopyOnlyNativeCapture(OriginalG1ReadOnlyCapture):
    """Preserve callback data even when callback is outside scene.update.

    No new physical step is created. All packet time labels remain untrusted.
    A malformed callback must never break physics or suppress 29DoF sampling.
    """

    def __init__(self, *, mode, sampler, stage_reader=None, native_subscribe=None,
                 max_packets=None):
        super().__init__(
            mode=mode, sampler=sampler,
            stage_reader=stage_reader or live_stage_with_schema_census,
            native_subscribe=native_subscribe,
            max_packets=max_packets or frozen()["max_packets_per_episode"],
        )
        self.after_terminal = False
        self.event_phase_counts = {}
        self.max_disclosed_candidates = 128

    def before_native_update(self, dt):
        if self.pending_step is not None:
            self.errors.append("V149_NESTED_SCENE_UPDATE")
            return
        if (isinstance(dt, bool) or not isinstance(dt, (int, float))
                or not math.isfinite(float(dt)) or abs(float(dt) - 0.005) > 1e-7):
            self.errors.append("V149_NOT_5MS_SCENE_UPDATE")
            return
        self.pending_step = len(self.rows) + 1

    def after_native_update(self, dt):
        if self.pending_step is None:
            self.errors.append("V149_SCENE_UPDATE_MISSING_BEGIN")
            return
        step = self.pending_step
        self.pending_step = None
        try:
            if abs(float(dt) - 0.005) > 1e-7:
                raise ValueError("V149_DT_CHANGED")
            row = self.sampler.sample(step)
            verify_original_state(row)
            self.rows.append(copy.deepcopy(row))
        except Exception as e:
            self.errors.append("V149_STATE_COPY_ERROR:" + type(e).__name__ + ":" + str(e)[:140])

    def mark_terminal(self):
        if self.pending_step is not None:
            self.errors.append("V149_TERMINATION_INSIDE_SCENE_UPDATE")
        self.after_terminal = True

    def _phase(self):
        if self.after_terminal:
            return "AFTER_TERMINAL_SCENE_UPDATE"
        if self.pending_step is not None:
            return "INSIDE_SCENE_UPDATE"
        if not self.rows:
            return "BEFORE_FIRST_SCENE_UPDATE"
        return "BETWEEN_SCENE_UPDATES"

    def on_callback(self, headers, points):
        self.callback_count += 1
        phase = self._phase()
        self.event_phase_counts[phase] = self.event_phase_counts.get(phase, 0) + 1
        if self.mode != "on" or self.stage_id is None or not hasattr(self, "_decode"):
            self.errors.append("V149_UNOBSERVABLE_CALLBACK_WITHOUT_STAGE_OR_MODE")
            return
        if len(self.packets) >= self.max_packets:
            self.errors.append("V149_NATIVE_EVENT_MEMORY_CAP_EXHAUSTED")
            return
        last = len(self.rows)
        # Adjacent completed-state candidates, NOT an authoritative time label.
        candidates = ([self.pending_step] if self.pending_step is not None
                      else ([1] if last == 0 else [last, last + 1]))
        hint = max(candidates)
        try:
            packet = native.copy_contact_vector_frame(
                headers, points, decode_path=self._decode,
                step_hint=hint, expected_stage_id=self.stage_id,
            )
            packet["v149_callback_sequence"] = self.callback_count
            packet["v149_arrival_phase"] = phase
            packet["v149_adjacent_step_candidates_unattested"] = candidates
            packet["v149_original_completed_rows_at_callback"] = last
            packet["v149_untrusted_step_hint_not_native_clock"] = True
            packet["clock_basis"] = "SCENE_UPDATE_ADJACENCY_ONLY"
            self.packets.append(packet)
        except Exception as e:
            self.errors.append("V149_CALLBACK_COPY_INVALID:" + type(e).__name__ + ":" + str(e)[:160])

    def review(self):
        census = self.stage_review.get("v149_existing_report_schema_census", {
            "status": "SCHEMA_CENSUS_NOT_RUN",
            "pair_report_schema_eligible": False,
            "no_events_means_no_contact": False,
            "runtime_positive_negative_controls_passed": False,
        })
        orphaned = sum(not any(1 <= s <= len(self.rows)
                               for s in p["v149_adjacent_step_candidates_unattested"])
                       for p in self.packets)
        ambiguous = sum(len(p["v149_adjacent_step_candidates_unattested"]) > 1
                        for p in self.packets)
        path_pairs = []
        for p in self.packets:
            for h in p["headers"]:
                try:
                    owners = [self.owner_map.get(x) for x in h["collider_paths"]]
                    if (frozenset(x.rsplit("/", 1)[-1] for x in owners
                                  if isinstance(x, str)) == TARGET_NAMES
                            and any(p["impulse_vector_magnitude_Ns"] > 0
                                    for p in h["manifold_points"])
                            and h["event_type"] in ("CONTACT_FOUND", "CONTACT_PERSISTS")):
                        path_pairs.append({
                            "colliders": h["collider_paths"],
                            "rigid_owner_paths": owners,
                            "arrival_phase": p["v149_arrival_phase"],
                            "adjacent_step_candidates_unattested":
                                p["v149_adjacent_step_candidates_unattested"],
                            "source_is_real_physx_certified": False,
                        })
                except (TypeError, ValueError):
                    self.errors.append("V149_OWNER_REVIEW_FAILED")
        return {
            "schema": "v149_uncalibrated_anyphase_native_copy",
            "mode": self.mode,
            "original_29dof_completed_5ms_rows": len(self.rows),
            "original_full_state_checksum": json_sha(self.rows),
            "copied_native_callback_packets": len(self.packets),
            "callback_invocations_not_step_count": self.callback_count,
            "copied_manifold_points": sum(p["total_points"] for p in self.packets),
            "arrival_phase_counts": self.event_phase_counts,
            "uncertain_step_adjacency_packets": ambiguous,
            "orphaned_callback_packets": orphaned,
            "wrist_hip_owner_path_event_candidates_not_truth":
                path_pairs[:self.max_disclosed_candidates],
            "wrist_hip_candidate_total": len(path_pairs),
            "readout_incomplete": bool(
                self.errors or self.pending_step is not None
                or (self.mode == "on" and not census.get("pair_report_schema_eligible", False))
                or orphaned > 0 or self.failed
            ),
            "errors": self.errors,
            "existing_report_api_census": census,
            "left_wrist_hip_native_physx_truth": "UNKNOWN",
            "same_native_physics_tick_calibrated": False,
            "off_on_physical_equality_certified": False,
            "runtime_positive_negative_controls_passed": False,
            "negative_event_absence_certified": False,
            "cpu_fixtures_cannot_certify_physx": True,
            "motor_execution_authorized": False,
            "new_physx_episode_success": None,
        }

    def persist_once(self, output):
        return safe_write_once(Path(output), {
            "summary": self.review(),
            "collider_to_rigid_owner": self.owner_map,
            "raw_callback_copies": self.packets,
            "original29_5ms_state_rows": self.rows,
            "named_max_nonfoot_net_body":
                getattr(self.sampler, "nonfoot_peak_names", []),
        })

#!/usr/bin/env python3
"""Copy-only episode sidecar for the original G1 PhysX runner.

The owning runner invokes on_pre_physics_step(step), advances ITS UNCHANGED
world/controller, and calls on_post_physics_step(state). Callbacks may happen
multiple times per step. This module never advances physics, sets motor
targets, edits USD prims or claims live evidence from CPU fixtures.

The G1 integration is not certified until actual GPU positive/negative
calibration and full original 3x paired OFF/ON physical equivalence.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path

import v134_native_contact_vector_gate as native
from v142_native_pair_ledger import (packet_review, verify_original_state,
                                     safe_write_once, TARGET)


def stable_hash(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,allow_nan=False,
                                     separators=(",",":")).encode("utf8")).hexdigest()


class NativeReadOnlyEpisodeSidecar:
    """Caller owns original physics step clock; callbacks are NOT steps."""

    def __init__(self, *, stage_id, owner_map, decode_path):
        if not isinstance(stage_id,int) or stage_id<0:
            raise ValueError("UNVERIFIED_STAGE_ID")
        if not owner_map or not isinstance(owner_map,dict):
            raise ValueError("MISSING_REAL_STAGE_OWNER_MAP")
        if not callable(decode_path):
            raise ValueError("MISSING_NATIVE_PATH_DECODER")
        self.stage_id=stage_id
        self.owner_map=copy.deepcopy(owner_map)
        self.decode_path=decode_path
        self._clock_step=None
        self._last_step=0
        self.packets=[]
        self.state_rows=[]
        self.errors=[]
        self.subscription=None

    def before_physics_step(self, step):
        if (isinstance(step, bool) or not isinstance(step, int)
                or self._clock_step is not None or step != self._last_step + 1):
            raise ValueError("PRE_STEP_MISSING_OR_REORDERED")
        self._clock_step=step

    def on_native_contact(self, headers, points):
        """Called inside PhysX callback before it returns: copy borrowed data."""
        if self._clock_step is None:
            self.errors.append("CONTACT_OUTSIDE_OR_UNCALIBRATED_STEP")
            return
        try:
            packet=native.copy_contact_vector_frame(
                headers,points,decode_path=self.decode_path,
                step_hint=self._clock_step,expected_stage_id=self.stage_id)
            self.packets.append(packet)
        except (Exception) as exc:
            # Do not change control or simulation when diagnostics fail.
            self.errors.append("UNTRUSTED_NATIVE_CALLBACK:"+type(exc).__name__+":"+str(exc)[:200])

    def after_physics_step(self, original_state):
        if self._clock_step is None:
            raise ValueError("POST_STEP_WITHOUT_PRE_STEP")
        verify_original_state(original_state)
        if original_state["step"] != self._clock_step:
            raise ValueError("POST_STEP_CLOCK_MISMATCH")
        self.state_rows.append(copy.deepcopy(original_state))
        self._last_step = self._clock_step
        self._clock_step=None

    def attach_existing_physx_read_only(self, subscribe=None):
        if self.subscription is not None:
            raise ValueError("DUPLICATE_NATIVE_CONTACT_SUBSCRIPTION")
        if subscribe is None:
            from omni.physx import get_physx_simulation_interface
            subscribe=get_physx_simulation_interface().subscribe_contact_report_events
        if not callable(subscribe):
            raise ValueError("MISSING_NATIVE_CONTACT_SUBSCRIBE")
        # This subscribes only. It does not apply PhysxContactReportAPI or
        # change collision filters. The existing scene MUST already report.
        self.subscription=subscribe(self.on_native_contact)
        if self.subscription is None:
            raise ValueError("SUBSCRIBE_RETURNED_NO_HANDLE")
        return self.subscription

    def review(self):
        if self._clock_step is not None or not self.state_rows:
            raise ValueError("INCOMPLETE_REAL_PHYSICS_EPISODE")
        if [row["step"] for row in self.state_rows] != list(range(1,self._last_step+1)):
            raise ValueError("MISSING_REAL_200HZ_SAMPLES")
        rows_by_step={x["step"]:x for x in self.state_rows}
        candidates=[]
        rejected=[]
        for packet in self.packets:
            step=packet["step_hint"]
            if step not in rows_by_step:
                rejected.append("CALLBACK_BEYOND_REAL_PHYSICS_HORIZON")
                continue
            try:
                r=packet_review(packet,rows_by_step[step],self.owner_map,
                                expected_stage_id=self.stage_id)
                candidates.extend(r["left_wrist_hip_colliding_pair_candidates"])
            except (KeyError,TypeError,ValueError) as exc:
                rejected.append("UNREVIEWABLE_PACKET:"+str(exc)[:200])
        return {
            "source": "IN_PROCESS_COPY_ONLY_RUNTIME_OR_SYNTHETIC_NOT_AUTHENTICATED",
            "original_physics_steps_input": self._last_step,
            "original_full_state_sha256": stable_hash(self.state_rows),
            "native_packets_copied": len(self.packets),
            "native_manifold_points_copied": sum(p["total_points"] for p in self.packets),
            "g1_wrist_hip_candidate_events": candidates,
            "callback_errors":list(self.errors),
            "unreviewable_packet_reasons":rejected,
            "no_native_callbacks_is_no_contact_proof":False,
            "native_pair_manifold_certified":False,
            "read_only_off_on_invariance_certified":False,
            "same_step_backend_clock_calibrated":False,
            "physical_kick_success":None,
            "motor_execution_authorized":False,
        }

    def persist_write_once(self, output, *, include_full_physical_states=True):
        r=self.review()
        record={"metadata":r,
                "read_only_usd_collider_to_actor":self.owner_map,
                "copied_callback_packets":self.packets,
                "full_original_physics_rows":
                    self.state_rows if include_full_physical_states else None}
        h=safe_write_once(output,record)
        return {"output":str(output),"sha256":h,"review":r}

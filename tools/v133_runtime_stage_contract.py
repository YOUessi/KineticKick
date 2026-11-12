#!/usr/bin/env python3
"""Read-only USD collider ancestry and report-API preflight for native PhysX.

Only use inside an initialized Isaac Sim stage; imports pxr lazily. This tool
never applies a schema, creates a prim, adjusts a collision setting or starts
simulation. Runtime readback proves USD configuration, not an actual contact.
"""
from __future__ import annotations

from pathlib import Path
import json

import v133_native_contact_frame as contract


def ancestry_owner_map(
    prims, *, path_of, parent_of, collision_enabled, rigid_body_enabled,
    report_enabled,
):
    """Collect every collision prim and its nearest rigid-body ancestor.

    The input is a *live USD stage traversal* in a real run; fake unit-test
    prims are not attested. Excludes ambiguous or disabled collision shapes.
    """
    available = list(prims)
    ids = {}
    for obj in available:
        path = contract.canonical_path(path_of(obj))
        if path in ids:
            raise ValueError("DUPLICATE_STAGE_PRIM_PATH")
        ids[path] = obj
    colliders = {}
    missing_owner = []
    missing_report = []
    disabled = []
    for path, prim in ids.items():
        if not collision_enabled(prim):
            continue
        cur = prim
        owner = None
        seen=set()
        while cur is not None:
            parent_id = contract.canonical_path(path_of(cur))
            if parent_id in seen:
                raise ValueError("RIGID_BODY_PARENT_CYCLE")
            seen.add(parent_id)
            if rigid_body_enabled(cur):
                owner = parent_id
                if not report_enabled(cur):
                    missing_report.append(owner)
                break
            cur = parent_of(cur)
        if owner is None:
            missing_owner.append(path)
            continue
        colliders[path]=owner
    return {
        "collider_to_nearest_rigidbody_owner":dict(sorted(colliders.items())),
        "colliders_with_no_rigidbody_owner":sorted(set(missing_owner)),
        "rigid_body_owners_without_report_API":sorted(set(missing_report)),
        "disabled_colliders_not_counted_as_contacts":sorted(disabled),
        "all_owners_have_report_API":not missing_report,
        "missing_owner_is_not_no_collision":bool(missing_owner),
        "native_pair_detected":False,
        "actuation_authorized":False,
    }


def inspect_actual_usd_stage(stage):
    """Actual USD APIs; remains a READ of the live stage, never a mutation.

    Report API on an articulation root rather than a child rigid body may
    still be valid in a specific backend; this preflight flags ambiguity for
    dedicated calibration rather than falsely declaring no contact.
    """
    from pxr import UsdPhysics, PhysxSchema

    def valid(obj):
        return obj is not None and obj.IsValid()

    def path_of(obj):
        return str(obj.GetPath())

    def parent_of(obj):
        parent=obj.GetParent()
        return parent if valid(parent) else None

    def collision_enabled(obj):
        if not obj.HasAPI(UsdPhysics.CollisionAPI):
            return False
        attr=UsdPhysics.CollisionAPI(obj).GetCollisionEnabledAttr()
        enabled=attr.Get() if attr else None
        return enabled is not False

    def rigid_body_enabled(obj):
        if not obj.HasAPI(UsdPhysics.RigidBodyAPI):
            return False
        attr=UsdPhysics.RigidBodyAPI(obj).GetRigidBodyEnabledAttr()
        enabled=attr.Get() if attr else None
        return enabled is not False

    def report_enabled(obj):
        return bool(obj.HasAPI(PhysxSchema.PhysxContactReportAPI))

    out=ancestry_owner_map(stage.Traverse(),path_of=path_of,
        parent_of=parent_of,collision_enabled=collision_enabled,
        rigid_body_enabled=rigid_body_enabled,report_enabled=report_enabled)
    out["actual_usd_read_only_inspection"]=True
    out["native_contact_collider_pair_verified"]=False
    out["runtime_callback_and_clock_still_required"]=True
    return out


def save_preflight(result, output: Path):
    output=Path(output)
    if not result.get("actuation_authorized") is False:
        raise ValueError("UNSAFE_STAGE_PREFLIGHT_CLAIM")
    with output.open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write("\n")

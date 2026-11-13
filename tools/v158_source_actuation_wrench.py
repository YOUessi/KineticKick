#!/usr/bin/env python3
"""v158: original 29-DoF G1 URDF actuator-limited offline 35D stance QP.

This is a CPU-only inverse-dynamics screening experiment, NOT a simulator
controller, not a runtime ground-force/collision witness, and not a learned
human-football trajectory. URDF actuator limits are model parameters, not
hardware calibration. Pinocchio LOCAL_WORLD_ALIGNED uses linear/ angular order.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import osqp
import scipy.sparse as sp

from wholebody_safety_interface_v055 import G1_JOINT_ORDER
from v156_real_urdf_pin_model import OriginalUrdfModel, FEET, urdf_path
from v156_urdf_collision_footprint_audit import foot_collision_bbox

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = "aa9deaeba02024f175af1d7b9918d1764d5cd01a"
PROTOCOL_PATH = "configs/v158_frozen_original_actuator_urdf_wrench.json"
G1_ORDER = tuple(G1_JOINT_ORDER)
DT = 0.005


def frozen():
    raw = subprocess.check_output(["git", "-C", str(ROOT), "show",
                                  f"{PROTOCOL_SHA}:{PROTOCOL_PATH}"])
    if raw != (ROOT / PROTOCOL_PATH).read_bytes():
        raise ValueError("V158_PROTOCOL_CHANGED_AFTER_FREEZE")
    c = json.loads(raw)
    if (c["parent_sha"] != "998c17b93619637443f883784e9daa2539348423"
            or c["expected_source_states"] != 48
            or c["expected_qp_solutions"] != 96
            or c["expected_negative_rejections"] != 16
            or len(c["negative_kinds"]) != 8
            or c["source_urdf_sha256"] !=
            "dbc8471004e9b0b85c27a4ebc1f0d90a727c1cf7ca9ef507b805596f2004d64b"):
        raise ValueError("V158_PROTOCOL_DENOMINATOR_DRIFT")
    return c


def arr(value, shape, name):
    x = np.asarray(value, dtype=float)
    if x.shape != shape or not np.all(np.isfinite(x)):
        raise ValueError("V158_INVALID_" + name)
    return x


def urdf_joint_limits(model):
    """In ORIGINAL IsaacLab order, never Pinocchio's internal joint order."""
    root = ET.parse(model.source).getroot()
    entries = {}
    for item in root.findall("joint"):
        name = item.get("name")
        if name not in G1_ORDER:
            continue
        if item.get("type") != "revolute":
            raise ValueError("V158_NOT_SINGLE_REVOLUTE_" + str(name))
        limit = item.find("limit")
        if limit is None or len(limit.attrib) < 4:
            raise ValueError("V158_MISSING_SOURCE_ACTUATOR_LIMIT")
        entries[name] = [float(limit.get(k)) for k in ("lower", "upper", "effort", "velocity")]
    if len(entries) != 29:
        raise ValueError("V158_WRONG_ORIGINAL29_LIMITS")
    v = np.asarray([entries[n] for n in G1_ORDER])
    if not np.isfinite(v).all() or np.any(v[:, 0] >= v[:, 1]) or np.any(v[:, 2:] <= 0):
        raise ValueError("V158_BAD_SOURCE_LIMIT_VALUES")
    return v


def phase_reference(cfg, phase, seed):
    """All 29 joints: deterministic EXPLICITLY SYNTHETIC phase coordination."""
    rng = np.random.default_rng(158000 + 119 * cfg["phases"].index(phase) + seed)
    a = rng.uniform(-.30, .30, 29)
    phase_amplitude = cfg["reference_acceleration_amplitude_rad_s2"][phase]
    for i, name in enumerate(G1_ORDER):
        if name.startswith("right_hip") or name == "right_knee_joint":
            a[i] += 0.85
        elif name.startswith("left_hip") or name == "left_knee_joint":
            a[i] -= 0.38
        elif name.startswith("waist"):
            a[i] += .35
        elif "shoulder" in name or "elbow" in name:
            a[i] += -.27 if name.startswith("left") else .32
        elif "wrist" in name:
            a[i] += .08
        else:
            a[i] += .15
    return a * phase_amplitude


def make_case(model, phase, mode, seed, effort_condition):
    cfg = frozen()
    if (phase not in cfg["phases"] or mode not in cfg["support_modes"]
            or seed not in cfg["seeds"] or effort_condition not in cfg["effort_conditions"]):
        raise ValueError("V158_UNDECLARED_CASE")
    limits = urdf_joint_limits(model)
    rng = np.random.default_rng(20261114 + 1000 * cfg["phases"].index(phase) + seed * 41)
    q29 = .07 * rng.uniform(-1, 1, 29)
    # Real source joint limits, with >=0.08rad interior margin; not observed motion.
    q29 = np.maximum(limits[:, 0] + .08, np.minimum(limits[:, 1] - .08, q29))
    dq29 = rng.uniform(-.035, .035, 29)
    q = model.pose(q29)
    v = model.speed(dq29, base_body6=(.007, -.003, .002, .004, -.004, .003))
    out = model.evaluate(q, v)
    mu = cfg["nominal_friction_mu_assumed_not_measured"]
    collision = {}
    for foot in FEET:
        bbox = foot_collision_bbox(model.source, foot)
        low = np.asarray(bbox["enclosing_min_xyz_m"])
        high = np.asarray(bbox["enclosing_max_xyz_m"])
        p = model.pin
        d = model.model.createData()
        p.forwardKinematics(model.model, d, q)
        p.updateFramePlacements(model.model, d)
        R = np.asarray(d.oMf[model.frames[foot]].rotation)
        # Outer rotated source collision bbox; upper bound is a necessary bound,
        # never a feasible sole/CoP polygon in PhysX.
        rad = np.abs(R) @ np.maximum(np.abs(low), np.abs(high))
        collision[foot] = rad
    feet = FEET[:1] if mode == "left" else FEET
    multiple = 1. if effort_condition == "urdf_effort" else float(
        cfg["relaxed_effort_multiplier_for_ablation_only"])
    problem = dict(
        q36=q.tolist(), v35=v.tolist(), M=out["M"].tolist(), h=out["h"].tolist(),
        source_urdf_sha256=model.sha, original_joint_names=list(G1_ORDER),
        q29=q29.tolist(), dq29=dq29.tolist(), source_limits=limits.tolist(),
        torque_caps=(limits[:, 2] * multiple).tolist(),
        a_ref29=phase_reference(cfg, phase, seed).tolist(),
        phase=phase, mode=mode, seed=seed, effort_condition=effort_condition,
        source_kind="original_G1_URDF_OFFLINE_only",
        human_training_qualified=False,
        native_contact_pair_certified=False,
        runtime_ground_force_certified=False,
        can_issue_control=False,
        contacts=[dict(body=foot,
                       J6=out["J6"][foot].tolist(),
                       Jdotv6=out["Jdotv6"][foot].tolist(),
                       envelope_xyz=np.asarray(collision[foot]).tolist(),
                       mu=mu, source_is_URDF_collision_outer_bbox=True,
                       ground_only_signed_normal_measurement=False)
                  for foot in feet])
    validate_case(problem, model)
    return problem


def validate_case(p, model):
    cfg = frozen()
    if p.get("source_kind") != "original_G1_URDF_OFFLINE_only":
        raise ValueError("V158_RUNTIME_SOURCE_NOT_ADMITTED")
    if p.get("human_training_qualified") is not False:
        raise ValueError("V158_NO_QUALIFIED_HUMAN_LABEL")
    if any(p.get(k) is not False for k in
           ("native_contact_pair_certified", "runtime_ground_force_certified",
            "can_issue_control")):
        raise ValueError("V158_UNCERTIFIED_RUNTIME_CLAIM")
    if p.get("source_urdf_sha256") != cfg["source_urdf_sha256"] or model.sha != cfg["source_urdf_sha256"]:
        raise ValueError("V158_UNVERIFIED_URDF")
    if p.get("original_joint_names") != list(G1_ORDER):
        raise ValueError("V158_ORIGINAL_JOINT_ORDER_MISMATCH")
    qq = arr(p.get("q36"), (36,), "Q36")
    vv = arr(p.get("v35"), (35,), "V35")
    if not np.isclose(np.linalg.norm(qq[3:7]), 1., atol=1e-8):
        raise ValueError("V158_UNNORMALIZED_QUATERNION")
    actual_limits = urdf_joint_limits(model)
    limits = arr(p.get("source_limits"), (29, 4), "SOURCE_LIMITS")
    if not np.array_equal(limits, actual_limits):
        raise ValueError("V158_LIMITS_NOT_FROM_ORIGINAL_URDF")
    if not np.array_equal(arr(p.get("q29"), (29,), "Q29"), qq[model.idxq]):
        raise ValueError("V158_BAD_ISAAC_TO_PIN_POSITION_MAP")
    if not np.array_equal(arr(p.get("dq29"), (29,), "DQ29"), vv[model.idxv]):
        raise ValueError("V158_BAD_ISAAC_TO_PIN_VELOCITY_MAP")
    q29 = np.asarray(p["q29"])
    dq29 = np.asarray(p["dq29"])
    if (np.any(q29 < limits[:, 0]) or np.any(q29 > limits[:, 1])
            or np.any(np.abs(dq29) > limits[:, 3])):
        raise ValueError("V158_ORIGINAL_JOINT_POSITION_VELOCITY_UNSAFE")
    condition = p.get("effort_condition")
    if condition not in cfg["effort_conditions"]:
        raise ValueError("V158_UNDECLARED_EFFORT_CONDITION")
    multiple = 1. if condition == "urdf_effort" else cfg["relaxed_effort_multiplier_for_ablation_only"]
    if not np.array_equal(arr(p.get("torque_caps"), (29,), "TAU_CAP"), limits[:, 2] * multiple):
        raise ValueError("V158_TORQUE_CAP_NOT_FROM_FROZEN_SOURCE")
    M = arr(p.get("M"), (35, 35), "M")
    h = arr(p.get("h"), (35,), "H")
    if not np.allclose(M, M.T, atol=1e-9, rtol=0) or np.linalg.eigvalsh(M)[0] < 1e-9:
        raise ValueError("V158_BAD_ORIGINAL_MASS_MATRIX")
    arr(p.get("a_ref29"), (29,), "REF")
    if p.get("phase") not in cfg["phases"] or p.get("mode") not in cfg["support_modes"] or p.get("seed") not in cfg["seeds"]:
        raise ValueError("V158_UNDECLARED_MATRIX_CASE")
    expected = list(FEET[:1] if p["mode"] == "left" else FEET)
    contacts = p.get("contacts")
    if not isinstance(contacts, list) or [c.get("body") for c in contacts] != expected:
        raise ValueError("V158_UNQUALIFIED_STANCE")
    for c in contacts:
        arr(c.get("J6"), (6, 35), "FOOT_J6")
        arr(c.get("Jdotv6"), (6,), "FOOT_JDOTV")
        rad = arr(c.get("envelope_xyz"), (3,), "SOLE_BBOX")
        if np.any(rad <= 0) or c.get("source_is_URDF_collision_outer_bbox") is not True:
            raise ValueError("V158_UNTRUSTED_FOOT_ENVELOPE")
        if c.get("ground_only_signed_normal_measurement") is not False:
            raise ValueError("V158_NOT_REAL_GROUND_GRF")
        if not np.isclose(c.get("mu"), cfg["nominal_friction_mu_assumed_not_measured"]):
            raise ValueError("V158_UNFROZEN_FRICTION")
    # Strengthen the source contract: a claimed SHA must not be allowed to
    # authenticate arbitrary caller-supplied M/h/J/contact envelope/reference.
    if hashlib.sha256(model.source.read_bytes()).hexdigest() != cfg["source_urdf_sha256"]:
        raise ValueError("V158_SOURCE_URDF_CHANGED_DURING_EXPERIMENT")
    expected_ref = phase_reference(cfg, p["phase"], p["seed"])
    if not np.allclose(p["a_ref29"], expected_ref, rtol=0, atol=1e-12):
        raise ValueError("V158_REFERENCE_MUTATED_AFTER_FREEZE")
    actual = model.evaluate(qq, vv)
    if (np.max(np.abs(M - actual["M"])) > 1e-9
            or np.max(np.abs(h - actual["h"])) > 1e-9):
        raise ValueError("V158_SOURCE_MASS_BIAS_NOT_ACTUAL_URDF")
    d = model.model.createData()
    model.pin.forwardKinematics(model.model, d, qq)
    model.pin.updateFramePlacements(model.model, d)
    for c in contacts:
        foot = c["body"]
        if (np.max(np.abs(np.asarray(c["J6"]) - actual["J6"][foot])) > 1e-9
                or np.max(np.abs(np.asarray(c["Jdotv6"]) - actual["Jdotv6"][foot])) > 1e-9):
            raise ValueError("V158_UNTRUSTED_SOURCE_STANCE_JACOBIAN")
        bbox = foot_collision_bbox(model.source, foot)
        hi = np.asarray(bbox["enclosing_max_xyz_m"])
        lo = np.asarray(bbox["enclosing_min_xyz_m"])
        R = np.asarray(d.oMf[model.frames[foot]].rotation)
        expected_radius = np.abs(R) @ np.maximum(np.abs(lo), np.abs(hi))
        if not np.allclose(c["envelope_xyz"], expected_radius, rtol=0, atol=1e-9):
            raise ValueError("V158_FOOT_WRENCH_ENVELOPE_NOT_FROM_SOURCE_GEOMETRY")
    return M, h


def solve(p, model):
    """Constrained offline inverse-dynamics optimization.

    6D contact wrenches have a loose URDF collision outer envelope only.
    Solved means model feasibility under HYPOTHETICAL stance, not physical pass.
    """
    M, h = validate_case(p, model)
    cfg = frozen()
    nc = len(p["contacts"])
    dim = 64 + 6 * nc
    entries, lows, highs = [], [], []

    def row(v, lo, hi):
        entries.append(v)
        lows.append(lo)
        highs.append(hi)

    J = np.vstack([np.asarray(c["J6"]) for c in p["contacts"]])
    JD = np.hstack([np.asarray(c["Jdotv6"]) for c in p["contacts"]])
    for i in range(35):
        r = np.zeros(dim)
        r[:35] = M[i]
        if i >= 6:
            r[35+i-6] = -1.
        r[64:] = -(J.T)[i]
        row(r, -h[i], -h[i])
    for i in range(6 * nc):
        r = np.zeros(dim)
        r[:35] = J[i]
        row(r, -JD[i], -JD[i])
    limits = np.asarray(p["source_limits"])
    q29, dq29 = np.asarray(p["q29"]), np.asarray(p["dq29"])
    amax = float(cfg["assumed_acceleration_limit_rad_s2"])
    for i in range(29):
        a_lo = max(-amax, (-limits[i,3]-dq29[i])/DT,
                   2*(limits[i,0]-q29[i]-DT*dq29[i])/DT**2)
        a_hi = min(amax, (limits[i,3]-dq29[i])/DT,
                   2*(limits[i,1]-q29[i]-DT*dq29[i])/DT**2)
        r = np.zeros(dim); r[6+i] = 1.
        row(r, a_lo, a_hi)
        r = np.zeros(dim); r[35+i] = 1.
        row(r, -p["torque_caps"][i], p["torque_caps"][i])
    for k, c in enumerate(p["contacts"]):
        base = 64 + 6*k
        mu = c["mu"]
        X,Y,Z = c["envelope_xyz"]
        r = np.zeros(dim);r[base+2] = 1.
        row(r, 0., np.inf)
        for j in (0,1):
            for sign in (-1.,1.):
                r = np.zeros(dim);r[base+j] = sign;r[base+2] = -mu
                row(r, -np.inf, 0.)
        # OUTER necessary wrench envelope; cannot certify contact realization.
        for j, radius in ((3,Y+Z*mu),(4,X+Z*mu),(5,(X+Y)*mu)):
            for sign in (-1.,1.):
                r = np.zeros(dim);r[base+j] = sign;r[base+2] = -radius
                row(r, -np.inf, 0.)
    weights = np.concatenate([np.full(6,20.),np.full(29,2.),
                              np.full(29,.0005),np.full(6*nc,.0002)])
    desired = np.zeros(dim)
    desired[6:35] = p["a_ref29"]
    opt = osqp.OSQP()
    opt.setup(P=sp.diags(2*weights,format="csc"), q=-2*weights*desired,
              A=sp.csc_matrix(np.vstack(entries)),
              l=np.array(lows),u=np.array(highs),
              verbose=False, polish=True, eps_abs=1e-6, eps_rel=1e-6,
              max_iter=12000, adaptive_rho=True)
    res = opt.solve()
    ok = res.info.status in ("solved","solved inaccurate") and res.x is not None and np.isfinite(res.x).all()
    result = {
        "phase": p["phase"], "mode":p["mode"], "seed":p["seed"],
        "effort_condition":p["effort_condition"], "status":res.info.status,
        "solved_numerically":bool(ok), "source_urdf_sha256":p["source_urdf_sha256"],
        "provenance":"original_G1_URDF_offline_35D_hypothetical_stance",
        "cpu_optimization_not_PhysX":True, "real_physics_episodes":0,
        "can_issue_control":False,"actual_contact_manifold":None,
        "ground_reaction_or_CoP_certified":False,
        "human_elite_learned":False,"strict_instep_success":None,
    }
    if ok:
        x = np.asarray(res.x)
        a, tau, wrench = x[:35], x[35:64], x[64:].reshape(nc,6)
        result.update(a35=a.tolist(),tau29=tau.tolist(),wrenches6=wrench.tolist(),
                      tracking_rmse_rad_s2=float(np.sqrt(np.mean((a[6:35]-p["a_ref29"])**2))),
                      maximum_effort_ratio=float(np.max(np.abs(tau)/p["torque_caps"])),
                      maximum_original_effort_ratio=float(np.max(np.abs(tau)/np.asarray(p["source_limits"])[:,2])),
                      resultant_fz_N=float(np.sum(wrench[:,2])),
                      objective=float(res.info.obj_val))
    return result


def independently_audit_solution(p, r, model, tol=0.008):
    """Independent residual evaluation; never trust OSQP's status alone."""
    if not r["solved_numerically"]:
        return {"audited":False, "reason":r["status"]}
    M,h = validate_case(p,model)
    a = arr(r["a35"],(35,),"SOLUTION_A")
    tau = arr(r["tau29"],(29,),"SOLUTION_TAU")
    w = arr(r["wrenches6"],(len(p["contacts"]),6),"SOLUTION_WRENCH")
    J = np.vstack([np.asarray(c["J6"]) for c in p["contacts"]])
    JD = np.concatenate([np.asarray(c["Jdotv6"]) for c in p["contacts"]])
    dyn = M@a+h-np.concatenate([np.zeros(6),tau])-J.T@w.reshape(-1)
    contact = J@a+JD
    limits = np.asarray(p["source_limits"])
    acc = a[6:35]
    pos_next=np.asarray(p["q29"])+DT*np.asarray(p["dq29"])+.5*DT**2*acc
    vel_next=np.asarray(p["dq29"])+DT*acc
    violation = [
        np.max(np.abs(tau)-np.asarray(p["torque_caps"])),
        np.max(np.abs(acc)-frozen()["assumed_acceleration_limit_rad_s2"]),
        np.max(pos_next-limits[:,1]),np.max(limits[:,0]-pos_next),
        np.max(np.abs(vel_next)-limits[:,3])]
    for i,c in enumerate(p["contacts"]):
        f=w[i];mu=c["mu"];X,Y,Z=c["envelope_xyz"]
        violation.extend([-f[2],abs(f[0])-mu*f[2],abs(f[1])-mu*f[2],
                          abs(f[3])-(Y+Z*mu)*f[2],
                          abs(f[4])-(X+Z*mu)*f[2],
                          abs(f[5])-(X+Y)*mu*f[2]])
    mr=float(np.max(np.abs(dyn)))
    cr=float(np.max(np.abs(contact)))
    bound=float(max(0.,max(violation)))
    if mr>tol or cr>tol or bound>tol:
        raise AssertionError(f"V158_INDEPENDENT_QP_RESIDUAL dynamics={mr}, support={cr}, bound={bound}")
    return {"audited":True,"max_dynamics_residual":mr,
            "max_hypothetical_6d_stance_residual":cr,
            "max_inequality_violation":bound,
            "source_URDF_not_same_tick_PhysX":True,
            "outer_bbox_does_not_certify_true_CoP":True}

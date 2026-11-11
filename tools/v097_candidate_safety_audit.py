#!/usr/bin/env python3
"""Independent, read-only all-29 URDF joint-angle screen for learned robot proposals.

Recomputes the same preregistered full-body support-conditioned ridge predictor
from SHA-certified v84 C/D original G1 PhysX. The v97 G/H holdout is NEVER used
for fitting or tuning. This is a conservative rejection screen, NOT a permission
to actuate; actual G1 inverse dynamics, motor effort, support slip, and certified
collider geometry are unavailable to this offline predictor.
"""
from __future__ import annotations
import argparse, hashlib, json, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import v097_trajectory_generator as gen
from v097_coordination_observer import load
from wholebody_safety_interface_v055 import G1_JOINT_ORDER

ROOT=Path(__file__).resolve().parents[1]


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def qualified_limits(urdf_path:Path,names) -> tuple[np.ndarray,np.ndarray]:
    """Verify real 29 joint name correspondence, do not make up G1 angle limits."""
    if tuple(names)!=tuple(G1_JOINT_ORDER):
        raise ValueError('ACTUAL_G1_29_JOINT_ORDER_MISMATCH')
    root=ET.parse(urdf_path).getroot()
    joints={j.attrib['name']:j for j in root.findall('joint')}
    low=[];high=[]
    for name in names:
        if name not in joints:raise ValueError('MISSING_ORIGINAL_URDF_JOINT:'+name)
        limit=joints[name].find('limit')
        if limit is None: raise ValueError('MISSING_ORIGINAL_URDF_LIMIT:'+name)
        try:
            lo=float(limit.attrib['lower']);hi=float(limit.attrib['upper'])
        except (ValueError,KeyError) as e:raise ValueError('INVALID_URDF_LIMIT:'+name) from e
        if not np.isfinite([lo,hi]).all() or lo>=hi:
            raise ValueError('INVALID_REAL_URDF_ANGLE_BOUNDS:'+name)
        low.append(lo);high.append(hi)
    return np.array(low),np.array(high)


def verify_candidate_screen(predicted,measured_q0,qdot_limit,lower,upper,actual_current_qdot):
    """Reject position/speed violations, never return motor-ready status."""
    if predicted.ndim!=3 or predicted.shape[1:]!=(gen.H,29):
        raise ValueError('NOT_20_FRAME_ALL29_PROPOSAL')
    if measured_q0.shape!=(len(predicted),29) or actual_current_qdot.shape!=(len(predicted),29):
        raise ValueError('BAD_ACTUAL_JOINT_STATE')
    if any(x.shape!=(29,) for x in (qdot_limit,lower,upper)):
        raise ValueError('ORIGINAL_G1_JOINT_LIMITS_MISSING')
    if not all(np.isfinite(x).all() for x in (predicted,measured_q0,qdot_limit,lower,upper,actual_current_qdot)):
        raise ValueError('INVALID_ORIGINAL_SOURCE_OR_GENERATION')
    if np.any(qdot_limit<=0) or np.any(lower>=upper):
        raise ValueError('UNQUALIFIED_ORIGINAL_LIMITS')
    q=gen.integrate_measured_q0_29dof(predicted,measured_q0,qdot_limit)
    clipped=np.maximum(np.maximum(lower[None,None,:]-q,0),np.maximum(q-upper[None,None,:],0))
    violation=clipped>1e-8
    qdot_actual=predicted*qdot_limit[None,None,:]
    speed_violation=np.abs(predicted)>1.000001
    velocity_jump=np.max(np.abs(qdot_actual[:,0,:]-actual_current_qdot),axis=1)
    return {
        'proposal_windows':int(len(predicted)),
        'candidate_steps_per_window':gen.H,
        'original_all29_urdf_angle_violating_windows':int(violation.any(axis=(1,2)).sum()),
        'original_all29_urdf_angle_violating_samples':int(violation.sum()),
        'maximum_original_urdf_angle_excess_rad':float(clipped.max(initial=0.)),
        'first_generated_velocity_jump_from_real_qdot_max_rad_s':float(velocity_jump.max(initial=0.)),
        'maximum_normalized_original_qdot_ratio':float(np.abs(predicted).max(initial=0.)),
        'qdot_limit_violating_samples':int(speed_violation.sum()),
        'requires_original_200hz_support_contact_jacobians':True,
        'requires_authenticated_native_collider_distance_gradient':True,
        'requires_actual_original_joint_torque_inverse_dynamics':True,
        'true_human_expert_force_impact_supervision':False,
        'physical_actuation_authorized':False,
        'conclusion':'FAIL_CLOSED_OFFLINE_ROBOT_ONLY_PROPOSAL',
    }


def run(out:Path):
    if out.exists():raise FileExistsError('WRITE_ONCE_CANDIDATE_REJECTION_AUDIT')
    old_report=ROOT/'reports/v097_29dof_heldout_GH.audit.json'
    source=json.loads(old_report.read_text())
    train=[load(c) for c in gen.TRAIN]
    holdout=[load(c,'v97') for c in gen.TEST]
    if source['training_case_sha256']!={z['case']:z['source_sha256'] for z in train}:
        raise ValueError('FROZEN_TRAIN_SOURCE_SHA_MISMATCH')
    if source['holdout_case_sha256']!={z['case']:z['source_sha256'] for z in holdout}:
        raise ValueError('FROZEN_EVAL_SOURCE_SHA_MISMATCH')
    model=gen.learn(train,'full')
    src=ROOT/'configs/paid_g1_main.urdf'
    low,high=qualified_limits(src,train[0]['joint_names'])
    results={}
    for z in holdout:
        pred,truth,starts=gen.infer(model,z)
        q0=2*z['base'][starts,6:35]
        measured_current_qdot=z['base'][starts,35:64]*z['qdot_limit'][None,:]
        report=verify_candidate_screen(pred,q0,z['qdot_limit'],low,high,measured_current_qdot)
        original=source['holdout'][z['case']]['metrics']['state_and_support']
        new_mae=float(np.abs(pred-truth).mean())
        if not np.isclose(new_mae,original['mean_abs_29_qdot_normalized'],rtol=1e-9,atol=1e-12):
            raise ValueError('SOURCE_GENERATOR_OUTPUT_DRIFT:'+z['case'])
        results[z['case']]=report
    result=dict(version='v0.1.97',audit_type='ORIGINAL_ALL29_URDF_ANGLE_AND_SPEED_REJECTION_SCREEN',
                trained_on='original G1 robot-only v84 C/D, NOT elite human',
                base_generation_sha256=sha(old_report),original_urdf_sha256=sha(src),
                sources_exact_sha_verified=True,case_results=results,
                additional_unverified_requirements=['joint acceleration','joint torque','motor effort','contact forces',
                   'ground reaction force','support Jacobian','self-collision geometry','14s recovery','human naturalness'],
                authorization_to_actuate_original_G1=False)
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('V097_SAFETY_AUDIT',json.dumps({k:{kk:v for kk,v in val.items() if kk in (
        'proposal_windows','original_all29_urdf_angle_violating_windows',
        'maximum_original_urdf_angle_excess_rad','qdot_limit_violating_samples')}
        for k,val in results.items()}),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    run(a.output)

"""Full-clip contact-aware retargeting of frozen, source-derived CMU references.

Solve a sparse sequence QP with original-G1 capsule clearance, source-marker
support anchors, inter-frame velocity bounds, and human anatomical directions.
Root orientation and all 50 Hz timestamps remain the marker-derived observations.
"""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import osqp
import pinocchio as pin
from scipy import sparse
from scipy.spatial.transform import Rotation

import v160_human_reference as source


ROOT = Path(__file__).resolve().parents[1]
SIDES = ('left', 'right')
N_STATE = 32


def contiguous(mask):
    edge = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)))


class Geometry:
    def __init__(self, robot):
        self.robot = robot
        self.feet = [robot.index[side + '_ankle_roll_link'] for side in SIDES]

    def at(self, state, root_rot, gradients=True):
        robot = self.robot
        q, root = state[:29], state[-3:]
        if gradients:
            feat, jf = robot.features(q)
            ja = np.stack([pin.getFrameJacobian(robot.model, robot.data, robot.ids[k],
                           pin.LOCAL_WORLD_ALIGNED).copy() for k in self.feet])
            p = np.stack([robot.data.oMf[j].translation for j in robot.ids])
            rot = np.stack([robot.data.oMf[j].rotation for j in robot.ids])
        else:
            p, rot = robot.poses(q)
        world = root + p @ root_rot.T
        rotations = root_rot @ rot
        centers = [world[k] + foot[:, :3] @ rotations[k].T
                   for k, foot in zip(self.feet, robot.soles)]
        heights = np.stack([x[:, 2] - shapes[:, 3] for x, shapes in zip(centers, robot.soles)])
        if not gradients:
            return world, rotations, centers, heights
        jacobians = []
        for k, foot, joint_jac in zip(self.feet, robot.soles, ja):
            r = foot[:, :3] @ rot[k].T
            position_j = joint_jac[:3].T[None] + np.cross(joint_jac[3:].T[None], r[:, None])
            position_j = np.einsum('ab,knb->kan', root_rot, position_j)
            jacobians.append(np.concatenate((position_j, np.tile(np.eye(3), (len(foot), 1, 1))), axis=2))
        return feat, np.pad(jf, ((0, 0), (0, 0), (0, 3))), centers, heights, jacobians


def anchors(geom, states, orientations, support, minimum):
    n = len(states)
    lock = np.zeros((n, 2), dtype=bool)
    index = np.full((n, 2), -1, dtype=int)
    targets = np.zeros((n, 2, 2))
    sample = [geom.at(x, r, gradients=False) for x, r in zip(states, orientations)]
    for side in range(2):
        points = np.stack([frame[2][side] for frame in sample])
        heights = np.stack([frame[3][side] for frame in sample])
        for start, end in contiguous(support[:, side]):
            if end - start < minimum:
                continue
            # Standardized lowest capsule cap for the whole stance segment.
            k = int(np.argmin(np.median(heights[start:end], axis=0)))
            lock[start:end, side] = True
            index[start:end, side] = k
            targets[start:end, side] = np.median(points[start:end, k, :2], axis=0)
    return lock, index, targets


def local_terms(geom, state, original, orientation, targets, lock, index, anchor, cfg):
    feat, jf, centers, heights, jp = geom.at(state, orientation)
    weights = np.sqrt(cfg['anatomical_direction_weights']) * cfg['direction_weight']
    mats = [(weights[:, None, None] * jf).reshape(-1, N_STATE)]
    wanted = [((targets - feat) * weights[:, None]).ravel()]
    reg = np.r_[np.full(29, cfg['joint_reference_weight']), np.full(3, cfg['root_reference_weight'])]
    mats.append(np.diag(reg))
    wanted.append(reg * (original - state))
    for j in range(2):
        if lock[j]:
            k = index[j]
            # A material point is fixed horizontally; allow orientation/rolling about that point.
            mats.append(cfg['support_xy_weight'] * jp[j][k, :2])
            wanted.append(cfg['support_xy_weight'] * (anchor[j] - centers[j][k, :2]))
            lowest = heights[j].argmin()
            mats.append(cfg['support_z_weight'] * jp[j][lowest, 2:3])
            wanted.append(np.array([cfg['support_z_weight'] * (cfg['stance_floor_m'] - heights[j, lowest])]))
    a, b = np.vstack(mats), np.concatenate(wanted)
    jz = np.vstack([j[:, 2] for j in jp])
    return a.T @ a, -(a.T @ b), jz, heights.ravel()


def difference(n, order):
    weights = [-1., 1.] if order == 1 else [1., -2., 1.]
    return sparse.diags(weights, np.arange(order + 1), shape=(n - order, n), format='csc')


def solve(geom, old, orientations, targets, support, cfg):
    robot = geom.robot
    n = len(old)
    state = old.copy()
    lock, endpoints, anchored = anchors(geom, old, orientations, support, cfg['minimum_support_samples'])
    identity = sparse.eye(n * N_STATE, format='csc')
    select = sparse.hstack([sparse.eye(29), sparse.csc_matrix((29, 3))], format='csc')
    diff1 = difference(n, 1)
    speed = sparse.kron(diff1, select, format='csc')
    edge_bound = np.tile(robot.vlim * cfg['velocity_limit_fraction'] / cfg['fps'], n - 1)
    motion_reg = sparse.csc_matrix((n*N_STATE, n*N_STATE))
    for order in (1, 2):
        coeff = np.r_[np.full(29, cfg[f'joint_corr_diff{order}_weight']),
                      np.full(3, cfg[f'root_corr_diff{order}_weight'])]
        diff = sparse.kron(difference(n, order), sparse.diags(coeff), format='csc')
        motion_reg += diff.T @ diff
    linear_reg = motion_reg + cfg['step_diagonal'] * identity
    lower = np.c_[np.tile(robot.low, (n, 1)), old[:, -3:] - cfg['root_max_offset_m']]
    upper = np.c_[np.tile(robot.high, (n, 1)), old[:, -3:] + cfg['root_max_offset_m']]
    step = np.tile(np.r_[np.full(29, cfg['joint_step_rad']), np.full(3, cfg['root_step_m'])], n)
    stages = []
    for count in range(cfg['sqp_iterations']):
        pieces = [local_terms(geom, x, orig, o, t, l, k, a, cfg)
                  for x, orig, o, t, l, k, a in zip(state, old, orientations, targets, lock, endpoints, anchored)]
        h, lin, floor_j, height = [list(g) for g in zip(*pieces)]
        floor_j = sparse.block_diag([sparse.csc_matrix(f) for f in floor_j], format='csc')
        p = sparse.block_diag([sparse.csc_matrix(v) for v in h], format='csc') + linear_reg
        grad = np.concatenate(lin) + motion_reg @ (state - old).ravel()
        constraints = sparse.vstack((identity, floor_j, speed), format='csc')
        floors = cfg['minimum_floor_m'] - np.concatenate(height)
        velocity = speed @ state.ravel()
        lo = np.r_[np.maximum((lower-state).ravel(), -step), floors, -edge_bound-velocity]
        hi = np.r_[np.minimum((upper-state).ravel(), step), np.full(len(floors), np.inf), edge_bound-velocity]
        solver = osqp.OSQP()
        solver.setup(P=sparse.triu(p).tocsc(), q=grad, A=constraints, l=lo, u=hi,
                     verbose=False, eps_abs=cfg['qp_tolerance'], eps_rel=cfg['qp_tolerance'],
                     max_iter=cfg['qp_max_iterations'], polish=True)
        sol = solver.solve()
        status = sol.info.status
        if status not in ('solved', 'solved inaccurate') or sol.x is None:
            raise RuntimeError(f'Iteration {count}: OSQP {status}')
        violation = max(0., float((lo-constraints@sol.x).max()), float((constraints@sol.x-hi).max()))
        if violation > cfg['max_qp_primal_residual']:
            raise RuntimeError(f'Iteration {count}: excessive linear violation {violation}')
        state += sol.x.reshape(n, N_STATE)
        stages.append({'status': status, 'iterations': sol.info.iter,
                       'max_linear_constraint_violation': violation, 'max_step': float(np.max(np.abs(sol.x)))})
    return state, lock, endpoints, anchored, stages


def metrics(geom, states, old, orient, targets, support, lock, endpoints, anchors_xy, cfg):
    robot = geom.robot
    dt = 1. / cfg['fps']
    poses = [geom.at(x, o, gradients=False) for x,o in zip(states, orient)]
    world = np.stack([v[0] for v in poses])
    rotation = np.stack([v[1] for v in poses])
    height = np.stack([v[3].min(axis=1) for v in poses])
    centers = np.stack([np.stack(v[2]) for v in poses])
    k = np.maximum(endpoints, 0)
    selected = centers[np.arange(len(states))[:, None], np.arange(2)[None], k, :2]
    both = lock[1:] & lock[:-1] & (k[1:] == k[:-1])
    point_speed = np.linalg.norm(np.diff(selected, axis=0), axis=-1) / dt
    point_speed = point_speed[both]
    dist_to_anchor = np.linalg.norm(selected - anchors_xy, axis=-1)[lock]
    ankle_v = np.linalg.norm(np.gradient(world[:, geom.feet, :2], dt, axis=0), axis=-1)
    foot_xyz_speed = np.linalg.norm(np.gradient(world[:, geom.feet], dt, axis=0), axis=2)
    joint_v = np.diff(states[:, :29], axis=0) / dt
    joint_acc = np.diff(joint_v, axis=0) / dt
    value = np.stack([robot.features(x[:29])[0] for x in states])
    err = np.degrees(np.arccos(np.clip(np.sum(value*targets, axis=-1), -1, 1)))
    # Measures the temporal evolution of the 18 true source directions independently of their offsets.
    robot_delta, human_delta = np.diff(value,axis=0) / dt, np.diff(targets,axis=0) / dt
    cos = np.sum(robot_delta*human_delta) / (np.linalg.norm(robot_delta)*np.linalg.norm(human_delta))
    def quant(raw, fraction=.95):
        return float(np.quantile(raw, fraction)) if raw.size else None
    metrics = dict(
        frames=len(states), raw_source_support_samples=int(support.sum()), locked_source_support_samples=int(lock.sum()),
        feet_below_plane_over_10mm_frames=int((height.min(axis=1) < -.01).sum()),
        lowest_sole_capsule_m=float(height.min()), source_support_penetration_over_1mm_samples=int((height[support] < -.001).sum()),
        source_support_hover_over_10mm_samples=int((height[support] > .01).sum()),
        source_support_hover_over_20mm_samples=int((height[support] > .02).sum()),
        source_support_height_abs_p95_m=quant(abs(height[support])),
        human_airborne_proxy_frames=int((~support.any(axis=1)).sum()),
        source_any_support_both_robot_feet_above_10mm_frames=int(np.sum(support.any(axis=1)&(height.min(axis=1)>.01))),
        ankle_xy_speed_on_all_source_quiet_p95_mps=quant(ankle_v[support]),
        locked_material_point_xy_speed_p95_mps=quant(point_speed), locked_material_point_xy_speed_max_mps=float(point_speed.max()),
        locked_material_point_displacement_to_anchor_p95_m=quant(dist_to_anchor),
        direction_mean_deg=float(err.mean()), direction_p95_deg=quant(err), direction_mean_by_group_deg=err.mean(axis=0).tolist(),
        human_robot_direction_velocity_cosine=float(cos),
        max_joint_speed_ratio=float(np.max(np.abs(joint_v) / robot.vlim)),
        joint_position_bound_excess_rad=float(np.maximum(np.maximum(robot.low-states[:,:29], states[:,:29]-robot.high), 0).max()),
        joint_acceleration_rms_rad_s2=float(np.sqrt(np.mean(joint_acc ** 2))),
        joint_acceleration_abs_p95_rad_s2=quant(abs(joint_acc)),
        root_xyz_correction_rms_m=float(np.sqrt(np.mean(np.sum((states[:,-3:]-old[:,-3:])**2, axis=1)))),
        root_z_correction_range_m=[float(x) for x in [np.min(states[:,-1]-old[:,-1]), np.max(states[:,-1]-old[:,-1])]],
        peak_left_right_ankle_3d_speed_mps=foot_xyz_speed.max(axis=0).tolist())
    return world, rotation, height, err, metrics


def build(clip, split, prev_dir, outdir, meta, cfg, geom, validations):
    original_file = prev_dir / (clip+'.npz')
    if source.sha(original_file) != meta['npz_sha256']:
        raise ValueError('Frozen reference digest mismatch: '+clip)
    with np.load(original_file, allow_pickle=False) as npz:
        arr = {k: npz[k].copy() for k in npz.files}
    robot = geom.robot
    if (arr['joint_names'].tolist()!=robot.joint_names or arr['body_names'].tolist()!=robot.body_names
            or arr['split'].item()!=split):
        raise ValueError('Incompatible original ordering/split: '+clip)
    orient=Rotation.from_quat(arr['root_quat_wxyz'][:, [1,2,3,0]]).as_matrix()
    angles = arr['joint_pos'][:, np.argsort(robot.runtime_indices)].astype(float)
    old=np.c_[angles, arr['root_pos_w'].astype(float)]
    target=arr['human_segment_direction_root'].astype(float)
    target/=np.linalg.norm(target,axis=-1,keepdims=True)
    support=arr['source_low_quiet_foot_proxy']
    state, lock, ep, anchor, trace = solve(geom, old, orient, target, support, cfg)
    args=(old, orient, target, support, lock, ep, anchor, cfg)
    prior=metrics(geom, old, *args)[-1]
    w,r,h,e,new=metrics(geom, state, *args)
    dt=1./cfg['fps']
    q=state[:,robot.runtime_indices]
    quat=source.angle_quaternions(r)
    arr.update(joint_pos=q.astype('f4'), joint_vel=np.gradient(q,dt,axis=0,edge_order=2).astype('f4'),
               root_pos_w=state[:,-3:].astype('f4'), root_quat_wxyz=quat[:,0].astype('f4'),
               body_pos_w=w.astype('f4'), body_quat_w=quat.astype('f4'),
               body_lin_vel_w=np.gradient(w,dt,axis=0,edge_order=2).astype('f4'),
               body_ang_vel_w=source.world_angular_velocity(r,dt).astype('f4'),
               urdf_sole_height_m=h.astype('f4'), ik_feature_error_deg=e.astype('f4'),
               locked_source_support_proxy=lock, locked_capsule_endpoint_index=ep,
               support_point_anchor_xy_m=anchor.astype('f4'))
    if max(new['joint_position_bound_excess_rad'], new['max_joint_speed_ratio']-1.) > 1e-5:
        raise ValueError(f'Original limit violation in {clip}')
    if not all(np.isfinite(arr[k]).all() for k in ['joint_pos','body_pos_w','body_quat_w','joint_vel']):
        raise ValueError('Invalid reference output')
    result=outdir/(clip+'.npz')
    np.savez_compressed(result,**arr)
    validations.append(dict(clip=clip, split=split, before=prior, after=new, iterations=trace,
          npz_file=result.name, npz_sha256=source.sha(result), original_npz_sha256=source.sha(original_file),
          source_c3d_sha256=meta['source_c3d_sha256'], source_processed_sha256=meta['source_processed_sha256'],
          minimum_source_frame=meta['source_frame_range_inclusive'][0],maximum_source_frame=meta['source_frame_range_inclusive'][1],
          player_id=meta['subject_id'], verified_first_contact_s=None, player_level_verified=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/v162_human_support.json')
    parser.add_argument('--original',type=Path,default=ROOT/'experiments/v160_human')
    parser.add_argument('--asset-root',type=Path,default=Path('/home/you/projects/kinetickick'))
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/v162_human/references')
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    basefile=args.original/'manifest.json'
    if (source.sha(basefile)!=cfg['original_manifest_sha256']
            or source.sha(args.asset_root/cfg['urdf'])!=cfg['urdf_sha256']):
        raise ValueError('Frozen source or robot changed')
    base=json.loads(basefile.read_text())
    schema=json.loads((args.asset_root/cfg['schema']).read_text())
    geom=Geometry(source.Robot(args.asset_root/cfg['urdf'], schema))
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    records=[]
    with tempfile.TemporaryDirectory(prefix='human-qp-',dir=args.output.parent) as tmp:
        dest=Path(tmp)
        for split, clips in cfg['splits'].items():
            for clip in clips:
                build(clip,split,args.original,dest,base['clips'][clip],cfg,geom,records)
                done=records[-1]['after']
                print(clip,'min z',round(done['lowest_sole_capsule_m'],5),
                      'p95 locked material speed',round(done['locked_material_point_xy_speed_p95_mps'],3),flush=True)
        report=dict(version='v0.1.162', method='full-clip contact-aware floating-root and articulated SQP',
            source_qualification=base['source_qualification'], source_license=base['source_license'],
            splits=cfg['splits'], registered_features=base['registered_task_names'], records=records,
            protocol_sha256=source.sha(args.config), core_code_sha256=source.sha(Path(__file__)),
            base_code_sha256=source.sha(Path(source.__file__)), source_manifest_sha256=source.sha(basefile),
            osqp_version=osqp.__version__, pinocchio_version=pin.__version__, trackability_qualification=None,
            physics_clip=cfg['physics_clip'], known_limits=cfg['scope'])
        (dest/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        dest.rename(args.output)


if __name__=='__main__':
    main()

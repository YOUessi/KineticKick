"""CPU-only checks for PAiD's imported capsules and sphere, not dynamics.

The runtime replaces each URDF cylinder by a capsule with the same segment
length and radius, confirmed against converted USD. Ground support and signed
segment-distance clearance are exact for these imported capsules/spheres.
Directly adjacent links are excluded: converted elbow joints explicitly have
physics:collisionEnabled=False. The pelvis/thigh pair is not adjacent.
"""
from itertools import combinations
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def segment_distance(p1, q1, p2, q2):
    """Minimum distance between two closed line segments, including points."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a < 1e-12 and e < 1e-12:
        return float(np.linalg.norm(r))
    if a < 1e-12:
        s, t = 0., np.clip(f / e, 0., 1.)
    else:
        c = d1 @ r
        if e < 1e-12:
            t, s = 0., np.clip(-c / a, 0., 1.)
        else:
            b = d1 @ d2
            denominator = a * e - b * b
            s = np.clip((b * f - c * e) / denominator, 0., 1.) if denominator > 1e-12 else 0.
            t = (b * s + f) / e
            if t < 0.:
                t, s = 0., np.clip(-c / a, 0., 1.)
            elif t > 1.:
                t, s = 1., np.clip((b - c) / a, 0., 1.)
    return float(np.linalg.norm(p1 + s * d1 - p2 - t * d2))



class ReferenceGeometry:
    """Read the repository's verified runtime schema and original PAiD URDF."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        tools_path = str(self.root / 'tools')
        if tools_path not in sys.path:
            sys.path.insert(0, tools_path)
        from audit_paid_urdf_reference import load_urdf, fk_at_frame

        self.fk_at_frame = fk_at_frame
        self.urdf_path = self.root / 'configs/paid_original/main.urdf'
        self.schema_path = self.root / 'reports/paid_physics_parity_clean_paid.json'
        schema = json.loads(self.schema_path.read_text())
        self.joint_names, self.body_names = schema['joint_names'], schema['body_names']
        self.tree, urdf_joints = load_urdf(self.urdf_path)
        if (len(self.joint_names) != 29 or len(self.body_names) != 30
                or len(set(self.joint_names)) != 29 or len(set(self.body_names)) != 30
                or set(self.joint_names) != set(urdf_joints)
                or self.body_names[0] != 'pelvis'):
            raise ValueError('Verified PAiD runtime schema mismatch')
        self.xml = ET.parse(self.urdf_path).getroot()
        self.adjacent = {
            frozenset((j.find('parent').attrib['link'], j.find('child').attrib['link']))
            for j in self.xml.findall('joint')
        }
        self.colliders = []
        for link in self.xml.findall('link'):
            for index, collision in enumerate(link.findall('collision')):
                origin = collision.find('origin')
                attrs = origin.attrib if origin is not None else {}
                translation = np.fromstring(attrs.get('xyz', '0 0 0'), sep=' ')
                rotation = Rotation.from_euler('xyz', np.fromstring(attrs.get('rpy', '0 0 0'), sep=' '))
                shape = list(collision.find('geometry'))[0]
                if shape.tag not in ('cylinder', 'sphere'):
                    raise ValueError('Unsupported collision primitive: ' + shape.tag)
                self.colliders.append((link.attrib['name'], index, translation, rotation,
                                       shape.tag, float(shape.attrib['radius']),
                                       float(shape.attrib.get('length', 0.))))

    def body_poses(self, q, position, quaternion_wxyz):
        return {name: (xyz, rotation) for name, xyz, rotation in self.fk_at_frame(
            self.tree, dict(zip(self.joint_names, q)), position, quaternion_wxyz)}

    def primitives(self, q, position, quaternion_wxyz):
        poses = self.body_poses(q, position, quaternion_wxyz)
        result = []
        for name, index, translation, local_rotation, kind, radius, length in self.colliders:
            xyz, body_rotation = poses[name]
            center = xyz + body_rotation.apply(translation)
            rotation = body_rotation * local_rotation
            axis = rotation.apply([0., 0., 1.])
            if kind == 'cylinder':
                lowest = center[2] - abs(axis[2]) * length / 2 - radius
            else:
                lowest = center[2] - radius
            result.append(dict(name=name, index=index, center=center, rotation=rotation,
                               kind=kind, radius=radius, half_length=length / 2,
                               start=center - axis * length / 2,
                               end=center + axis * length / 2, lowest=float(lowest)))
        return result

    def pair_gaps(self, primitives):
        for a, b in combinations(primitives, 2):
            if (a['name'] == b['name']
                    or frozenset((a['name'], b['name'])) in self.adjacent):
                continue
            gap = (segment_distance(a['start'], a['end'], b['start'], b['end'])
                   - a['radius'] - b['radius'])
            yield dict(body_a=a['name'], body_b=b['name'], capsule_gap_m=float(gap))

    def check(self, joints, positions, quaternions_wxyz):
        ground, nonfoot, overlaps = [], [], []
        minimum_pair = None
        hip_gaps = {'left': [], 'right': []}
        for frame, q in enumerate(joints):
            primitives = self.primitives(q, positions[frame], quaternions_wxyz[frame])
            pelvis = next(g for g in primitives if g['name'] == 'pelvis')
            for side in hip_gaps:
                hip = next(g for g in primitives if g['name'] == side + '_hip_roll_link')
                gap = segment_distance(pelvis['start'], pelvis['end'], hip['start'], hip['end']) - pelvis['radius'] - hip['radius']
                hip_gaps[side].append(float(gap))
            ground.append(min(g['lowest'] for g in primitives if 'ankle_roll' in g['name']))
            nonfoot.append(min(g['lowest'] for g in primitives if 'ankle_roll' not in g['name']))
            for pair in self.pair_gaps(primitives):
                pair['frame'] = frame
                if minimum_pair is None or pair['capsule_gap_m'] < minimum_pair['capsule_gap_m']:
                    minimum_pair = pair
                if pair['capsule_gap_m'] < 0.:
                    overlaps.append(pair)
        return dict(
            pelvis_hip_capsule_gaps_m={side: {'initial': values[0], 'min': min(values), 'min_frame': int(np.argmin(values)), 'max': max(values)} for side, values in hip_gaps.items()},
            runtime_primitive_model='URDF cylinders become capsules: segment height=URDF length; radius unchanged. Sphere unchanged.',
            min_foot_z=float(min(ground)), initial_foot_z=float(ground[0]),
            max_minfoot_z=float(max(ground)), min_nonfoot_z=float(min(nonfoot)),
            min_nonadjacent_capsule_gap_m=minimum_pair['capsule_gap_m'],
            closest_nonadjacent_pair=minimum_pair,
            nonadjacent_capsule_overlap_frames=len({p['frame'] for p in overlaps}),
            nonadjacent_capsule_overlap_examples=overlaps[:8],
            caveat='URDF cylinders are treated as runtime capsules with unchanged cylinder-segment length, matching PAiD replace_cylinders_with_capsules=True. Signed segment-distance gaps are exact for capsules/spheres. Directly adjacent links excluded.'
        ), np.asarray(ground)

"""Render paired original-mesh PhysX traces with a separately labelled source inset."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from free_shooting_metrics import goal_crossing
from v160_tracking_preview import (AQUA, GOLD, MUTED, RED, WHITE, Camera, load_geometry,
                                   panel_background, point_marker, render_robot, unit, write_text)
from v161_preview import goal_mesh

FPS = 25
ROOT = Path(__file__).resolve().parents[1]
DARK = (15, 26, 33)
RULE = (62, 87, 90)
FOOT_NAMES = {'left_ankle_roll_link': '左脚', 'right_ankle_roll_link': '右脚'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrices(quaternions):
    return Rotation.from_quat(quaternions[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(*quaternions.shape[:-1], 3, 3)


def load_run(folder):
    report = json.loads((folder/'tracking_result.json').read_text())
    trial_path = folder/'trial.json'
    trial = json.loads(trial_path.read_text()) if trial_path.exists() else report
    with np.load(folder/'rollout.npz', allow_pickle=False) as z:
        trace = {k: z[k] for k in ('time_s', 'body_position_w_m', 'body_quat_wxyz', 'ball_position_w_m', 'ball_velocity_w_mps',
                                   'foot_ground_normal_load_N', 'robot_ball_contact')}
    for p, h in report['local_artifact_sha256'].items():
        if digest(folder/p) != h:
            raise ValueError('Trace changed relative to its run summary')
    t = trace['time_s']
    if not np.array_equal(t, np.arange(1, 2801) * .005) or report['additional_state_writes_after_initialization']:
        raise ValueError('Expected a single uninterrupted 14-second 200Hz episode')
    data = dict(t=t, body=trace['body_position_w_m'], rot=matrices(trace['body_quat_wxyz']),
                ball=trace['ball_position_w_m'], vel=trace['ball_velocity_w_mps'], load=trace['foot_ground_normal_load_N'],
                names=report['source_body_names'], digests=report['local_artifact_sha256'], trial=trial)
    geom = dict(goal_xyz_m=trial['goal_xyz_m'], goal_forward_xy=trial['goal_forward_xy'],
                goal_width_height_m=trial.get('goal_width_height_m', [1.6, 1.1]))
    data['geometry'] = geom
    events = [(0.0, '跟踪参考动作', MUTED)]
    contacts = np.flatnonzero(trace['robot_ball_contact'])
    impact = None
    if contacts.size:
        tick = int(contacts[0] + 1)
        with gzip.open(folder/'native_contacts.jsonl.gz', 'rt') as file:
            for line in file:
                i, rows = json.loads(line)
                if i == tick:
                    break
            else:
                raise ValueError('Missing first contact log')
        actors = set()
        for pair in rows:
            pair_actors = set(pair['actors'])
            if 'SoccerBall' in pair_actors and 'ground' not in pair_actors:
                net = np.array(pair['points'])[:, 6:9].sum(axis=0)
                if np.linalg.norm(net) / .005 > 2.:
                    actors.update(pair_actors - {'SoccerBall'})
        if not actors:
            raise ValueError('First-contact pair cannot be authenticated')
        label = '、'.join(FOOT_NAMES.get(n, '其他部位') for n in sorted(actors))
        impact = tick * .005
        events.append((impact, f'{label}首次触球  {impact:.3f} s', GOLD))
        if contacts[-1] + 1 < len(t):
            i = contacts[-1] + 1
            events.append((float(t[i]), f'脱离时球速  {np.linalg.norm(data["vel"][i, :2]):.2f} m/s', GOLD))
    data['impact'] = impact
    crossed = False
    for i in range(1, len(t)):
        old, new = data['ball'][i-1:i+1]
        f = np.array(geom['goal_forward_xy'], dtype=float)
        g = np.array(geom['goal_xyz_m'][:2], dtype=float)
        if goal_crossing(old, new, goal=g, forward=f, width=geom['goal_width_height_m'][0],
                         height=geom['goal_width_height_m'][1], radius=.11):
            frac = (.11 - (old[:2] - g) @ f) / ((new[:2] - old[:2]) @ f)
            cross = float(t[i-1] + frac * .005)
            events.append((cross, f'整球进门  {cross:.3f} s', AQUA))
            crossed = True
            break
    if crossed != report['goal_crossed_whole_ball']:
        raise ValueError('Native ball path disagrees with reported goal')
    fall = min([v for v in (report['first_height_below_0p45m_s'], report['first_tilt_over_60deg_s']) if v is not None], default=None)
    if fall is not None:
        events.append((fall, f'触及跌倒阈值  {fall:.3f} s', RED))
    tail = report['final_standing_seconds_cpu_definition']
    score_file = folder/'score.json'
    if score_file.exists():
        score = json.loads(score_file.read_text())
        if 'strict_standing_tail_s' in score:
            checksums = {'rollout.npz': score['rollout_sha256'],
                         'native_contacts.jsonl.gz': score['contact_sha256']}
            tail = score['strict_standing_tail_s']
        else:
            checksums = score['input_sha256']
            tail = score['terminal_standing_seconds']
        for name, checksum in checksums.items():
            if digest(folder/name) != checksum:
                raise ValueError('Standing score belongs to a different run')
    data.update(events=sorted(events, key=lambda x:x[0]), tail=tail,
                fall=fall, goal=crossed)
    return data


def camera():
    cam = Camera()
    cam.depth = unit(np.array([-.85, -3.8, 1.22]))
    cam.right = unit(np.cross([0, 0, 1], cam.depth))
    cam.up = unit(np.cross(cam.depth, cam.right))
    cam.proj = np.stack((cam.right, -cam.up, cam.depth))
    cam.center = np.array([.04, -.30, .58])
    cam.scale = 161.
    return cam


def source_pose(cam, shape, p, r):
    cam.center = np.r_[p[0, :2], .58]
    panel = panel_background(cam, False)
    render_robot(panel, cam, shape, p, r)
    # Crop around the robot with the source camera tracking the reference pelvis.
    return Image.fromarray(panel[83:313, 175:455]).resize((201, 165), Image.Resampling.LANCZOS)


def overhead(run, i):
    """Map at the goal end keeps the ball measurable when it leaves the close-up."""
    fwd = np.asarray(run['geometry']['goal_forward_xy'])
    lateral = np.array([-fwd[1], fwd[0]])
    center = np.array(run['geometry']['goal_xyz_m'][:2])
    def px(xy):
        d = np.asarray(xy) - center
        return np.rint(np.stack((134 + 22*(d @ fwd), 46 - 22*(d @ lateral)), axis=-1)).astype(np.int32)
    tile = np.full((89, 157, 3), (21, 45, 47), dtype=np.uint8)
    width = run['geometry']['goal_width_height_m'][0]
    g = np.stack((center-lateral*width/2, center+lateral*width/2))
    cv2.polylines(tile, [px(np.stack((g[0],g[0]+.25*fwd,g[1]+.25*fwd,g[1])))], False, (90,145,128), 2, cv2.LINE_AA)
    cv2.line(tile, tuple(px(g[0])), tuple(px(g[1])), (183,217,203), 2, cv2.LINE_AA)
    trail = px(run['ball'][:i+1:5, :2])
    if len(trail)>1:
        cv2.polylines(tile, [trail], False, GOLD, 1, cv2.LINE_AA)
    cv2.circle(tile, tuple(px(run['body'][i,0,:2])), 3, AQUA, -1, cv2.LINE_AA)
    cv2.circle(tile, tuple(px(run['ball'][i,:2])), 4, GOLD, -1, cv2.LINE_AA)
    cv2.rectangle(tile, (0,0), (156,88), (70,97,95), 1)
    return Image.fromarray(tile)


def scene(run, index, shapes, cam, bg):
    panel = bg.copy()
    render_robot(panel, cam, shapes, run['body'][index], run['rot'][index])
    point = run['ball'][index]
    loc = cam.pixels(point)
    trail = cam.pixels(run['ball'][max(0, index-50):index+1:2])
    if len(trail) > 1:
        cv2.polylines(panel, [trail], False, (193, 148, 66), 2, cv2.LINE_AA)
    if 0 <= loc[0] < panel.shape[1] and 0 <= loc[1] < panel.shape[0]:
        radius = int(.11 * cam.scale)
        point_marker(panel, cam, point, GOLD, radius, -1)
        point_marker(panel, cam, point, (255, 230, 164), radius, 2)
        if run['impact'] is not None and abs(run['t'][index] - run['impact']) < .10:
            point_marker(panel, cam, point, AQUA, radius + 5, 2)
    return Image.fromarray(panel)


def draw_stats(d, run, index, x, source_ended):
    t = run['t'][index]
    active = [e for e in run['events'] if e[0] <= t + 1e-8]
    _, label, color = active[-1]
    if source_ended and run['fall'] is None and len(active) == 1:
        label = '已到真人动作末帧'
    write_text(d, (x, 551), label, 24, color, True)
    left, right = run['load'][index]
    write_text(d, (x, 591), f'足底载荷    左 {left:.0f} N   右 {right:.0f} N', 19, WHITE)
    write_text(d, (x, 621), f'骨盆高度    {run["body"][index, 0, 2]:.2f} m', 19, MUTED)
    tail = np.clip(t - (14.0 - run['tail']), 0., run['tail'])
    if tail > 0:
        write_text(d, (x, 651), f'末段连续站稳    {tail:.2f} s', 19, AQUA)
    else:
        write_text(d, (x, 651), f'实时水平球速    {np.linalg.norm(run["vel"][index, :2]):.2f} m/s', 19, MUTED)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ('before', 'after', 'reference', 'output'):
        cli.add_argument('--' + key, type=Path, required=True)
    cli.add_argument('--urdf', type=Path, default=Path('/home/you/projects/kinetickick/configs/paid_original/paid_main_resolved.urdf'))
    cli.add_argument('--camera-scale', type=float, help='Pixels per world metre for both side-by-side panels')
    cli.add_argument('--camera-center-xy', type=float, nargs=2, metavar=('X', 'Y'),
                     help='World-coordinate center for off-origin source scenes')
    cli.add_argument('--after-label', default='改进后')
    cli.add_argument('--reference-clock', type=Path, help='NPZ with time_s and source_time_s mapping elapsed simulation seconds to original 50 Hz reference time')
    cli.add_argument('--frame-only', type=float, help='Render just the given physical time to a PNG for review')
    a = cli.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    runs = [load_run(a.before), load_run(a.after)]
    same_goal = all(np.allclose(runs[0]['geometry'][k], runs[1]['geometry'][k], atol=1.e-5, rtol=0) for k in runs[0]['geometry'])
    if runs[0]['names'] != runs[1]['names'] or not np.allclose(runs[0]['trial']['ball_initial_xyz_m'], runs[1]['trial']['ball_initial_xyz_m'], atol=1.e-5, rtol=0) or not same_goal:
        raise ValueError('Side-by-side requires the same robot, ground ball and goal')
    with np.load(a.reference) as ref:
        rp = ref['body_pos_w']
        rr = matrices(ref['body_quat_w'])
        rt = ref['time_s']
        if runs[0]['names'] != ref['body_names'].tolist():
            raise ValueError('Source and robot use different body order')
    if a.reference_clock:
        with np.load(a.reference_clock) as clock:
            clock_t, clock_src = clock['time_s'], clock['source_time_s']
        if not (len(clock_src) == len(clock_t) and np.all(np.diff(clock_t)>0) and np.all(np.diff(clock_src)>=-1e-8)):
            raise ValueError('Invalid reference clock')
    def source_time(t):
        return np.clip(t if a.reference_clock is None else np.interp(t, clock_t, clock_src), 0, rt[-1])
    cv2.setNumThreads(1)
    shapes = load_geometry(a.urdf, runs[0]['names'])
    cam = camera()
    if a.camera_center_xy is not None:
        if not np.isfinite(a.camera_center_xy).all():
            cli.error('Camera center must be finite')
        cam.center[:2] = a.camera_center_xy
    if a.camera_scale is not None:
        if not np.isfinite(a.camera_scale) or a.camera_scale <= 0:
            cli.error('Camera scale must be finite and positive')
        cam.scale = a.camera_scale
    cam_ref = camera()
    cam_ref.scale = 136.
    bg = panel_background(cam, False)
    goal_mesh(bg, cam, runs[0]['geometry'])
    def render(i):
        t = runs[0]['t'][i]
        src = source_time(t)
        ri = min(len(rt)-1, round(src * 50))
        image = Image.new('RGB', (1280, 720), DARK)
        image.paste(scene(runs[0], i, shapes, cam, bg), (0, 101))
        image.paste(scene(runs[1], i, shapes, cam, bg), (640, 101))
        image.paste(source_pose(cam_ref, shapes, rp[ri], rr[ri]), (541, 538))
        image.paste(overhead(runs[1],i),(654,111))
        d = ImageDraw.Draw(image)
        write_text(d, (23, 4), '真人射门动作迁移', 33, WHITE, True)
        write_text(d, (1255, 15), f'{t:05.2f} / 14.00 s', 24, AQUA, True, 'ra')
        d.line((0, 57, 1280, 57), fill=RULE, width=2)
        write_text(d, (24, 62), '改进前  ·  PAiD', 24, WHITE, True)
        write_text(d, (666, 62), a.after_label, 24, AQUA, True)
        d.line((640, 57, 640, 531), fill=RULE, width=2)
        write_text(d, (661, 111), '俯视球路', 15, MUTED)
        d.line((0, 532, 1280, 532), fill=RULE, width=2)
        draw_stats(d, runs[0], i, 24, t >= rt[-1])
        draw_stats(d, runs[1], i, 778, src >= rt[-1])
        write_text(d, (641, 539), '真人重定向参考', 15, WHITE, True, 'ma')
        write_text(d, (641, 676), f'末帧 {rt[-1]:.2f} s' if src >= rt[-1] - 1e-8 else f'源时间 {src:.2f} s', 15, MUTED, True, 'ma')
        d.line((20, 708, 1260, 708), fill=RULE, width=3)
        d.line((20, 708, 20+1240*t/14., 708), fill=AQUA, width=3)
        return image
    a.output.parent.mkdir(exist_ok=True, parents=True)
    if a.frame_only is not None:
        if a.output.suffix.lower() != '.png' or not 0.005 <= a.frame_only <= 14:
            raise ValueError('Expected a PNG and time inside a physical episode')
        idx = int(np.argmin(abs(runs[0]['t'] - a.frame_only)))
        render(idx).save(a.output)
        return
    temp = a.output.with_name('.' + a.output.stem + '.encoding.mp4')
    if temp.exists():
        raise FileExistsError(temp)
    cmd = ['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-f','rawvideo','-pix_fmt','rgb24','-video_size','1280x720',
           '-framerate',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)]
    child = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for idx in range(7, 2800, 8):
            child.stdin.write(render(idx).tobytes())
        child.stdin.close()
        if child.wait():
            raise RuntimeError('Encoding failed')
        os.link(temp, a.output)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        temp.unlink(missing_ok=True)
    print(json.dumps(dict(video=str(a.output), sha256=digest(a.output), before=runs[0]['digests'], after=runs[1]['digests'],
                          source_sha256=digest(a.reference), frames=350, duration_s=14), ensure_ascii=False))


if __name__ == '__main__':
    main()

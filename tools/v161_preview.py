"""Render a measured v161 CPU trajectory, annotating only events read from its matching score."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation

from v160_tracking_preview import (W,H,PANEL_HEIGHT,FPS,WHITE,AQUA,GOLD,MUTED,RED, Camera, load_geometry,
                                    plot_world_line, render_robot, point_marker, write_text, panel_background)
from v161_evaluate import digest, DEFAULT_URDF

BODY_LABEL={'left_wrist_yaw_link':'左腕','right_wrist_yaw_link':'右腕','left_hip_roll_link':'左髋',
            'right_hip_roll_link':'右髋','pelvis':'骨盆','torso_link':'躯干','ground':'地面','SoccerBall':'足球'}


def goal_mesh(canvas,cam,score):
    forward=np.asarray(score['goal_forward_xy']);lateral=np.array([-forward[1],forward[0]])
    center=np.asarray(score['goal_xyz_m'])[:2];width,height=score['goal_width_height_m']
    def p(x,depth,z):return np.r_[center+lateral*x+forward*depth,z]
    for x in (-width/2,width/2):
        plot_world_line(canvas,cam,p(x,0,0),p(x,0,height),(180,216,203),3)
        plot_world_line(canvas,cam,p(x,0,height),p(x,.32,height),(101,151,134),2)
    plot_world_line(canvas,cam,p(-width/2,0,height),p(width/2,0,height),(180,216,203),3)
    for x in np.linspace(-width/2,width/2,9):
        plot_world_line(canvas,cam,p(x,.32,0),p(x,.32,height),(91,130,116))
    for z in np.linspace(0,height,7):
        plot_world_line(canvas,cam,p(-width/2,.32,z),p(width/2,.32,z),(91,130,116))


def event_labels(score):
    events=[(0.,'助跑、摆腿与支撑',WHITE)]
    hit=score['earliest_loaded_right_foot_s']
    if hit is not None:events.append((hit,f'{hit:.3f} 秒  右脚触球',AQUA))
    if score['contact_episodes']:
        episode=score['contact_episodes'][-1];release=episode['first_unqualified_bin_s'];speed=score['last_exit_speed_xy_mps']
        if release is not None and speed is not None:events.append((release,f'{release:.3f} 秒  末段触球结束后水平球速 {speed:.2f} 米/秒',WHITE))
    crossing=score['whole_ball_crossing']
    if crossing is not None:
        time=crossing['interpolated_time_s'];msg='整球进门' if score['checks']['goal'] else '越过测量平面，未取得有效进球'
        events.append((time,f'{time:.3f} 秒  {msg}',AQUA if score['checks']['goal'] else GOLD))
    nonfoot=score['native_pair_events']['first_nonfoot_pair']
    if nonfoot is not None:
        t=nonfoot['time_s'];labels=[BODY_LABEL.get(n,n.replace('_link','')) for n in nonfoot['actors']]
        events.append((t,f'{t:.3f} 秒  {labels[0]}与{labels[1]}接触  {nonfoot["force_N"]:.1f} N',RED))
    wrong=score['native_pair_events']['first_wrong_ball_contact']
    if wrong is not None:
        t=wrong['time_s'];events.append((t,f'{t:.3f} 秒  非指定部位触球',RED))
    fall=score['first_fall_s']
    if fall is not None:events.append((fall,f'{fall:.3f} 秒  触及跌倒阈值',RED))
    return sorted(events,key=lambda event:event[0])


def render(folder,score_path,output,assets,urdf,label):
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    score=json.loads(score_path.read_text())
    for key in ('rollout.npz','native_contacts.jsonl.gz','tracking_result.json'):
        if digest(folder/key)!=score['input_sha256'][key]:raise ValueError('Score belongs to a different run: '+key)
    if digest(urdf)!=score['urdf_sha256']:raise ValueError('Visual URDF differs from scored asset')
    saved=json.loads((folder/'tracking_result.json').read_text())
    names=saved.get('source_body_names')
    if not names:names=json.loads((folder/'source_inventory.json').read_text())['body_names']
    with np.load(folder/'rollout.npz',allow_pickle=False) as tr:
        times=tr['time_s'];pos=tr['body_position_w_m'];rot=Rotation.from_quat(tr['body_quat_wxyz'][...,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(len(times),len(names),3,3)
        ball=tr['ball_position_w_m'];vel=tr['ball_velocity_w_mps']
        commands=tr['policy_commands_time_frame_action_target']
    clip=Path(score.get('source_clip') or saved['source_clip']);refpath=clip if clip.is_absolute() else assets/clip
    with np.load(refpath,allow_pickle=False) as ref:
        rp=ref['body_pos_w'];rr=Rotation.from_quat(ref['body_quat_w'][...,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(len(rp),len(names),3,3)
    geom=load_geometry(urdf,names);events=event_labels(score)
    hits=score['contact_episodes'];contact_times=[x['first_loaded_bin_s'] for x in hits]
    marker=score['native_pair_events']['first_nonfoot_pair']
    outcome='本地动作指标通过' if score['cpu_proxy_task_pass'] else f'未通过 {len(score["failed_checks"])} 项'
    temp=output.with_name(output.stem+'.encoding.mp4')
    if temp.exists():raise FileExistsError(temp)
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-f','rawvideo','-pix_fmt','rgb24','-video_size',f'{W}x{H}','-framerate',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-threads','3','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)]
    child=subprocess.Popen(cmd,stdin=subprocess.PIPE);cv2.setNumThreads(1)
    try:
        cam=Camera();goal=np.asarray(score['goal_xyz_m']);start=np.asarray(score['initial_ball_xyz_m']);center=(pos[:,:1].mean(axis=(0,1))[:2]+start[:2]+goal[:2])/3.
        cam.center=np.r_[center,.68]
        action=np.vstack([pos[:,0,:],rp[:,0,:],goal,start]);projection=(action-cam.center)@cam.proj.T
        scale=min(145.,260./max(abs(projection[:,0]).max(),.3),165./max(abs(projection[:,1]).max(),.4))
        cam.scale=scale;bgs=[panel_background(cam,False) for _ in range(2)];goal_mesh(bgs[1],cam,score)
        count=max(1,int(np.ceil(times[-1]*FPS-1e-8)))
        for frame in range(count):
            t=min((frame+1)/FPS,times[-1]);idx=min(int(np.searchsorted(times,t-1e-8)),len(times)-1)
            ci=np.clip(np.searchsorted(commands[:,0],t-.005,side='right')-1,0,len(commands)-1)
            ri=min(max(0,int(commands[ci,1])),len(rp)-1)
            panels=[image.copy() for image in bgs]
            render_robot(panels[0],cam,geom,rp[ri],rr[ri]);render_robot(panels[1],cam,geom,pos[idx],rot[idx])
            point=ball[idx];xy=cam.pixels(point)
            trail=cam.pixels(ball[max(0,idx-25):idx+1:3])
            if len(trail)>1:cv2.polylines(panels[1],[trail],False,(192,156,78),2,cv2.LINE_AA)
            if 0<=xy[0]<640 and 0<=xy[1]<PANEL_HEIGHT:
                radius=max(2,round(cam.scale*score['ball_radius_m']))
                point_marker(panels[1],cam,point,GOLD,radius,-1)
                if any(abs(t-c)<.10 for c in contact_times):point_marker(panels[1],cam,point,AQUA,radius+4,2)
            if marker and 0<=t-marker['time_s']<.32:point_marker(panels[1],cam,marker['world_m'],RED,11,3)
            canvas=Image.new('RGB',(W,H),(15,26,33));canvas.paste(Image.fromarray(panels[0]),(0,117));canvas.paste(Image.fromarray(panels[1]),(640,117));draw=ImageDraw.Draw(canvas)
            write_text(draw,(26,13),label,33,bold=True);write_text(draw,(1250,25),'G1  /  CPU PhysX',21,AQUA,anchor='ra')
            draw.line((0,68,W,68),fill=(63,91,96),width=2)
            write_text(draw,(25,80),'输入动作素材',24,bold=True);write_text(draw,(620,85),f'源帧 {ri} / {len(rp)-1}',17,MUTED,anchor='ra')
            write_text(draw,(662,80),'实际物理轨迹',24,bold=True);write_text(draw,(1252,85),f'{t:05.2f} 秒',21,AQUA,anchor='ra')
            stamp,msg,color=next(e for e in reversed(events) if t+1e-8>=e[0]);write_text(draw,(27,562),msg,24,color,bold=True)
            draw.line((0,606,W,606),fill=(63,91,96),width=2)
            write_text(draw,(27,623),f'骨盆高度 {pos[idx,0,2]:.2f} 米   ·   当前球速 {np.linalg.norm(vel[idx,:2]):.2f} 米/秒',19,MUTED)
            if frame==count-1:write_text(draw,(1250,624),outcome,21,AQUA if score['cpu_proxy_task_pass'] else RED,True,'ra')
            write_text(draw,(27,668),'关节和足球按记录的逐帧物理状态绘制',17,MUTED)
            child.stdin.write(canvas.tobytes())
        child.stdin.close()
        if child.wait()!=0:raise RuntimeError('Video encoder failed')
        os.link(temp,output);temp.unlink()
    finally:
        if child.poll() is None:child.kill();child.wait()
        if temp.exists():temp.unlink()
    return dict(video=str(output),frames=count,duration_s=count/FPS,video_sha256=digest(output),evidence_sha256=score['input_sha256'],score_sha256=digest(score_path),clip=str(refpath),clip_sha256=digest(refpath))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--score',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--assets',type=Path,default=Path('/home/you/projects/kinetickick'));p.add_argument('--urdf',type=Path,default=DEFAULT_URDF);p.add_argument('--label',default='KineticKick  /  实测射门与恢复')
    a=p.parse_args();print(json.dumps(render(a.run,a.score,a.output,a.assets,a.urdf,a.label),ensure_ascii=False))

if __name__=='__main__':main()

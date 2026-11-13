"""CPU-only movie from stored 200 Hz V160 poses and original G1 visual meshes."""
import argparse
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
import vtk
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[1]
W,H = 1280,720
PANEL_HEIGHT = 430
FPS = 25
GOLD=(250,183,73);RED=(255,117,110);AQUA=(107,224,204);MUTED=(172,191,194);WHITE=(239,245,243)
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-'
FONTS={(s,b): ImageFont.truetype(FONT+('Bold' if b else 'Regular')+'.ttc',s,index=2) for s in [15,17,19,21,24,26,29,33] for b in [0,1]}

def array3(element, field):
    return np.fromstring(element.get(field,'0 0 0') if element is not None else '0 0 0',sep=' ')

def transform(element):
    return Rotation.from_euler('xyz',array3(element,'rpy')).as_matrix(),array3(element,'xyz')

def unit(x):return x/np.linalg.norm(x)

def load_geometry(urdf, names):
    xml=ET.parse(urdf).getroot(); owned={n:(i,np.eye(3),np.zeros(3)) for i,n in enumerate(names)}
    fixed=[j for j in xml.findall('joint') if j.get('type')=='fixed']
    for _ in range(len(fixed)):
        for j in fixed:
            link=j.find('child').get('link');parent=j.find('parent').get('link')
            if link in owned or parent not in owned:continue
            idx,R,p=owned[parent];local,t=transform(j.find('origin'));owned[link]=(idx,R@local,p+R@t)
    shapes=[]
    for link in xml.findall('link'):
        for vis in link.findall('visual'):
            mesh=vis.find('geometry/mesh');filename=mesh.get('filename');idx,R,p=owned[link.get('name')]
            local,t=transform(vis.find('origin'));t=p+R@t;local=R@local
            rd=vtk.vtkSTLReader();rd.SetFileName(filename);rd.Update();dec=vtk.vtkQuadricClustering();dec.SetInputData(rd.GetOutput());dec.SetNumberOfDivisions(8,8,8);dec.Update();poly=dec.GetOutput()
            verts=vtk_to_numpy(poly.GetPoints().GetData()).astype(float); faces=vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1,4)[:,1:];tri=(verts@local.T+t)[faces]
            color=vis.find('material/color');v=np.fromstring(color.get('rgba','0.7 0.7 0.7 1'),sep=' ')[:3] if color is not None else np.full(3,.7)
            base=(np.array([112,139,149]) if v.mean()<.4 else np.array([213,226,228]));shapes.append((idx,tri,base))
    return shapes

class Camera:
    def __init__(self):
        self.depth=unit(np.array([4.3,-2.15,1.36]));self.right=unit(np.cross([0,0,1],self.depth));self.up=unit(np.cross(self.depth,self.right));self.proj=np.stack((self.right,-self.up,self.depth));self.center=np.array([.18,-1.85,.56]);self.scale=125.
    def pixels(self,pts):
        out=(pts-self.center)@self.proj.T
        return np.rint(out[...,:2]*self.scale+np.array([327,207])).astype(np.int32)

def plot_world_line(img,cam,a,b,color,width=1):
    pts=cam.pixels(np.array([a,b]));cv2.line(img,tuple(pts[0]),tuple(pts[1]),color,width,cv2.LINE_AA)

def panel_background(cam,with_goal):
    arr=np.full((PANEL_HEIGHT,640,3),(27,55,53),np.uint8)
    for x in np.arange(-6,7,.5):plot_world_line(arr,cam,[x,-10,0],[x,5,0],(44,75,70))
    for y in np.arange(-8,6,.5):plot_world_line(arr,cam,[-8,y,0],[8,y,0],(44,75,70))
    if with_goal:
        line=(175,214,196);mesh=(87,131,114);s=-3.7;back=-4.04
        for x in np.linspace(-.8,.8,9):plot_world_line(arr,cam,[x,back,0],[x,back,1.1],mesh)
        for z in np.linspace(0,1.1,7):plot_world_line(arr,cam,[-.8,back,z],[.8,back,z],mesh)
        for x in (-.8,.8):
            plot_world_line(arr,cam,[x,s,0],[x,s,1.1],line,3);plot_world_line(arr,cam,[x,s,1.1],[x,back,1.1],mesh,2)
        plot_world_line(arr,cam,[-.8,s,1.1],[.8,s,1.1],line,3);plot_world_line(arr,cam,[-.8,s,0],[.8,s,0],(99,144,123),2)
    return arr

def render_robot(canvas,cam,shapes,positions,rotations):
    all_points=[];all_depth=[];all_colors=[];light=unit(np.array([.2,-.4,.92]));root=positions[0];xy=cam.pixels(np.array([root[0],root[1],0.]));cv2.ellipse(canvas,tuple(xy),(34,8),-6,0,360,(23,46,44),-1,cv2.LINE_AA)
    for idx,tri,base in shapes:
        world=tri@rotations[idx].T+positions[idx];normals=np.cross(world[:,1]-world[:,0],world[:,2]-world[:,0]);normals/=np.maximum(1e-10,np.linalg.norm(normals,axis=1))[:,None]
        front=(normals@cam.depth)>=0;world=world[front];n=normals[front];shade=.52+.44*np.clip(n@light,0,1)+.10*np.clip(n@cam.depth,0,1)
        screen=cam.pixels(world);visible=(screen[:,:,0].max(1)>=0)&(screen[:,:,0].min(1)<640)&(screen[:,:,1].max(1)>=0)&(screen[:,:,1].min(1)<PANEL_HEIGHT)
        all_points.append(screen[visible]);all_depth.append((world[visible].mean(axis=1)-cam.center)@cam.depth);all_colors.append(np.clip(shade[visible,None]*base,0,255).astype(np.uint8))
    polygons=np.concatenate(all_points);depth=np.concatenate(all_depth);colors=np.concatenate(all_colors).tolist()
    for i in depth.argsort():cv2.fillConvexPoly(canvas,polygons[i],colors[i],cv2.LINE_8)

def point_marker(img,cam,point,color,radius,thickness):
    x,y=cam.pixels(point);cv2.circle(img,(int(x),int(y)),radius,color,thickness,cv2.LINE_AA)

def write_text(draw,xy,string,size,color=WHITE,bold=False,anchor='la'):
    draw.text(xy,string,font=FONTS[size,int(bold)],fill=color,anchor=anchor)

def get_contacts(path):
    ball={};wrist=None
    with gzip.open(path,'rt') as stream:
        for line in stream:
            index,groups=json.loads(line);time=index*.005
            for g in groups:
                actors=set(g['actors']);pts=np.asarray(g['points']);force=np.linalg.norm(pts[:,6:9].sum(0))/.005
                if actors=={'SoccerBall','right_ankle_roll_link'} and force>2:ball.setdefault(index,pts[0,:3])
                if actors=={'left_wrist_yaw_link','left_hip_roll_link'} and force>20 and wrist is None:wrist=(time,pts[0,:3])
    if wrist is None or not ball:raise ValueError('Missing native marker data')
    return ball,wrist

def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--assets',type=Path,default=Path('/home/you/projects/kinetickick'));cli.add_argument('--run',type=Path,default=ROOT/'experiments/v160_tracking/official_001_cpu');cli.add_argument('--output',type=Path,default=ROOT/'media/v160_paid_tracking_reference_vs_physics.mp4');args=cli.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=json.loads((args.run/'tracking_result.json').read_text());audit=json.loads((args.run/'independent_audit.json').read_text());names=result['source_body_names']
    with np.load(args.run/'rollout.npz') as rec:
        times=rec['time_s'];pos=rec['body_position_w_m'];rot=Rotation.from_quat(rec['body_quat_wxyz'][...,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(len(times),30,3,3);ball_xyz=rec['ball_position_w_m'];ball_vel=rec['ball_velocity_w_mps']
    with np.load(args.assets/result['source_clip']) as ref:
        rp=ref['body_pos_w'];rr=Rotation.from_quat(ref['body_quat_w'][...,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(len(rp),30,3,3)
    sys.path.insert(0,str(ROOT/'tools'));from free_shooting_metrics import goal_crossing
    goal=[0.,-3.7];forward=[0.,-1.];radius=.11
    first=next(i for i in range(1,len(times)) if goal_crossing(ball_xyz[i-1],ball_xyz[i],goal=goal,forward=forward,width=1.6,height=1.1,radius=radius))
    old,new=ball_xyz[first-1,1],ball_xyz[first,1];cross=float(times[first-1]+.005*(goal[1]-radius-old)/(new-old));contacts,wrist=get_contacts(args.run/'native_contacts.jsonl.gz');hits=sorted(contacts);exit_index=hits[-1]+1;exit_speed=float(np.linalg.norm(ball_vel[exit_index-1,:2]));first_hit=hits[0]*.005
    if abs(first_hit-audit['first_native_ball_contact']['time_s'])>1e-6 or abs(wrist[0]-audit['first_native_left_wrist_hip_over20N']['time_s'])>1e-6:raise ValueError('Native contact timestamp disagrees with independent audit')
    fall=result['first_height_below_0p45m_s'];end=result['reference_last_timestamp_s'];g=load_geometry(args.assets/'configs/paid_original/paid_main_resolved.urdf',names);cam=Camera();back=[panel_background(cam,b) for b in (False,True)]
    args.output.parent.mkdir(exist_ok=True,parents=True);temporary=args.output.with_name('.'+args.output.stem+'.writing.mp4');
    if temporary.exists():raise FileExistsError(temporary)
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{W}x{H}','-framerate',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-threads','3','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)]
    events=[(0.,'起始姿态'),(first_hit,'右脚触球'),(cross,'整球过门'),(end,'参考结束'),(wrist[0],'左腕碰左髋'),(fall,'跌倒阈值'),(14.,'回合结束')]
    anchors=[(0.,'观察助跑与摆腿'),(first_hit,f'{first_hit:.3f} s  右踝与球首次接触'),(exit_index*.005,f'{exit_index*.005:.3f} s  末段触球离开 2 N 阈值，水平出球 {exit_speed:.2f} m/s'),(cross,f'{cross:.3f} s  整球穿过球门'),(end,'参考已到最后一帧，继续观察实际恢复'),(wrist[0],f'{wrist[0]:.3f} s  左腕碰左髋，接触力约 {audit["first_native_left_wrist_hip_over20N"]["net_impulse_norm_over_dt_N"]:.1f} N'),(fall,f'{fall:.3f} s  骨盆低于 0.45 m，达到跌倒阈值'),(14.,'回合结束，机器人仍倒地')]
    process=subprocess.Popen(cmd,stdin=subprocess.PIPE);cv2.setNumThreads(1)
    try:
        # End-of-video sample: each of 350 display intervals shows its observed end state.
        for f in range(350):
            idx=(f+1)*8-1;t=times[idx];ri=min(round(t*50),len(rp)-1)
            if t>5.2:
                zoom=float(np.clip((t-5.2)/1.2,0.,1.));zoom=zoom*zoom*(3.-2.*zoom)
                cam.center=(1.-zoom)*np.array([.18,-1.85,.56])+zoom*np.array([.62,-.58,.57]);cam.scale=125.+zoom*66.
                if t<6.5 or t<=6.5+.04:back=[panel_background(cam,b) for b in (False,True)]
            frames=[x.copy() for x in back];render_robot(frames[0],cam,g,rp[ri],rr[ri]);render_robot(frames[1],cam,g,pos[idx],rot[idx]);
            ball_pos=ball_xyz[idx];bp=cam.pixels(ball_pos)
            if 0<bp[0]<640 and 0<bp[1]<PANEL_HEIGHT:
                trail=cam.pixels(ball_xyz[max(0,idx-21):idx+1:3]);cv2.polylines(frames[1],[trail],False,(193,150,76),2,cv2.LINE_AA)
                ball_radius=round(radius*cam.scale)
                point_marker(frames[1],cam,[*ball_pos[:2],0.],(28,49,43),max(3,ball_radius-2),-1);point_marker(frames[1],cam,ball_pos,(245,181,71),ball_radius,-1);point_marker(frames[1],cam,ball_pos,(255,231,174),ball_radius,2)
            if wrist[0]<=t<wrist[0]+.28:point_marker(frames[1],cam,wrist[1],RED,12,3)
            image=Image.new('RGB',(W,H),(15,26,33));image.paste(Image.fromarray(frames[0]),(0,117));image.paste(Image.fromarray(frames[1]),(640,117));draw=ImageDraw.Draw(image)
            draw.line((640,80,640,559),fill=(154,183,180),width=2)
            write_text(draw,(30,13),'KineticKick  /  基础动作跟踪',33,bold=True);write_text(draw,(1250,24),'v0.1.160   ·   001 右脚',21,AQUA,anchor='ra')
            draw.line((0,68,W,68),fill=(63,91,96),width=2)
            write_text(draw,(24,80),'参考动作',24,bold=True);write_text(draw,(615,85),f'{min(t,end):.2f} s  /  '+('保持末帧' if t>end else '原始 50 Hz'),17,MUTED,anchor='ra')
            write_text(draw,(663,80),'物理回合',24,bold=True);write_text(draw,(1253,85),'SAPIEN CPU PhysX  ·  200 Hz 实测',17,AQUA,anchor='ra')
            label=next(message for stamp,message in reversed(anchors) if t+1e-9>=stamp)
            write_text(draw,(27,567),f'{t:05.2f} s',26,AQUA,bold=True);write_text(draw,(190,566),label,26,RED if t>=wrist[0] else WHITE,bold=True)
            write_text(draw,(1252,570),f'骨盆 {pos[idx,0,2]:.2f} 米',17,MUTED,anchor='ra')
            # Timeline labels below the axis deliberately use an independent layout at nearby event times.
            a,b,y=35,1242,638;draw.line((a,y,b,y),fill=(75,101,103),width=3);cursor=a+(b-a)*t/14;draw.line((a,y,cursor,y),fill=AQUA,width=4)
            for ei,(et,lab) in enumerate(events):
                x=a+(b-a)*et/14;draw.ellipse((x-4,y-4,x+4,y+4),fill=RED if ei in (4,5) else GOLD)
                if ei not in (1,4):write_text(draw,(x,654),lab,15,MUTED,anchor=('la' if ei==0 else 'ra' if ei==6 else 'ma'))
                else:write_text(draw,(x,611),lab,15,RED if ei==4 else GOLD,anchor='ma')
            write_text(draw,(27,689),'机器人由原 URDF 网格与逐帧位姿绘制；足球及球路仅在右侧显示。',15,MUTED)
            process.stdin.write(image.tobytes())
        process.stdin.close()
        if process.wait():raise RuntimeError('Encoding failed')
        temporary.rename(args.output)
    finally:
        if process.poll() is None:process.kill();process.wait()
    print(json.dumps(dict(frames=350,duration=14,mesh_links=len(g),impact_s=first_hit,last_release_sample_s=exit_index*.005,speed_mps=exit_speed,goal_s=cross,wrist_s=wrist[0],fall_s=fall,output=str(args.output)),ensure_ascii=False))

if __name__=='__main__':main()

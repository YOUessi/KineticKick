"""Research candidate: original RUN natural deceleration -> standing, no state writes."""
from pathlib import Path
import argparse
import sys,json,hashlib
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation,RotationSpline
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--standing-hold-seconds',type=float,default=1.);args=parser.parse_args()
if not 1<=args.standing_hold_seconds<=3:parser.error('Standing hold must be1..3seconds')
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'tools'))
if args.output.exists() or args.output.with_suffix('.json').exists():raise FileExistsError(args.output)
from running_reference_geometry import ReferenceGeometry
from g1_retarget_kinematics import JOINTS
geom=ReferenceGeometry(root);names=geom.joint_names;bodies=geom.body_names
raw=np.load(root/'data/human_motion/straight_reference29/run_straight.npz');f=raw['frames'].astype(float);ts=np.arange(len(f))/30.;t=np.arange(148)/50.
qi=CubicHermiteSpline(ts,f[:,7:][:,[JOINTS.index(n) for n in names]],np.gradient(f[:,7:][:,[JOINTS.index(n) for n in names]],1/30,axis=0))
pi=CubicHermiteSpline(ts,f[:,:3],np.gradient(f[:,:3],1/30,axis=0));yaw=Rotation.from_euler('z',-np.pi/2)
q=qi(t);dq=qi(t,1);p=yaw.apply(pi(t));v=yaw.apply(pi(t,1));rot=yaw*RotationSpline(ts,Rotation.from_quat(f[:,3:7]))(t)
stand=np.load(root/'third_party/HumanoidSoccer/motions/soccer-standard/soccer-standard-010_right.npz')
qend=stand['joint_pos'][0].astype(float);pend=p[-1]+v[-1]*.2;pend[2]=stand['body_pos_w'][0,0,2]
rend=Rotation.from_quat(stand['body_quat_w'][0,0,[1,2,3,0]])
tailt=np.arange(1,51)/50.;hold=round(args.standing_hold_seconds*50)
qtail=CubicHermiteSpline([0,1],np.stack([q[-1],qend]),np.stack([dq[-1],np.zeros(29)]))(tailt)
ptail=CubicHermiteSpline([0,1],np.stack([p[-1],pend]),np.stack([v[-1],np.zeros(3)]))(tailt)
u=tailt**2*(3-2*tailt);delta=(rend*rot[-1].inv()).as_rotvec();rtail=Rotation.from_rotvec(u[:,None]*delta)*rot[-1]
q=np.concatenate([q,qtail,np.tile(qend,(hold,1))]);p=np.concatenate([p,ptail,np.tile(pend,(hold,1))]);rall=Rotation.concatenate([rot,rtail,Rotation.from_quat(np.tile(rend.as_quat(),(hold,1)))])
# Same bounded hip correction as the verified running prefix.
j=names.index('left_hip_pitch_joint');floor=-1.312411756;q[:,j]=floor+.01*np.logaddexp(0.,(q[:,j]-floor)/.01)
quats=rall.as_quat()[:,[3,0,1,2]]
_,bottom=geom.check(q,p,quats);zshift=np.maximum(.010516867291,.001-bottom)
# During recovery the original standing q/root combination leaves both
# capsule feet about4cm high. Blend the root reference down to true support
# height, retaining the natural RUN part and the verified prefix.
stance_weight=np.r_[np.zeros(148),u,np.ones(hold)]
zshift=(1-stance_weight)*zshift+stance_weight*(.001-bottom)
p[:,2]+=zshift
bp=[];bq=[]
for i in range(len(q)):
 fk=geom.body_poses(q[i],p[i],quats[i]);bp.append([fk[n][0] for n in bodies]);bq.append([fk[n][1].as_quat()[[3,0,1,2]] for n in bodies])
bp=np.array(bp);bq=np.array(bq);lv=np.gradient(bp,.02,axis=0,edge_order=2);av=np.zeros_like(lv)
for j in range(len(bodies)):
 rr=Rotation.from_quat(bq[:,j,[1,2,3,0]]);av[1:-1,j]=(rr[2:]*rr[:-2].inv()).as_rotvec()*25;av[0,j]=(rr[1]*rr[0].inv()).as_rotvec()*50;av[-1,j]=0
limits={j.attrib['name']:j.find('limit') for j in geom.xml.findall('joint') if j.find('limit') is not None}
lo=np.array([float(limits[n].attrib['lower']) for n in names]);hi=np.array([float(limits[n].attrib['upper']) for n in names]);assert np.max(np.maximum(lo-q,q-hi))<=1e-6
out=args.output;out.parent.mkdir(parents=True,exist_ok=True)
arrays=dict(fps=np.array([50]),joint_pos=q.astype('f4'),joint_vel=np.gradient(q,.02,axis=0,edge_order=2).astype('f4'),body_pos_w=bp.astype('f4'),body_quat_w=bq.astype('f4'),body_lin_vel_w=lv.astype('f4'),body_ang_vel_w=av.astype('f4'),kick_leg=np.array('right'))
# Keep the verified 2.2-second prefix bit-for-bit; extend commands, never robot state.
prefix_path=root/'experiments/running_v010/run_capsule_clearance1mm.npz'
prefix=np.load(prefix_path)
for key in ('joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w'):
    arrays[key][:111]=prefix[key]
with out.open('xb') as stream:np.savez_compressed(stream,**arrays)

# Audit the arrays actually written, including the preserved prefix.
a,_=geom.check(arrays['joint_pos'],arrays['body_pos_w'][:,0],arrays['body_quat_w'][:,0])
meta=dict(purpose='Candidate natural RUN deceleration plus one-second blend and finite standing hold; unvalidated physical transition',standing_hold_seconds=args.standing_hold_seconds,duration_s=(len(q)-1)/50,frames=len(q),fps=50,joint_names=names,body_names=bodies,angular_velocity_frame='world',output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),geometry=a,max_root_height_correction_m=float(zshift.max()),min_root_height_correction_m=float(zshift.min()),standing_capsule_clearance_m=.001,prefix_sha256=hashlib.sha256(prefix_path.read_bytes()).hexdigest(),prefix_frames=111,warning='Only a kinematic candidate. Root-height adjustment varies after the learned prefix; no complete run-to-shot claim.')
out.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps({k:meta[k] for k in ['duration_s','frames','max_root_height_correction_m','geometry']}))

"""Optional read-only ground contact/anchor evidence around a fixed time window.

Raw SDK points retain unknown solver staging; pre/post link states bracket each
step. Normal points and friction anchors are separate and never paired by index.
"""
from pathlib import Path
import hashlib
import numpy as np
from strike_dynamics_telemetry import StrikeDynamicsObserver


def indexed_entries(fields,counts,starts):
    counts=np.asarray(counts);starts=np.asarray(starts);fields={k:np.asarray(v) for k,v in fields.items()}
    if counts.shape!=starts.shape or not fields:raise ValueError('Consistent contact pair metadata required')
    capacity={len(v) for v in fields.values()}
    if len(capacity)!=1:raise ValueError('Consistent buffer capacities required')
    capacity=capacity.pop();rows=[]
    for pair in np.ndindex(counts.shape):
        n,s=counts[pair],starts[pair]
        if not np.isfinite([n,s]).all() or int(n)!=n or int(s)!=s or n<0:raise ValueError('Invalid pair metadata')
        n,s=int(n),int(s)
        if not n:continue
        if s<0 or s+n>capacity:raise ValueError('Contact buffer overflow')
        for index in range(s,s+n):
            if any(not np.isfinite(v[index]).all() for v in fields.values()):raise ValueError('Nonfinite active contact')
            rows.append(dict(pair_index=list(pair),buffer_index=index,**{k:v[index].tolist() for k,v in fields.items()}))
    return rows


def measured_wrench(normal,friction,origin_w):
    origin=np.asarray(origin_w,float);force=np.zeros(3);moment=np.zeros(3)
    for row in normal+friction:
        f=np.asarray(row['force_w_N'],float);p=np.asarray(row['point_w'],float)
        force+=f;moment+=np.cross(p-origin,f)
    return dict(force_w_N=force.tolist(),moment_w_Nm=moment.tolist(),origin_w=origin.tolist())


def raw_ground_contacts(view,dt,friction_sign):
    cpu=lambda x:x.detach().cpu().numpy().copy()
    f,p,n,d,c,s=map(cpu,view.get_contact_data(dt))
    normal=indexed_entries(dict(force_scalar_N=f.reshape(-1),point_w=p,normal_w=n,separation_m=d.reshape(-1),force_w_N=f.reshape(-1,1)*n),c,s)
    ff,fp,fc,fs=map(cpu,view.get_friction_data(dt))
    friction=indexed_entries(dict(point_w=fp,force_raw_w_N=ff,force_w_N=friction_sign*ff),fc,fs)
    matrix=cpu(view.get_contact_force_matrix(dt)).reshape(-1,3).sum(0)
    summed=np.sum([r['force_w_N'] for r in normal],axis=0) if normal else np.zeros(3)
    # Keep the raw normal orientation visible, independent of the matrix used
    # by the previously certified observer. Do not silently flip vector data.
    return dict(normal_contacts=normal,friction_anchors=friction,
        normal_counts=c.tolist(),normal_starts=s.tolist(),friction_counts=fc.tolist(),friction_starts=fs.tolist(),
        normal_buffer_capacity=len(f),friction_buffer_capacity=len(ff),
        force_matrix_w_N=matrix.tolist(),raw_normal_sum_w_N=summed.tolist(),
        raw_normal_matrix_error_N=float(np.linalg.norm(summed-matrix)),
        capacity_exhausted=bool(c.sum()>=len(f) or fc.sum()>=len(ff)))


class GroundContactObserver(StrikeDynamicsObserver):
    def __init__(self,env,probe_path,window):
        super().__init__(env,probe_path)
        self.window=tuple(map(float,window))
        if not np.isfinite(self.window).all() or not 0<=self.window[0]<self.window[1]<=14.:raise ValueError('Finite bounded contact detail window required')
        self.feet_indices={side:self.robot.body_names.index(side+'_ankle_roll_link') for side in self.views}
        self.previous_feet=None;self.detail_samples=0

    def feet_state(self):
        r=self.robot.data
        return {side:dict(body_name=side+'_ankle_roll_link',position_w_m=r.body_pos_w[0,i].cpu().tolist(),
            quaternion_wxyz=r.body_quat_w[0,i].cpu().tolist(),
            link_origin_linear_velocity_w_mps=r.body_link_lin_vel_w[0,i].cpu().tolist(),
            angular_velocity_w_rad_s=r.body_ang_vel_w[0,i].cpu().tolist()) for side,i in self.feet_indices.items()}

    def after_update(self,dt):
        if not self.active or dt<=0:
            self.previous_feet=self.feet_state()
            return
        before=self.previous_feet
        super().after_update(dt)
        after=self.feet_state();now=self.rows[-1]['time_s']
        if self.window[0]-1e-9<=now<=self.window[1]+1e-9:
            if before is None:raise ValueError('Missing prior link state')
            contacts={side:raw_ground_contacts(view,dt,self.friction_sign) for side,view in self.views.items()}
            self.rows[-1]['ground_contact_detail']=dict(step_start_s=now-dt,step_end_s=now,
                feet_before=before,feet_after=after,ground_filtered_contacts=contacts)
            self.detail_samples+=1
        self.previous_feet=after

    def attach(self,result):
        super().attach(result)
        result['dynamics_telemetry']['ground_contact_detail']=dict(window_s=list(self.window),samples=self.detail_samples,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            point_stage='Raw SDK solver contact/anchor points; exact material point and solver staging not independently certified',
            force_units='Step-average force N from dt=0.005; friction sign from original validated probe',
            contact_pair_identity='Separate per-foot rigid view filtered to the recorded ground_collision_paths; pair indices and raw buffer counts retained. Shape ids are not exposed by this API.',
            foot_velocity_origin='ArticulationData.body_link_lin_vel_w: link actor origin, not centre of mass',
            foot_states='Previous and current post-scene-update poses/velocities bracket this physics step',physical_writes=False)
        return result

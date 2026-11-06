"""Read-only support/segment telemetry with independently probed force semantics.

Import in simulator processes only after AppLauncher. No physics state writes.
"""
import hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def aggregate_buffer(values,counts,starts):
    values=np.asarray(values,float);counts=np.asarray(counts);starts=np.asarray(starts)
    if values.ndim!=2 or values.shape[1]!=3 or counts.shape!=starts.shape:raise ValueError('Invalid contact buffer shapes')
    result=np.zeros(3)
    for count,start in zip(counts.flat,starts.flat):
        if count!=int(count) or start!=int(start) or count<0:raise ValueError('Invalid contact buffer index')
        count,start=int(count),int(start)
        if not count:continue
        if start<0 or start+count>len(values):raise ValueError('Contact buffer overflow')
        part=values[start:start+count]
        if not np.isfinite(part).all():raise ValueError('Nonfinite contact buffer')
        result+=part.sum(0)
    return result


def contact_components(view,dt):
    cpu=lambda x:x.detach().cpu().numpy().copy()
    matrix=cpu(view.get_contact_force_matrix(dt)).reshape(-1,3).sum(0)
    force,point,normal,separation,count,start=view.get_contact_data(dt)
    force,normal,count,start=map(cpu,[force,normal,count,start])
    reconstructed=aggregate_buffer(force.reshape(-1,1)*normal,count,start)
    friction,points,nfriction,friction_start=view.get_friction_data(dt)
    friction,nfriction,friction_start=map(cpu,[friction,nfriction,friction_start])
    tangential=aggregate_buffer(friction,nfriction,friction_start)
    exhausted=bool(count.sum()>=len(force) or nfriction.sum()>=len(friction))
    return dict(force_matrix_w_N=matrix.tolist(),normal_w_N=reconstructed.tolist(),friction_raw_w_N=tangential.tolist(),
        normal_contact_count=int(count.sum()),friction_contact_count=int(nfriction.sum()),capacity_exhausted=exhausted)


def certify_force_probe(rows,mass_kg,initial_velocity,gravity,tolerance_Ns=.02):
    if not rows or not np.isfinite([mass_kg,tolerance_Ns]).all() or mass_kg<=0 or tolerance_Ns<=0:raise ValueError('Positive finite probe parameters required')
    dt=np.array([r['dt_s'] for r in rows]);matrix=np.array([r['force_matrix_w_N'] for r in rows]);normal=np.array([r['normal_w_N'] for r in rows]);friction=np.array([r['friction_raw_w_N'] for r in rows])
    values=np.concatenate([dt,matrix.ravel(),normal.ravel(),friction.ravel(),np.asarray(initial_velocity),np.asarray(gravity),np.asarray(rows[-1]['velocity_w'])])
    if not np.isfinite(values).all() or np.any(dt<=0):raise ValueError('Finite contiguous probe values required')
    expected=mass_kg*(np.asarray(rows[-1]['velocity_w'])-initial_velocity)-mass_kg*np.asarray(gravity)*dt.sum()
    normal_impulse=(matrix*dt[:,None]).sum(0);friction_impulse=(friction*dt[:,None]).sum(0)
    errors={sign:float(np.linalg.norm(normal_impulse+sign*friction_impulse-expected)) for sign in [1,-1]}
    sign=min(errors,key=errors.get)
    reconstruction=min(float((np.linalg.norm(matrix-normal,axis=-1)*dt).sum()),float((np.linalg.norm(matrix+normal,axis=-1)*dt).sum()))
    adequate_friction=float(np.linalg.norm(friction_impulse))>max(10*tolerance_Ns,1e-8)
    verified=bool(errors[sign]<=tolerance_Ns and reconstruction<=tolerance_Ns and adequate_friction and not any(r['capacity_exhausted'] for r in rows))
    return dict(verified=verified,friction_sign=sign,matrix_semantics='normal_only' if reconstruction<=tolerance_Ns else 'unresolved',
        expected_ground_impulse_Ns=expected.tolist(),normal_impulse_Ns=normal_impulse.tolist(),raw_friction_impulse_Ns=friction_impulse.tolist(),
        impulse_error_Ns=errors[sign],opposite_sign_impulse_error_Ns=errors[-sign],normal_reconstruction_error_Ns=reconstruction,
        tolerance_Ns=tolerance_Ns,friction_signal_adequate=adequate_friction,interpretation='Mass times velocity change minus gravity impulse, isolated yawed sliding body; solver impulse balance, not hardware force calibration')


def validated_force_probe(probe):
    """Recompute the sign certificate from recorded probe samples, never trust a flag."""
    try:
        stored=probe['certificate'];tolerance=float(stored['tolerance_Ns'])
        if stored['verified'] is not True or stored['friction_sign'] not in (-1,1) or not 0<tolerance<=.02:
            raise ValueError('Invalid force certificate')
        computed=certify_force_probe(probe['samples'],probe['mass_kg'],probe['initial_velocity_w'],probe['gravity_w'],tolerance)
        if not computed['verified'] or computed['matrix_semantics']!='normal_only':
            raise ValueError('Force certificate cannot be reproduced')
        for key,value in computed.items():
            if key=='interpretation':continue
            observed=stored[key]
            if isinstance(value,(bool,str)):
                same=type(observed) is type(value) and observed==value
            else:
                same=np.shape(observed)==np.shape(value) and np.allclose(observed,value,rtol=1e-10,atol=1e-12)
            if not same:raise ValueError('Force certificate differs at '+key)
    except (KeyError,TypeError,ValueError,OverflowError) as exc:
        raise ValueError('Invalid or stale force certificate: '+str(exc)) from exc
    return computed


class StrikeDynamicsObserver:
    def __init__(self,env,probe_path):
        import json
        from running_balance import setup_feedback
        self.env=env;self.robot=env.scene['robot'];self.active=False;self.rows=[]
        if env.num_envs!=1:raise ValueError('Independent dynamics recording requires one environment')
        self.probe_path=Path(probe_path);self.probe=json.loads(self.probe_path.read_text())
        self.probe_certificate=validated_force_probe(self.probe)
        self.friction_sign=self.probe_certificate['friction_sign']
        self.ground=setup_feedback(env)['ground']
        self.views={side:env.scene['contact_forces']._physics_sim_view.create_rigid_contact_view('/World/envs/env_*/Robot/'+side+'_ankle_roll_link',self.ground,max_contact_data_count=128) for side in ['left','right']}
        self.segments={'pelvis':'pelvis','thorax':'torso_link','left_upper_arm':'left_shoulder_yaw_link','right_upper_arm':'right_shoulder_yaw_link'}
        self.indices={key:self.robot.body_names.index(name) for key,name in self.segments.items()}
        self.mass=float(self.robot.root_physx_view.get_masses().sum())

    def after_update(self,dt):
        if not self.active or dt<=0:return
        if abs(dt-.005)>1e-8:raise ValueError('Expected original5ms physics sample')
        r=self.robot.data;forces={side:contact_components(view,dt) for side,view in self.views.items()}
        normal={side:entry['force_matrix_w_N'] for side,entry in forces.items()}
        friction={side:(self.friction_sign*np.asarray(entry['friction_raw_w_N'])).tolist() for side,entry in forces.items()}
        row=dict(time_s=(len(self.rows)+1)*dt,physics_dt_s=dt,
            root_velocity_w=r.root_lin_vel_w[0].cpu().tolist(),
            support_foot_normal_force_w_N=normal,support_foot_friction_force_w_N=friction,
            support_foot_ground_force_w_N={side:(np.array(normal[side])+friction[side]).tolist() for side in forces},
            support_contact_capacity_exhausted=any(x['capacity_exhausted'] for x in forces.values()),
            segment_quaternions_wxyz={key:r.body_quat_w[0,index].cpu().tolist() for key,index in self.indices.items()},
            segment_angular_velocity_w_rad_s={key:r.body_ang_vel_w[0,index].cpu().tolist() for key,index in self.indices.items()})
        self.rows.append(row)

    def attach(self,result):
        from strike_baseline_metrics import validated_rows
        rows=validated_rows(result['traces'][0])
        if len(rows)!=len(self.rows):raise ValueError('Dynamics observer and original physical recorder differ in coverage')
        for original,sample in zip(rows,self.rows):
            if abs(original['time_s']-sample['time_s'])>1e-7:raise ValueError('Dynamics observer time differs')
            original.update({k:v for k,v in sample.items() if k not in ['time_s','physics_dt_s']})
        if len(result['traces'][0])>len(rows):
            terminal=result['traces'][0][-1]
            terminal.update({k:v for k,v in self.rows[-1].items() if k not in ['time_s','physics_dt_s']})
        complete=not any(r['support_contact_capacity_exhausted'] for r in self.rows)
        result['dynamics_telemetry']=dict(schema_version=1,robot_total_mass_kg=self.mass,segment_body_names=self.segments,
            support_ground_force=dict(includes_normal=True,includes_friction=True,force_on='robot',frame='world',ground_filtered=True,coverage_complete=complete,sign_probe_verified=True),
            ground_collision_paths=self.ground,probe_path=str(self.probe_path),probe_sha256=hashlib.sha256(self.probe_path.read_bytes()).hexdigest(),
            force_semantics='Ground-filtered normal force matrix plus separately aggregated friction anchors with isolated momentum-probed sign',
            torque_semantics='Existing implicit actuator clipped-PD estimate only; no new pure actuator torque measurement claimed',
            observer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),physical_writes=False,controller_changed=False)
        return result

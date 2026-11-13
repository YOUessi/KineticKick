"""Development-only trajectory search around an unchanged PAiD action sequence."""
import numpy as np

LEG = ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')

def residual_weights(t):
    # Raise and turn the foot during the backswing, preserve a forward swing after impact.
    return np.array([np.interp(t,[2.08,2.19,2.34,2.46,2.61],[0,1,1,.65,0]),
                     np.interp(t,[2.19,2.32,2.41,2.55,2.67],[0,1,1,.70,0])])

def make_controller(ref,schema,config):
    indices=[schema['joint_names'].index('right_'+j+'_joint') for j in LEG]
    coeff=np.asarray(config['coefficients']).reshape(2,len(indices))
    shoulder=schema['joint_names'].index('left_shoulder_roll_joint')
    wrist=schema['body_names'].index('left_wrist_yaw_link')
    hip=schema['body_names'].index('left_hip_roll_link')
    prior=None
    def control(ctx,target):
        nonlocal prior
        target[indices]+=residual_weights(ctx['t'])@coeff
        cap=config.get('backswing_pitch_target_cap')
        if cap is not None and 2.12<=ctx['t']<=2.30:
            # Shape the commanded backswing while leaving the actuator limits unchanged.
            target[indices[4]]=min(target[indices[4]],cap)
        relief=config.get('roll_setpoint_relief',0.)
        if relief and 2.10<=ctx['t']<=2.55:
            j=indices[5]
            lower,upper=ctx['limits'][j,:2]
            # Taper outward commands using the measured joint state and motion,
            # retaining plenty of margin for the original finite-effort drive.
            predicted=ctx['q'][j]+.020*ctx['dq'][j]
            for direction,distance in ((1,predicted-lower),(-1,upper-predicted)):
                proximity=np.clip(1.-distance/.06,0.,1.)
                if direction*(target[j]-ctx['q'][j])<0.:
                    target[j]+=direction*relief*proximity
        clearance=config.get('wrist_clearance')
        if clearance and ctx['first_contact_s'] is not None:
            # Apply to the recovery of the arms after release, using only current and past links.
            gap=float(np.linalg.norm(ctx['body_pos'][wrist]-ctx['body_pos'][hip]))
            rate=0. if prior is None else (gap-prior)/ctx['dt']
            prior=gap
            elapsed=ctx['t']-ctx['first_contact_s']
            window=np.interp(elapsed,[.18,.34,1.25,1.6],[0.,1.,1.,0.],left=0.,right=0.)
            near=np.clip((.22-gap-.08*min(0.,rate))/.07,0.,1.)
            target[shoulder]+=clearance*window*near
        return target
    return control

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
    def control(ctx,target):
        target[indices]+=residual_weights(ctx['t'])@coeff
        return target
    return control

"""Bounded early actuator feedback from a measured lateral capture error."""
import torch


def approach_features(reference,reference_velocity,root,velocity,ball_shift,ramp,initial_velocity_delta,lookahead,scale):
    # This is a local capture-error feature, not a guarantee of dynamic stability.
    error=reference[:,0]+ball_shift[:,0]*ramp-root[:,0]+lookahead*(reference_velocity[:,0]-velocity[:,0])
    return (error/scale).clamp(-1.,1.)*(initial_velocity_delta/.2).clamp(0.,1.)


def approach_command(weights,coefficients,feature,previous,maximum,maximum_step):
    desired=(weights[:,:,None]*coefficients).sum(1)*feature[:,None]
    desired=desired.clamp(-maximum,maximum)
    return previous+(desired-previous).clamp(-maximum_step,maximum_step)

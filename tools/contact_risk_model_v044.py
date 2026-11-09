"""v44 planned-action-conditioned full-29-joint future-speed forecaster.

This small supervised model estimates the largest future qdot ratio over a
fixed 1.25s postcontact horizon using present state AND planned style flag.
A learned estimate is not a physics barrier, a safe robot action or human
coordination. All reported MSE/gates use independent initial-condition groups.
"""
import torch
from torch import nn

class ActionConditionedRiskMLP(nn.Module):
    def __init__(self,width=67,hidden=128,output=29):
        super().__init__()
        if width!=67 or output!=29:raise ValueError("Frozen G1 input/output contract")
        self.net=nn.Sequential(nn.Linear(width,hidden),nn.ELU(),
            nn.Linear(hidden,hidden),nn.ELU(),
            nn.Linear(hidden,hidden//2),nn.ELU(),
            nn.Linear(hidden//2,output))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,x,feature_mean,feature_std,target_mean,target_std):
        if x.ndim!=2 or x.shape[-1]!=67 or not all(torch.isfinite(z).all() for z in (
             x,feature_mean,feature_std,target_mean,target_std)):
            raise ValueError("Finite current G1 measured-state and committed-controller input required")
        if (feature_std<=0).any() or (target_std<=0).any():raise ValueError("Nonzero train-only scale")
        normalized=(x-feature_mean)/feature_std
        estimated=target_mean+target_std*self.net(normalized)
        return torch.nn.functional.softplus(estimated*5.)/5.

def conditional_risk_score(q,qd,qd_limits,left_right_support_N,
                           ball_rel_root_xy,ball_vel_xyz,time_s,style_enabled):
    """Exact 67-d vector from measured robot state and already-planned style action.

    qdot / original limits; force is original ground-contact vertical N.
    No future trajectory or post-contact labels may be part of input.
    """
    import numpy as np
    def arr(x,n):
        v=np.asarray(x,dtype=np.float32)
        if v.shape!=(n,) or not np.isfinite(v).all():raise ValueError("Bad G1 measured state layout")
        return v
    angles=arr(q,29)
    speeds=arr(qd,29)
    limits=arr(qd_limits,29)
    if np.any(limits<=0):raise ValueError("Original joint qdot limits must be unchanged and positive")
    loads=arr(left_right_support_N,2)
    position=arr(ball_rel_root_xy,2)
    ball=arr(ball_vel_xyz,3)
    if not isinstance(style_enabled,(bool,int)) or style_enabled not in (False,True,0,1):
        raise ValueError("Known planned controller indicator required")
    if not np.isfinite(time_s):raise ValueError("Current control time required")
    result=np.concatenate([angles,speeds/limits,np.clip(loads,0,5000)/1500.,
                           position/3.,ball/5.,np.asarray([time_s/2.,float(style_enabled)],dtype=np.float32)])
    if result.shape!=(67,) or not np.isfinite(result).all():raise ValueError("Invalid 67D present-only input")
    return result

def risk_metrics(pred,target):
    import numpy as np
    p=np.asarray(pred,float);t=np.asarray(target,float)
    if p.shape!=t.shape or p.ndim!=2 or p.shape[1]!=29:raise ValueError("29 physical G1 qdot targets")
    if not np.isfinite(p).all() or not np.isfinite(t).all():raise ValueError("Nonfinite ratio")
    return dict(mae=float(abs(p-t).mean()),mse=float(((p-t)**2).mean()),
       worst_qdot_ratio_mae=float(np.mean(abs(p.max(axis=-1)-t.max(axis=-1)))),
       underpredicted_unsafe_window_count=int(((t.max(axis=-1)>1.)&(p.max(axis=-1)<1.)).sum()),
       all_future_unsafe_window_count=int((t.max(axis=-1)>1.).sum()))

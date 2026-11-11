#!/usr/bin/env python3
"""Exact 1.25x within original 75 mrad envelope; SYNTHETIC dose and reversal falsification.

This is not a human-learned policy, a certified collision controller, or real GRF.
Physical 29DoF limits and existing stance, qdot and nonfoot rejection remain untouched.
"""
from __future__ import annotations
import math
import torch
from phase_coordinated_v063 import PhaseCoordinatedTask, AMPLITUDE, SLEW, WINDOW, phase_trajectory

DOSE=1.25
FIRST=round(WINDOW[0]/0.02)
LAST=round(WINDOW[1]/0.02)
KINDS=('forward','reverse_window')
MARKER_BASE=dict(enabled=True,kind='bounded_synthetic_dose_falsification',
                 multiplier=DOSE,human_force_groundtruth=False)

def curve_contract(original, order):
    if order not in KINDS: raise ValueError('Unregistered v69 time order')
    if tuple(original.shape)!=(700,29) or FIRST!=43 or LAST!=78:
        raise ValueError('Unexpected original temporal phase lattice')
    curve=original.clone()*DOSE
    if order=='reverse_window':
        curve[FIRST:LAST+1]=torch.flip(curve[FIRST:LAST+1],dims=(0,))
    if not bool(torch.isfinite(curve).all()):raise ValueError('Nonfinite dose waveform')
    peak=float(curve.abs().max())
    if peak>AMPLITUDE+1e-7 or peak<.05:
        raise ValueError('Dose is outside preregistered physical 75mrad envelope')
    if bool((curve[:FIRST].abs()>1e-9).any()) or bool((curve[LAST+1:].abs()>1e-9).any()):
        raise ValueError('New dose leaks into unregistered time windows')
    return curve

class DoseTask(PhaseCoordinatedTask):
    def __init__(self,env,config):
        marker=config.get('phase_dose_v069')
        if not isinstance(marker,dict) or marker not in (
                {**MARKER_BASE,'order':'forward'},
                {**MARKER_BASE,'order':'reverse_window'}):
            raise ValueError('v069 dose marker must match frozen contract')
        if config.get('phase_reversal_v067') or config.get('phase_dose_v068'):
            raise ValueError('Conflicting phases cannot coexist')
        super().__init__(env,config)
        self.dose_order=marker['order']
        before=self.phase_curve.clone()
        self.phase_curve=curve_contract(before,self.dose_order)
        self.phase_reference_max_rad=float(self.phase_curve.abs().max())
        self.phase_original_reference_max_rad=float(before.abs().max())
        assert self.phase_reference_max_rad>self.phase_original_reference_max_rad
        assert math.isclose(self.phase_reference_max_rad/self.phase_original_reference_max_rad,DOSE,rel_tol=1e-6)

def get_task(env):
    if not hasattr(env,'_phase_dose_v069_task'):
        env._phase_dose_v069_task=DoseTask(env,env.cfg.joint_strike)
    return env._phase_dose_v069_task

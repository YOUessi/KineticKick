#!/usr/bin/env python3
"""Pre-registered, energy-matched reversal of SYNTHETIC v63 29DoF phase.

A control, not human imitation: reverses only the 0.86--1.56s temporal
phase samples of the same pre-existing 29-DoF intent, retaining every
support, joint-speed, proximity and real nonfoot rejection in the parent.
It cannot certify collider clearance or human-like movement.
"""
from __future__ import annotations
import torch
from phase_coordinated_v063 import PhaseCoordinatedTask, WINDOW

FIRST=round(WINDOW[0]/.02)
LAST=round(WINDOW[1]/.02)
class ReversedPhaseTask(PhaseCoordinatedTask):
    def __init__(self, env, config):
        marker=config.get('phase_reversal_v067')
        if marker != {'enabled': True, 'operation': 'reverse_existing_synthetic_phase_index_only',
                      'human_contact_groundtruth': False, 'unchanged_original_g1_physics': True}:
            raise ValueError('v67 preregistered negative-control marker mismatch')
        super().__init__(env, config)
        if self.phase_curve.shape!=(700,29) or FIRST!=43 or LAST!=78:
            raise ValueError('Frozen temporal lattice drift')
        original=self.phase_curve.clone()
        self.phase_curve[FIRST:LAST+1]=torch.flip(original[FIRST:LAST+1], dims=[0])
        if not torch.allclose(self.phase_curve[:FIRST],original[:FIRST]) or not torch.allclose(self.phase_curve[LAST+1:],original[LAST+1:]):
            raise RuntimeError('Control changed outside pre-impact phase')
        if not torch.allclose(self.phase_curve[FIRST:LAST+1].abs().sort(dim=0).values,
                              original[FIRST:LAST+1].abs().sort(dim=0).values):
            raise RuntimeError('Phase reversal changed per-joint amplitude multiset')

def get_task(env):
    if not hasattr(env,'_phase_reversal_v067_task'):
        env._phase_reversal_v067_task=ReversedPhaseTask(env,env.cfg.joint_strike)
    return env._phase_reversal_v067_task

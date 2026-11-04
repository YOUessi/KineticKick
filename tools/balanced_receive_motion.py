"""Keep impact retreat through the ball's flight/ground transition.

After a confirmed first touch, the foot can still move with inertia while contact
force is zero. Restoring its old position then can strike the descending ball.
"""
import numpy as np
from inside_receive_motion import InsideReceiveMotion


class BalancedReceiveMotion(InsideReceiveMotion):
    def yield_to_contact(self, target, actual_position, force):
        if not self.cfg.get('retain_retreat_after_contact',False):
            return super().yield_to_contact(target,actual_position,force)
        # Called only after the actor has measured a real first contact, and
        # before deliberate foot separation. No contact force is synthesized.
        away=-self.normal
        displacement=max(0.,float((actual_position-target[:3,3])@away))
        increment=min(displacement,max(0.,self.cfg['absorption_limit_m']-self.absorbed_displacement))
        self.absorbed_displacement+=increment
        result=target.copy();result[:3,3]+=increment*away
        return result

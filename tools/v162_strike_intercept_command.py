"""Connect early rolling-ball interception to an actual velocity/goal command in one episode."""
import json
from pathlib import Path

from v162_moving import stance_planner, support_clearance
from v162_strike_power import make_controller as command_controller

ROOT=Path(__file__).resolve().parents[1]

def make_controller(ref,schema,config):
    planner_cfg=json.loads((ROOT/config['moving_config']).read_text())
    power_cfg=json.loads((ROOT/config['power_config']).read_text())
    plan=stance_planner(ref,schema,planner_cfg)
    clearance=support_clearance(schema,dict(urdf=planner_cfg['urdf'],**planner_cfg['left_clearance']))
    swing=command_controller(ref,schema,power_cfg)
    def control(ctx,target):
        return swing(ctx,clearance(ctx,plan(ctx)))
    return control

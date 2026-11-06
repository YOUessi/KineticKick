"""Prespecified cases shared by serial execution and independent auditing."""
import json

COMBINATIONS = [
    dict(id='combined_left_slower',ball_offset_xy_m=[-.025,0.],root_velocity_y_delta_mps=.10),
    dict(id='combined_right_faster',ball_offset_xy_m=[.025,0.],root_velocity_y_delta_mps=-.10),
    dict(id='heldout_left_near_slower',ball_offset_xy_m=[-.018,.020],root_velocity_y_delta_mps=.075),
    dict(id='heldout_right_far_slower',ball_offset_xy_m=[.022,-.025],root_velocity_y_delta_mps=.125),
    dict(id='heldout_left_far_faster',ball_offset_xy_m=[-.015,-.030],root_velocity_y_delta_mps=-.075),
    dict(id='heldout_right_near_faster',ball_offset_xy_m=[.020,.025],root_velocity_y_delta_mps=-.125),
    dict(id='heldout_confirm_left_far_slower',ball_offset_xy_m=[-.020,-.020],root_velocity_y_delta_mps=.100),
    dict(id='heldout_confirm_right_near_slower',ball_offset_xy_m=[.015,.015],root_velocity_y_delta_mps=.150),
    dict(id='heldout_v028_near_slow_13',ball_offset_xy_m=[.010,.020],root_velocity_y_delta_mps=.130),
    dict(id='heldout_v028_near_slow_17',ball_offset_xy_m=[.020,.010],root_velocity_y_delta_mps=.170),
    dict(id='heldout_v028_left_near_slow',ball_offset_xy_m=[-.010,.020],root_velocity_y_delta_mps=.150),
]


def load_cases(root):
    frozen=json.loads((root/'configs/strike_baseline_v1.json').read_text())['cases']
    return {case['id']:case for case in frozen+COMBINATIONS}


EVALUATION_CASES={
    '0.1.26':{'combined_left_slower','combined_right_faster'},
    '0.1.27':{'heldout_confirm_left_far_slower','heldout_confirm_right_near_slower'},
    '0.1.28':{'heldout_v028_near_slow_13','heldout_v028_near_slow_17','heldout_v028_left_near_slow'},
}

def case_group(name,version):
    return 'evaluation' if name in EVALUATION_CASES.get(version,set()) else 'development'

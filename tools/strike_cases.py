"""Prespecified cases shared by serial execution and independent auditing."""
import json

COMBINATIONS = [
    dict(id='combined_left_slower',ball_offset_xy_m=[-.025,0.],root_velocity_y_delta_mps=.10),
    dict(id='combined_right_faster',ball_offset_xy_m=[.025,0.],root_velocity_y_delta_mps=-.10),
    dict(id='heldout_left_near_slower',ball_offset_xy_m=[-.018,.020],root_velocity_y_delta_mps=.075),
    dict(id='heldout_right_far_slower',ball_offset_xy_m=[.022,-.025],root_velocity_y_delta_mps=.125),
    dict(id='heldout_left_far_faster',ball_offset_xy_m=[-.015,-.030],root_velocity_y_delta_mps=-.075),
    dict(id='heldout_right_near_faster',ball_offset_xy_m=[.020,.025],root_velocity_y_delta_mps=-.125),
]


def load_cases(root):
    frozen=json.loads((root/'configs/strike_baseline_v1.json').read_text())['cases']
    return {case['id']:case for case in frozen+COMBINATIONS}

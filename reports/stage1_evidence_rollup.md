# Stage 1 — Free-root G1 walking and soccer positioning

These are observed runs, not a trained natural humanoid kicking controller.

## walking
- duration_s: 4.0
- min_root_height_m: 0.6244
- reset_count: 0
- displacement_x_m: 1.3358

## goal_approach_no_ball
- duration_s: 7.6
- min_root_height_m: 0.62
- resets: 0
- final_distance_to_approach_xy_m: 0.234
- final_heading_error_deg: -0.2

## goal_approach_with_ball
- duration_s: 3.0
- min_root_height_m: 0.623
- resets: 0
- final_distance_to_approach_xy_m: 0.17
- final_heading_error_deg: 0.45
- max_foot_ball_contact_force_N: 0.0

## preplant_24cm_stop
- duration_s: 10.4
- min_root_height_m: 0.624
- resets: 0
- final_distance_to_approach_xy_m: 0.24
- final_heading_error_deg: -0.39
- min_support_left_foot_target_error_m: 0.319
- support_left_foot_stationary_verified_step: None
- max_foot_ball_contact_force_N: 0.0

## preplant_8cm_stop
- duration_s: 10.4
- min_root_height_m: 0.625
- resets: 0
- final_distance_to_approach_xy_m: 0.082
- final_heading_error_deg: -0.45
- min_support_left_foot_target_error_m: 0.2242
- support_left_foot_stationary_verified_step: None
- max_foot_ball_contact_force_N: 0.0

## preplant_24cm_stop — foot constraints
Not an achieved planned foot plant unless 15 consecutive frames satisfy all criteria.
- steps: 520
- min_left_foot_target_error_m: 0.319
- final_left_foot_target_error_m: 0.433
- median_foot_slip_mps: 0.262
- number_of_frames_satisfying_all_support_constraints: 0
- episode_reset_count: 0
- unexpected_ball_contact_frames: 0
- note: Support-ready frames not necessarily consecutive; need 15 consecutive for policy transition.

## preplant_8cm_stop — foot constraints
Not an achieved planned foot plant unless 15 consecutive frames satisfy all criteria.
- steps: 520
- min_left_foot_target_error_m: 0.2242
- final_left_foot_target_error_m: 0.2665
- median_foot_slip_mps: 0.282
- number_of_frames_satisfying_all_support_constraints: 0
- episode_reset_count: 0
- unexpected_ball_contact_frames: 0
- note: Support-ready frames not necessarily consecutive; need 15 consecutive for policy transition.

## Next gate
Learn support-foot placement/hold with explicit left-foot support contact
and measurable naturalness, before inserting an aggressive shot into the walking policy.
Never treat a fixed-root shot demo as free-root humanoid control.

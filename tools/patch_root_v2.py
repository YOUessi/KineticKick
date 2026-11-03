"""One-off reproducible diff to create global-root v2 from v1."""
from pathlib import Path
p=Path("/home/you/projects/kinetickick/tools/train_paid_stage1_global_root_v2.py")
s=p.read_text()
changes=[
("ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm",
 "ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm"),
("global_torso_planar_huber, global_torso_linear_velocity_tracking)",
 "global_torso_planar_huber, global_torso_linear_velocity_tracking,\n    global_anchor_xy_out_of_bounds)"),
("pa.add_argument(\"--iters\",type=int,default=240)",
 "pa.add_argument(\"--iters\",type=int,default=200)"),
("pa.add_argument(\"--seed\",type=int,default=85)",
 "pa.add_argument(\"--seed\",type=int,default=86)"),
("pa.add_argument(\"--phase-zero-prob\",type=float,default=0.65)",
 "pa.add_argument(\"--phase-zero-prob\",type=float,default=0.80)\n    pa.add_argument(\"--root-xy-threshold\",type=float,default=0.70)"),
("pa.add_argument(\"--root-exp-weight\",type=float,default=8.0)",
 "pa.add_argument(\"--root-exp-weight\",type=float,default=16.0)"),
("pa.add_argument(\"--root-huber-weight\",type=float,default=-4.0)",
 "pa.add_argument(\"--root-huber-weight\",type=float,default=-10.0)"),
("pa.add_argument(\"--root-velocity-weight\",type=float,default=3.0)",
 "pa.add_argument(\"--root-velocity-weight\",type=float,default=6.0)"),
("pa.add_argument(\"--init-checkpoint\",type=str,default=str(MIGRATION))",
 "pa.add_argument(\"--init-checkpoint\",type=str,default=str(ROOT/\"experiments/paid29_global_root_v1/seed85_n96_iter240/last_model.pt\"))"),
("    cfg.terminations.ee_body_pos.params[\"threshold\"]=0.4",
 "    cfg.terminations.ee_body_pos.params[\"threshold\"]=0.4\n    cfg.terminations.global_anchor_xy=DoneTerm(func=global_anchor_xy_out_of_bounds,\n        params={\"command_name\":\"motion\",\"threshold\":a.root_xy_threshold})"),
("run_dir=ROOT/f\"experiments/paid29_global_root_v1/seed{a.seed}_n{a.num_envs}_iter{a.iters}\"",
 "run_dir=ROOT/f\"experiments/paid29_global_root_v2/seed{a.seed}_n{a.num_envs}_iter{a.iters}\""),
("\"purpose\":\"GLOBAL WORLD ROOT TRAJECTORY EXPERIMENT, NOT ORIGINAL PAiD\",",
 "\"purpose\":\"GLOBAL ROOT V2, STRONGER PROGRESS + PLANAR TERMINATION, NOT ORIGINAL PAiD\","),
("\"root_velocity_weight\":a.root_velocity_weight,",
 "\"root_velocity_weight\":a.root_velocity_weight,\n        \"root_xy_termination_threshold_m\":a.root_xy_threshold,\n        \"reference_end_wrap\":\"frozen last frame; no forced teleport\","),
("    assert torch.equal(motion.time_steps,phase)\n    assert obs.shape[1]==166,obs.shape",
 "    assert torch.equal(motion.time_steps,phase)\n    # Prevent stock PAiD MotionCommand from teleporting the robot to frame 0 at frame 260.\n    orig_update=motion._update_command\n    def no_reference_wrap(self):\n        cap=self.motion_length-1\n        at_end=self.time_steps>=cap\n        if bool(at_end.any()):\n            self.time_steps[at_end]=cap[at_end]-1\n        return orig_update()\n    motion._update_command=types.MethodType(no_reference_wrap,motion)\n    assert obs.shape[1]==166,obs.shape"),
]
for before, after in changes:
    n=s.count(before)
    assert n==1,(n,before)
    s=s.replace(before,after,1)
p.write_text(s)
import ast
ast.parse(s)
print("ROOT_V2_PATCHED",p,len(s.splitlines()),"lines")

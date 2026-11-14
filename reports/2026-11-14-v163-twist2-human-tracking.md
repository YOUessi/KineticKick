# TWIST2 independent tracker comparison, CMU 10_02

The MIT-licensed upstream project is [Amazon FAR TWIST2](https://github.com/amazon-far/TWIST2/tree/b06178f19a22f2138cbd31f60c6d494bc263f67d). The local asset directory is ignored by Git. Obtain the unchanged model and accompanying license from the same revision.

```bash
src=$(mktemp -d)
git -C "$src" init --quiet
git -C "$src" fetch --depth=1 --filter=blob:none https://github.com/amazon-far/TWIST2.git b06178f19a22f2138cbd31f60c6d494bc263f67d
git -C "$src" checkout FETCH_HEAD -- LICENSE assets/ckpts/twist2_1017_25k.onnx
mkdir -p experiments/v163_policy/assets
cp "$src/LICENSE" "$src/assets/ckpts/twist2_1017_25k.onnx" experiments/v163_policy/assets/
sha256sum experiments/v163_policy/assets/twist2_1017_25k.onnx
```

The released checkpoint hash is `126ab9089b6ca0be8b6582fc9ef17b0b6c6d6577378f9324a2e10d580fe48f2e`. The Tang-local CMU reference hash is `3b29d1a49287e45d15052ade9241ad9aab20b6411307de030f8603330a61479c`. At 50 Hz the actor reads a 35-dimensional whole-body command (local root XY velocity, height, roll/pitch, yaw rate and 29 joints), current proprioception, ten historical samples, and a repeat of the current command as the future input. It returns all 29 named PD commands. The simulation preserves original URDF effort/velocity/position limits, original armatures, collisions and free root, with the TWIST2 gains. The ball follows physical contact alone.

```bash
env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/you/conda_env/maniskill/bin/python tools/v163_policy_twist.py \
  --model experiments/v163_policy/assets/twist2_1017_25k.onnx \
  --reference experiments/v162_human/armclear/10_02.npz \
  --root-lead .22 --lead .08 --feedback 3.5 --yaw-feedback 2 \
  --placement .05 -.15 --guard .055 --arm-gap --rest \
  --output experiments/repro_v163_twist
```

The unassisted 50 Hz actor followed the complete 4.90 s human sequence and stayed upright for 14 s. It missed the registered ball with root position RMSE 32.94 cm and pelvis-relative body MPJPE 3.73 cm during the source clip. A neutral 0.65 s transition after the last source frame produced a continuous terminal bilateral stance of 8.135 s. Pose feedback reduced the source-clip root RMSE to 16.96 cm and yielded 8.035 s terminal stance. Both variants missed the ball.

In the command above, approach and body preview brought the right foot to the registered ball at 2.960 s. The ball entered the 2.8 m-distant goal at 2.781 m/s exit speed, with terminal bilateral stance 8.44 s, source-clip root RMSE 20.15 cm and pelvis-relative body MPJPE 4.71 cm. The maximum non-foot contact was 5.13 N. The primary trial had 96 joint-limit sample/axis excursions and 0% instep impulse by all four conventions, and failed the paper's combined qualification. All measured arrays from the frozen repeat matched the first run exactly. The complete TWIST2 actor trace contains 700 x 1432 input observations and 700 x 29 output actions.

At a separately calibrated nearer ball (.71, -.67) one development attempt reached 5.368 m/s, scored and stood for 8.70 s. Its four instep shares were 93.97%, 93.97%, 89.05% and 0%, and it had eight joint-limit excursions. An original-ball variant reached 3.825 m/s but missed the goal. Development includes 18 complete 14 s CPU PhysX runs including the repeat, all on the same training clip. Every trial and source hash is in Tang’s `experiments/v163_policy/ledger.json`, SHA-256 `621916b263c96dacda0babac7672a6f03ef015dd3c1721247c3184270d44a8dd`. The wider failure distribution is relevant when interpreting the calibrated scores.

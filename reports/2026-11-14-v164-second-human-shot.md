# CMU 10_05 自由根 G1 正脚背射门与恢复

冻结控制在 SAPIEN 3.0.3 CPU PhysX 的 14 秒单回合通过既有 12 项门槛。右脚在 2.580 秒触球，水平出球速度 3.9131 m/s，3.5804 秒整球越过球门平面，末段连续站稳 10.075 秒。真实物理轨迹只在回合开始时初始化。对照与修复使用同一机器人、球、球门和 3.62 秒人体重定向参考。

| 同场控制 | 物理结果 | 首次摔倒 | 末段连续站稳 |
|---|---|---|---|
| 原 PAiD | 3/12，未触球，1.075 秒右腕与右髋碰撞 106.7 N，1.420 秒首次关节位置越界 | 2.905 秒 | 0 秒 |
| 脚面与接近动作已标定，未接随挥 | 11/12，3.9131 m/s 进球，右脚相对骨盆前移 6.54 cm | 无 | 9.595 秒 |
| 冻结的右脚触球门控随挥 | 12/12，3.9131 m/s 进球，右脚相对骨盆前移 15.42 cm | 无 | 10.075 秒 |

击球前脚尖向下 31.33°。接触前后两种脚姿与两种接触点定义的组合，都把 100% 的测得法向冲量归于正脚背，累计 1.6915 N·s。击球后 250 ms 内右脚向前 29.02 cm，120 ms 时前向速度 0.800 m/s，240 ms 时开始持续回摆。全段没有额外碰球、非足部接触超限或实际关节位置、速度越界。球进门时中心横向距门心 22.64 cm。原 Isaac GPU 碰撞几何等价性与实物部署待验证。

## 迁移质量与开发分母

原始 436 帧、120 Hz 的人体动作来自 CMU Graphics Lab Motion Capture Database 的受试者 10，实际竞技水平没有独立认证。原始 C3D SHA-256 为 `1389f02ff7e20a9bb1d536cf4a2351abca887810f13ab6d113ff4f9272bf7337`。数据授权、来源和处理散列见 Tang 的 `experiments/v160_human/manifest.json`，研究使用应注明 CMU 来源。GMT 模型使用 v163 冻结权重及其作者的研究使用条件。[原始 C3D](https://mocap.cs.cmu.edu/subjects/10/10_05.c3d) · [模型与许可证记录](2026-11-14-v163-human-motion-balance.md)

来球位置取源发射前连续可见的 48 个三标记帧，中位数经同一髋部尺度变换对齐，球心高度取物理半径 0.11 m。出球方向取第 303 至 309 帧的标记位移。人体测得的是未校准标记团运动，缺少同步力测量与人体真实撞击时刻。重定向事件时刻为 2.52151724137931 秒，配置内摆动相位中心为 2.5212278920274844 秒。参考右脚于 2.54 秒经过已注册球位置，左脚在事件窗内最近仍相距 20.0 cm。双臂自碰处理保留原腿部与骨盆参考。

冻结权重读取来源动作，前段利用实测骨盆跟踪误差，实际右脚法向/切向合成的球接触力超过 2 N 后启用有限时长随挥目标。髋部、膝、踝和支撑腿补偿在这个开发片上标定。2175 条有完整参数与结果记录的短物理候选分布于 14 批次，见本地 `experiments/v164_strike/development/index.json`。其中包含脚面方向、力度、开始卸载支撑和落地时机的失败尝试。PAiD、跨片直接复制和若干开发代表另留 14 秒全程记录。10_05 是 train 分组，与 10_02 同一人。这次回合给出第二条经单片开发的成功轨迹，本片的 2,175 个开发候选均采用中心球位。冻结后另做的两次球位扰动记录在[总报告](2026-11-14-v164-human-transfer.md)。独立跨人射门测试数为零。

182 帧的 50 Hz 源参考走完。相对骨盆的身体位置平均误差为 4.79 cm，世界骨盆位置 RMSE 为 32.56 cm，最后一个源帧的水平骨盆误差约 79.35 cm。全局落点、末段动作对齐与跨来源泛化仍需研究。冻结后独立重跑的 18 个轨迹数组逐元素相同，解压后 2800 步原生接触日志 SHA-256 均为 `b188b6bef920f3a1a9dc003fe66d17c69eaab1e91053058ad1ffc19d34b6f7e6`。

## 在 Tang 复现

本地要保留原 G1 资产、确认散列的 GMT 权重及 `experiments/v164_clear/references/10_05.npz`。下列命令从仓库根目录执行并写入新的时间戳目录。CLI 输入使用实际运行的十位小数方向 `(.9890973568, .1472630352)`。场景注册脚本的 float32 方向为 `(.9890973567962646, .1472630351781845)`，物理运行归一化结果为 `(.9890973653235156, .1472630364690346)`。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
run_dir="experiments/v164_strike/replay_$(date +%Y%m%d_%H%M%S)"
/usr/bin/python3 tools/v164_strike_scene.py --clip 10_05
/home/you/conda_env/maniskill/bin/python tools/v163_gmt.py \
  --reference experiments/v164_clear/references/10_05.npz \
  --drive native --settle --feedback 2.5 --adapter configs/v164_strike_10_05.json \
  --ball 2.1207752227783203 -.8806997537612915 \
  --direction .9890973568 .1472630352 --goal-distance 2.8 --output "$run_dir"
/usr/bin/python3 tools/v161_evaluate.py --run "$run_dir" --output "$run_dir/score.json"
/usr/bin/python3 tools/v161_surface_audit.py --run "$run_dir"
```

## 视频

两份视频使用同一已计分的比较轨迹，均为 14 秒、350 帧、1280×720 H.264，中央小窗为人体动作重定向。球门从人体标记估算的出球方向往前 2.8 m 搭建。Tang 本地的 `experiments/v164_strike/full_field_preview.json` 与 `preview.json` 保存视频与源轨迹的散列。

- [全场固定机位](../media/v164_human_10_05_shot_full_field.mp4) 同时呈现原 PAiD、改进后的机器人、整座球门与触球到过线的球路，保留右上方俯视轨迹。SHA-256 为 `972e5d0751afa1fddc21d611a762d71f467fd088e0971af6454b22f793e0d828`。
- [触球近景](../media/v164_human_10_05_shot_comparison.mp4) 放大 2.58 秒的触球姿态，部分后续球路离开主画面。SHA-256 为 `bb3496ce7f87840d57e41749026dd3ec733a5e61b2d6b1f7e4896685f4d3e644`。

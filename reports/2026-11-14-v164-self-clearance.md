# v0.1.164 真人动作跟踪的自碰修复

CMU 10_01 全速跟踪时，左腿卸载，左胫与右支撑脚在 5.620 s 开始碰撞，原生接触对峰值为 610.39 N。摆腿期间髋滚转改变间距的方向会翻转。四组单方向髋滚转改动仍产生 307–1,215 N 非足净力。

参考预处理对两侧肩滚转做平滑外展，根据原碰撞体消除腕髋和臂躯干穿插。运行时读取原 G1 双腿位姿与足底载荷，以 0.14 s 预测左右胫骨及对侧鞋底之间的距离。两脚横向间距低于 0.10 m 且预计胶囊间距低于 0.11 m 时，使用实际最近点对的间距雅可比修正卸载腿的髋俯仰、滚转、偏航与膝关节。配置来自 10_01 的候选比较，随后固定。冻结参数保存在 `configs/v164_clear_support.json`。

| 源片 | 全部七项通过 | 最大非足净力 N | 严格末段站稳 s | 位置 / 速度越界数 | 根轨迹 RMSE m | 骨盆相对全身 MPJPE mm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 10_01，训练开发 | 是 | 10.65 | 5.125 | 0 / 0 | 0.243 | 29.9 |
| 10_06，曾用于跟踪开发 | 是 | 0.00 | 7.095 | 0 / 0 | 0.105 | 30.6 |
| 11_01，历史跨被试，冻结确认 | 否 | 0.00 | 8.095 | 1 / 0 | 0.268 | 29.1 |

三段均完成 14 s，自由基无跌倒，未发生超过 20 N 的危险非足接触。11_01 在 10 ms 时右踝滚转超出原上限 0.000004700 rad，严格门槛保留失败。足部与足球在三段均无接触。这里验证整段跟踪，射门需要另行评测。

## Tang 复现

从版本工作树运行。两套 Python 分别提供 Pinocchio 和 SAPIEN。`mktemp` 为每轮另建目录，保留既有实验结果。

```bash
run=$(mktemp -d)
prep=/home/you/conda_env/env_isaaclab/bin/python
phys=/home/you/conda_env/maniskill/bin/python
for clip in 10_01 10_06 11_01; do
  source_ref="experiments/v162_human/references/$clip.npz"
  PYTHONNOUSERSITE=1 "$prep" tools/v164_clear_reference.py \
    --input "$source_ref" --output "$run/$clip.npz" > "$run/$clip.json" || break
  PYTHONNOUSERSITE=1 "$phys" tools/v164_support_track.py \
    --reference "$run/$clip.npz" --config configs/v164_clear_support.json \
    --output "$run/guard_$clip" || break
  PYTHONNOUSERSITE=1 "$phys" tools/v164_support_score.py \
    --run "$run/guard_$clip" --reference "$source_ref" --clearance "$run/$clip.json" \
    --output "$run/guard_$clip/score.json" || break
done
```

各片的 `score.json`、完整物理状态与接触流存于 Tang 的 `experiments/v164_clear/guard_10_01/`、`guard_10_06/` 和 `guard_11_01/`。参考的输入、输出与 URDF 哈希在 `experiments/v164_clear/references/` 的同名 JSON 中。九个开发候选的配置、指标和输出哈希保存在 `experiments/v164_clear/development/train_candidates.json`，同目录保留每次运行的原始记录。

整理几何反馈的变量名后重新运行 10_01。18 个物理状态数组、3 个控制器数组和整个解压后的原生接触流与冻结实验逐项一致。对应源代码和对照结果保存在 `experiments/v164_clear/development/v164_clear_legs_frozen.py`、`experiments/v164_clear/development/readable_replay/array_parity.json`。

## 10_05 失衡与六片固定分母

同一初态修正后，10_05 原来在 2.915 s 左上臂碰躯干（272.11 N），3.770 s 骨盆低于 0.45 m。失衡早于碰撞。2.20 s 网络预览中的前向速度指令达到 3.64 m/s；2.60 s 骨盆倾角达 23.0°，原片为 8.5°；2.90 s 实测与参考侧向速度为 1.07 与 0.20 m/s，骨盆水平偏差已达 0.337 m。单独取消额外步幅需求或取消支撑位置与速度反馈均消除摔倒和碰撞，分别站稳 7.970 与 9.510 s。第三个实验按实测沿源速度方向的进展逐渐撤去步幅需求，也通过七项，站稳 8.315 s。三条回放支持总速度需求过高参与了失衡。

在同一六片固定分母上比较两个候选。先冻结两组的输入和代码哈希，先跑零额外步幅。由于 10_03 回退，再执行已冻结的速度反馈候选。三组分别对应原冻结的 0.7 倍额外步幅、额外步幅为零以及实测速度接近参考时逐步撤去额外步幅。下表列出每片的原始骨盆轨迹 RMSE（m）和七项验收结果。

| 源片 | 原固定 0.7 | 固定零额外步幅 | 随实测速度衰减 |
| --- | ---: | ---: | ---: |
| 10_01 | 0.254 通过 | 0.206 通过 | 0.235 通过 |
| 10_02 | 0.363 通过 | 0.356 通过 | 0.298 通过 |
| 10_03 | 0.138 通过 | 0.615 失败 | 0.964 失败 |
| 10_05 | 0.411 失败 | 0.171 通过 | 0.203 通过 |
| 10_06 | 0.162 通过 | 0.309 通过 | 0.221 通过 |
| 11_01 | 0.289 通过 | 0.154 通过 | 0.134 通过 |
| 六片均值与严格通过数 | 0.270，5/6 | 0.302，5/6 | 0.343，5/6 |

后两组均有 6/6 完成 14 s、全段未跌倒、位置和速度保持原限位且末段站稳。它们均在 10_03 触发右腕与同侧髋的危险接触，分别为 2.760 s、32.39 N 和 2.780 s、38.33 N。原固定增益保住 10_03，在 10_05 摔倒。预先约定的规则以七项通过数排序，相同时比较所有六片的根轨迹均方根误差均值。在两个新候选中选额外步幅为零。原版保留为基准，三者按全部六片的同一规则比较仍是原版均值最低。该组包含已用于开发的运动及历史评测过的 11_01，无法据此估计对新被试的泛化。所有候选均按相同代码对六片执行。

### 重跑冻结六片对照

在 Tang 的项目工作树执行。代码、配置和六个既有净化源的 SHA 位于 `experiments/v164_clear/stride_matrix/cohort/preregister.json`。十二次回放的原始状态、原生接触流及逐片 `score.json` 均在同目录的 `no_stride/` 和 `speed_matched/` 下。`no_stride_results.json`、`speed_matched_results.json` 含两项完整分母。下面的命令恢复相同参考与初始化。

```bash
run=$(mktemp -d)
phys=/home/you/conda_env/maniskill/bin/python
for mode in no_stride speed_matched; do
  runner=tools/v164_support_track.py
  cfg=configs/v164_clear_no_stride.json
  if [ "$mode" = speed_matched ]; then
    runner=tools/v164_clear_stride.py
    cfg=configs/v164_clear_support.json
  fi
  for clip in 10_01 10_02 10_03 10_05 10_06 11_01; do
    ref=experiments/v164_clear/references/$clip.npz
    clear=experiments/v164_clear/references/$clip.json
    case "$clip" in
      10_02) ref=experiments/v162_human/armclear/10_02.npz
             clear=experiments/v164_initial/legacy_transform_provenance.json ;;
      10_03) ref=experiments/v164_landing/fixed_10_03/reference.npz
             clear=experiments/v164_landing/fixed_10_03/clearance.json ;;
    esac
    dest="$run/$mode/$clip"
    PYTHONNOUSERSITE=1 "$phys" "$runner" --reference "$ref" --config "$cfg" \
      --initial-state-config configs/v164_initial_braking.json --output "$dest" || break
    PYTHONNOUSERSITE=1 "$phys" tools/v164_support_score.py --run "$dest" \
      --clearance "$clear" --reference "experiments/v162_human/references/$clip.npz" \
      --output "$dest/score.json" || break
  done
done
```

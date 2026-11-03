# v0.1.2：方向与力度控制原型，定点精度验收未通过

2026-11-04，Tang 本地 Isaac Lab / PhysX。分支 `version/v0.1.2`，从已验证的 `version/v0.1.1` 开始。本版本增加了指定方向和力度的实验接口；**尚未实现稳定的自由状态定点射门，不应把软件测试通过解释为机器人目标达成。**

## 已实现的接口与测量

- 三种触球方式继续使用自由根部机器人和真实脚球接触；正常关节驱动负责动作，没有给球施加人为冲量，没有在回合内重置机器人或球。
- 固定球门中心 `(0, -3.7)` m、法向 `(0, -1)`、开口 `1.6 × 1.1` m。目标为世界坐标 x=`-0.30 / 0 / +0.30` m。改变目标不会移动或转动球门。这是横向低球落点，不是任意高度角落，也不是最终停球位置。
- 轻／中／重请求由版本化参数表转换成策略瞄准输入、触球时序和有限关节修正。少量组合使用触球前的足端反馈：每个关节修正最多 0.30 rad，逆运动学位置误差超过 15 mm 或姿态误差超过 0.15 rad 时不覆盖原策略。
- 每 5 ms 记录物理接触、球状态和站立状态。出球力度定义为第一次脚球接触后 **0.10 s** 的水平球速，按相邻样本插值；落点定义为球心首次正向穿过固定球门平面的位置。
- 只开放已标定的离散组合；不支持的中间位置或球速会在启动仿真前报错，不会悄悄夹紧成另一个请求。

下表是**请求值**，不是保证实现的球速：

| 踢法 | 轻 | 中 | 重 |
|---|---:|---:|---:|
| 脚内侧 `inside` | 3.8 m/s | 4.3 m/s | 4.9 m/s |
| 正脚背 `instep` | 1.6 m/s | 2.0 m/s | 2.5 m/s |
| 脚外侧 `outside` | 1.4 m/s | 1.8 m/s | 2.2 m/s |

## 冻结后的独立验证：27/54 达标

控制参数根据标定种子 801–808 的数据选择，随后冻结。验收使用新种子 **811、812**，每种踢法 3 个目标 × 3 档力度，共 54 回合。初始根部 xy 各扰动 ±1 cm、偏航 ±0.01 rad、关节角 ±0.005 rad。没有挑选成功重跑替换失败。

每回合同时要求：走近球、指定脚面承担主要接触冲量、整球通过球门、连续站稳至少 1 秒、落点误差 ≤0.15 m、球速误差 ≤`max(0.20 m/s, 请求球速的12%)`。所有成功判定从完整轨迹重新计算。每种踢法至少 15/18、每种踢法的每档力度至少 5/6，且实测方向与力度均有次序和间隔，才通过版本验收。

| 踢法 | 种子 811 | 种子 812 | 合计 | 判定 |
|---|---:|---:|---:|---|
| 脚内侧 | 5/9 | 6/9 | **11/18** | 未通过 |
| 正脚背 | 3/9 | 1/9 | **4/18** | 未通过 |
| 脚外侧 | 5/9 | 7/9 | **12/18** | 未通过 |
| 全部 | 13/27 | 14/27 | **27/54（50%）** | **未通过** |

54 回合都有指定脚面触球；51 回合进球；53 回合完成站稳；1 回合发生跌倒/终止，不能算恢复成功。单看落点误差达标 32/54，单看球速达标 37/54。软件回放检查器正确返回退出码 **1**。

三档实测平均出球速度（包含失败回合的有效速度观测）：

| 踢法 | 轻档实测均值 | 中档实测均值 | 重档实测均值 |
|---|---:|---:|---:|
| 脚内侧 | 3.63 m/s | 4.27 m/s | 4.77 m/s |
| 正脚背 | 2.07 m/s | 1.43 m/s | 2.27 m/s |
| 脚外侧 | 1.42 m/s | 1.67 m/s | 2.11 m/s |

脚内侧和脚外侧具有平均意义上的力度调节效果。正脚背中档比轻档慢，方向与力度尚未解耦。不同档位成功率详见本地 `reports/v012_targeted_acceptance_summary.json`。

典型失败保留在原始记录中：

- `inside_medium_x0 / seed811`：目标 x=-0.30 m，实际 x≈+0.076 m、球速 3.706 m/s，方向和力度都偏离。
- `instep_medium_x2 / seed811、812`：出球分别仅 0.092、0.203 m/s，没有到达球门平面。
- `outside_low_x2 / seed811`：触球后跌倒/终止；完整终止样本保留，没有自动重置掩盖。

## 诊断与当前边界

仅改变策略虚拟瞄准点，能够改变方向，但会同时改变摆腿和球速；脚外侧可调球速范围也过窄。改变触球时序能扩大力度范围，但相同指令在重复物理回合中的横向差异约 20 cm。接触记录显示脚外侧原动作经常击中脚掌前缘附近。

尝试完整足端姿态反馈后，较大的动作覆盖破坏了原策略的平衡，也出现不可达目标。改成短时、有限修正并拒绝不可达目标，恢复了该轮 12 个标定回合的稳定性，但没有解决独立扰动验收中的精度问题。不能依据这些实验声称查表或足端修正已经实现稳定定点控制。

本次停止继续同类参数搜索，保留可复现的控制原型和负结果。稳定定点射门仍是未完成工作。下一项有依据的工作是让策略显式接收目标球速和触球方式，结合目标方向、真实接触面及恢复站稳的奖励进行条件训练，并用不同初始姿态的独立回合验证；这项训练尚未运行，也没有训练成功的证据。

## 在 Tang 上运行

每次只开一个仿真器。以下是**实验命令**，结果应以窗口和 JSON 中的实际误差为准：

```bash
cd /home/you/projects/kinetickick
/home/you/conda_env/env_isaaclab/bin/python tools/show_free_shooting_live.py \
  --style inside --target-x -0.3 --power high --output reports/targeted_demo.json

# 数值球速必须是该踢法已标定的档位；例如脚外侧的 1.8 m/s。
/home/you/conda_env/env_isaaclab/bin/python tools/show_free_shooting_live.py \
  --style outside --target-x 0.3 --speed-mps 1.8 --output reports/targeted_demo_outside.json
```

黄色标记表示目标横向 ±15 cm 的区间。窗口显示请求值和实测误差；未满足定点指标时不会显示完整任务成功。没有 `--target-x`、`--power` 或 `--speed-mps` 时，仍运行原 v0.1.1 流程。

复现验收（六次顺序运行，不能并行使用同一 GPU）：

```bash
for seed in 811 812; do
  for style in inside instep outside; do
    /home/you/conda_env/env_isaaclab/bin/python tools/eval_free_shooting.py \
      --targeted-acceptance --style "$style" --seed "$seed" \
      --output "reports/reproduced_${style}_${seed}.json"
  done
done
/home/you/conda_env/env_isaaclab/bin/python tools/check_targeted_shooting_acceptance.py \
  reports/reproduced_*_81[12].json --output reports/reproduced_targeted_summary.json
```

当前版本的精度验收预期不通过，不能为得到绿色结果修改阈值。CPU 单元测试与上述物理验收分别报告。

## 冻结与证据

- 控制实现冻结提交：`340f188`（后续只增加回放检查、显示和验证记录）。
- 配置：`configs/targeted_shooting_v1.json`，SHA256 `b2917d32ddc620e05f876dac5dc348e5b97a14d07f34c50cbb1738028e3b6ba2`。
- 策略：`489711f8654faf8aba49584a39266a33a9068a8a6bc8122e88f2bc7f06739ab1`。
- 原脚掌资产：`34ef7f147b91af2615ca5c88f42ad8fd5d1a00fa63ca66d8e01ec9245a2c7761`。
- `reports/v012_accept_inside_811.json`：`a5b5df54a8bd1130339fc67e4c7027ef88ed417e8fb94c0043d561c0356b8814`。
- `reports/v012_accept_instep_811.json`：`092316e65db390dbd9c8ae35b9ae82ebdc08e26360ad23e0ef79528e9d3bb83f`。
- `reports/v012_accept_outside_811.json`：`23d1b2ed7df818dd8adf7ee3dac91ddd477e5e84b95f95920ea06ca33a63f426`。
- `reports/v012_accept_inside_812.json`：`8250b2583d38785a3f6eeae7d683a44c2dc7a4e5fc9cf671f73cfe583a7540b2`。
- `reports/v012_accept_instep_812.json`：`c65bdb53a3c22fea7fe71b84be488a8bfc3815fe02cb44700272902d2a3d8837`。
- `reports/v012_accept_outside_812.json`：`af9d83fe409461fcb94cb8b58ac82241aa2b177483a3e47af8b3392d434f4c61`。

原始轨迹、模型、第三方资产和录像留在 Tang 本地；GitHub 只发布源码、配置和本报告。完整运行及测试仍依赖 Tang 的本地资产。

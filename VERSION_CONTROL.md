# KineticKick 版本管理

远程仓库：[YOUessi/KineticKick](https://github.com/YOUessi/KineticKick)。
Tang 本地目录：`/home/you/projects/kinetickick`。

## 分支约定

- `main`：集成基线，首次导入的代码快照保存在这里。
- `version/v0.1.0`：首次导入版本，与最初的 `main` 指向同一个提交。
- 每个后续版本使用独立的 `version/v<主版本>.<次版本>.<修订号>` 分支，在开始修改前创建。
- 优先使用用户指定的版本号；未指定时，检查本地和远端分支，采用下一个未使用的修订号。
- 同一版本内的后续修改继续提交到对应分支；开始新版本时创建新分支，保留旧分支历史。
- 经过验证的版本可以再集成到 `main`。保留版本分支，不强制推送或覆盖历史。

例如，从首次导入版本开始下一版（确认工作区干净且该分支名尚未使用）：

```bash
git fetch origin
git switch -c version/v0.1.1 version/v0.1.0
# 修改代码并运行相应检查后：
git add <本次修改的文件>
git diff --cached
git commit -m "Describe the version change"
git push -u origin version/v0.1.1
```

## 首次纳入 Git 的范围

版本控制包含项目脚本、测试、项目自身的 JSON 配置、README、研究计划和 Markdown 报告。
以下内容保留在 Tang，由 `.gitignore` 排除：

- `data/`、`third_party/`：外部动作、机器人资源与上游源码。
- `.pretrained_checkpoints/`、`experiments/`：预训练权重、训练检查点、运行参数和训练日志。
- `media/`，以及 `reports/` 内除顶层 Markdown 文档以外的生成结果。
- `configs/paid_g1_main.urdf`、`configs/paid_original/`：上游 URDF 副本及包含本机网格路径的派生文件。
- Python 缓存、虚拟环境、环境变量文件和密钥文件。

因此，GitHub 保存代码和研究文档的版本，不是 Tang 全部实验素材的备份。
文档中引用的本地数据、历史测量和权重不会随克隆自动取得。

## 上游代码定位

首次导入时，两个上游工作区均没有未提交修改：

| 本地目录 | 上游仓库 | 提交 |
| --- | --- | --- |
| `third_party/HumanoidSoccer` | [TeleHuman/HumanoidSoccer](https://github.com/TeleHuman/HumanoidSoccer) | `e72e470230047dedaf66df0983f1d0ab746faeb5` |
| `third_party/whole_body_tracking` | [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) | `9db915838b17785c6fec90c91fe80b95b147e1bd` |

在其他机器复现时，应按对应提交准备上游代码和所需资源，保留上游许可，并参考 README 的环境约束。
PAiD 原始 URDF 位于其仓库的
`source/whole_body_tracking/soccer/assets/unitree_description/urdf/g1/main.urdf`。
本地 `configs/paid_g1_main.urdf` 与 `configs/paid_original/main.urdf` 是相同副本，SHA-256 为
`dbc8471004e9b0b85c27a4ebc1f0d90a727c1cf7ca9ef507b805596f2004d64b`。
恢复这两个副本并准备上游网格后，运行 `tools/prepare_paid_original.py` 可生成本机路径对应的 `paid_main_resolved.urdf`。
该脚本要求先创建 `configs/paid_original/` 和 `reports/` 目录。

## 验证范围

Tang 上的完整测试需要本地数据、历史测量与检查点，使用现有环境运行：

```bash
/home/you/conda_env/env_isaaclab/bin/python -m unittest discover -s tests -v
```

这些测试运行在 CPU 上，不启动 Isaac Sim 或重新训练策略。
新克隆尚未准备 Tang 素材时，可以在装有 NumPy 的 Python 环境中运行以下独立测试：

```bash
python -m unittest discover -s tests -p 'test_goal_planner.py' -v
python -m unittest discover -s tests -p 'test_motion_audit.py' -v
python -m unittest discover -s tests -p 'test_soccer_visual.py' -v
```

完整测试中的一部分依赖 Tang 的绝对路径与忽略文件；它们不是跨机器即开即用的验收套件。

## v0.1.1 的球位感知策略依赖

`tools/eval_free_shooting.py` 使用上表 PAiD 提交中的 `ckp/policy_30000.onnx`。
本地放在 `.pretrained_checkpoints/paid_official/policy_30000.onnx`，不提交模型文件。
SHA-256：`489711f8654faf8aba49584a39266a33a9068a8a6bc8122e88f2bc7f06739ab1`。
模型沿用上游许可与来源信息。

Tang 的 ONNX Runtime 安装在忽略目录中，保持现有 Isaac Lab 环境不变：

```bash
/home/you/conda_env/env_isaaclab/bin/python -m pip install --no-deps \
  --target experiments/runtime/onnxruntime onnxruntime==1.23.2
```

仿真器必须在导入 NumPy 等数值库之前启动，否则系统 NumPy 与仿真器插件可能出现二进制不兼容。

## v0.1.7 连续短程助跑

本版本基于`version/v0.1.6`创建独立`version/v0.1.7`，不更新main或旧版本历史。继续使用相同PAiD官方ONNX、原始29DoF资产及已记录的上游提交；没有重新训练或更换模型。增加使用上游`soccer-standard-009_right.npz`和`soccer-standard-010_right.npz`，仍保留在被忽略的third_party目录。

`configs/runup_shooting_v1.json`记录该版本控制源码、物理资产、权重、参考内容指纹；预览入口校验它们。生成的完整物理遥测、帧与视频保存在reports/media，不进入Git。CPU测试仍有Tang-local依赖。新流程验收范围、未通过的组合与未重标定的定点／力度能力见`reports/2026-11-04-runup-sequence.md`。

## v0.1.8 脚外侧与原生力度定点

本版本从已推送的v0.1.7创建`version/v0.1.8`，保留main和所有旧版本历史。继续使用原29DoF资产、官方ONNX和原始力矩/速度上限；未重新训练或替换权重。新增LAFAN RUN参考转换仅供有界物理诊断，保留原始数据，生成参考置于ignored experiments/running_v018。

默认预览使用`configs/runup_shooting_v2.json`：脚外侧完整流程6/6、三种动作原生力度定点18/18。持续奔跑和三档联合定点尚未通过，不发布虚假校准表。旧控制新种子回归5/6及失败细节均记录在`reports/2026-11-04-running-aim.md`。新源码使v0.1.7配置指纹不再匹配；旧配置及视频仍可在原版本分支重现。新生成的物理遥测/媒体继续由.gitignore排除。

## v0.1.9 高球快速试跑

从v0.1.8创建独立`version/v0.1.9`。仅新增挑球参数、试跑入口和首次飞行统计，不改原控制器、物理资产或上游版本。按用户要求快速跑通，不重复完整回归和扰动矩阵。三组inside-foot示例及实测范围见`reports/2026-11-04-lofted-shooting.md`；它们不是精确轻/中/重或三维定点校准。原始报告和媒体继续留在ignored reports/media。

## v0.1.10 跑步策略训练与实际碰撞体修正

从v0.1.9建立独立`version/v0.1.10`。继续使用记录的同一上游提交、原29DoF资产、关节与执行器限制。新增官方ONNX到可训练循环策略的转换、有限参考生成、有界训练、严格ONNX导出及跑步回放入口；新训练权重、参考和诊断证据留在ignored experiments/reports/media。源码指纹绑定的v0.1.8/v0.1.9射门配置保持一致。

参考检查按上游`replace_cylinders_with_capsules=True`使用实际胶囊几何；通过有界髋屈曲修正避开起步自碰撞，没有关闭碰撞。原数据和失败训练均保留。两次有界训练及实际回放的结果、未完成目标见`reports/2026-11-04-running-training-pilot.md`；完整目标尚不能由训练回报或短段生存率推出。

## v0.1.11 跑步反馈与物理过渡

从已推送的v0.1.10建立独立`version/v0.1.11`，不更新main或旧分支历史。新增176观测循环策略、真实脚底支撑/胶囊高度反馈、有界训练、减速/零速度起跑候选和实测状态触发的控制交接。沿用同一上游提交、原29DoF资产及执行器限制；原runup/loft配置指纹保持一致。

六次交替腾空及6.94秒无摔倒回放已有物理证据，但最终站稳、静止起跑、射门衔接和联合目标控制仍未通过，见`reports/2026-11-04-running-feedback-and-transitions.md`。训练权重、原始参考、日志、数值报告与媒体继续仅保存在Tang的ignored目录；GitHub代码不构成完整素材备份。

## v0.1.12 连续奔跑后的真实射门

从v0.1.11建立独立`version/v0.1.12`。新增一个实际含球的奔跑／射门策略交接入口、组合指令参考生成器和冻结的固定场景配置，原控制器和旧版本不变。无新增训练，沿用原模型与资产限制。

同一14秒物理回合已完成6次交替腾空、脚内侧触球进球及末段7.185秒站稳。运动初态、固定球位和未完成项目，以及第一次末段失稳的修复依据，见`reports/2026-11-04-running-to-real-shot.md`。原始数据、参考、权重、数值证据及媒体继续保留在Tang的ignored目录。

## v0.1.13 连续跑动中的射门

从已推送且验证的v0.1.12建立`version/v0.1.13`，保留main和全部旧版本。新增连续跑步控制与右腿足球策略的有限窗口协同、专门拒绝停步射门的判据和固定场景配置，旧控制源码及其指纹保持不变。沿用同一上游提交、原29DoF资产与执行器限制，无新增训练。

同一14秒物理回合中，触球前3次交替腾空，触球时向前2.299m/s，脚内侧进球，继续迈步并在末段站稳9.575秒。范围、失败诊断和本地证据见`reports/2026-11-04-stride-shot.md`。原始动作、权重、运行报告和媒体仍保留在Tang的ignored目录，新克隆不包含这些素材。


## v0.1.14 正脚背与脚外侧移动射门

从已推送的v0.1.13创建独立`version/v0.1.14`，保留main与全部旧版本历史。新增两种固定场景、以实际支撑状态触发的收势及固定世界坐标恢复提示。新增接触采样器校正球面/脚面与碰撞时刻姿态，保留原始点、间隙和姿态；旧分类规则与旧入口源码不变。无新增训练，继续沿用同一上游提交、原29DoF资产、策略和执行器限制。

两段最终14秒实际渲染回合均完成合法对应脚面触球、进球与末段站稳；7项针对性CPU检查通过，未跑完整套件或扰动矩阵。数据、失败记录、适用范围及复现入口见`reports/2026-11-04-more-foot-surfaces.md`。本地权重、动作、生成报告与媒体继续由.gitignore排除。

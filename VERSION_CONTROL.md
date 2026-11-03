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

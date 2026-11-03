# KineticKick：G1 自然步态 Stage 2 实际进展（2026-10-21）

## 目标与边界

研究目标是让 G1 在真实动力学中生成自然的人类步行、转向、慢跑，再迁移至自然足球助跑、支撑脚落地和射门。
“动作捕捉的3D重放”和“机器人真正通过控制在重力下自然步行”严格区分。当前前者已有原型，后者尚未完成。

## 1. 官方37DoF PPO与V1/V2物理训练

- 保留 NVIDIA 官方 G1 37DoF PPO 权重，不覆盖。
- V1和V2各完成64环境、120 PPO迭代、184320采样步，来自同一官方起点。
- 通过9环境同场景匹配测试：3控制器×0.25/0.5/0.8 m/s命令，各410步；2个评估种子42、45增加三模型一致的初始速度扰动，避免纯确定性重复。
- 联合守门指标包括膝关节坐标、速度误差、底座/躯干姿态、真实左右脚接触力以及承重脚踝刚体速度代理。
- V1、V2在3档速度下均未满足完整守门标准：膝关节坐标较小、速度误差有所减少，但躯干倾角或脚部承重滑动速度恶化。不能推广为更好的人类自然步态。
- 注意：只有一个训练种子seed40；评估多种子不等于独立训练种子。
- 证据：reports/gait_robustness_matrix_20261021.md 与对应JSON，reports/gait_perturbed_seed42/45.json。

## 2. Tang 上新增真实人类步态重定向参考

Tang直连 huggingface.co 不通，但 hf-mirror.com 可访问：
- 下载5段G1 29DoF retargeted LAFAN1动作CSV：walk1_subject1、walk2_subject3、walk3_subject5、walk4_subject1、run1_subject2。
- 合计34438帧，30FPS，约1147.9秒，即约19.13分钟。
- 36列严格对应基座XYZ、四元数XYZW及29个G1关节。
- 逐帧检查有限数值、四元数归一化及URDF关节限位，均合格。
- 跑步参考重新选择连续运动较快的8秒片段（所选样本根部速度中位约1.53m/s），避免把低速过渡误标成跑步。
- 下载G1 29DoF rev1_0机器人描述URDF和完整STL，共39份文件26.82MB。29活动关节名字与参考文件完全对应，全部59处URDF引用的网格文件存在。
- 资产存在Tang的 data/assets/g1_29dof_lafan1，动作存在 data/human_motion/lafan1_g1_29dof。
- 原始素材和资产保留在gitignore中的data/，不会上传GitHub或复制Mac。
- 数据集包含BSD 3-Clause文件，但人体原始LAFAN1动作版权和使用许可须在对外发布前另行审核。

## 3. 可用于之后追踪训练的动作参考银行

- 通过URDF前向运动学计算39个连杆的真实几何变换；并在5段8秒片段上提取跨髋、肩部协调的相关性线索。
- 例如walk1、walk2样本左右髋pitch相关约-0.616/-0.862，左右肩pitch相关约-0.703/-0.842。两种机械结构关节轴不同，不能直接转移原始关节角。
- 已建立reference bank（5×240×29关节），包括关节位置和时间导数、基座位姿与速度、左右足部的运动学轨迹和伪支撑相位标签。
- 伪支撑相位从脚高度与移动速度推断，不是真实物理接触传感器或人体测力台标签。
- 数据：data/human_motion/g1_29dof_reference_bank_v0.npz
- 清单：reports/g1_29dof_reference_bank_manifest.json
- 源码：tools/g1_retarget_kinematics.py、tools/audit_lafan1_g1_29dof.py、tools/extract_human_coordination_priors.py、tools/build_g1_29dof_reference_bank.py

## 4. Tang 真实可观看的人体步态参考

- 运行脚本：tools/show_g1_human_reference_walk.py
- GUI标题：KineticKick | LAFAN1 HUMAN-GAIT REFERENCE
- 默认WALK-1，可点WALK-2、WALK-3、RUN-1，也可暂停、慢放、恢复播放。按钮已在真实Tang GUI点击验证。
- 完整G1 29DoF模型（39 links、35 USD meshes、约393270三角形）显示正常，原始G1头部外观正常，不含训练时绿蓝速度箭头。
- 这仅是渲染网格运动学回放，新增物理刚体/碰撞体为0；本演示不验证机器人在重力和电机控制下能跟随动作。绝不可把它冒称为已成功训练的自然步态。
- 截图只在Tang：media/g1_29_human_walk_reference_solid.jpg

## 5. 下一工程验收关口

Gate A：将已验证的29DoF URDF / meshes 转成有质量、惯性、关节轴、限位、脚底接触与PD控制的PhysX articulation。在Tang当前Isaac Sim安装中暂未找到URDF importer扩展；不能认为29DoF物理本体已完成。

Gate B：在自由根部、正常重力下，先站立、再跟踪 walk1/walk2/walk3/run1 参考；对照 BeyondMimic 类型tracking baseline。

Gate C：研究 phase-aware whole-body coordination 先验，利用真实人体步态相位和跨身体部位耦合，带足部接触、滑动、摔倒、躯干姿态的刚性约束。

Gate D：独立训练种子、多速度和转身泛化、人体自然度盲评。只有通过才把自然步态policy用于助跑和足球技术动作。

## 工程验收

- 程序：tools/eval_gait_multi_condition.py 和 tools/summarize_gait_robustness.py。
- 自动化测试：tests/test_lafan1_reference.py及现有测试。
- 保留物理行走GUI：tools/show_g1_gait_policy_ab.py（三个策略选择，默认OFFICIAL）。
- 当前 Tang 可视化GUI为29DoF人体动作参考；物理37DoF比较GUI与参考viewer不要同时运行以避免额外GPU负担。

**没有宣称已成功训练人体自然步态，所有研究数据只保存在 Tang，不复制至 Mac。**

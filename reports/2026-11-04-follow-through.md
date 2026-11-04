# v0.1.5 正脚背随摆与未通过的脚外侧试验

2026-11-04，Tang。基础版本：`version/v0.1.4`（8943ce6），物理控制基线：v0.1.3。

## 结论：未完成稳定随摆

能在单次仿真中做出明显的触球后随摆，但冻结扰动复核未通过。**没有发布新的稳定控制配置，普通预览仍使用v0.1.3**。正脚背、脚外侧的稳定随摆均不能宣称完成；近竖直触球和准确方向／力度也仍未完成。

单次演示中的正脚背踝向前随摆由约15cm增加到25cm，且进球并恢复站稳。对应实验释放末端为frame137，髋膝参考在frame122–158平滑变化。最后复核的是137.5帧候选，只有3/6同时满足新增随摆、实际触球、进球和站稳。

`tools/shooting_control.py` 新增可选 `continuation` 后续关节目标阶段，带独立余弦混合窗口、原有限位和原电机驱动，用于脚外侧诊断。**默认配置没有启用它**。其多环境独立时序、释放、限位、禁止嵌套及禁止误用为参考偏移均有测试。

没有修改物理阻尼、质量、关节限位或终止规则，没有给球或身体写入速度、位置。仿真本身已有惯性，问题在于位置目标、离球时机和后续平衡动作的配合。

## 验收定义与范围

固定真实球门中心(0,-3.7)m，球(0.14,-0.93)m与左右偏移2cm；初始根部XY各±1cm、朝向±0.01rad、关节角±0.005rad。每种踢法三个球位×两个独立种子，8.2秒／回合。沿真实球到球门方向投影测量，不使用虚拟策略目标。

候选要通过的正脚背检查：

- 首次真实触球后0.25秒内，踝向前位移≥22cm、扣除骨盆平移后的相对位移≥15cm。
- 触球后0.12秒的20ms差分速度≥0.60m/s；连续两个采样间隔向后运动不得早于0.20秒。
- 0.22–0.45秒内回摆峰值≤2m/s；延长至0.75秒的恢复回摆峰值≤2.5m/s。
- 0.08–0.75秒内不得再出现超过2N的球接触；20ms运动采样必须有限、连续并覆盖整个窗口。
- 原有实际触球面、脚尖下压／脚背朝向、进球、连续站稳及失败判定全部保留；至少5/6同时通过，任何物理失败或原生终止都会使矩阵失败。

后期候选的脚内侧、脚外侧作为未修改的物理／姿势回归，不计作新增随摆成功；其审计输出显式列出 `follow_through_required_styles=["instep"]`，最终 `passed=false`。这些阈值是本轮工程检查，不是人体自然性的证明，也不代表任意球位或实机能力。

## 为什么未交付脚外侧改进

仅改变释放时机／参考不能消除急收腿；加入独立膝关节保持后，早期回摆峰值从约3.7m/s降至0.35–0.77m/s，首批冻结矩阵18/18完成物理流程。但单环境渲染暴露了后续缺陷：腿保持在前方近地面，骨盆向后移动，放开后出现约6.4m/s的追赶式回摆；该渲染回合在3.56秒原生终止。终止时根部高0.763m、倾角4.76°，不能直接称为摔倒。

因此把检查延长至0.75秒；原候选的脚外侧6/6都未通过延长后的运动检查。六个提前释放组合、六个髋膝配合组合仍未消除此问题，停止这条参数调整路线并恢复旧控制。不能用只看触球附近的慢放宣称恢复自然。

正脚背的第二次独立复核18/18物理流程成功，但新增随摆仅4/6，两个回合在0.12秒的速度为0.592、0.594m/s，未达0.60门槛。没有放宽门槛；另做两个小范围释放试验，137帧候选的后续独立复核又出现了二次脚尖接触，只有3/6同时通过。最终选择此前已单独试过的137.5帧中间时机，重新冻结并使用全新种子946/947；所有失败候选保留，不混入最终验收。

## 冻结结果与证据

| 候选 | 独立种子 | 完整物理流程 | 正脚背新增随摆 | 结论 |
|---|---|---:|---:|---|
| 138帧；含脚外侧膝过渡 | 935/936 | 18/18 | 6/6 | 脚外侧延长检查0/6，另有渲染终止，拒绝 |
| 138帧；恢复旧脚外侧 | 940/941 | 18/18 | 4/6 | 速度门槛不足，拒绝 |
| 137帧；旧脚外侧 | 943/944 | 17/18 | 3/6 | 二次触球、姿势等失败，拒绝 |
| 137.5帧；旧脚外侧 | 946/947 | 17/18 | 3/6 | 仍未稳定通过，拒绝 |

这里新增随摆均按最后的0.75秒覆盖要求重放，不把初版只看0.45秒的通过结果当作交付依据。首个候选的6/6不能覆盖后续独立检查中的失败。最后候选6次正脚背均站稳，没有摔倒或原生终止，但有1次未满足完整物理射门要求。

44个诊断回合、4轮各18回合的独立矩阵、3个渲染回合，共119个实际仿真回合；未训练模型、未调用付费推理服务。源码测试123项通过，仅证明代码检查通过，不代表控制任务通过。

本地关键证据：

- `reports/v015_follow_probe1.json` 至 `probe7.json`：全部44个诊断回合。
- `experiments/follow_through/candidate1_frozen.json` 至 `candidate4_frozen.json`：四轮对应的原始冻结清单，保留原哈希和控制内容。
- `reports/v015_follow_final946.json`、`reports/v015_follow_final947.json`、`reports/v015_follow_final_summary.json`：最后一轮，`passed=false`。
- `reports/v015_follow_demo_outside.json`：导致扩大恢复检查的3.56秒原生终止。
- `reports/v015_follow_final_demo_instep.json`：137帧单次演示，seed945；只证明该回合成功。

最后冻结清单的SHA256与源码指纹见下面的证据清单。JSON、媒体和大体积数据留在Tang，GitHub保留代码、试验入口和本记录。

## 预览与复现

`media/kick_follow_through_v0.1.5_experiment.mp4` 左侧为v0.1.3、右侧为137帧实验。两侧初始机器人状态、真实球／球门、模型和相机相同；包含完整8.2秒和同一时间段的0.25倍局部慢放。录像标明“单次成功演示，稳定性验收未通过”。脚外侧未展示为成功改进。

```bash
cd /home/you/projects/kinetickick
# 显式实验，普通预览的默认配置没有改变
/home/you/conda_env/env_isaaclab/bin/python tools/show_free_shooting_live.py \
  --trials configs/follow_through_feasibility_trial.json --seed 945 \
  --output reports/follow_through_experiment_live.json
# 重放最后冻结矩阵；预期返回1（未通过）
/home/you/conda_env/env_isaaclab/bin/python tools/follow_through.py \
  --config experiments/follow_through/candidate4_frozen.json \
  reports/v015_follow_final946.json reports/v015_follow_final947.json \
  --output reports/v015_follow_final_summary.json
/home/you/conda_env/env_isaaclab/bin/python -m unittest discover -s tests
```

需要Tang本地资产、模型、动作和IsaacLab环境，每次只运行一个仿真器。下一步应针对真实离球事件、脚球间隙与平衡状态做反馈控制；当前固定帧释放参数对微小初态变化敏感，继续细扫相同参数已不能证明稳健性。

## 版本与兼容性

候选冻结控制器源码、动作参考和配置哈希，运行器记录实际哈希，审计核对完整矩阵与5ms接触记录。旧v0.1.2定点／力度标定保持原指纹，因本版源码扩展而在仿真启动前拒绝运行；不要把旧报告替换上新哈希。要复现旧定点实验，用保留的 `version/v0.1.4` 或更早分支。

原始报告、录像、模型和数据留在Tang，不上传GitHub。主分支及以往版本不改写。

## 证据哈希

| 本地文件 | SHA256 |
|---|---|
| `experiments/follow_through/candidate4_frozen.json` | `eac86403ea15c5d7131c9722a6c486a20b89ff752b93087a478716fafa0d3190` |
| `configs/follow_through_feasibility_trial.json` | `daaf25b940a9dcd25358083ab8172df48395fa1e0ccce7b2a3c907346b3730da` |
| `reports/v015_follow_final946.json` | `99460da3fe1e39daefff202dcbefb8a65fb8de86286cce5d60bc348c1a13ad6f` |
| `reports/v015_follow_final947.json` | `5a2c8c5fc25701394a1954abc09671f24d837a0333aaa1783b8f82ec641210db` |
| `reports/v015_follow_final_demo_instep.json` | `31bb313b2fd55f80e6f68a01c7808e4bf7622e2f5dbb5d10fdc7b50ef4ad39b5` |
| `media/kick_follow_through_v0.1.5_experiment.mp4` | `efe5e4cea814f7799c3516d4859bab5c45890ae2f76feb6066703111670bad28` |

# v0.1.15：正脚背加力射门

新增一组可复现的正脚背加力参数，仍使用v0.1.14的移动接近、真实触球、随摆和站稳流程。最终14秒实际录像中球进门，末段连续站稳8.190秒，没有非脚部碰球或非脚部碰撞。

| 同场景测量 | 原正脚背 | 加力正脚背 |
| --- | ---: | ---: |
| 首次触球(s) | 1.580 | 1.570 |
| 水平出球速度(m/s) | 3.772 | 4.630 |
| 三维出球速度(m/s) | 3.922 | 4.753 |
| 首段受力采样时长(ms) | 15 | 15 |
| 进球 | 是 | 是 |
| 末段连续站稳(s) | 8.810 | 8.190 |

水平球速增加**22.8%**。比较使用相同球位[-0.12,-4.02]、球门[0.7,-6.52]、种子1201、资产及策略；只改变关节控制与恢复参数。速度取首段连续正受力结束后第一次零足部力采样时的实球速度，采样间隔5ms。该时刻不单独证明几何分离，也不把受力时间变长当成加力依据。

## 实现

新增`configs/stride_instep_strong_v1.json`，复用未修改的`tools/run_stride_styles.py`及`StrikeVelocityPulse`。在参考帧[74,77,78,79]内短时请求右髋俯仰速度峰值-12rad/s，左脚地面支撑至少80N时才生效，之后及时撤去加速。控制仍通过原关节PD执行器和原速度/力矩限制，没有修改机器人/球状态、球质量、碰撞参数或添加外部球冲量。

以实测偏出球门的轨迹为依据，小幅调整脚背朝向；使用逆运动学计算目标修正，踝俯仰目标仍受原边界限制，实际方向以球轨迹为准。恢复提示取自v0.1.14末段已稳定站姿的局部目标，并按原入口转成固定场地坐标、地面球心高度的提示。

## 有界试跑中的淘汰项

- `v015_instep_hip_drive.json`：持续髋部加速虽使球更快，但大腿过摆，骨盆与右髋碰撞；未通过。
- `v015_instep_braked_drive.json`：缩短加速后合法触球进门，但14秒结束时倾角过大，未通过恢复。
- `v015_outside_drive.json`：加力脚外侧射门后失衡，未发布该组合。
- `v015_instep_knee_drive.json`：伸膝方案出现小腿碰球，未采用。
- `v015_instep_strong_candidate.json`：出球更快并站稳，但偏出原球门；随后调整脚背朝向，没有移动球门来计分。
- `v015_instep_strong_aimed.json`及`v015_instep_strong_peak12.json`：完整流程通过，选用后者制作最终录像。

原始失败报告继续保存在Tang的ignored目录。本版只有参数与说明变更，没有修改旧入口或旧配置；不重新训练，不跑完整CPU测试套件或扰动矩阵。最终发布依据为完整14秒物理录像及同场景球速比较。

## 使用与本地证据

```bash
# 原正脚背
/home/you/conda_env/env_isaaclab/bin/python tools/run_stride_styles.py --style instep \
  --output reports/new_normal.json
# 加力正脚背
/home/you/conda_env/env_isaaclab/bin/python tools/run_stride_styles.py \
  --plan configs/stride_instep_strong_v1.json \
  --output reports/new_strong.json --video-dir media/new_strong_frames
```

最终遥测为`reports/v015_instep_strong_preview.json`，对比为`reports/v015_power_comparison.json`；其SHA-256以及录像实际使用的原始计划保存在配置中。视频为`media/stride_instep_strong_v0.1.15.mp4`，包含完整14秒正常速度与同回合5.6秒近景慢放。

范围是运动初态、固定球位的右脚正脚背加力档；不是机器人最大功率测试，也不代表三种脚面、轻中重及方向/高度联合标定全部完成。所有旧版本和本地研究资产保留；数据、模型与生成媒体仍遵守.gitignore。

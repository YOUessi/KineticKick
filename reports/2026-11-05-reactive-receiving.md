# v0.1.21：观测来球后抬脚、脚弓卸力与恢复站稳

六个有界的地面/高空来球组合均有通过的6秒实际物理录像：从同一个双脚落地姿态开始，观测球的当前位置与速度，再预测接触高度、时刻和方向。脚持续移动到接触位置并随球回撤；地面球通过脚面制动降低余速，随后分离、落脚、恢复支撑。收脚后没有再次触球，也没有非接球脚或其他身体部位碰球。

“停稳”仍是近静止标准，不是数学上的零速度。本版将门槛收紧到球速 **<0.02m/s**、总角速度 **<0.3rad/s**，在身体0.65m范围内且球心低于0.145m，末段连续保持至少0.75s。站稳要求根部高度>0.65m、倾角<12°、水平速度<0.15m/s、双脚地面支撑各>20N，末段连续至少0.5s。没有为失败场景放宽门槛。

## 最终录像结果

速度栏是来球初始水平速度；高空球还带有2.943m/s的初始向上速度，实测最高球心约1.043m。三个速度与方向是三个组合，并非完整速度×方向矩阵。

| 场景 | 水平来速 m/s / 偏转 | 开始响应 / 触球 s | 最终球速 m/s / 总角速度 rad/s | 末段近静止 s | 末段站稳 s |
| --- | --- | --- | --- | --- | --- |
| `ground_center` | 1.200 / 0° | 0.64 / 1.175 | 0.00385 / 0.0911 | 4.410 | 3.370 |
| `ground_left_slow` | 0.900 / -8° | 1.04 / 1.560 | 0.01099 / 0.1059 | 3.990 | 2.950 |
| `ground_right_fast` | 1.500 / +8° | 0.40 / 0.945 | 0.00662 / 0.1570 | 4.635 | 3.590 |
| `aerial_center` | 3.975 / 0° | 0.04 / 0.690 | 0.00857 / 0.1304 | 5.160 | 4.010 |
| `aerial_left_slow` | 3.600 / -8° | 0.04 / 0.690 | 0.01479 / 0.1577 | 5.065 | 2.820 |
| `aerial_right_fast` | 4.300 / +8° | 0.04 / 0.685 | 0.00980 / 0.0995 | 5.150 | 4.010 |

六段均满足真实脚内侧冲量占比100%、触球前主动回撤、初段减速超过40%、安全分离以及无摔倒终止。高空球在首次落地前完成脚弓触球，初段减速只比较落地前的球速；地面球减速包含地面作用。恢复交接的原生胶囊-球间隙至少2.5cm，交接后最大球速不超过0.1m/s，从开始分离到结束均没有右脚再次接触。原射门分数不用于接球验收。

## 动作怎样响应来球

- 地面与高空使用同一个初始关节姿态和参考，右脚原点高度3.5cm。初始右脚有约3.4N地面支撑，但身体已偏重左支撑腿；这是双脚落地的接球站姿，尚未验证从均匀负重的普通站姿转换。
- 每20ms读取仿真中的真实球位置与速度，至少三次观测后才触发运动。根据当前高度和竖直速度判断滚动/空中轨迹，预测接触位置、下降速度与脚面方向。没有读取场景名、预设发球时刻或球的初始参数来决定动作；不是摄像头视觉识别。
- 只在预测时间和脚部工作空间可达时准备接触。地面提前量0.55s，高空0.72s；最后80ms冻结接触预测，避免临近碰撞时目标跳动。高空在0.04s触发之后持续抬脚到接触，而非先把脚摆在空中等待。不同高度改变脚部目标，不同速度改变动作时刻；停止/远离的球不触发盲目抬脚。
- 空中回撤速度比例以实测水平速度修正，参考速度3.974566m/s时为0.1014，斜率−0.11，限定在0.04–0.20。脚面的横向速度阻尼通过原关节力矩执行。接触法线实测偏差在正偏转方向较大，因此按实测入射角从−0.13到+0.13rad逐渐加入最多−0.008rad（约−0.46°）的脚面修正。该补偿经过这几个局部方向标定，不能外推为任意角度精确接球。
- 地面球实际触碰后才启动低速制动；高空球落地后的脚面接近速度受到限制，避免再次把球推走。高空案例的主要减速来自首次空中卸力和地面接触，不能把没有发生的二次触球算作成功制动。
- 恢复时先向远离球的方向分离，再放回脚。检测到脚真实落地后固定其实际落点，逐渐转移10%的支撑负载，避免追逐理想落点而拖脚失衡。最终使用关节位置/速度/力矩控制维持站姿；原ONNX仍为现有运行框架提供元数据/观测兼容，但本版 `motor_stand` 模式的策略混合权重为0，不宣称由学习策略完成恢复。

相较v0.1.20的0.034/0.039m/s，两类球余速进一步降低，但初始姿态、控制和部分来球路径已变化，不能把它当作严格控制变量的A/B实验。

## 验证、失败与证据范围

本轮保留 **30个有完整数值报告的有界物理回合**，其中21个按各自报告中的完整判据未通过；另有一次控制代码异常在生成完整报告前结束，日志仍保留。没有训练新模型或做全量参数扫描。最终交付只采用上表六个通过录像，失败尝试不重标为成功。

主要失败包括：移动右脚时支撑补偿切换不一致；脚落地后仍追逐目标导致拖脚失衡；恢复策略继续行走撞球；高空落地后的主动接近把球推走；提前翻平脚面引起碰撞或增加自旋。固定回撤比例在慢球产生反弹、在快球使球滚向腿；改为实测速度调节后修复。提高偏航反馈增益未改善接触法线；统一脚面角度修正改善快球却增加另一方向反弹，最终改为实测入射方向驱动的小范围补偿。所有失败报告、日志和源码快照保存在ignored的 `reports/v021_*` 与 `experiments/inside_v021/source_*`。

12项针对性CPU检查通过，覆盖实时高度/速度/方向响应、停止与远离来球、制动衔接、拒绝预摆脚和残余滚动、实际双支撑恢复的重力补偿与脚底边界。没有运行完整Tang资产测试套件。最终录像均有150帧、完整300控制步和1200物理步；对应仿真进程退出码均为0。

为避免重复运行已通过的地面案例，地面录像保留于高空改动之前，两侧高空录像保留于方向补偿最终扩展之前。因此六段录像的计划/运动源码指纹并非全部相同。保留原源码快照，并在最终源码下回放其测得的预测输入：三个地面场景共173次预测和两侧高空场景各31次预测，脚位姿、速度、阶段输出逐元素相同；运行器源码也与这些录像的指纹一致。后续变更仅在对应输入下未激活的分支或等价计算/遥测字段，未将旧录像冒称为重新运行。中央高空录像直接来自最终源码和配置。详见本地 `reports/v021_release_equivalence.json`。

这一证据只覆盖同一初始姿态、六条确定性来球、地面0.9–1.5m/s和高空水平3.6–4.3m/s、偏转±8°。它们参与过控制器标定，不是独立泛化测试集。没有证明任意球位、任意高度、随机扰动、视觉延迟、行走接球或连续颠球。余速门槛虽更低，仍可能产生缓慢位移；没有删除或改写这种物理运动。

## 复现与预览

```bash
/home/you/conda_env/env_isaaclab/bin/python tools/run_reactive_receive.py \
  --plan configs/reactive_receive_v1.json --scenario ground_center \
  --output reports/new_reactive_ground.json --video-dir media/new_reactive_ground_frames

/home/you/conda_env/env_isaaclab/bin/python tools/run_reactive_receive.py \
  --plan configs/reactive_receive_v1.json --scenario aerial_right_fast \
  --output reports/new_reactive_aerial.json --video-dir media/new_reactive_aerial_frames
```

依次运行，使用新输出路径保留证据。其余场景名见上表。若本地参考缺失，从配置的 `controller` 对象导出临时计划，再用原生成器重建：

```bash
/home/you/conda_env/env_isaaclab/bin/python - <<'PYCODE'
import json
from pathlib import Path
p=json.loads(Path('configs/reactive_receive_v1.json').read_text())
Path('/tmp/kinetickick_v021_reference.json').write_text(json.dumps(p['controller']))
PYCODE
/home/you/conda_env/env_isaaclab/bin/python tools/prepare_aerial_reference.py \
  --plan /tmp/kinetickick_v021_reference.json
```

生成器拒绝覆盖已有参考。脚的运动通过原电机与碰撞求解实现，参考只初始化自由机器人。球位置/速度只在开始时写入；六个最终报告均为 `root_fixed=false`、`post_initialization_state_writes=false`。原29DoF机器人、球质量/半径/材料、碰撞几何、关节/执行器限制和上游版本均保留。旧射门与接球入口源码没有修改。

预览 `media/reactive_receive_v0.1.21.mp4` 包含六个完整正常速度回合，以及地面/高空中心案例的0.25倍接触慢放。所有帧来自上表对应真实物理回合，缩放和字幕不改变动作。原始帧目录与指标见 `reports/reactive_v021_final_evidence.json`。

| 最终录像报告 | SHA-256 |
| --- | --- |
| `reports/v021_ground_center_preview.json` | `c5fc8d887d7c3de6a588eb1f9ef55c3caf732b9e3cb3db0848ab60e32e39cc73` |
| `reports/v021_ground_left_slow_preview.json` | `76a8628f4824050621cf9c517782a912b0a0c82e8e8c39a8ab6961de5945acea` |
| `reports/v021_ground_right_fast_preview.json` | `795123c3aebada74251213159ee8dc074e6c0e5575548b383b46c03330e1b821` |
| `reports/v021_aerial_center_published.json` | `52752e920080b067c3c50ec7a9946c8133c630a9a779d8af965d402a607c27ea` |
| `reports/v021_aerial_left_slow_release.json` | `c0cd1eab648289c24a43b7764d56b90c500184552a66bc234f039130d3add5c6` |
| `reports/v021_aerial_right_fast_final.json` | `9d8c2f050839fbb017e9e34825a12a149312f80b624660c9a7e71c8d709cadbd` |

| 发布文件 | SHA-256 |
| --- | --- |
| `configs/reactive_receive_v1.json` | `60098046156e5b9af422f492037d51f11f7ecd0eb93a816354462c9b1d1a4651` |
| `tools/run_reactive_receive.py` | `769d359c761003059d6d810098d1901425a6f86a4dd8881d4eedd95b45645c7c` |
| `tools/reactive_receive_motion.py` | `f87930dac055bf395506b0f05b4e0893243047fc84c35b58d1ae168ad4c88ed6` |
| `tools/reactive_receive_metrics.py` | `da65304518a7010560805a9c86203fbccade588d50d49213195a4b37f97e2bd8` |
| `tools/receiving_stance.py` | `2b4d434d049d7d4ece5799ae64857dccf3596a37cfae42999c49f681a9437170` |
| `experiments/inside_v021/common_stance.npz` | `d7b5b7be730ba841eab037c348a962abe0c4729d85140a28d7e502df43c8ee61` |
| `media/reactive_receive_v0.1.21.mp4` | `70946d3acaa3b703573ad6b08aa71b8e21fe330f8fd0378842626fa24e9fe813` |

原始报告、参考、模型、失败快照和媒体由.gitignore保留在Tang。GitHub保存代码、配置和本记录；新克隆仍依赖Tang-local资产与Isaac Lab安装，不是自包含运行包。

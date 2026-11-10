# v0.1.55 — KineticKick 200Hz 原 PhysX 批量物理复现与方法边界

源码基准 `3cac1736f72218cf7bb864059f3e0a21c08b7da6`；实验配置 `configs/physx_pair_dynamics_v055.json` 的 SHA 在本地进度记录里独立锁定。实测只在 Tang Ubuntu 的 `/home/you/projects/kinetickick` 跑，Mac 无需搬运 30MB 的原物理 JSON。

## 环境预检

```bash
cd /home/you/projects/kinetickick
git status --short && git rev-parse HEAD && git branch --show-current
df -h . && nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv
ls -l experiments/kinetickick_physics_gpu.lock
/home/you/conda_env/env_isaaclab/bin/python -c 'import rsl_rl, ezc3d, torch; print(torch.__version__)'
```

如磁盘可用空间 <20GiB、有其他运行中的GPU作业，**不要**启动新PhysX或重训练。原CMU数据、G1原权重/PhysX JSON与媒体全部属于本机本地文件，严格遵守 `.gitignore`。本批已完成的 15 条记录绝不能重复覆盖。

## SHA 冻结审计与原始批次

```bash
cd /home/you/projects/kinetickick
/home/you/conda_env/env_isaaclab/bin/python - <<'PY'
import json,sys
sys.path.insert(0,'tools')
from run_pair_dynamics_batch_v055 import freeze
cfg=json.load(open('configs/physx_pair_dynamics_v055.json'))
print('frozen protocol valid',freeze(cfg),len(cfg['cases']))
PY

# 此命令只在全量报告不存在、确需补跑时使用：
# /home/you/conda_env/env_isaaclab/bin/python -u tools/run_pair_dynamics_batch_v055.py
# 运行器在同一文件夹使用 flock，逐条检查旧结果 SHA，不覆盖历史。
```

单回合原 200Hz 时步=0.005s、控制20ms，原29DoF、左右脚地面过滤法向力、3D质心等随 `traces` 一起写入。若已存在 JSON，直接读原始源做 CPU 审计，不重跑 GPU。

```bash
# 已经有以下原审计文件，请直接查看，原生成器故意 FileExistsError 防止覆盖：
python3 - <<'PY'
import json
for x in ['v055_full_physics_dynamics_audit','v055_seed_replay_audit',
          'v055_nonfoot_net_force_pair_candidates']:
 d=json.load(open('reports/'+x+'.json'))
 print(x, 'loaded', len(d.get('results',[])))
PY

# 只在这些分析结果全都尚不存在且原 15 个物理 JSON hash 完整时：
# python3 tools/audit_pair_dynamics_v055.py
# python3 tools/audit_seed_replay_v055.py
# python3 tools/scan_nonfoot_force_pairs_v055.py
```

CPU 快速及全测试 **必须使用原已安装依赖的 IsaacLab conda 环境**，不要误用系统 `/usr/bin/python3`（系统缺少 `ezc3d` / `rsl_rl`）：

```bash
/home/you/conda_env/env_isaaclab/bin/python -m pytest -q tests/
# 本版实际结果: 676 passed, 69 subtests passed
```

## 控制接口的使用限制

`tools/wholebody_safety_interface_v055.py` 仅是CPU可执行的29DoF物理观测门控接口，**不是已训练或 PhysX 验证过的真实足球全身动作模型**。必须提供经过认证的 `G1_JOINT_ORDER` 和 `pair_evidence_available=True` 的**真实直接接触对观测**；本版拿不到，所以在实际机器人必须始终返回旧控制器原动作。仅有左腕-左髋相反净力/连杆原点距离不足以激活。

当前所有 `reports/v055_*.{json,log}` 与 `media/v055_*png` 为本机保留证据（忽略目录），**不提交GitHub**。GitHub上传的是冻结协议、执行器/测量脚本、CPU测试、失败归因和文字研究报告。

## 需要重新设计后才能推进的部分

1. 在**隔离非 G1 小场景**明确 Isaac Sim/IsaacLab 的安装版本、GPU/contact filter 支持，再做带接触对 prim 路径/接触法线/点位的 positive & negative 校验；不能直接再次在主生产G1场景上建原生自碰撞 view。
2. 原中心3次 pair-on/off 逐个5ms全字段一致后才考虑读到的contact对是真实非侵入测量。尚不成立时始终报 **unverified**。
3. 将不同 seed 下完全相同物理轨迹标记为 deterministic replay，不能作为独立分母。下一期预注册新球位/速度、摩擦扰动与人类受试者划分；不能把已观察本版holdout继续当未见测试集。
4. 人体素材可考虑 CMU 标记数据作**弱阶段代理**、广播视频作视觉参考，既无直接高频脚背接触标定也无人类GRF/受力真值，必须单独收集资格合格的多被试接触与力板资料才能开展“人类强力抽射”假设验证。

详见研究主报告 `reports/2026-11-11-v055-physx-dynamics-and-pair-evidence-boundary.md`。

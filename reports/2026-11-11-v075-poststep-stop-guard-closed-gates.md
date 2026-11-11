# v0.1.75｜真实 GPU STOP 关停保护对照 + 793项 CPU 回归

2026-11-11 Tang RTX4090 / Ubuntu 运行；采样时 CUDA compute PID 0、原 GPU 排他锁获独占、剩余磁盘约38.58GiB。

## 协议预冻结
- parent: 141046798584e7108dbab25733abba69c94a5f9e
- freeze commit: 501f29919214996a9cfdcd57c8e195aade3ef073
- frozen config SHA256 5f29b907198014fdbd77c4fcbd80b774db63c94ee2d53fb6828dda1dfe17aa00
- controller / IsaacLab core 未改；原 v54 三个日志每个 SHA 复核完好；所有数据链接指向既有源，仅用于CPU测试。

## 两个新增真实 GPU 诊断（不是G1）
| 实验 | 步数 | 关停路径 | exit | watchdog | 显式返回 | 严格判定 |
|--|--:|--|--:|--|--|--|
| guarded_close | 8 | 仅最后 step 后才设置STOP禁用位, app.close | 0 | 无，4.025s | 无APP_CLOSE_RETURN | UNKNOWN（冻结门禁拒绝） |
| guarded_stop | 8 | 最后 step 后设位, sim.stop→app.close | 0 | 无，3.774s | SIM_STOP_RETURN存在，APP_CLOSE_RETURN缺失 | UNKNOWN（冻结门禁拒绝） |

v74 无禁用位的两次8步诊断皆陷入 STOP 回调，约50s被watchdog截断；本轮 guard 改为约4s、正常进程exit0，强烈支持STOP callback是原问题，但缺少APP_CLOSE_RETURN；当前不能认证 Python 清洁关闭，更不能声称已修复G1生产管线。未修改v74证据，不改用户IsaacLab安装。

## 完整CPU测试及数据
首跑因新 worktree 缺少55条既有原始只读数据/URDF/NPZ软链接，得到45 failed,732 passed,87 skipped；仅在本新 worktree 恢复55个指向原文件的链接后重新测试，获得 **793 passed,71 skipped,51 subtests passed,0 failed**。初始失败为测试依赖挂载缺失，不是算法退化；原文件未改、旧证据未覆盖。

**新增原G1射门=0、严格射门成功率不适用；原生左腕-左髋collider pair未认证。** 原 v54 3+3/原 v67 9的历史结果仍独立，不算本批新测。所有合法射门必须满足脚背冲量≥95%、全部原29关节qdot、球门与速度、非脚碰撞、14s恢复，进球不能替代门禁。

## 下一依赖
下一个单独冻结版本应以 atexit、SystemExit 跟踪重新区分 SimulationApp.close()内部发出正常进程退出（无返回）和真正清洁的Python返回。由独立协议决定是否可接受无 APP_CLOSE_RETURN 但exit0的退出语义；不得事后更改本版冻结门禁。只有证明G1启动/退出安全，才能锁互斥跑v54原三回合off/on原生只读contact pair，按200Hz逐字段严格parity与法向/点/冲量判断碰撞。人体跨被试性能仍失败，正脚背/GRF真值仍不可得，不宣称已学会优秀球员式全身抽射。

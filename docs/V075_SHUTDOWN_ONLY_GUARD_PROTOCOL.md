# v0.1.75｜仅在物理结束后阻断 IsaacLab STOP 自旋回调

冻结提交：501f29919214996a9cfdcd57c8e195aade3ef073；configs/v075_frozen_shutdown_native_pair.json SHA256 5f29b907198014fdbd77c4fcbd80b774db63c94ee2d53fb6828dda1dfe17aa00。
源自 v74 已有的真实 GPU 8-step Isaac 诊断，停机时 simulation_context.py 的 STOP 回调在 timeline 非 playing 时执行无界 while: render 循环。

本实验不得更改 USD、机器人、控制器、物理步长、模型、原传感器或 IsaacLab 源文件。只允许在最后物理步完成之后设置 sim._disable_app_control_on_stop_handle=True，分别观察直接 close 与先 stop 再 close 两条路径。单试验 85s watchdog、SIGUSR1 可诊断栈、独立进程组、GPU 原锁、磁盘和 CUDA 互斥、原始 v54 逐 SHA 校验。所有输出采用排他创建，结果不许覆盖。tools/v075_guarded_batch.py 严格要求进程状态为 0、8 步完整、APP_CLOSE_RETURN 与 PYTHON_MAIN_RETURN 两个明确标记，否则只能判为未通过，不能把 exit0 偷换为可证明的 Python 控制流返回。

这两个试验仅为空仿真器真实 Isaac GPU step，绝不是原 29DoF G1 踢球。若清理标记不齐，保留 fail-closed，不准执行新的 G1 物理，也不能解释为两次踢球失败。

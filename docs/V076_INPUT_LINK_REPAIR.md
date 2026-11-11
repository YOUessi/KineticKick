# v76 输入依赖补齐审计
首个G1启动尝试因新 worktree 缺少源冻结数据而在0.06s内失败，未进入 Isaac/PhysX、未生成G1轨迹。错误为 experiments/running_v013/stride_reference.npz 的 SHA 校验失败（实际文件缺失）。检查三个冻结源协议仅发现3个工作树本地缺失项：stride_reference.npz、run_stop last_model.pt、paid_official policy_30000.onnx；每个已有原源文件SHA与冻结配置完全一致。因此仅在本 v76 工作树建立3个原文件**软链接**（不复制/覆盖/改写权重），原错误日志及审计文件完整保留。第二个attempt使用新的唯一 _r1 后缀继续完全相同的种子/控制器/球位与先验硬门禁。新增代码只改变输出路径命名，不改变控制器。

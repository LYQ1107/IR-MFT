# 六组原生融合均完成九轮，完整Dev正在运行

六组原模型已全部完成原九轮训练并实际退出0：quality × RGB／gray／IR、temporal × RGB／gray／IR。每组九遍全部27251个Train帧，共245259逻辑帧次、61326次更新；九轮交付地址完全匹配，同模式初始化相同。旧六轮、八轮及原NAS恢复的额外物理重放继续保留，不将重放计作额外逻辑epoch或随机种子。

最后结束的是temporal_ir。原进程退场后，独立CPU只读诊断实际退出0：最终459657个FP32参数、46个张量均有限，最终快照与稳定恢复状态相同；46/46张量有历史非零梯度的AdamW二阶矩证据，更新计数61326，实际保存的学习率与衰减和temporal RGB一致。该证据不代表红外收益、持续有效梯度或收敛。原八轮公开CPU证明、历史及快照SHA保持不变，与新第九轮证明合并，确认该IR组九份原生状态均有限且逐轮不同；没有重新读取活跃优化器或复制权重。

| 原九轮组 | 完整轮数 | 逻辑帧次 | 更新数 | 原始实际退出 |
|---|---:|---:|---:|---:|
| quality_rgb | 9 | 245259 | 61326 | 0 |
| quality_gray | 9 | 245259 | 61326 | 0 |
| quality_ir | 9 | 245259 | 61326 | 0 |
| temporal_rgb | 9 | 245259 | 61326 | 0 |
| temporal_gray | 9 | 245259 | 61326 | 0 |
| temporal_ir | 9 | 245259 | 61326 | 0 |

原控制器自行进入dev_normal_09_NN：GPU4评估完整SN-013，GPU5评估完整BT-005，各31条生成流，包含原始缺失／错位干预及新鲜RGB参考；随后仍由原流程完成新增30项官方Dev评估，连同此前62项累计92项。观察时两个原始进程的PID／启动tick／完整argv／CUDA及NVML均匹配。其余七个后续控制器身份核对通过，仍CPU等待、尚未开始正式实验。旧九轮训练只读观察器26625因目标达到而正常退出0，无需续接；完整Dev自动CPU导出等待器继续运行。原整个研究控制器仍未退出。

这是训练完成与完整Dev启动证明，九轮完整HOTA和原输入字节摘要还没有出齐。最近完整指标仍是前六轮Dev：IR时序50.115210，对直接RGB47.579162为+2.536048，对同结构RGB48.097061为+2.018149，对跨结构最强RGB49.918478仅+0.196732。上述Dev数不能与Test基线29.529跨划分相减。此前正式主方案Test29.834177也不是本批原生结果。

后续是否继续十二轮完全沿用原规则：任一正常组截至九轮的Dev最佳比截至六轮提高>0.2点，则六组共同延长至十二轮；保持原十二轮学习率时间轴，不加随机种子或重置。最终选模、触发的同预算模块去除、全部七段15985帧Test及逐方法成本仍待原流程完成。

独立创新计数仍0，Test至少+5点及2–3项独立贡献目标未达，Goal保持active。全部正负结果、强RGB／灰度对照、历史Test暴露、RGB预训练见Dev、RGB派生伪IR等限制继续保留。

[机器报告](PERCEPTUAL_MFT_20261126_NATIVE_ALL_SIX_NINE_EPOCHS.json) · [九轮交付与IR状态核对](perceptual_20261126_native_all_six_nine_epochs/actual_all6_delivery_and_IR_states.json) · [真实CPU诊断](perceptual_20261126_native_all_six_nine_epochs/actual_IR_CPU_learning_diagnostic.json) · [CPU实际执行](perceptual_20261126_native_all_six_nine_epochs/actual_IR_CPU_learning_execution.json) · [原Dev两卡及后续队列身份](perceptual_20261126_native_all_six_nine_epochs/actual_live_Dev_and_future7_queues.json) · [原观察器实际退出0](perceptual_20261126_native_all_six_nine_epochs/actual_nine_epoch_observer_exit0.json) · [原件SHA清单](perceptual_20261126_native_all_six_nine_epochs/publication_manifest.json) · [此前完整六轮Dev](PERCEPTUAL_MFT_20261126_NATIVE_MINIMUM6_FULL_DEV_ANALYSIS_ZH.md)

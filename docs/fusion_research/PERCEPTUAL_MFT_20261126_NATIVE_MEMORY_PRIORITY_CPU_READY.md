# 固定伪红外主模型：历史外观匹配的实际CPU检查与后续队列

当前目标保持原始无红外YOLOX-X＋ByteTrack完整七段Test HOTA29.528964，最低34.528964、进一步39.528964。新增工作不训练RGB／灰度结构，不追加种子，也不改变固定11月DiffV2IR PNG或已按Dev选出的NP_temporal_ir_e06。当前七段Test继续优先占用4/5卡。

已有Dev逐目标分析显示身份分配是明显瓶颈。后续方向复用当前检测／融合分支的P3目标区域向量，在线保存每条轨迹的归一化指数移动平均外观，用固定余弦距离惩罚辅助ByteTrack主关联阶段。它不训练ReID模型，不修改检测框、原检测阈值或其他关联阶段；没有磁盘外观库，也不读取未来或重写历史轨迹。余弦匹配、EMA和门控均为已有算法，不因组合或CPU通过就计为创新。

只比较同一个冻结伪红外模型的三个固定外观权重0／0.05／0.1，复用此前F4组件。质量策略为uniform，保留score与当前候选重叠形成的可靠性代理，不把已有全局注意力当成已校准可靠性。接受阈值0.2、EMA系数0.2、IoU下限0.1、衰减25帧固定，无额外阈值搜索。三个权重须完成两段4834帧Dev，按完整精度HOTA／AssA／IDF1及较小权重选择。只有相对同视觉零权重的Dev HOTA严格提高，才冻结并评估新的完整七段Test；否则保留负结果，沿用原主方案既有Test，不重复零权重神经推理。

实际CPU检查已退出0：四个不同Train帧，历史记忆引擎调用5次、独立原视觉参考3次。对BT-003前三帧，三种权重的当前检测包均与原视觉参考逐元素一致，零权重的字面MOT逐字节一致；切换BT-001再返回BT-003第一帧，所有检测和MOT恢复相同。观察到28次非空单位向量记忆状态，包含同一轨迹在不同帧和权重下的重复观察，不是28个独立目标。两个非零条件分别在2次主代价调用中改变8条边。改变代价不代表改变了正确匹配或提高HOTA。

源检测器与融合权重前后未变。CPU缩小输入320×576，实际子进程计时28.220秒、包装到退出32.960秒，峰值RSS1.550GiB；这些是工程检查成本，不是方法端到端FPS。正式800×1440的GPU检查尚未运行，不能由CPU结果代替。

新流程已锁定430项源码／配置，真实CPU控制器PID2442817、启动ticks97382911和原始命令摘要均核实，CUDA_VISIBLE_DEVICES为空、阶段为空、GPU子进程0。实际调用正式CUDA入口时，因当前原始baseline Test仍活跃而正确拒绝，CUDA未初始化；这项故意拒绝检查本身退出0。必须等当前Test控制器和所有worker实际退出、完整结果写出后，才在4/5卡各做12帧真实Train GPU检查，然后开始上述完整Dev。不会让第三个GPU任务进入。

这是用户最新“先比较原始baseline、暂缓额外RGB结构和消融”的简化研究流程。原先同时要求native和spatial两家族的广范围F4协议及七个旧后续队列保持原件与未完成状态，没有伪造空间家族完成文件、绕过其原控制器或宣称原F4完成。新入口只用已完成九轮Dev并早已冻结的native主模型，其来源、零权重轨迹和完整图像字节摘要须匹配既有原件。完整Test值不能进入三个权重的Dev选择。

截至2026-11-26T15:30:24.100006+00:00没有新增Dev／Test准确率，没有证明记忆收益或+5目标。当前Test仍运行；本后续只在CPU等待。工作目录59.989GiB／64GiB，独立空闲2641.310GiB，高于40GiB保留值；权重、图像、MOT及特征缓存不推送。历史Test暴露、RGB预训练见过Dev、伪IR源于RGB、单seed0及双Dev视频限制保持。

[实际CPU结果与来源](PERCEPTUAL_MFT_20261126_NATIVE_MEMORY_PRIORITY_CPU_READY.json)、[CPU原件](perceptual_20261126_native_memory_priority_CPU_ready/CPU_engineering_r1.json)、[实际CUDA拒绝检查](perceptual_20261126_native_memory_priority_CPU_ready/active_predecessor_CUDA_refusal_CPU_proof_r1.json)、[锁定登记](perceptual_20261126_native_memory_priority_CPU_ready/pipeline_registered.json)、[固定科学配置](../../configs/perceptual_native_memory_priority_20261126.json)、[完整执行代码](../../tools/run_mft_native_memory_priority.py)。原记录中父工具退出自字段null保持不改，实际工具0由相邻外部收据及本报告所列工具回复证明。

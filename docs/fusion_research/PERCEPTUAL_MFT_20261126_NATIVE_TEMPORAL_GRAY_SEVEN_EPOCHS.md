# 时序灰度完成七遍完整 Train，原九轮预算继续执行

原 temporal_gray 已保存第 7 个完整 Train 遍历，并继续第 8 轮。每轮完整交付六个 Train 视频的 27251 帧、6814 个片段，累计记录 190757 逻辑帧次、47698 次优化器更新，没有重复补齐帧。原 GPU 4 worker 仍在运行，尚未完成共同九轮预算；实际原 worker 退出码仍未知，不能把 CPU 审计退出 0 当作原训练阶段完成。

既有只读 CPU 前缀审计实际退出 0。七轮实际训练地址与共同日程一致，七份不可变模块状态均有限、参数摘要两两不同；前六轮快照、模块参数摘要及训练历史与已发布的灰度六轮审计逐项一致。seed0 初始化与时序 RGB／IR 相同，冻结 RGB 和输入内容约束保持。参数有限或发生变化不构成模态收益或收敛证据。

本组辅助输入是灰度，审计明确记录 `IR_formal_Train_used=false`。它是匹配的灰度强对照，不提供 IR 训练或 IR 贡献证据。原固定新版伪 IR 的时序 IR 组仍在 GPU 5 继续原第 8 轮。

CPU child 与实际工具均退出 0，child 用时 6.79 秒，内部检查 1.01804 秒，峰值 RSS 0.57026 GiB。检查只读取原交付记录和不可变 epoch 快照，未读取活跃 optimizer latest，未停止或重启原 worker，没有新训练、前向、反向、优化器更新或 GPU 工作，也未新增随机种子。这是 CPU 审计成本，不是方法端到端速度。

| 原六组正常条件 | 已保存完整 Train 轮次 | 当前状态 |
| --- | ---: | --- |
| quality_rgb | 9 | 原九轮 worker 实际退出 0 |
| quality_gray | 9 | 原九轮 worker 实际退出 0 |
| quality_ir | 9 | 原九轮 worker 实际退出 0 |
| temporal_rgb | 9 | 原九轮 worker 实际退出 0 |
| temporal_gray | 7 | 原 GPU 4 进程继续第 8 轮 |
| temporal_ir | 7 | 原 GPU 5 进程继续第 8 轮 |

六组现在均完成至少七遍完整 Train，共同九轮阶段仍完成 4／6。两卡进程、原控制器及 GPU UUID 已重新核验，102 个原注册科学源文件和后续 TaskUtility 的 427 个源文件未变；后续七个已注册控制器仍为 CPU 等待、GPU 阶段为空。只保存小型文本证明，未新增图像、特征或权重副本。

**本节点没有新的 Dev／Test 指标。** 最近完整 Dev 仍为最低六轮的 62 条件；累计 92 条件的九轮 Dev 等待六组共同预算完成，再按原规则决定是否全部继续到 12 轮。正式选择、触发的同预算模块去除／组合、完整七段 15985 帧 Test 与独立逐方法成本仍须完成。

固定 Test 基线为直接 RGB 微调 YOLOX-X＋ByteTrack 的 29.529，目标 34.529 及 2–3 项可独立归因的多模态贡献保持。独立创新计数仍为 0。历史 Test 暴露、RGB 预训练见 Dev、RGB 派生伪 IR、单 seed0 等限制继续保留。

[完整来源证明](PERCEPTUAL_MFT_20261126_NATIVE_TEMPORAL_GRAY_SEVEN_EPOCHS.json) · [原 CPU 前缀审计](perceptual_20261126_native_temporal_gray_seven_epochs/actual_CPU_prefix_audit.json) · [实际执行收据](perceptual_20261126_native_temporal_gray_seven_epochs/actual_CPU_execution.json) · [原七轮交付记录](perceptual_20261126_native_temporal_gray_seven_epochs/original_temporal_gray_progress7.json) · [既有灰度六轮审计](PERCEPTUAL_MFT_20261126_NATIVE_TEMPORAL_GRAY_COMPLETE_TRAIN.json) · [当前六组状态](PERCEPTUAL_MFT_20261126_NATIVE_EXTENDED_TRAIN_STATUS.json)

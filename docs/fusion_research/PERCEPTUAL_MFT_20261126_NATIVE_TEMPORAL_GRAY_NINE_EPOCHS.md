# 时序灰度控制：原九轮训练完成并实际退出 0

原 temporal_gray 第九轮已完整保存，原 worker PID2005053 已退出并由原控制器记录退出码 0。该组是同预算灰度强对照，不是 IR 训练；完整研究、完整九轮 Dev 和创新贡献仍未完成。

| 项目 | 实际结果 |
| --- | --- |
| 完整 Train 遍历 | 9 × 27,251 帧 |
| 逻辑帧次 / 优化更新 | 245,259 / 61,326 |
| 第九轮文件 SHA256 | `b991dfb14c0ca05043a0c74ec399683cf5738527efa10e3689721c0459358447` |
| 第九轮实际模块参数 SHA256 | `864bfe2d0497539186e992c5c9182adae9aeda77c45babb7c4c5409c4d8e1dd9` |
| 初始化 | `e0ea1c8f8646a784bfc74f0076da0afb5fc0d65c1e68243df543b3ffc7ddfc8a` |
| FP32 原生参数 / 参数张量 | 459,657 / 46 |
| 已存 AdamW 更新次数 | 61,326 |
| 已存 LR / weight decay | 5.2118310001287937e-05 / 0.0001 |

全部九轮每轮 6,814 个 clip、27,251 帧、无重复补齐，实际交付地址与已完成的 quality RGB/灰度/IR 及 temporal RGB 对照一致。旧前八轮历史、元数据和快照文件摘要与已公开八轮证明完全一致。旧八轮独立状态审计与本次最终第九轮诊断共同确认九份原生状态有限，参数摘要逐轮不同；没有把仅仅文件摘要不同当作参数变化。

仅在原训练实际退出 0 且进程消失后，CPU 才读取稳定的 latest 优化器和不可变第九轮快照。最终状态与可续跑状态一致，已存 LR 与 weight decay 精确等于已完成的时序 RGB 对照；46/46 张量的 AdamW 二阶矩表明有过非零梯度。历史梯度、参数变化和有限值均不证明有效灰度/IR 信息、HOTA 提升或收敛。活动中的时序 IR 优化器未读取。

CPU 实际会话 9170 与子进程均退出 0；子进程墙钟 5.114028 秒，内部 0.192085 秒，峰值 RSS 0.579765 GiB。没有新的前向、反向、优化或 GPU 工作，也没有复制模型权重。

六组完整轮次当前为 **9 / 9 / 9 / 9 / 9 / 8**；五组原九轮任务实际完成，剩余 temporal_ir 在原 GPU5 进程继续第九轮。原 GPU4 灰度进程已退场；后七个登记控制器均仍是 CPU 等待、阶段为空。原只读训练观察器和完整九轮 Dev CPU 导出等待器的实际 PID/启动时刻/命令摘要也核对存活，不能把旧 PID 消失视为需要重启训练。

共同九轮 Dev 需等全部六组完成，再累计评估 92 个条件；按预先规则判断是否所有组继续十二轮，随后冻结模型、执行条件触发的去除消融和完整七段 Test。当前最近完整结果仍是六轮 Dev，未新增准确率、独立贡献或 Test 达标证据。

原 native102 项和最终 TaskUtility427 项文件重算摘要均未变。工作目录约 59.809402 GiB / 64 GiB，独立可用约 2662.851994 GiB，高于 40 GiB 保留值；源数据、断点与未来缓存依赖保持。

[机器可读报告](PERCEPTUAL_MFT_20261126_NATIVE_TEMPORAL_GRAY_NINE_EPOCHS.json) · [原完整九轮进度](perceptual_20261126_native_temporal_gray_nine_epochs/original_temporal_gray_progress9.json) · [原 worker 日志](perceptual_20261126_native_temporal_gray_nine_epochs/original_temporal_gray_worker_log9.txt) · [实际 CPU 诊断](perceptual_20261126_native_temporal_gray_nine_epochs/actual_CPU_learning_diagnostic.json) · [交付及旧八轮核验](perceptual_20261126_native_temporal_gray_nine_epochs/actual_delivery_and_snapshot_verification.json) · [原进程与后续队列](perceptual_20261126_native_temporal_gray_nine_epochs/actual_one_GPU_and_future_CPU_queue.json) · [完整 Dev 门控问题诊断](PERCEPTUAL_MFT_20261126_NATIVE_DEV_GATE_DIAGNOSIS.md)

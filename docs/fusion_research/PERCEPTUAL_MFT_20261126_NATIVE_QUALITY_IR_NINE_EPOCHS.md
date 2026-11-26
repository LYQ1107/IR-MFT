# 质量 IR 完成九遍完整 Train：共同九轮阶段已完成 4／6 组

原 quality_ir worker 已完成 9 次完整 Train 遍历，原控制器记录实际退出码 0，旧进程已退场。质量模式的 RGB、灰度、IR 三组现在均完成相同九轮预算；时序 RGB 也已完成九轮。原控制器仍在执行其余两组，整个研究尚未结束。

本组每轮完整交付六个 Train 视频的 27251 帧、6814 个片段，九轮共 245259 逻辑帧次、61326 次优化器更新。各轮视频长度、实际地址摘要和累计更新均核对通过，与已完成的质量 RGB／灰度对照全部相同，三组 seed0 初始化相同。短片段保留真实长度，没有重复补齐帧。旧六轮快照与原公开 Dev 配置字节一致，第 7、8 轮快照及交付历史也与此前保存证明一致。

九轮延续同一条预先固定的 12 轮学习率时间轴，未重置优化器或学习率。已存学习率为 5.211831000128794e-5，weight decay 为 0.0001，与匹配 RGB／灰度相同。三组最终模块参数摘要不同，但参数或文件摘要不同不构成模态收益证据，也不证明全部中间检查点参数两两不同。

原质量 IR 进程退出后，独立 CPU 检查核验了最终九轮快照和稳定的 optimizer latest：459171 个参数均有限、为 FP32，33 个参数张量的已存优化器 step 均为 61326。33／33 张量的 AdamW 二阶矩支持其历史上收到过非零梯度，区分于单纯权重衰减；这不证明梯度持续足够大、IR 信息有用、预测发生有益变化或已经收敛。质量 IR 的第 7、8 轮参数有限性没有在本次检查中独立补查。

CPU child 和实际工具均退出 0。该检查 child 用时 7.53 秒，内部诊断用时 0.23052 秒，峰值 RSS 0.58021 GiB；没有新前向、反向、优化器更新或 GPU 调用，没有图像／特征／权重复制。为重建原初始化而使用 seed0 是只读诊断，不是新增随机种子的训练重复。上述成本不是方法的端到端速度。

| 共同九轮阶段 | 已保存完整轮次 | 原阶段状态 |
| --- | ---: | --- |
| quality_rgb | 9 | 实际退出 0 |
| quality_gray | 9 | 实际退出 0 |
| quality_ir | 9 | 实际退出 0 |
| temporal_rgb | 9 | 实际退出 0 |
| temporal_gray | 6 | 原 GPU 4 进程继续训练 |
| temporal_ir | 7 | 原 GPU 5 进程继续训练 |

GPU 4 已由原队列接入 temporal_gray，GPU 5 继续 temporal_ir。两组是原六轮状态的九轮预算续训，没有增加种子或修改队列。原训练控制器及两个实际 GPU 进程的 PID、启动 tick、argv、CUDA 环境和 GPU UUID 已重新核对；原质量 IR worker 退出后才读取其优化器。原 102 个科学源文件、后续 TaskUtility 的 427 个源文件保持不变，后续七个已注册控制器均核验仍为 CPU 等待、GPU 阶段为空。

原只读观察器在六小时窗口结束后实际退出 0，原训练持续运行；新只读观察器继承旧 worker 身份信息并继续观察。该观察续接没有重启原模型或改变科学设定，相关完成收据、源代码快照和新句柄均保留。

随后时序 IR 也完成第 7 个完整 Train 遍历，共记录 190757 逻辑帧次、47698 次更新，并继续原第 8 轮。另一次既有 CPU 前缀审计实际退出 0：七轮训练地址均与控制日程一致，七份不可变模块状态均有限、参数摘要两两不同，前六轮快照、模块状态摘要及训练历史与已发布审计逐项一致，初始化与时序 RGB／灰度相同。该审计只读取不可变 epoch 快照，没有读取活跃 optimizer latest，没有停止原 worker；它不把时序 IR 的共同九轮阶段标记完成。CPU child 用时 11.05 秒，内部检查 1.05245 秒，峰值 RSS 0.57340 GiB，仍非方法端到端成本。

**本节点没有任何新的 Dev 或 Test 准确率。** 完整九轮 Dev 尚未开始，仍须等六组共同预算完成，再执行累计 92 条件的完整 Dev 和原冻结的 9／12 轮决定。最近完整 Dev 仍是最低六轮的 62 条件；已有时序 IR 相对跨结构最强 RGB 仅 +0.196732 HOTA，不能把九轮训练完成解释成已经扩大优势。

正式选模、触发的同预算模块去除／组合实验、全部正常和干预条件的七段 15985 帧 Test、独立逐方法成本仍未完成。固定 Test 基线 29.529、目标 34.529 与 2–3 项可独立归因的贡献要求保持；独立创新计数仍为 0。历史 Test 暴露、RGB 预训练见 Dev、RGB 派生伪 IR、单 seed0 等限制继续保留。

[完整九轮来源证明](PERCEPTUAL_MFT_20261126_NATIVE_QUALITY_IR_NINE_EPOCHS.json) · [原交付／快照核验](perceptual_20261126_native_quality_ir_nine_epochs/actual_nine_epoch_delivery_and_snapshot_verification.json) · [CPU 参数／优化器诊断](perceptual_20261126_native_quality_ir_nine_epochs/actual_CPU_learning_diagnostic.json) · [实际 CPU 执行收据](perceptual_20261126_native_quality_ir_nine_epochs/actual_CPU_learning_execution.json) · [时序 IR 七轮前缀审计](perceptual_20261126_native_quality_ir_nine_epochs/actual_temporal_ir_epoch7_prefix_audit.json) · [当前六组状态](PERCEPTUAL_MFT_20261126_NATIVE_EXTENDED_TRAIN_STATUS.json) · [最近完整 Dev](PERCEPTUAL_MFT_20261126_NATIVE_MINIMUM6_FULL_DEV.json)

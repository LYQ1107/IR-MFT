# 时序灰度与新版伪 IR：两个完整八轮前缀

原时序灰度与时序新版伪红外均已保存第 8 个完整 Train 遍历。每组累计 218,008 逻辑帧次、54,512 次更新；独立 CPU 快照检查及实际子进程均退出 0。两组原 GPU 进程继续第 9 轮，没有停止或重启。

| 条件 | 完整 Train 轮次 | 当前物理 GPU | 第 8 轮文件 SHA256 | 第 8 轮实际模块参数 SHA256 |
| --- | ---: | ---: | --- | --- |
| temporal_gray | 8 | 4 | `161c88c1d6a53377351b2c9ec9a8464e98a9a97cdbb192811e08b92d2052e3c3` | `ca64644c0ba314bd715e2bed0fb7026924b433b9a4bc12bf57e6f8da487ba1b4` |
| temporal_ir | 8 | 5 | `c14923de1741044436087ce4060dde522d0a6ef7a07da8afcbcf24b6d8bf8bf9` | `12bd814e42af148b5c5a11a2ef4c20c4f632264cc3f2a0a6d47f922302f5ec32` |

每组的八份原生参数状态逐一在 CPU 读取，均有限且其参数摘要各自不同；旧第 1–7 轮快照及历史与已公开证明完全一致。八轮训练地址与已完成的时序 RGB 控制一致，均为每轮 27,251 帧、6,814 个无重复补齐的 clip；三种时序输入共享原初始化 `e0ea1c8f8646a784bfc74f0076da0afb5fc0d65c1e68243df543b3ffc7ddfc8a`。原 RGB 冻结和逐次读图摘要约束由实际执行源与原参考清单绑定；没有事后虚构逐图消费列表。

灰度条件只使用 RGB/灰度，不把它计作伪 IR 训练或模态收益。两组都只检查不可变的完成轮次权重和原历史，没有读取活动中的 latest 优化器，也没有新增神经网络前向、反向、优化或 GPU 工作。

实际 CPU 子进程墙钟 6.943453 秒，审计内部 1.643638 秒，峰值 RSS 0.572369 GiB。实际进程身份、退出码和日志均随附。

六组当前完整轮次依 quality_rgb / quality_gray / quality_ir / temporal_rgb / temporal_gray / temporal_ir 顺序为 **9 / 9 / 9 / 9 / 8 / 8**。原九轮阶段仍只有 4/6 组实际完成，完整九轮 Dev 尚未启动；最近的完整 Dev 仍是六轮的 62 个条件。不能从训练计数、有限参数或损失曲线推断 HOTA、收敛或独立贡献。

原控制器与两个 GPU 进程的 PID、启动时刻、命令摘要、CUDA 环境和 GPU UUID 均直接核对。七个后续控制器仍为 CPU 等待且阶段为空；native 的 102 项冻结文件与最终 TaskUtility 的 427 项冻结文件均重新核对无变化。工作目录约 59.808454 GiB / 64 GiB，独立空闲约 2666.551594 GiB，高于 40 GiB 保留值；只复制小型文本证明，没有权重或图像副本。

最近完整 Dev：时序 IR HOTA 50.115210，直接 RGB 47.579162，相差 +2.536048；对跨两种结构最强 RGB 49.918478 只有 +0.196732。现有质量 IR 还低于自身 RGB 控制，时序变化也未显示 IR 独有增益。新版原生 Test、触发的去除实验及最终成本仍需完成；独立创新计数保持 0，Test +5 点目标未达。历史 Test 暴露、RGB 预训练见 Dev 与 RGB 派生伪 IR 的限制保持。

[机器可读报告](PERCEPTUAL_MFT_20261126_NATIVE_TEMPORAL_PAIR_EIGHT_EPOCHS.json) · [原 CPU 审计](perceptual_20261126_native_temporal_pair_eight_epochs/actual_CPU_prefix_audit.json) · [实际执行](perceptual_20261126_native_temporal_pair_eight_epochs/actual_CPU_execution.json) · [当前两卡与后续队列](perceptual_20261126_native_temporal_pair_eight_epochs/actual_two_GPU_and_future_CPU_queue.json) · [最近完整 Dev](PERCEPTUAL_MFT_20261126_NATIVE_MINIMUM6_FULL_DEV_ANALYSIS_ZH.md) · [生成成本分项](PERCEPTUAL_MFT_20261126_NOVEMBER_AUXILIARY_COST.md)

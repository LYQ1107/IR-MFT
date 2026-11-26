# 当前任务：优先达到原始RGB baseline +5 HOTA

用户最新要求：先不用扩展RGB结构对照，优先让新版伪红外方法超过原始、不使用红外的跟踪baseline至少5点HOTA；进一步争取10点。该指令优先于此前较广的消融与对照计划。原始YOLOX-X＋ByteTrack完整七段15985帧Test HOTA为29.5289636096556，对应目标34.5289636096556／39.5289636096556。不会把增强后的RGB结构改成新主基线。

两项正在训练的扩展去除条件已按用户指令停止，实际SIGTERM退出均为-15；原控制器实际退出-15、包装工具143，不是实验失败。另四项延后。保存停止前完整原state、已有进度与全部正常九轮结果。七个旧后继队列仍受原完成条件约束，不能因此提前获得GPU。研究Goal继续active。

主方案仍为用户改范围前已冻结的NP_temporal_ir_e06，原正常选模SHA为`fd3b82e44e499f6e8d4488a5155a2ef55da75b4b4f15e6573307e1c2255857d9`。新版DiffV2IR图像、YOLOX-X、ByteTrack、seed0及ProtocolV2保持。新的Test freeze在任何本轮Test生产／评估前写入，只测主方案、新鲜B1 RGB和规范原始RGB，不再追加不同RGB结构或去除模型的Test。

发布观察时点：2026-11-26T15:11:00.745004+00:00，原始运行阶段`test_original_baseline_priority_20261126_NN`。仅物理GPU4/5，最多两张。尚无本轮Test精度，不能预先宣称+5或+10。原Test此前已被历史研究暴露，本轮严格不拿Test改检查点、阈值或结构。伪IR是由RGB生成的固定辅助表达，不是独立物理热测量。

现有Dev：原始直接RGB47.579162，主方案50.115210，相差+2.536048点。该差值不是Test提升。[完整九轮Dev与负结果](../PERCEPTUAL_MFT_20261126_NATIVE_STAGE9_FULL_DEV_ANALYSIS_ZH.md)保持公开。后续若当前主方案未达目标，按Train/Dev证据继续研究，不能通过换用Test最好的模型来凑目标。

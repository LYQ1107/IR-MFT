# 当前优先目标：原始RGB baseline至少+5 HOTA

用户最新要求暂缓扩展RGB结构比较与模块消融，先完成新版伪红外方法对原始无红外跟踪baseline的提升。固定比较对象仍是直接RGB微调YOLOX-X＋ByteTrack：完整七段15985帧Test HOTA29.528964，目标至少34.528964，进一步争取39.528964。此前增强RGB结构的结果保留，不替换主基线。

原六组正常模型各完成九遍27251帧Train、245259逻辑帧次、61326次更新。九轮完整Dev已出齐：两段完整视频4834帧、92项条件、97个Dev相关worker加6个续训worker均实际退出0。全部旧62项按case ID精确相同；追加第7–9轮没有刷新任何一组最佳，原规则不扩到12轮。这不是收敛证明，也不能单独认定IR无效。

| 完整Dev条件 | 原Dev最佳检查点 | HOTA |
| --- | --- | ---: |
| 原始直接RGB | NP_source_canonical_RGB | 47.579162 |
| 新鲜B1 RGB | NP_source_fresh_RGB | 47.181043 |
| 当前帧RGB辅助分支 | NP_quality_rgb_e02 | 49.918478 |
| 当前帧灰度辅助分支 | NP_quality_gray_e05 | 49.413533 |
| 当前帧IR辅助分支 | NP_quality_ir_e06 | 49.703387 |
| 时序RGB辅助分支 | NP_temporal_rgb_e05 | 48.097061 |
| 时序灰度辅助分支 | NP_temporal_gray_e03 | 48.941466 |
| 时序IR主方案 | NP_temporal_ir_e06 | 50.115210 |

时序IR对原始直接RGB的Dev提升为+2.536048点，尚未达到5点，不能与Test29.529跨划分相减。对同结构RGB为+2.018149、灰度+1.173744。已完成的跨结构RGB比较仅+0.196732，质量IR低于质量RGB0.215090，两段视频方向不全一致，负结果不删除。这些附加比较不再是当前优先验收标准。所有结构共享基础YOLOX-X与ByteTrack，quality与temporal是附加分支不同，而非另换跟踪方法。

[九轮官方47字段与原件](PERCEPTUAL_MFT_20261126_NATIVE_STAGE9_FULL_DEV.json)、[全量CSV](PERCEPTUAL_MFT_20261126_NATIVE_STAGE9_FULL_DEV.csv)、[完整中文分析](PERCEPTUAL_MFT_20261126_NATIVE_STAGE9_FULL_DEV_ANALYSIS_ZH.md)、[三张完整曲线](perceptual_20261126_native_stage9_dev_figures/README.md)均公开。全部18项缺失IR在47字段、逐视频、完整CSV与MOT精确回到新鲜B1；正确IR相对同权重错位IR在主检查点高2.241000点，支持当前组合使用对齐辅助输入，不替代独立模块证明。

[九轮门控诊断](PERCEPTUAL_MFT_20261126_NATIVE_STAGE9_DEV_GATE_DIAGNOSIS.md)保留540条层级记录、108配对与36个模型／视频比较。错位后平均门控更高76/108，时序49/54，不能当作已校准可靠性贡献。受保护特征单元记录未改变，不代表检测或身份均不变。没有新增训练、随机种子或Test准确率。

已按用户指令终止原扩展队列及两项活动消融，真实SIGTERM退出-15，另四项延后；训练原件和快照保留。原来源102项与TaskUtility427项仍逐字节不变。研究Goal继续active，旧后继CPU队列不能提前释放GPU。[最新范围、终止原件与七段Test冻结](perceptual_20261126_original_baseline_priority/README.md)记录全部改变。

新Test仅评估用户改范围前已选定的NP_temporal_ir_e06及原始／新鲜RGB基线，使用物理GPU4/5且最多两卡，seed0。观察时点2026-11-26T15:11:00.745004+00:00阶段`test_original_baseline_priority_20261126_NN`，当前无本轮Test精度，不宣称+5或+10已达到。不会用Test重新选择模型或阈值。当前独立贡献计数0是已有证据状态，最新优先目标是相对原始baseline的整体提升。

[固定11月生成因果小检查](PERCEPTUAL_MFT_20261126_NOVEMBER_GENERATOR_CAUSALITY.md)保留8次实际Train调用、5个不同RGB帧的IR／SAM缓存逐像素一致、未来后缀改变不影响所测前缀和重置检查。它不覆盖全部缓存或Test，不增加准确率与创新计数。[完整Test历史生成组件成本](PERCEPTUAL_MFT_20261126_NOVEMBER_AUXILIARY_COST.md)选定SAM加DiffV2IR区域均值4.102096秒／帧，不是完整生成加跟踪FPS。新版伪IR固定由RGB生成，并非独立热传感器信息；历史Test暴露及RGB预训练见Dev限制保持。

此前正式主方案Test29.834177，对原RGB29.528964约+0.305214点；它不是本批原生结果。旧的[六轮结果](PERCEPTUAL_MFT_20261126_NATIVE_MINIMUM6_FULL_DEV.json)、[共同九轮训练原件](PERCEPTUAL_MFT_20261126_NATIVE_ALL_SIX_NINE_EPOCHS.md)、[五种公开后端完整结果](PERCEPTUAL_MFT_20261126_ADDITIONAL_PUBLIC_TRACKERS_COMPLETE_ANALYSIS_ZH.md)、[逐身份误差分析](PERCEPTUAL_MFT_20261126_NATIVE_DEV_TRACK_ERRORS.md)继续保留。原NAS恢复中21次额外更新、84帧重放不改写为逻辑轮数或新种子。

最新磁盘工作目录59.983GiB／64GiB，独立空闲2641.638GiB，高于40GiB保留值。只推送小型代码、配置、JSON、CSV、文本及SVG，权重、图片、PNG预览、MOT与特征缓存不推送。[机器状态](PERCEPTUAL_MFT_20261126_NATIVE_EXTENDED_TRAIN_STATUS.json)记录本次时点。

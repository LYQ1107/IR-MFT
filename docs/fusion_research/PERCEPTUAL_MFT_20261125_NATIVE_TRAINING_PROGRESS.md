# 原生融合：六组九轮训练完成，完整Dev正在评估

六组同预算原模型已全部完成九遍27251帧Train，各245259逻辑帧次、61326次更新，原worker均实际退出0。九轮交付地址一致，同模式RGB／灰度／IR初始化匹配。原控制器已自行进入完整九轮Dev：GPU4处理SN-013，GPU5处理BT-005；各31条生成流，随后原流程新增30项官方Dev评估，累计92项。九轮完整Dev指标和原图输入字节核验尚未出齐，整个研究仍在运行。

[共同九轮训练与原件](PERCEPTUAL_MFT_20261126_NATIVE_ALL_SIX_NINE_EPOCHS.md)记录最后temporal_ir的真实CPU检查与原两卡Dev身份。原进程退场后读取稳定优化器，确认459657个参数有限，46/46张量有历史非零梯度证据；原八轮快照和历史不变，加新第九轮证明共九份有限且逐轮不同状态。该诊断不证明红外收益或收敛。训练只读观察器26625已因目标达到正常退出0；完整Dev自动CPU导出继续等待。其余七个后续队列均保持CPU等待，不提前启动额外GPU任务。

最新完整准确率仍来自前六轮的两段完整Dev视频4834帧、62项条件。65个原生产／规范源／官方评估worker、独立CPU审计与自动导出均实际退出0。

| 前六轮完整Dev条件 | 检查点 | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 规范直接RGB | NP_source_canonical_RGB | 47.579 | 68.864 | 32.908 | 55.865 | 90.200 | 94 |
| 新鲜B1 RGB | NP_source_fresh_RGB | 47.181 | 68.818 | 32.384 | 55.342 | 90.127 | 96 |
| quality_rgb | NP_quality_rgb_e02 | 49.918 | 67.931 | 36.715 | 58.576 | 89.884 | 93 |
| quality_gray | NP_quality_gray_e05 | 49.414 | 69.074 | 35.378 | 57.678 | 90.198 | 92 |
| quality_ir | NP_quality_ir_e06 | 49.703 | 68.231 | 36.238 | 57.982 | 90.125 | 88 |
| temporal_rgb | NP_temporal_rgb_e05 | 48.097 | 69.004 | 33.558 | 56.545 | 90.166 | 94 |
| temporal_gray | NP_temporal_gray_e03 | 48.941 | 68.910 | 34.791 | 57.044 | 90.261 | 89 |
| temporal_ir | NP_temporal_ir_e06 | 50.115 | 68.248 | 36.840 | 59.668 | 90.011 | 96 |

时序IR对规范直接RGB为+2.536048点，对同结构RGB为+2.018149，对同结构灰度为+1.173744；对跨结构最强RGB只高+0.196732。质量IR仍低于质量RGB约0.215090点。全部为Dev，不能与Test29.529跨划分相减。IR时序收益主要体现在关联，DetA／MOTA／IDSW并非全部改善；两个视频相对最强RGB方向相反，第6轮时序结构对IR与RGB的增量各约+0.412／+0.413，尚不能归因为IR特有贡献。

正确IR相对同检查点真实图像roll，质量／时序分别+1.691156／+2.241000点；全部12项缺失条件在47字段、完整CSV和MOT均精确回到新鲜B1 RGB。B1与规范直接RGB不同，两者均保留。[完整指标与原件](PERCEPTUAL_MFT_20261126_NATIVE_MINIMUM6_FULL_DEV.json)、[全部47字段](PERCEPTUAL_MFT_20261126_NATIVE_MINIMUM6_FULL_DEV.csv)、[中文分析](PERCEPTUAL_MFT_20261126_NATIVE_MINIMUM6_FULL_DEV_ANALYSIS_ZH.md)、[三张对照图](perceptual_20261126_native_minimum6_dev_figures/README.md)均已发布。

[逐身份误差分析](PERCEPTUAL_MFT_20261126_NATIVE_DEV_TRACK_ERRORS.md)保留294条目标记录和9702次逐阈值官方核验，显示收益集中与BT-005身份分配退化。[门控响应诊断](PERCEPTUAL_MFT_20261126_NATIVE_DEV_GATE_DIAGNOSIS.md)覆盖60条件、360条层级记录与72项配对；时序36项中31项错位后平均门控更高，不能把动态全局门控当成已验证可靠性创新。两项均没有新增训练、选模或Test结果。

原Dev规则触发六组共同6→9，保持单一12轮学习率时间轴，不加种子、不重置。九轮完整Dev结束后，仍按原规则判断是否六组共同9→12：任一正常组截至9轮Dev最佳比截至6轮提高>0.2点。当前不宣称收敛。[完整Dev独立导出与自动等待](PERCEPTUAL_MFT_20261126_NATIVE_EXTENDED_DEV_AUDIT_READY.md)继续适用。

[固定11月生成检查](PERCEPTUAL_MFT_20261126_NOVEMBER_GENERATOR_CAUSALITY.md)新增真实GPU8次Train调用：改变未来不改变所测两帧前缀，切换序列再返回相同，5个不同RGB帧对应的IR／SAM缓存均逐像素一致；权重逐字节校验，7.524GiB匿名RAM缓存已释放。该结果不覆盖全缓存或完整Test、不增加跟踪准确率或创新计数。[原完整Test生成组件成本](PERCEPTUAL_MFT_20261126_NOVEMBER_AUXILIARY_COST.md)平均SAM加DiffV2IR选定区域4.102096秒／帧，不是端到端FPS；完整生成加MFT成本仍需测量。[各家族选模范围](PERCEPTUAL_MFT_20261126_SELECTION_SCOPES.md)保持原冻结顺序，无新增全局选模规则。

此前各阶段原件继续保留：[首对RGB／灰度九轮](PERCEPTUAL_MFT_20261126_NATIVE_FIRST_PAIR_NINE_EPOCHS.md)、[时序RGB九轮](PERCEPTUAL_MFT_20261126_NATIVE_TEMPORAL_RGB_NINE_EPOCHS.md)、[质量IR九轮](PERCEPTUAL_MFT_20261126_NATIVE_QUALITY_IR_NINE_EPOCHS.md)、[时序灰度九轮](PERCEPTUAL_MFT_20261126_NATIVE_TEMPORAL_GRAY_NINE_EPOCHS.md)、[灰度／IR八轮前缀](PERCEPTUAL_MFT_20261126_NATIVE_TEMPORAL_PAIR_EIGHT_EPOCHS.md)、[原IR六轮完成](PERCEPTUAL_MFT_20261126_NATIVE_TEMPORAL_IR_COMPLETE_TRAIN.md)、[完整BT-005输入依赖](PERCEPTUAL_MFT_20261126_NATIVE_BT005_COMPLETE_PRODUCER.md)、[五种公开后端完整对比](PERCEPTUAL_MFT_20261126_ADDITIONAL_PUBLIC_TRACKERS_COMPLETE_ANALYSIS_ZH.md)。原NAS失败及恢复日志不删除：丢失保存点后21次更新、重放84帧，额外物理交付不计逻辑轮数或新种子。

独立创新计数仍0。本家族尚未完成最终预算、选模、触发的模块去除、全部七段15985帧Test及逐方法运行成本。此前正式主方案Test29.834177，对原RGB29.529约+0.305177；它不是本批原生结果。Test至少34.529及2–3项独立贡献尚未达到，Goal保持active。历史Test暴露、RGB预训练见Dev、伪IR由RGB生成等限制保持。

最近磁盘核对工作目录59.812GiB／64GiB，独立空闲2655.860GiB，高于40GiB保留值。仅推送小型代码、配置、文本、CSV与已有SVG，生成预览与权重留在工作区。[机器状态及原件](PERCEPTUAL_MFT_20261126_NATIVE_EXTENDED_TRAIN_STATUS.json)记录本次观察时点。

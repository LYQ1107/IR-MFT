# 原生融合：完整六轮Dev已出，六组统一延长到九轮

最低六轮的两个完整Dev视频4834帧、62项条件已完成；65个原生产／规范源／官方评估worker均实际退出0。独立CPU审计、自动导出入口和完整Dev只读观察器也均实际退出0；三张图已生成并检查。原整个研究控制器仍在运行，不能把阶段完成当成整个任务完成。

时序IR前六轮的Dev最佳为50.115210，对同模式RGB48.097061为+2.018149，对同模式灰度48.941466为+1.173744；对规范直接RGB47.579162为+2.536048。跨两种结构的最强RGB49.918478仍然接近，优势只有+0.196732。质量IR49.703387还低于对应RGB49.918478约0.215090点。

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

上表全部为Dev，不与Test基线29.529跨划分相减。时序IR在BT-005仍低于同模式灰度；相对跨结构最强RGB，两段方向反转。第6轮中时序结构对IR与RGB的增量各约+0.412／+0.413，不能把它当IR独有的时序贡献。所有曲线、负结果与同轮对照保留。

正确IR相对同检查点真实图像roll，质量／时序模式分别+1.691156／+2.241000点；时序IR两段各自也为正。全部12个缺失条件在两个视频的全部47字段、逐序列、完整官方CSV及MOT均精确等于新鲜B1 RGB。B1与B4数值并不等价，两种RGB参考均继续保留。

质量灰度、质量IR、时序IR的后3轮最佳值分别比前3轮高0.947、1.642、2.707点，触发原固定>0.2点规则。六组共同从6继续到9个完整Train遍历，沿单一12轮学习率时间轴，不新增种子或重置学习率。当前4／5号卡分别继续temporal_gray／temporal_ir，quality_ir已完成原九轮预算；当前不宣称收敛，后续9／12决定仍由完整Dev规则给出。

六组已完成最低六遍27251帧，每组保留模型逻辑交付163506帧次、40884次优化器更新；36份模块状态有限且逐轮不同，同一模式三种输入初始化匹配，各轮地址摘要一致。新增九轮阶段并未把旧六轮检查点替换。IR的旧NAS失败及恢复原件仍保留：保存点后21次更新丢失并重放84帧，额外物理交付不计作逻辑epoch或新种子。原RGB冻结、原图SHA和源代码约束保持。

完整[阶段指标与原件](PERCEPTUAL_MFT_20261126_NATIVE_MINIMUM6_FULL_DEV.json)、[中文科研分析](PERCEPTUAL_MFT_20261126_NATIVE_MINIMUM6_FULL_DEV_ANALYSIS_ZH.md)、[47字段总表](PERCEPTUAL_MFT_20261126_NATIVE_MINIMUM6_FULL_DEV.csv)、[三张完整对照图](perceptual_20261126_native_minimum6_dev_figures/README.md)均已发布。原[IR六轮证明](PERCEPTUAL_MFT_20261126_NATIVE_TEMPORAL_IR_COMPLETE_TRAIN.md)、[BT-005整段输入依赖核对](PERCEPTUAL_MFT_20261126_NATIVE_BT005_COMPLETE_PRODUCER.md)、[五个公开后端完整比较](PERCEPTUAL_MFT_20261126_ADDITIONAL_PUBLIC_TRACKERS_COMPLETE_ANALYSIS_ZH.md)保留。

旧完整Dev观察器14350和自动CPU导出77049已正常完成，不再当作活跃训练句柄。九轮只读观察器86394按六小时窗口实际退出0后，已续接为26625，继承原worker身份信息并核验原控制器与两卡进程的真实PID／启动tick／argv／CUDA环境；原模型训练未重启。日志中的批次仅是周期记录，progress.json还可能停留在上一个完整epoch；二者均不能冒充共同九轮已完成。

独立创新计数仍0。新原生家族尚未完成最终预算、选模、触发的模块去除、完整七段Test与独立逐方法成本。既有正式主方案Test仍29.834177，对原RGB29.529约+0.305177；这不是本批原生结果。Test目标34.529及2–3项独立贡献未达，Goal继续active。历史Test暴露、RGB预训练见Dev和RGB派生伪IR限制保持。
本次发布核对：工作目录约59.802GiB／64GiB，独立空闲约2680.410GiB，高于40GiB保留值。只发布小型文本、SVG和CSV；PNG预览留在工作目录，不上传权重、图像数据集、FPN或原始轨迹。

后续九／十二轮完整Dev的[独立导出检查与自动等待](PERCEPTUAL_MFT_20261126_NATIVE_EXTENDED_DEV_AUDIT_READY.md)已接好；CPU工程检查实际退出0，九轮实际结果仍待原实验完成。

已核对：[首批质量模式RGB／灰度完成九轮且实际退出0](PERCEPTUAL_MFT_20261126_NATIVE_FIRST_PAIR_NINE_EPOCHS.md)、[时序RGB完成九轮且实际退出0](PERCEPTUAL_MFT_20261126_NATIVE_TEMPORAL_RGB_NINE_EPOCHS.md)，以及新完成的[质量IR九轮及CPU检查](PERCEPTUAL_MFT_20261126_NATIVE_QUALITY_IR_NINE_EPOCHS.md)。每组245259逻辑帧次、61326次更新，CPU参数／优化器诊断实际退出0；质量RGB／灰度／IR三组已完成相同预算，九轮阶段当前4／6完成。

最新完整轮次核对：quality_ir已完成原第9轮并实际退出0，九轮训练地址与质量RGB／灰度对照全部一致，旧六轮及此前第7、8轮历史／快照保持字节一致。原进程退场后读取其稳定优化器，确认最终459171个参数有限、已存学习率与匹配控制相同；历史非零梯度不作为IR收益或收敛证据。随后temporal_ir完成第7轮，190757逻辑帧次、47698次更新；独立CPU前缀审计核验七份模块状态有限且不同、旧六轮历史保持不变，未读取活跃优化器，并继续原第8轮。

当前六组完整轮次按quality_rgb／quality_gray／quality_ir／temporal_rgb／temporal_gray／temporal_ir顺序为9／9／9／9／6／7；[当前状态与原件](PERCEPTUAL_MFT_20261126_NATIVE_EXTENDED_TRAIN_STATUS.json)。4号卡已按原队列接入temporal_gray，5号卡继续temporal_ir。完整九轮Dev尚未开始，上述六轮指标仍是最近完整Dev结果。原历史NAS恢复中的额外物理重放继续单独保留，不计成新的逻辑epoch。

新增[完整Dev逐目标身份误差诊断](PERCEPTUAL_MFT_20261126_NATIVE_DEV_TRACK_ERRORS.md)：14个完整条件、294条目标记录及9702次逐阈值官方CSV核验均完成；发现SN-013收益集中于少数身份、BT-005存在明显身份分配退化。全部正负记录与复现代码已发布。该诊断没有新训练、选模或Test读取，不增加准确率或独立创新计数。

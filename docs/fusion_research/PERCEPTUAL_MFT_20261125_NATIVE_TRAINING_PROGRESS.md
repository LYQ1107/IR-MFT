# 原生融合：六组最低六轮完成，完整Dev进行中

六组正常条件quality／temporal×RGB／灰度／IR均已完成六遍27251帧，六个最终worker均实际退出0。每组保留模型累计163506逻辑帧次、40884次实际更新；36份模块状态有限、逐轮不同且偏离初始化，六组每轮地址摘要相同、没有尾批padding，原RGB冻结日志逐轮通过。两个模式各自三种输入的初始模块相同。

| 完整Train轮次 | quality_rgb | quality_gray | quality_ir | temporal_rgb | temporal_gray | temporal_ir |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.57960204 | 2.57888883 | 2.57926598 | 2.58042841 | 2.58023237 | 2.58052280 |
| 2 | 2.57919934 | 2.57904583 | 2.57899623 | 2.57961410 | 2.57923564 | 2.58045331 |
| 3 | 2.57954909 | 2.57884048 | 2.57922564 | 2.58110718 | 2.57895533 | 2.58128197 |
| 4 | 2.57744853 | 2.57810849 | 2.57777712 | 2.57879639 | 2.57704012 | 2.57918316 |
| 5 | 2.57830708 | 2.57856514 | 2.57829463 | 2.57967240 | 2.57889570 | 2.58158252 |
| 6 | 2.57770683 | 2.57693930 | 2.57791129 | 2.57765829 | 2.57691833 | 2.57843980 |

最后一组IR经历一次NAS输入错误：旧worker实际退出1，保存点后21次更新丢失；源版本固定的恢复入口在原图像SHA重新匹配后，从原优化器／时序记忆／RNG状态接续。恢复worker最终完成六轮并退出0。重放21批次／84帧记作基础设施额外开销，不计为额外逻辑epoch或新的随机种子。所有原失败与恢复证据保持。

完整Dev正在进行。BT-005已完成整段1834帧、61路输出，原5号卡工作进程实际退出0；独立CPU核对61份MOT及全部12组缺失／图像错位配对通过，未计算部分Dev精度。SN-013仍由原4号卡进程继续。冻结规格有61路输出：36个正常检查点、12个IR空间错位、12个辅助缺失、新鲜B1 RGB；规范RGB缓存对照另行构建。全部条件仍需完整4834帧、完整官方指标及缺失分支全CSV／MOT一致性核对；完整Dev预算决策尚未结束。

| 正常条件 | 当前最低训练状态 | 完整Train轮数 | 独立模块核对 |
| --- | --- | ---: | --- |
| quality_rgb | 最终worker完成并退出0 | 6 | 全部六轮 |
| quality_gray | 最终worker完成并退出0 | 6 | 全部六轮 |
| quality_ir | 最终worker完成并退出0 | 6 | 全部六轮 |
| temporal_rgb | 最终worker完成并退出0 | 6 | 全部六轮 |
| temporal_gray | 最终worker完成并退出0 | 6 | 全部六轮 |
| temporal_ir | 最终worker完成并退出0 | 6 | 全部六轮 |

新增[IR六轮原件](PERCEPTUAL_MFT_20261126_NATIVE_TEMPORAL_IR_COMPLETE_TRAIN.json)、[实际CPU执行](PERCEPTUAL_MFT_20261126_NATIVE_TEMPORAL_IR_COMPLETE_TRAIN_EXECUTION.json)、[完整核对](PERCEPTUAL_MFT_20261126_NATIVE_TEMPORAL_IR_COMPLETE_TRAIN_VERIFICATION.json)、[科研分析](PERCEPTUAL_MFT_20261126_NATIVE_TEMPORAL_IR_COMPLETE_TRAIN.md)。旧[时序灰度六轮](PERCEPTUAL_MFT_20261126_NATIVE_TEMPORAL_GRAY_COMPLETE_TRAIN.json)、[质量IR六轮](PERCEPTUAL_MFT_20261126_NATIVE_QUALITY_IR_COMPLETE_TRAIN.json)、[时序RGB六轮](PERCEPTUAL_MFT_20261125_NATIVE_TEMPORAL_RGB_COMPLETE_TRAIN.json)、[两个质量对照](PERCEPTUAL_MFT_20261125_NATIVE_TRAINING_PROGRESS.json)保持原件。

最低六轮不等于收敛证明。后续是否六组统一延长到九／十二轮，只由原冻结完整Dev曲线规则决定；所有正式正常条件即使较差仍须完整七段Test。只有正常IR完整Dev超过直接RGB及同模式RGB／灰度，才触发同预算额外机制去除拟合。no_RGB_protection同时去掉教师置信特征mask与可靠RGB保持loss，属于联合机制消融。

输入仍为固定11月伪IR及原RGB；原读图SHA断言绑定原成功内容清单。原生训练没有逐图消费列表，不把参考清单当成保存的消费明细。原102份源码及独立恢复源码保持哈希，原恢复体直接执行冻结的Dev／Test控制体。历史[梯度诊断](PERCEPTUAL_MFT_20261126_NATIVE_PARAMETER_LEARNING_DIAGNOSTIC.md)与[固定8帧CPU舍入诊断](PERCEPTUAL_MFT_20261126_NATIVE_TRAIN_FEATURE_ROUNDING_DIAGNOSTIC.md)保持原四组范围，不扩写成六组精度证据。

五公开后端完整40组Dev／Test对照及47字段见[分析](PERCEPTUAL_MFT_20261126_ADDITIONAL_PUBLIC_TRACKERS_COMPLETE_ANALYSIS_ZH.md)。NAS[原故障与入口](PERCEPTUAL_MFT_20261126_NATIVE_NAS_IO_RECOVERY.md)及[实际恢复核对](PERCEPTUAL_MFT_20261126_NATIVE_NAS_IO_RESUMED.md)均保留。

最新发布核对：工作目录约59.576GiB／64GiB，独立空闲约2683.444GiB，高于40GiB保留值。Git仅发布文本，不包含图像、FPN或checkpoint。

固定主方案仍29.834177 HOTA，对原RGB29.529约+0.305177；本批原生方案尚无完整Dev／Test精度。S不稳定、R有负贡献、C只有条件消融支持，独立创新计数仍0。历史Test已暴露、原RGB预训练见Dev、伪IR由RGB派生的限制继续披露。目标34.529及2–3项独立贡献未达，所需完整评估、消融与成本继续，Goal保持active。

新增[BT-005完整输出与实际IR输入依赖核对](PERCEPTUAL_MFT_20261126_NATIVE_BT005_COMPLETE_PRODUCER.md)：原完整生产清单、逐层门控、真实MOT及排除分数后的输出列比较均保留；这是单视频工程证据，不能当成完整Dev／Test收益或创新证明。

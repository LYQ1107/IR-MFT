# 原生融合正式训练：五组六轮完成，时序IR继续训练

quality_rgb／quality_gray／quality_ir／temporal_rgb／temporal_gray各完成六轮全部27251帧，五个原worker均实际退出0。每组163506帧次、40884次实际更新；每轮6814批次，无padding重复。30份模块状态有限、各自逐轮不同且偏离初始化，五组每轮实际交付地址摘要一致，原RGB冻结日志逐轮通过核对。

| 完整Train轮次 | quality_rgb loss | quality_gray loss | quality_ir loss | temporal_rgb loss | temporal_gray loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.57960204 | 2.57888883 | 2.57926598 | 2.58042841 | 2.58023237 |
| 2 | 2.57919934 | 2.57904583 | 2.57899623 | 2.57961410 | 2.57923564 |
| 3 | 2.57954909 | 2.57884048 | 2.57922564 | 2.58110718 | 2.57895533 |
| 4 | 2.57744853 | 2.57810849 | 2.57777712 | 2.57879639 | 2.57704012 |
| 5 | 2.57830708 | 2.57856514 | 2.57829463 | 2.57967240 | 2.57889570 |
| 6 | 2.57770683 | 2.57693930 | 2.57791129 | 2.57765829 | 2.57691833 |

当前仅剩temporal_ir在原GPU5进程继续训练：已完成四遍，进入第五遍。原temporal_gray已完成第六遍并退出0；新只读观察器接续跟踪原IR进程。六组完整Train共同预算结束前，完整Dev尚未启动；完整Dev曲线决定是否统一延长九／十二轮，正式正常条件即使较差仍须完整七段Test。

| 正常条件 | 当前状态 | 完整Train轮数 | 独立模块核对 |
| --- | --- | ---: | --- |
| quality_rgb | 原worker完成并退出0 | 6 | 全部六轮 |
| quality_gray | 原worker完成并退出0 | 6 | 全部六轮 |
| quality_ir | 原worker完成并退出0 | 6 | 全部六轮 |
| temporal_rgb | 原worker完成并退出0 | 6 | 全部六轮 |
| temporal_gray | 原worker完成并退出0 | 6 | 全部六轮 |
| temporal_ir | GPU5原进程第五遍正式训练 | 4 | 尚无完整六轮资格 |

时序灰度新增[六轮原件核对](PERCEPTUAL_MFT_20261126_NATIVE_TEMPORAL_GRAY_COMPLETE_TRAIN.json)、[执行与成本](PERCEPTUAL_MFT_20261126_NATIVE_TEMPORAL_GRAY_COMPLETE_TRAIN_EXECUTION.json)、[原始日志](PERCEPTUAL_MFT_20261126_NATIVE_TEMPORAL_GRAY_COMPLETE_TRAIN_CPU_LOG.txt)、[完成及发布核对](PERCEPTUAL_MFT_20261126_NATIVE_TEMPORAL_GRAY_COMPLETE_TRAIN_VERIFICATION.json)、[科研分析](PERCEPTUAL_MFT_20261126_NATIVE_TEMPORAL_GRAY_COMPLETE_TRAIN.md)。首次审计因依赖路径漏项在导入阶段退出1，第二次恢复原环境路径后实际退出0；原训练与源码均保持，失败原件完整保留。

旧[两个质量对照](PERCEPTUAL_MFT_20261125_NATIVE_TRAINING_PROGRESS.json)、[质量IR六轮](PERCEPTUAL_MFT_20261126_NATIVE_QUALITY_IR_COMPLETE_TRAIN.json)、[时序RGB六轮](PERCEPTUAL_MFT_20261125_NATIVE_TEMPORAL_RGB_COMPLETE_TRAIN.json)、[首轮IR／时序](PERCEPTUAL_MFT_20261125_NATIVE_MODALITY_FIRST_EPOCH.json)原件保持。

完整Train loss变化很小且非单调，不能代替收敛或HOTA结论。正常原生方案保留(1−p)×1[p<0.6]教师置信特征mask及可靠RGB保持loss；已登记no_RGB_protection同时取消mask与保持loss，是联合机制消融，仅当正常IR完整Dev超过直接RGB和同模式RGB／灰度时触发。当前配方、种子与选择规则未更改。

实际IR仍为固定11月DiffV2IR PNG；原读图SHA断言绑定已成功IR参考清单，IR预期54502个RGB＋IR条目，RGB／灰度预期RGB集合27251项。原生训练未另存逐图消费列表；编码字节RAM容量8GiB不是整个worker RSS上限。

已完成四组的[历史梯度诊断](PERCEPTUAL_MFT_20261126_NATIVE_PARAMETER_LEARNING_DIAGNOSTIC.md)与[固定8帧CPU特征舍入诊断](PERCEPTUAL_MFT_20261126_NATIVE_TRAIN_FEATURE_ROUNDING_DIAGNOSTIC.md)保持原范围，不扩写为五组证明，也不作为精度或模态贡献。

并行公开后端补测已完成BoT-SORT／HybridSORT的8项Dev及8项Test，16个worker与原CPU控制器均实际退出0。五后端40组官方47字段及原CSV见[完整分析](PERCEPTUAL_MFT_20261126_ADDITIONAL_PUBLIC_TRACKERS_COMPLETE_ANALYSIS_ZH.md)。BoT的Test IR−RGB约−0.924点；Hybrid约+0.759点却低于灰度约0.311点，未据Test改选原主方案。

最新发布核对：工作目录约59.444GiB／64GiB，独立空闲约2689.519GiB，保留40GiB规则通过。只发布文本，不保存图像、FPN或权重副本。

固定主方案仍29.834177 HOTA，对原RGB29.529约+0.305177；本轮原生家族尚无完整Dev／Test精度。S不稳定、R有负贡献、C只有条件消融支持，独立创新计数仍0。历史Test已暴露、原RGB预训练见Dev及伪IR由RGB派生的限制继续披露。目标34.529及2–3项独立贡献尚未达；所需训练、完整评估、消融与成本继续，Goal保持active。

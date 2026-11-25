# 原生质量IR六轮完整训练完成

quality_ir已在原GPU4进程完成六轮全部27251帧，累计163506帧次、40884次实际更新；原worker1089904实际退出0。原队列随即在GPU4启动temporal_gray，至少32次真实更新已确认；GPU5原temporal_ir继续第二轮。当前四个正常条件完成六轮，全部六组及完整Dev尚未完成。

| 完整Train轮次 | quality_rgb loss | quality_gray loss | quality_ir loss | temporal_rgb loss |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2.57960204 | 2.57888883 | 2.57926598 | 2.58042841 |
| 2 | 2.57919934 | 2.57904583 | 2.57899623 | 2.57961410 |
| 3 | 2.57954909 | 2.57884048 | 2.57922564 | 2.58110718 |
| 4 | 2.57744853 | 2.57810849 | 2.57777712 | 2.57879639 |
| 5 | 2.57830708 | 2.57856514 | 2.57829463 | 2.57967240 |
| 6 | 2.57770683 | 2.57693930 | 2.57791129 | 2.57765829 |

独立CPU调用实际退出0，耗时约5.360秒、观测峰值RSS0.579GiB。六份IR模块权重均有限、每轮状态不同且均已偏离初始化；原RGB冻结日志及每轮全帧地址摘要通过核对。与此前三组一起，24份模块快照通过检查，四组每轮交付顺序一致，无padding重复；质量模式的RGB／灰度／IR初始参数相同。旧16份证明、原第一轮IR快照及进度原件保持字节不变。首次导出在文件写完后因最后一行print参数错误退出1；随后逐字节独立核对现有产物通过，没有重跑训练或审计，也未改动原结果。

六轮IR总loss变化很小且非单调，末轮2.577911；同预算quality_rgb末轮2.577707。此处不据训练loss认定收敛、模态增益或HOTA提升。共同最低六轮及完整Dev预算规则保持，是否延长到九／十二轮只由完整Dev决定。

真实IR输入仍为固定11月DiffV2IR PNG，实际读图SHA断言绑定原四份成功IR参考清单、54502个RGB＋IR条目；原生训练没有另存完整逐图消费清单，未补造或用参考清单冒充。编码字节RAM容量8GiB不等于整个worker RSS上限。

[六轮原件核对](PERCEPTUAL_MFT_20261126_NATIVE_QUALITY_IR_COMPLETE_TRAIN.json)、[实际CPU执行](PERCEPTUAL_MFT_20261126_NATIVE_QUALITY_IR_COMPLETE_TRAIN_EXECUTION.json)、[CPU原始日志](PERCEPTUAL_MFT_20261126_NATIVE_QUALITY_IR_COMPLETE_TRAIN_CPU_LOG.txt)、[交接及发布核对](PERCEPTUAL_MFT_20261126_NATIVE_QUALITY_IR_COMPLETE_TRAIN_VERIFICATION.json)、[完整六轮progress原件](perceptual_20261125_native_training_progress/quality_ir_six_epoch_original_progress.json)。

现有固定主方法仍为29.834177 HOTA，对原RGB29.529为+0.305177；没有本轮原生方案的新Dev／Test准确率。目标34.529及2–3项独立贡献未达，创新计数仍0，Goal保持active。物理4／5卡、seed0及存储约束保持，Git仅保存文本证明。

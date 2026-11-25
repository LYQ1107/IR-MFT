# 原生时序灰度：六轮完整训练及第五组核对完成

原GPU4的temporal_gray已完成六轮全部27251帧，累计163506帧次、40884次实际更新。原worker1330025在原控制器中记录实际退出0，原进程已结束；只读观察器52730也实际退出0。完整journal交付计数包含日志最后6784批次之后的30次更新，没有把最后一条周期日志误当作总量。

| 完整Train轮次 | quality_rgb loss | quality_gray loss | quality_ir loss | temporal_rgb loss | temporal_gray loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.57960204 | 2.57888883 | 2.57926598 | 2.58042841 | 2.58023237 |
| 2 | 2.57919934 | 2.57904583 | 2.57899623 | 2.57961410 | 2.57923564 |
| 3 | 2.57954909 | 2.57884048 | 2.57922564 | 2.58110718 | 2.57895533 |
| 4 | 2.57744853 | 2.57810849 | 2.57777712 | 2.57879639 | 2.57704012 |
| 5 | 2.57830708 | 2.57856514 | 2.57829463 | 2.57967240 | 2.57889570 |
| 6 | 2.57770683 | 2.57693930 | 2.57791129 | 2.57765829 | 2.57691833 |

独立CPU审计实际退出0，整个调用约6.240秒，观测峰值RSS约0.579GiB。新增六份模块快照均有限、逐轮不同且偏离初始化；连同此前四组，共30个独立模块状态通过核对。五组每轮实际交付地址摘要一致，6814批次、无padding重复；时序RGB／灰度的初始模块相同。原RGB冻结日志逐轮通过检查。

首次CPU审计因独立环境漏掉HybridSORT中的vendored motmetrics而在导入阶段退出1；原源码、日志及执行记录完整保留。第二次只恢复原native worker的PYTHONPATH，未修改源码、快照或训练；实际退出0。第二次首个cmdline快照为空，随后取得非空且精确匹配的原始argv，两个原始观察均保留。

灰度末轮完整Train loss为2.57691833，与其他同预算条件接近；此处不据训练loss宣称收敛、IR增益或HOTA改善。这是灰度强对照的训练完成证据，尚非新方法精度或完整研究完成。

当前五组正常条件完成六轮；GPU5原temporal_ir已完成四遍、继续第五遍。新只读观察器84844接续观察原PID1248117，原观察器已结束，未启动重复训练或发送训练信号。六组共同完成前尚无本家族完整Dev／Test；是否统一延长到九／十二轮仍由冻结的完整Dev曲线规则决定。

固定11月DiffV2IR、Protocol V2及seed0保持。灰度条件只读原RGB并按登记转换；原生训练没有逐图消费列表，不用参考清单冒充已保存的消费明细。审计只读已有小型模块和记录，不执行NN前向或优化器更新，旧24份模块证明保持。

发布核对时工作目录约59.444GiB／64GiB，独立空闲约2689.519GiB，高于40GiB保留值。Git仅发布文本证明，不包含权重、图像或FPN副本。

原Dev选定主方案完整Test仍29.834177 HOTA，相对固定RGB29.529约+0.305177；目标34.529及2–3项独立贡献尚未达，创新计数仍0。历史Test暴露、RGB预训练见Dev及RGB派生伪IR限制保持。Goal继续active。

[原始CPU审计](PERCEPTUAL_MFT_20261126_NATIVE_TEMPORAL_GRAY_COMPLETE_TRAIN.json) · [实际执行](PERCEPTUAL_MFT_20261126_NATIVE_TEMPORAL_GRAY_COMPLETE_TRAIN_EXECUTION.json) · [原始日志](PERCEPTUAL_MFT_20261126_NATIVE_TEMPORAL_GRAY_COMPLETE_TRAIN_CPU_LOG.txt) · [独立核对](PERCEPTUAL_MFT_20261126_NATIVE_TEMPORAL_GRAY_COMPLETE_TRAIN_VERIFICATION.json) · [完整六轮progress原件](perceptual_20261125_native_training_progress/temporal_gray_six_epoch_original_progress.json)

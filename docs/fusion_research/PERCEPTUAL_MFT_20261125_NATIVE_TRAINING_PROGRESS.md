# 原生融合正式训练：两个对照已完成两遍，六遍预算继续

原控制器及同两个GPU进程持续训练，quality_rgb／quality_gray已各完成两遍全部27251帧，累计54502帧次、13628个片段／实际更新。每遍都无padding重复，两遍各自的实际顺序摘要在两组间完全一致；原源RGB权重／BN逐遍检查通过。当前进入第三遍，最低六遍共同预算未变，其他四个正常条件仍在原队列。

| 条件 | 物理GPU | 第一遍完整loss | 第二遍完整loss | 累计实际更新 | 第二遍checkpoint SHA256 |
| --- | ---: | ---: | ---: | ---: | --- |
| quality_rgb | 4 | 2.57960204 | 2.57919934 | 13628 | 39688fb9b2dac625af3b8390f59ee6053776da3154d036cce54536706d104318 |
| quality_gray | 5 | 2.57888883 | 2.57904583 | 13628 | e7fa36d488cc95213fb3a2456800f7d297155089ca7bfec2fbb7de6d53c2ae35 |

两个完整Train loss均值变化很小，不能据此宣称泛化改善或模型未学习：四个快照均有限，第二遍参数与第一遍确有变化，但参数变化不等于HOTA提升。它们是同容量RGB／灰度控制，原生IR及时间分支尚未正式调度，不能外推其成败。没有原生候选的新Dev/Test结果，所有选择继续等待完整登记的Dev。

源代码中的正常方案保留RGB保护：特征修正受(1−p)×1[p<0.6]约束，另有可靠RGB保持loss。现有loss不足以把平坦表现归因给该约束、优化或特征表达中的任何一项。已登记的no_RGB_protection会同时去掉该特征mask并把保持loss权重置0，属于联合保护机制消融；按原协议只有正常IR在完整Dev超过直接RGB及同模式RGB／灰度后才拟合，当前尚未触发。不根据训练途中或历史Test修改条件。

只读CPU核对实际退出0，完整调用5.340秒，观测峰值RSS0.580GiB；源码沿用此前同一个核对工具，没有新增训练或随机种子。原首遍报告及两份首遍快照记录保持不变；四个原生检查点SHA、历史、真实PID／启动tick／argv和原progress的checkpoint链接均核对一致。工作盘实测59.121GiB，上限64GiB，独立空闲2724.379GiB。权重留在工作盘，Git仅保存小型记录。

完整核对见[原始记录](PERCEPTUAL_MFT_20261125_NATIVE_TRAINING_PROGRESS.json)、[实际执行](PERCEPTUAL_MFT_20261125_NATIVE_TRAINING_PROGRESS_EXECUTION.json)、[发布核对](PERCEPTUAL_MFT_20261125_NATIVE_TRAINING_PROGRESS_VERIFICATION.json)及[两份progress原件](perceptual_20261125_native_training_progress)。本轮仅复用tools/audit_mft_native_epoch_prefix.py，把--through-epoch设为2，未做神经前向、优化更新或训练重启。

来源范围与首遍相同：实际执行的逐次读图SHA门槛绑定成功参考清单，原生epoch日志不另存完整逐图消费列表，本报告没有补造该列表。两组正式控制只消费RGB源字节；同图灰度派生不等于正式IR已训练。历史Test暴露、原RGB预训练见Dev和固定RGB派生IR的限制保持。

全套六组训练、完整Dev／预算决策／消融／七段Test／成本仍待完成。原冻结结构主方案仍29.834177 HOTA，目标34.529及2–3项独立贡献未达，创新计数0，Goal保持active。

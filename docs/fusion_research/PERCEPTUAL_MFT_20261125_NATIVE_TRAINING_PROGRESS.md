# 原生融合正式训练：三个对照六遍完成，两个IR分支正在训练

quality_rgb／quality_gray／temporal_rgb已各完成六遍全部27251帧，三个原worker均实际退出0。每组累计163506帧次、40884个片段／实际更新；每遍6814片段、无padding重复。三组六遍实际顺序摘要完全相同；十八份原生快照通过独立CPU有限性、历史及SHA核对，原RGB冻结检查逐遍通过。本轮新增核对temporal_rgb六份快照，旧两个对照和第一遍IR／时序证明保持原字节。

| 完整Train遍数 | quality_rgb loss | quality_gray loss | temporal_rgb loss |
| --- | ---: | ---: | ---: |
| 1 | 2.57960204 | 2.57888883 | 2.58042841 |
| 2 | 2.57919934 | 2.57904583 | 2.57961410 |
| 3 | 2.57954909 | 2.57884048 | 2.58110718 |
| 4 | 2.57744853 | 2.57810849 | 2.57879639 |
| 5 | 2.57830708 | 2.57856514 | 2.57967240 |
| 6 | 2.57770683 | 2.57693930 | 2.57765829 |

三组完整Train loss变化很小。拟合及参数变化不等于HOTA提高，也不构成数值收敛证明。六组正常条件尚未全部完成；最低共同预算六遍，之后仅按全部完整4834帧Dev及预登记规则共同决定九／十二遍，不按Test或部分Train结果延长。当前没有本轮原生方案的新Dev／Test准确率，所有较差正式正常条件仍须完整七段Test。

原控制器已自动把GPU5从完成的temporal_rgb交给temporal_ir。新任务是正式fit／through-epoch6，实际PID、启动tick、argv、CUDA_VISIBLE_DEVICES=5及GPU UUID均匹配。11月25日23:50（北京时间）的原进程只读快照显示：quality_ir已完成五遍，正在第六遍；temporal_ir正在第一遍；temporal_gray尚在原队列等待。早期观察器交接时句柄已不可用、数字退出码未知；当前只读观察器1254273及两个原训练进程均直接核对存活，原训练未重启。此处只是原progress／实际日志的进度，不增加六遍完成资格或后续快照有限性证明。

| 正常条件 | 当前状态 | 完整Train遍数 | 本次及原已公开的独立快照核对 |
| --- | --- | ---: | --- |
| quality_rgb | 六遍完成、原worker退出0 | 6 | 全部六遍 |
| quality_gray | 六遍完成、原worker退出0 | 6 | 全部六遍 |
| temporal_rgb | 六遍完成、原worker退出0 | 6 | 全部六遍 |
| quality_ir | GPU4原进程继续第六遍 | 5 | 原首遍完整核对；后续原日志／SHA记录保留 |
| temporal_ir | GPU5原队列正式训练中 | 0 | 工程门槛先前已通过；当前实际更新，不冒充完整一遍 |
| temporal_gray | 原队列等待 | 0 | 工程门槛先前已通过，正式训练未调度 |

正常原生方案仍保留(1−p)×1[p<0.6]教师置信特征mask及可靠RGB保持loss。当前loss不能将效果归因于这些约束或优化中的某项；已登记no_RGB_protection会同时移除mask并将保持loss权重置0，是联合机制消融，仅在正常IR完整Dev超过直接RGB和同模式RGB／灰度后触发。当前没有更改配方、种子或选择规则。

正式IR读图使用固定11月新版PNG，实际执行的SHA断言绑定四份成功参考清单，预期RGB＋IR集合54502项。RGB／灰度条件的预期RGB集合27251项。原生epoch日志不另存完整逐图消费列表，本报告未补造该列表，也未用参考清单冒充实际消费清单。reader的8GiB是编码字节RAM容量，不是整个worker RSS上限；实际RSS／GPU映射与主机内存见发布核对JSON。

新增CPU核对实际退出0，完整调用5.500秒，观测峰值RSS0.597GiB；复用原有同一工具，只读已存在小型原生快照与记录，没有NN前向或GPU调用。原第一遍及两对照六遍的JSON、执行、原件均保持，新时序六遍原件另存。

23:50快照工作盘按原guard口径lstat且不跟随符号链接，实测59.252GiB／上限64GiB，独立空闲2704.841GiB；主机MemAvailable及真实GPU进程见[状态快照](PERCEPTUAL_MFT_20261125_NATIVE_LIVE_STATUS_SNAPSHOT.json)，高于40GiB保留值。Git只保存轻量文本，不包含权重、图像、数组或缓存。

新增证据见[时序RGB六遍核对](PERCEPTUAL_MFT_20261125_NATIVE_TEMPORAL_RGB_COMPLETE_TRAIN.json)、[实际CPU执行](PERCEPTUAL_MFT_20261125_NATIVE_TEMPORAL_RGB_COMPLETE_TRAIN_EXECUTION.json)、[真实交接及发布核对](PERCEPTUAL_MFT_20261125_NATIVE_TEMPORAL_RGB_COMPLETE_TRAIN_VERIFICATION.json)和[完整六遍progress原件](perceptual_20261125_native_training_progress/temporal_rgb_six_epoch_original_progress.json)。原[两个六遍对照](PERCEPTUAL_MFT_20261125_NATIVE_TRAINING_PROGRESS.json)、[首遍IR／时序](PERCEPTUAL_MFT_20261125_NATIVE_MODALITY_FIRST_EPOCH.json)及其原件保持。

固定结构主方案仍29.834177 HOTA，对原RGB29.529为+0.305177；C只有条件消融支持，S不稳定、R有负贡献。本次进度不改变主模型、Test选择或独立创新计数。历史Test暴露、原RGB预训练见Dev、伪IR由RGB派生等限制保持。全套六组Train、完整Dev／预算决策／所需消融／七段Test／独立成本仍待完成，目标34.529及2–3项独立贡献未达，Goal保持active。

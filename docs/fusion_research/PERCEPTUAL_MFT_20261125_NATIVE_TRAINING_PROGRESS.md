# 原生融合正式训练：两个对照六遍完成，IR与时序正式接续

原控制器自动完成quality_rgb／quality_gray的六遍全部Train，两个原worker均实际退出0，任务记录为COMPLETE_REGISTERED_TRAIN_STAGE。每组累计163506帧次、40884个片段／实际更新；每遍27251帧、6814片段，无padding重复。两组六遍各自的实际顺序摘要相同，原RGB冻结检查逐遍通过；十二份原生快照均有限、各遍参数不同，第一／二遍记录与此前发布值完全一致。

| 完整Train遍数 | quality_rgb loss | quality_gray loss |
| --- | ---: | ---: |
| 1 | 2.57960204 | 2.57888883 |
| 2 | 2.57919934 | 2.57904583 |
| 3 | 2.57954909 | 2.57884048 |
| 4 | 2.57744853 | 2.57810849 |
| 5 | 2.57830708 | 2.57856514 |
| 6 | 2.57770683 | 2.57693930 |

完整Train loss变化很小。六遍拟合及参数变化不等于HOTA提高，也不构成数值收敛证明。六组同预算正常条件仍需全部完成；最低共同预算为六遍，之后仅按完整4834帧Dev的预登记规则共同决定九／十二遍。当前没有新原生方案的Dev／Test准确率，所有较差正式条件仍须完整七段Test。

正常原生方案仍保留(1−p)×1[p<0.6]教师置信特征mask及可靠RGB保持loss。当前loss不能把效果归因于这些约束或优化中的某一项；已登记no_RGB_protection会同时去掉mask并将保持loss权重置0，属于联合机制消融，只有正常IR在完整Dev超过直接RGB及同模式RGB／灰度后才触发，不按训练途中或历史Test修改条件。

原队列自动把GPU4交给quality_ir、GPU5交给temporal_rgb，两个原任务的实际PID／启动tick／argv／CUDA_VISIBLE_DEVICES及GPU UUID均核实，实际日志超过6814更新。两者均已完成至少一遍完整27251帧；本次独立核对各自第一遍，时序后续更新继续进行。temporal_gray／temporal_ir仍在原队列等待，没有新起重复训练或追加种子。

| 本次独立核对对象 | 完整遍数 | 每遍帧数 | 该前缀实际更新 | 最后一份已核对快照SHA256 |
| --- | ---: | ---: | ---: | --- |
| quality_rgb | 6 | 27251 | 40884 | da453beb4acdc6b6b2e0b411ddad09e70b8f6e4ff16dc83906218d676d379d66 |
| quality_gray | 6 | 27251 | 40884 | cd600d5ad015e102b939e311d33e29e6cbc761981ae358019112aa38e7b1becb |
| quality_ir | 1 | 27251 | 6814 | e4b297a4a5584ad4157c14f388cf17d86c6758105c38edbb38a0a78171930490 |
| temporal_rgb | 1 | 27251 | 6814 | 34caebfbd992dc9dda9a08a04c8728622146cba5c86e49a308d2934777b70384 |

quality_ir正式首遍已使用固定11月新版PNG；执行的逐次读图SHA断言绑定四份成功的原RGB／IR参考清单，预期输入集合为54502项；RGB／灰度条件预期RGB集合27251项。完整交付顺序和快照均已核对，但原生epoch日志没有另存完整逐图消费列表，本报告没有补造该列表，也不把参考清单冒充实际原生读取清单。原生reader登记的编码字节RAM容量为8GiB，与先前成本测量的2GiB解码缓存不同。 本次IR进程实际峰值RSS为10.127GiB，8GiB不是整个worker的RSS上限；主机实际MemAvailable约81.427GiB，高于40GiB保留值。

两次复用同一CPU核对工具均实际退出0，完整调用分别6.275秒、7.008秒，外层峰值RSS分别0.580、0.599GiB。它们只读已存在的十四份小型原生快照和记录，不进行NN前向、拟合或GPU调用。旧只读观察句柄在本次复查时已不可用；依据原状态和真实进程恢复观察，没有把观察结束当成训练失败，也没有回填旧观察器数字退出码。

工作盘实测59.268GiB／上限64GiB，独立空闲2724.144GiB；时间点见发布核对JSON。权重、图像、数组及缓存均留在工作盘，Git只保存小型文本。此前两遍版本保存在提交2e74363，原首遍报告不变。

完整核对见[两个六遍对照](PERCEPTUAL_MFT_20261125_NATIVE_TRAINING_PROGRESS.json)、[首遍IR／时序](PERCEPTUAL_MFT_20261125_NATIVE_MODALITY_FIRST_EPOCH.json)、[六遍CPU执行](PERCEPTUAL_MFT_20261125_NATIVE_TRAINING_PROGRESS_EXECUTION.json)、[首遍CPU执行](PERCEPTUAL_MFT_20261125_NATIVE_MODALITY_FIRST_EPOCH_EXECUTION.json)、[发布与真实调度核对](PERCEPTUAL_MFT_20261125_NATIVE_TRAINING_PROGRESS_VERIFICATION.json)及[四份progress原件](perceptual_20261125_native_training_progress)。

当前固定结构主方案仍29.834177 HOTA，对原RGB29.529为+0.305177；C只有条件消融支持，S不稳定、R有负贡献。本次训练进度不改变主模型、Test选择或独立创新计数。历史Test暴露、源RGB预训练见过Dev、伪IR由RGB派生的限制保持。全套六组拟合、完整Dev／预算决策／所需消融／七段Test／成本仍待完成；目标34.529及2–3项独立贡献未达，Goal保持active。

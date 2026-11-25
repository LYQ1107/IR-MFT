# 原生融合正式训练：四组六轮完成，时序灰度／IR继续训练

quality_rgb／quality_gray／temporal_rgb／quality_ir现已各完成六轮全部27251帧，四个原worker均实际退出0。每组累计163506帧次、40884个片段／实际更新；每轮6814片段，无padding重复。24份原生模块快照有限、每组逐轮状态均不同且偏离初始化，四组每轮实际交付地址摘要一致；质量模式RGB／灰度／IR初始参数相同。原RGB冻结日志逐轮核对通过。此次新增quality_ir六轮证明，旧16份相关证明及首轮IR原件保持字节不变。

| 完整Train轮次 | quality_rgb loss | quality_gray loss | quality_ir loss | temporal_rgb loss |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2.57960204 | 2.57888883 | 2.57926598 | 2.58042841 |
| 2 | 2.57919934 | 2.57904583 | 2.57899623 | 2.57961410 |
| 3 | 2.57954909 | 2.57884048 | 2.57922564 | 2.58110718 |
| 4 | 2.57744853 | 2.57810849 | 2.57777712 | 2.57879639 |
| 5 | 2.57830708 | 2.57856514 | 2.57829463 | 2.57967240 |
| 6 | 2.57770683 | 2.57693930 | 2.57791129 | 2.57765829 |

当前两组剩余任务均保持原PID：temporal_gray已完成四遍、继续第五遍，temporal_ir已完成三遍、继续第四遍。这是原journal、实际更新与进程身份的只读观察，不增加六轮张量资格；完整Dev尚未启动。

并行的BoT-SORT／HybridSORT公开后端补测已完成8项完整Dev及8项完整Test，16个worker和原CPU控制器均实际退出0，官方47字段与逐序列CSV独立核对退出0。五后端40组汇总见[完整分析](PERCEPTUAL_MFT_20261126_ADDITIONAL_PUBLIC_TRACKERS_COMPLETE_ANALYSIS_ZH.md)及[核对证据](PERCEPTUAL_MFT_20261126_ADDITIONAL_PUBLIC_TRACKERS_COMPLETE_VERIFICATION.json)。新增Test中BoT的IR−RGB约−0.924点，Hybrid约+0.759点却低于灰度约0.311点；未据此改选原主方案，也未增加创新点。

四组完整Train loss变化很小且非单调，不能代替收敛或HOTA结论。质量IR末轮2.577911，RGB末轮2.577707；未据此宣称IR增益。共同最低六轮及完整4834帧Dev规则保持，是否延长九／十二轮只依据完整Dev决定。全部较差正式正常条件仍须完整七段Test。

原GPU4的quality_ir完成并退出0后，同一个原控制器自动启动temporal_gray；真实PID1330025、启动tick、argv、CUDA_VISIBLE_DEVICES=4与GPU UUID持续匹配。GPU5原temporal_ir为PID1248117，身份与GPU UUID也持续匹配。只读观察器仅跟踪原剩余训练，不发送训练信号；六轮审计仍只在原worker完整结束后执行。

| 正常条件 | 当前状态 | 已完成全量Train轮数 | 独立模块快照核对 |
| --- | --- | ---: | --- |
| quality_rgb | 原worker完成并退出0 | 6 | 全部六轮 |
| quality_gray | 原worker完成并退出0 | 6 | 全部六轮 |
| quality_ir | 原worker完成并退出0 | 6 | 本次全部六轮 |
| temporal_rgb | 原worker完成并退出0 | 6 | 全部六轮 |
| temporal_gray | GPU4原进程第五遍正式训练 | 4 | 当前没有完整六轮资格 |
| temporal_ir | GPU5原进程第四遍正式训练 | 3 | 当前没有完整六轮资格 |

正常原生方案保留(1−p)×1[p<0.6]教师置信特征mask及可靠RGB保持loss。已登记no_RGB_protection会同时去掉mask并将保持loss权重置0，是联合机制消融；仅在正常IR完整Dev超过直接RGB和同模式RGB／灰度后触发。当前配方、种子及选择规则均未更改。

实际IR读图仍固定11月DiffV2IR PNG，执行的SHA断言绑定原四份成功IR参考清单、54502个RGB＋IR条目；RGB／灰度的预期RGB集合27251项。原生训练没有另存完整逐图消费列表，未补造或以参考清单冒充。编码字节RAM容量8GiB不是整个worker RSS上限；真实进程／GPU映射与主机内存见发布核对。

此次独立CPU审计实际退出0，完整调用5.360秒、观测峰值RSS0.579GiB；复用原工具，只读已有小型原生快照及记录，没有NN前向、GPU调用或新拟合。首次导出末行print参数错误退出1，已生成产物随后逐字节独立核对通过；原审计与训练结果没有重跑或改写。

发布核对时工作目录59.263GiB／上限64GiB，独立空闲2703.428GiB，主机MemAvailable86.441GiB，高于40GiB保留值。原guard按lstat且不跟随符号链接计数。Git仅保存文本证明，不包含权重、图像、数组或缓存。

新增[质量IR六轮核对](PERCEPTUAL_MFT_20261126_NATIVE_QUALITY_IR_COMPLETE_TRAIN.json)、[实际CPU执行](PERCEPTUAL_MFT_20261126_NATIVE_QUALITY_IR_COMPLETE_TRAIN_EXECUTION.json)、[交接及发布核对](PERCEPTUAL_MFT_20261126_NATIVE_QUALITY_IR_COMPLETE_TRAIN_VERIFICATION.json)、[完整六轮progress原件](perceptual_20261125_native_training_progress/quality_ir_six_epoch_original_progress.json)。旧[两个质量对照](PERCEPTUAL_MFT_20261125_NATIVE_TRAINING_PROGRESS.json)、[时序RGB六轮](PERCEPTUAL_MFT_20261125_NATIVE_TEMPORAL_RGB_COMPLETE_TRAIN.json)、[首轮IR／时序](PERCEPTUAL_MFT_20261125_NATIVE_MODALITY_FIRST_EPOCH.json)原件保持。

新增已完成四组的[参数／历史梯度诊断](PERCEPTUAL_MFT_20261126_NATIVE_PARAMETER_LEARNING_DIAGNOSTIC.md)：145个参数张量均保存正AdamW二阶矩，各40884步；三层输出投影均已离开零初始化。该证据排查整分支无历史梯度，不证明精度、推理修正或收敛；完整Dev及其真实门控／特征修正统计仍待原六组全部完成。

新增固定8帧Train的[CPU特征舍入诊断](PERCEPTUAL_MFT_20261126_NATIVE_TRAIN_FEATURE_ROUNDING_DIAGNOSTIC.md)：FP32 RGB参考FPN显式转FP16后，IR三层相对修正RMS约13.1%／4.1%／5.4%。这些样本的修正实际非零，但RGB／灰度也有明显修正，不证明精度或模态贡献；完整GPU Dev仍待完成。首次CPU FP16参考路径因成本结束的−15／外层241及原件完整保留，原GPU训练未改动。

固定结构主方案仍29.834177 HOTA，对原RGB29.529为+0.305177。本轮原生方案尚无完整Dev／Test准确率；S不稳定、R有负贡献、C只有条件消融支持，独立创新计数仍0。历史Test暴露、原RGB预训练见Dev及伪IR由RGB派生的限制保持。全六组完整Train、Dev／预算决策、所需消融、完整七段Test及独立成本仍待完成，目标34.529及2–3项独立贡献未达，Goal保持active。

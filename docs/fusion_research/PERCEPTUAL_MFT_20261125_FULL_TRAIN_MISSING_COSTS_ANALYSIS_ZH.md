# 三个缺测控制的真实成本已补齐

六次独立GPU4执行全部退出0；实际GPU UUID、PID／启动时刻／argv、CUDA设备及FP16精度一致。两段各32帧网络输出与冻结proposal逐数组相同，独立ByteTrack前缀行也完全一致。每次计时48帧／12个B4批次；RSS和解码图像缓存均在本成本worker的8GiB／2GiB范围内。完整验证见[独立来源与交接核对](PERCEPTUAL_MFT_20261125_LEGACY_COST_NATIVE_HANDOFF_AUDIT.json)，六份逐字节原报告及原日志见[原始成本目录](perceptual_20261125_full_train_missing_costs)。

| 配置 | 内存输入 | B4 FPS | 输入准备占比 | NN／NMS占比 | 跟踪占比 | RSS GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FTT_IR_image_roll | False | 2.401142 | 91.18% | 8.12% | 0.70% | 1.531 |
| FTT_IR_unavailable | False | 2.170845 | 92.31% | 6.95% | 0.75% | 1.467 |
| FTT_source_RGB_canonical | False | 16.306708 | 69.18% | 24.92% | 5.89% | 1.391 |
| FTT_IR_image_roll | True | 2.977525 | 87.88% | 10.94% | 1.17% | 1.971 |
| FTT_IR_unavailable | True | 3.050973 | 88.65% | 10.05% | 1.30% | 1.897 |
| FTT_source_RGB_canonical | True | 16.586532 | 67.63% | 25.70% | 6.67% | 1.745 |

本批控制的主要计时开销位于输入准备／传输段，含原图解码（读图模式）、CPU预处理、结构描述及张量传输；NN／NMS段包括实际backbone、head、适配器及后处理。内存预读只移除解码／读图，不移除结构计算。因此不能把NN／NMS子段FPS当作完整方法FPS，也不能归因给ByteTrack。新原生CNN路线的实际速度仍待其正式独立测量。

实际缺失IR仍编码IR的实现、短前缀B4范围、排除IR生成／模型加载／预读、共享负载和文件缓存未控制等限制保持。与旧十项合并后每模式13项均有独立成本，旧十项未重写；见[完整结果与成本](PERCEPTUAL_MFT_20261125_FULL_TRAIN_COMPLETE_RESULTS.md)。

原成本控制器875865及六个子进程均已实际退出。它自动恢复同一个原生CPU等待器3943138；后者已经启动六组真实GPU工程检查，并接续正式拟合，没有另开第三张卡或中断健康训练。本轮不新增HOTA或独立贡献。

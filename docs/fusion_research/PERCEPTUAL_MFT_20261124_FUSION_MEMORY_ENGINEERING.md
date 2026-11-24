# F4 融合记忆：工程原型与未完成的效果验证

本轮补上任务书指定的 `FusionMemoryTrack` / `FusionMemoryTracker` 接口，
并验证零权重路径。**尚未传入学习到的融合特征，尚无新的 HOTA，不能
据此增加已验证创新数或宣布第四个当前 IR 方向已完成。**

条件协议见 [prototype config](../../configs/perceptual_fusion_track_memory_prototype_20261124.json)。
正式使用须先有新版 November IR 视觉融合模型在完整 Dev 上胜过原始／
新鲜 RGB 及同模式 RGB／灰度强对照，再冻结视觉主模型，并登记完整
F4 Dev/Test/成本流程。当前没有启动此类正式流程或额外 GPU 作业。

实现使用原 BYTETracker 已有的 factory/hook，原文件字节保持不变。
每条轨迹分别保存 RGB、辅助和融合记忆；可靠性代理
`r = score * (1 - max_other_detection_IoU) * q` 决定更新资格与更新量。
重叠不是标注遮挡，q 也不是已校准的真实不确定性。高分、可靠观测才更新，
其余冻结；记忆按实际距离上次可靠更新的帧数衰减。当前模态缺失时不读取
辅助／融合数组，使用 RGB 记忆。视频切换清空状态，禁止读取未来帧。

外观仅对原本可匹配且预测 IoU 至少 0.1 的第一阶段边增加非负、小权重
余弦代价；不降低原代价，不开放原阈值之外的边。第二阶段和 pending
代价保持原逻辑。`appearance_weight=0` 不读取特征或更新特征记忆。
未来必须同时比较相同检测、相同 q 下的 RGB 与融合记忆，以及普通
RGB 的 q=1 强对照和融合记忆的 q=1 消融。仅普通 RGB 记忆有效时，
不能算融合贡献。外观 EMA 和置信度参与跟踪有既有先例：
[BoT-SORT 固定版本](https://github.com/NirAharon/BoT-SORT/blob/251985436d6712aaf682aaaf5f71edb4987224bd/tracker/bot_sort.py)、
[HybridSORT 固定版本](https://github.com/ymzis69/HybridSORT/blob/396f8d30db13304c0cbaf1dcf2e16ded93ce1701/trackers/hybrid_sort_tracker/hybrid_sort.py)。
本实现没有复制这两个项目的源代码，也不把这些基础操作单独称为创新。

[CPU 原始证明](PERCEPTUAL_MFT_20261124_FUSION_MEMORY_CPU_PROOF.json)记录了：

- 11 项有状态、更新、缺失模态、代价及 ROI 检查通过；启用记忆的输入
  为合成向量，ROI 使用已知线性特征平面。
- 对原 RGB Dev 检测包回放完整 BT-005 1834 帧和 SN-013 3000 帧，
  总计 4834 帧。零权重输出与原已发布 MOT 文件逐字节一致；每帧
  tracked/lost/removed 的数值和 Kalman 状态摘要也严格一致。
- 沿用每段原始 ID 初始计数，分别为 0 和 22；不事后改写 ID，
  新 tracker 不修改原 BaseTrack 全局计数器。
- ROIAlign 确认实际 stride8、原图到网络坐标的 resize、3×3 输出、
  aligned 和 sampling_ratio=2。已知网络 ROI `[8,4,40,20]` 的均值
  正确为 `[24,12]`；不改变交给 tracker 的原图检测框。
- Torch 2.6.0、torchvision 0.21.0 的编译 CPU 算子可用，运行 39.477 秒，
  进程峰值 RSS 0.580 GiB。该时间是此 CPU 检查，不能写成系统 FPS。

原检测包和轨迹只读复用，无图像／FPN 副本，没有读取 Train/Test GT、
新 Test 回放或准确率评估。后续仍缺实际已学 RGB/辅助/融合 P3 及 q
的图像到实例接口、正式模型资格、完整 Dev 选择、全部正式 Test、
模态干预及真实组件成本。[来源与资源核查](PERCEPTUAL_MFT_20261124_FUSION_MEMORY_SOURCE_AUDIT.json)
确认八份现有登记的源文件全部未改，测速的动态来源集合仍为 137 个。
当时工作目录 58.294 GiB，小于 64 GiB；没有打断正在运行的两组训练。

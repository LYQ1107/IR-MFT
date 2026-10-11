# Codex 长期科研任务书：CFP-FishMOT

## 0. 角色与最终目标

你现在是本项目的高级计算机视觉研究员、PyTorch 工程师、实验负责人和严格论文审稿人。

请在现有 IR-MFT 项目基础上，完成一项有独立科学问题、真实创新模块、可复现实验和完整消融证据的多模态鱼类多目标跟踪研究。

### FINAL GOAL

**Reliable Synthetic-Modality Identity Association for Online Multi-Object Fish Tracking**

研究系统暂定名：CFP-FishMOT。

论文方向暂定：

**When Does a Synthetic Modality Help? Counterfactual Predictive Association for Underwater Multi-Object Tracking**

最终研究目标：

开发一个严格在线的 RGB + 构建 IR 多目标鱼类跟踪框架，研究如何估计合成模态对身份关联的条件收益，利用时序预测维持身份一致性，并在鱼群交叉、遮挡和证据冲突时避免错误关联与身份记忆污染。

重点不再是简单改善检测器，而是改善：

- HOTA：总体跟踪质量。
- AssA：身份关联质量。
- IDF1：身份保持能力。
- IDSW：身份切换次数。

同时完整报告 MOTA、DetA、FN、FP、Frag、运行速度和计算成本。

**FINAL GOAL 不是“代码能够运行”，也不是“做完三个模块”，而是形成一个经过公平实验验证、具有可解释贡献和可复现性的在线多模态鱼类跟踪方法。**

不能提前宣称超过 SOTA，也不能伪造任何指标。

---

## 1. 项目现状与工程边界

当前服务器项目：

`/data3/liuyeqiang/IR-MFT`

GitHub：

https://github.com/LYQ1107/IR-MFT

历史分支：

`feat/qrtm-progressive-v1`

历史提交：

`e0cf4c7`

本机已经恢复的数据资产位于：

`/data3/liuyeqiang/IR-MFT_assets`

具体权重、数据集和缓存路径必须通过实际文件检查确定，不允许凭空假设。

历史最佳 V3：

- RGB 与构建 IR 分别经过独立 YOLOPAFPN。
- P3/P4/P5 三个尺度采用 RGB 主导门控融合。
- V2 冻结 RGB 和检测头训练 IR 与门控。
- V3 解冻全部参数进行低学习率微调。
- V3 Epoch 3 为历史最佳。
- 关联器仍然是标准 ByteTrack。

已有结果：

| 方法 | HOTA | IDF1 | MOTA | IDSW |
|---|---:|---:|---:|---:|
| ByteTrack 历史基线 | 待核实 | 34.100 | 51.800 | 560 |
| HybridSORT-ReID | 待核实 | 39.000 | 50.800 | 519 |
| V3-E3 | 31.091 | 35.044 | 52.325 | 560 |

V3 历史文件：

`/data3/liuyeqiang/IR-MFT_assets/history/v3_eval_epoch3_tuned_b8/metrics.json`

首先读取其中的真实 HOTA、AssA、DetA、AssRe、AssPr、IDF1、MOTA、IDSW。

不能用 HOTA 推算 AssA。

历史结果有测试集参数调优，必须标记为历史参考结果，而不是完全独立的最终测试结论。

### 开发保护规则

1. 不覆盖 V1/V2/V3 原始代码和权重。
2. 不删除训练数据、检测缓存和历史日志。
3. 从当前分支创建新的研究分支，例如 `research/cfp-fishmot-v1`。
4. 所有实验配置、随机种子、运行命令、Git commit 和模型权重路径必须记录。
5. 大型数据、模型权重、推理缓存不提交到 GitHub。
6. 不随意安装全局依赖或覆盖其他 Conda 环境。
7. 不终止其他用户的 GPU 任务。
8. 优先复用当前服务器已有数据和依赖。
9. 如遇到权限、数据缺失、旧版本不兼容，先调查并修复，不直接宣布研究失败。

---

## 2. 首先深入研究 15 个开源项目

请在开发前完成代码级文献调研。

不是只阅读 README，而是需要查看网络结构、训练入口、关键损失函数、数据组织方式、推理逻辑和评估流程。

参考项目：

1. SU-T：https://github.com/vranlee/SU-T
2. TIDE：https://github.com/vranlee/TIDE
3. M4FT：https://github.com/vranlee/M-4-FT
4. DM3T：https://github.com/vranlee/DM-3-T
5. TFMFT：https://github.com/vranlee/TFMFT
6. PFTrack：https://github.com/wqw123wqw/PFTrack
7. MOTIP：https://github.com/MCG-NJU/MOTIP
8. TCEI：https://github.com/1941Zpf/TCEI
9. GOT-JEPA：https://github.com/chenshihfang/GOT
10. HybridSORT：https://github.com/ymzis69/HybridSORT
11. BoT-SORT：https://github.com/NirAharon/BoT-SORT
12. ByteTrack：https://github.com/FoundationVision/ByteTrack
13. FastReID：https://github.com/JDAI-CV/fast-reid
14. V-JEPA 2：https://github.com/facebookresearch/vjepa2
15. DINOv3：https://github.com/facebookresearch/dinov3

评估工具：

https://github.com/JonathonLuiten/TrackEval

研究要求：

- 把参考代码放在项目之外的独立参考目录或被 Git 忽略的目录。
- 保存每个项目的准确 commit。
- 检查源码许可证，不能无条件复制外部代码。
- 记录具体的模型类、网络输入输出形状、训练损失、采样器、优化器、训练入口和所需预训练权重。
- 明确哪些项目仅开放部分代码，哪些能够完整训练。
- 针对每篇论文，判断其研究问题与我们的区别。
- 不强行运行全部 15 个大型项目，不进行无意义的重复预训练。

特别注意：

**MOTIP**

检查 `models/motip/id_decoder.py`、`train.py` 和 `docs/GET_STARTED.md`。

重点研究：

- 如何将历史轨迹 ID 编码为上下文。
- 如何预测当前检测的 ID。
- 如何表示 unmatched/newborn。
- 如何利用时间掩码避免未来信息泄漏。
- 如何训练身份分类损失。
- 为什么直接 ID 预测可以替代部分手工关联规则。

**V-JEPA 2**

检查视频编码器、masked prediction、EMA target encoder、predictor、预训练配置及数据采样逻辑。

只借鉴隐空间预测思想，不直接从头训练数亿参数的视频大模型。

**GOT-JEPA**

检查 `pytracking/ltr/models/tracking/tompnet_JEPA.py`、`train_settings/tomp/GOT_JEPA_378_pretrain.py` 及微调配置。

重点学习遮挡下的预测式表征和自监督训练设计。

注意 GOT-JEPA 属于单目标跟踪，不能直接把单目标模板逻辑复制成多目标身份关联。

**TCEI**

研究 `TCEI/tcei.py` 中的熵度量、缓存更新和历史经验校准。

重点判断哪些机制能够用于在线关联置信度估计。

如果完整训练代码未公开，记录这个事实，不编造训练命令。

**SU-T / TIDE / HybridSORT**

重点研究：

- 鱼类非刚性形变。
- 多阶段关联。
- 低置信度候选。
- 几何与运动线索。
- ReID 和非 ReID 模式之间的差异。
- 相似鱼发生身份交换的典型原因。

**DM3T / PFTrack**

重点研究真正的 RGB-T 融合结构、训练方式和时序建模。

但必须区分真实热红外和构建 IR，不能直接假设其信息互补性相同。

**DINOv3 / FastReID**

使用轻量预训练模型作为强外观特征基线，检查冻结特征与微调特征的差别。

### 调研交付物

创建：

`docs/research/LITERATURE_15_CODE_AUDIT.md`

每个项目记录：

- 论文题目、年份、GitHub URL、commit。
- 代码可用状态。
- 关键源码位置。
- 模型输入和输出。
- 核心模块。
- 训练数据及样本构造。
- 损失函数和训练流程。
- 推理和关联过程。
- 计算量及部署限制。
- 可以借鉴的机制。
- 不适用于本任务的部分。
- 与我们拟提出方法的创新重叠。
- 许可证要求。

最后形成一个可执行的架构对照表，而不是只有文字综述。

---

## 3. Phase 0：恢复环境、复现 V3、冻结实验协议

这是后续一切工作的前提。

### 3.1 环境检查

自动调查：

- GPU 数量、型号、显存和空闲状态。
- CUDA、PyTorch、TorchVision。
- Conda 环境。
- YOLOX、TrackEval、FastReID 依赖。
- V3 最佳权重完整性。
- RGB/IR 数据完整性。
- 训练与测试视频数量。
- 视频、帧、GT 和 track ID 的映射。
- 检测缓存的格式及版本。

编写自动 preflight 工具，检查结果必须保存到文件。

### 3.2 复现基线

首先运行 V3-E3 的完整推理与评测。

要求核对：

- 七个历史测试序列。
- 历史总帧数 15,985。
- HOTA、AssA、DetA。
- IDF1、MOTA、IDSW。
- FP、FN、Frag。
- 每序列输出文件。
- 检测框数量。
- 推理耗时和环境版本。

如果不能完全复现，优先定位：

- 模型权重。
- 图像预处理和 RGB/IR 通道顺序。
- 浮点精度。
- 检测阈值。
- NMS。
- ByteTrack 配置。
- 不同 TrackEval 版本。
- 测试图像排序与帧 ID。
- GPU/软件版本导致的数值差异。

不能因为历史指标与本机不同，就直接进行新方法训练。

### 3.3 冻结独立开发协议

现有数据为 8 个训练视频和 7 个历史测试视频。

从训练视频中构建按完整视频隔离的 Train/Development 划分。

优先尝试六个训练视频、两个开发视频，并考虑小样本情况下的视频级交叉验证。

具体划分需要根据鱼的数量、遮挡分布、鱼种、光照及场景进行审计。

严格防止同一视频的相邻帧同时进入训练和开发集。

训练身份标签使用 `(video_id, track_id)` 作为复合主键，不允许不同视频中编号相同的 ID 被错误地认为是同一条鱼。

历史七个测试视频不得参与新模型的超参数选择、最佳 checkpoint 选择、失败诊断调参和模型结构选择。

由于历史测试集已被使用过，最终论文必须披露这一事实；如果能够获得新的外部测试视频，应优先建立真正未接触的最终评测集。

建立统一评估器，所有方法使用相同的 GT、序列、TrackEval 版本、IoU 匹配协议及指标聚合方式。

所有检测缓存保存来源权重和配置哈希，禁止跨模型错误复用。

---

## 4. 科学假设及原创架构

我们最终要验证三个主要假设。

### H1：合成模态的身份价值是条件性的

构建 IR 不一定对每条鱼都有效。

对于某些鱼，IR 表征可能改善身份区分；对于另一些鱼，IR 可能只是重复 RGB 信息，甚至引起错误匹配。

希望学习的是：

**给定当前鱼、历史轨迹和候选集合，加入 IR 证据究竟会降低还是增加身份关联风险？**

这不是普通图像清晰度预测，也不是只训练一个固定融合权重。

### H2：历史隐特征预测能够帮助遮挡后的身份恢复

只预测下一帧的检测框不足以解决相似鱼交叉。

尝试在潜在空间预测身份相关表征，并用于当前候选身份判断。

必须证明预测器优于普通时序平均、EMA、GRU 和运动预测，而不是单纯因为参数更多。

### H3：局部鱼群关系可以在高歧义事件中提供辅助证据

只在多条鱼密集交叉、运动关联模糊时，引入局部群体关系。

不假设整个鱼群总是朝相同方向运动，不让群体关系覆盖个体身份线索。

H3 是有条件的扩展模块，必须先证明其独立价值。

### 科学限制

首先核实 IR 的真实生成来源。

如果 IR 是仅由 RGB 生成的合成模态，其原始身份信息不可能凭空超过全部 RGB 输入所包含的信息。

我们的研究应当围绕有限模型能力、表征变换、优化难度、特征可靠性及关联误差改善展开，而非声称合成模态增加了独立传感器信息。

如果 IR 不是纯 RGB 生成，必须通过数据溯源重新确定科学表述。

---

## 5. Phase I：构建多模态身份信息实验平台

这是第一项真正的研究实验。

固定 V3 检测器，不改变检测框。

实现共享接口的 ReID 评估系统。

至少比较：

- 无 ReID 的 ByteTrack。
- HybridSORT-ReID。
- BoT-SORT-ReID。
- RGB-only ReID。
- IR-only ReID。
- RGB+IR 拼接融合。
- RGB+IR 固定加权融合。
- RGB+IR 可学习门控融合。
- 参数量匹配的 MLP/Attention 融合。

可以考虑 DINOv3 小模型冻结特征、FastReID 和 SU-T 的鱼类 ReID 实现作为参考。

### 训练数据

从训练视频中抽取真实 track ID 对应的图像块。

正样本：

同一视频、同一条鱼的不同时刻。

困难负样本：

- 同一帧中邻近的不同鱼。
- 外观高度相似的鱼。
- 交叉事件中的相似候选。
- 非刚性形变后的相似目标。
- 不同光照下的困难观测。

对正样本保留时间间隔分布，不能全部由紧邻帧组成。

训练使用身份分类、监督对比学习、Triplet 或合适的度量学习损失，具体组合必须经过对照。

禁止用测试集 track ID 训练。

### 先进行身份特征诊断

评估：

- Pairwise AUC。
- Recall@1。
- 正负样本余弦相似度分布。
- Hard-negative 区分能力。
- 跨帧身份一致性。
- 跨模态证据冲突率。
- 校准误差。
- 在相同检测框上的 HOTA、AssA、IDF1、IDSW。

关键问题：

**IR-only 是否具有可利用的身份信息？RGB+IR 是否在至少部分事件中优于 RGB-only？**

还应进行：

- 正确配对 IR。
- IR 置零。
- IR 重复 RGB。
- IR 特征随机扰动。
- IR 错误配对。
- 不同构建 IR 质量分组。

不能只证明增加一条网络分支能提高参数容量；必须证明生成模态有实际的条件收益。

输出：

`docs/research/PHASE_I_MODALITY_IDENTITY_DIAGNOSIS.md`

---

## 6. Phase II：实现创新点 1——Counterfactual Modality Utility

新模块建议命名：

**Counterfactual Modality Utility Estimator，CMUE**

### 6.1 基本输入

对历史轨迹 i 和当前候选检测 j，提取：

- RGB 特征。
- 构建 IR 特征。
- RGB 匹配分数。
- IR 匹配分数。
- IoU / 运动距离。
- 检测置信度。
- 外观相似度。
- 当前候选集合的分数差距。
- 模态之间的预测冲突。
- 可在线计算的图像质量与遮挡指标。
- 轨迹记忆的可靠性。

不能使用当前帧 GT、真实身份或未来帧作为推理输入。

### 6.2 反事实价值监督

在训练视频中，使用相同的检测框、候选集合、关联器结构及评价标签，分别进行：

A. 仅使用 RGB 身份证据的判断。

B. 在相同条件下引入 IR 身份证据的判断。

计算两者在监督关联损失或候选决策风险上的差别。

构建训练标签：

`utility = RGB-only association loss - RGB+IR association loss`

如果引入 IR 后损失降低，utility 为正。

如果引入 IR 后损失增大，utility 为负。

这是一种固定候选条件下的算法级反事实干预，不是声称观察到两个不同物理世界的因果结果。

必须使用训练集内的交叉拟合、冻结教师或其他防止自我强化的策略，避免用同一个过拟合模型生成伪标签并同时拟合这些标签。

训练 CMUE 预测 IR 在当前关联任务中的预期收益。

损失可以尝试：

- Utility regression。
- Utility sign classification。
- Association cross entropy。
- Calibration loss。
- Negative-transfer penalty。

不要第一版就使用大量复杂损失。先实现简单、可解释、容易消融的版本。

### 6.3 推理机制

先计算 RGB-only 关联分数。

CMUE 根据在线证据判断 IR 的可信价值。

当预期 IR 可能改善身份判断时，适量加入 IR。

当 IR 可能导致错误关联时，降低其贡献或完全回退 RGB。

必须存在可验证的 RGB-only fallback。

初始设计保守，防止负迁移。

### 6.4 对照实验

至少包括：

- RGB-only。
- IR-only。
- Fixed Gate。
- Learned Gate。
- Quality Gate。
- Uncertainty Gate。
- Same-evidence MLP。
- Capacity-matched MLP。
- CMUE。
- CMUE 不使用反事实监督。
- CMUE 不使用 fallback。
- CMUE 不使用校准约束。

关注 AssA、HOTA、IDF1、IDSW。

另外报告 IR 被接受、拒绝及纠错、反向改错的次数。

必须分析 CMUE 的实际决策是否集中在确实有收益的样本上，而不是仅仅学到一个接近常数的权重。

---

## 7. Phase III：实现创新点 2——Predictive Latent Identity Association

模块建议命名：

**Predictive Latent Identity Memory，PLIM**

核心思想：

根据鱼的历史观测，在隐特征空间预测当前或下一时刻的身份相关表示，用于与当前检测候选进行关联。

### 7.1 模型结构

首先使用轻量方案：

- RGB / IR 目标级编码器。
- Track-level temporal encoder。
- 轻量 GRU 或 Temporal Transformer predictor。
- 历史身份原型。
- 预测身份特征。
- Candidate identity scoring head。

可以先使用 4、8、16 帧历史长度进行开发集对照。

不直接移植完整 V-JEPA 2 主干。

### 7.2 训练方式

从训练视频构造有真实 ID 的时间序列。

输入历史帧的身份特征，预测后续时刻的潜在特征。

训练时，可以用后续真实帧通过停止梯度或 EMA teacher 生成监督目标。

但在线推理阶段必须只使用当前及过去观测。

可选损失：

- Latent cosine prediction。
- Identity contrastive loss。
- Hard-negative ranking loss。
- Association classification loss。

重点防止模型预测出模糊的平均鱼特征。

如果预测出来的特征对所有鱼几乎一样，应通过身份区分损失和困难负样本修正，而不是只盲目增加预测器深度。

### 7.3 与 MOTIP 思想结合

参考 MOTIP 的身份解码设计，但不要求完整重做 DETR 检测器。

在现有 V3 检测框和轨迹集合上，构造当前候选相对于历史轨迹的身份匹配分数。

包含 unmatched 和 newborn 选择。

如果传统 Hungarian 分配已经足够，先保留它。

只有在开发集显示直接 ID 解码具有实际收益时，才将可学习 ID Decoder 作为最终关联器。

### 7.4 记忆更新

严格区分：

A. 检测与轨迹几何更新。

B. 身份记忆写入。

一个检测成功匹配，不意味着其外观特征一定可靠。

针对遮挡、密集交叉、低置信匹配和模态冲突，设计延迟写入、拒绝写入或降低权重的机制。

实验中必须比较：

- Last feature。
- EMA feature。
- FIFO history。
- Short/long-term memory。
- Predictive memory。
- Predictive memory + reliability write policy。

注意区分“预测式身份关联”的收益与“安全记忆更新”的收益。

### 7.5 关键对照

- ByteTrack 原始运动预测。
- 普通 Kalman/轨迹外推。
- EMA 身份特征。
- GRU 时序建模。
- 参数量匹配的 Temporal Transformer。
- 仅预测 RGB。
- 仅预测 IR。
- RGB+IR 预测。
- PLIM。
- PLIM + CMUE。

重点测试交叉遮挡事件后的身份恢复能力。

---

## 8. Phase IV：实现创新点 3——Interaction-aware Association

这是可选增强，不应早于前两个核心创新的有效性验证。

研究密集鱼群中的局部交互关系。

参考 TIDE、HybridSORT 的关联机制，以及最新多目标协同运动建模思想。

不要直接引入庞大的图网络。

先实现局部轻量交互模型：

- 根据空间邻近、速度与候选歧义建立局部邻域。
- 计算轨迹间的相对运动特征。
- 为当前身份决策产生辅助约束。
- 只在候选分数接近或交叉遮挡时调用。
- 当关系预测不可信时回退到个体身份关联。

必须比较：

- 无交互模块。
- 普通邻域运动平均。
- 固定几何规则。
- 轻量邻域注意力。
- 自适应局部关系。
- 完整模型。

重点观察：

AssA、IDSW、Frag、HOTA、交叉事件 ID 恢复率和额外延迟。

如果发现鱼群关系不能稳定帮助身份恢复，可以将它保留为负结果，而不是强行纳入最终网络。

---

## 9. Phase V：整合完整在线架构

最终候选系统：

RGB + 构建 IR<br>
→ V3 双分支检测器<br>
→ RGB / IR 身份编码器<br>
→ Counterfactual Modality Utility Estimator<br>
→ Predictive Latent Identity Memory<br>
→ 歧义感知候选关联<br>
→ 条件式轨迹身份记忆更新<br>
→ 在线输出 bbox + persistent track ID

所有组件应设计为可开关的模块。

例如：

- `use_ir`
- `use_cf_utility`
- `use_latent_prediction`
- `use_safe_memory`
- `use_group_interaction`

具体配置名可根据项目代码调整。

最终程序必须保留一个能够严格退化为原始 V3 + ByteTrack 的基线模式。

### 训练策略

优先分阶段训练：

1. 冻结 V3 检测器。
2. 训练 RGB / IR ReID。
3. 训练 CMUE。
4. 训练 PLIM。
5. 验证联合关联器。
6. 仅在确实必要时，尝试低学习率联合微调身份适配层或检测器。

不允许一开始就端到端训练所有模块，因为无法确定增益或退化来自哪里。

必须保存每个阶段的 checkpoint，并支持 resume。

---

## 10. Phase VI：完整评估体系

统一使用 TrackEval。

所有可比实验必须保证：

- 相同数据集和序列。
- 相同评估代码。
- 相同 GT。
- 相同帧顺序。
- 相同检测器，除非该实验专门研究检测模块。
- 相同允许的候选范围。
- 相同或公平调优的开发集预算。

必须设置两个评估协议。

### Protocol A：相同检测框的关联能力比较

所有方法使用 V3-E3 的同一组冻结检测输出。

比较 ByteTrack、HybridSORT-ReID、BoT-SORT-ReID、RGB/IR ReID、CMUE、PLIM 和完整关联器。

这个协议主要回答：

**我们的新方法究竟有没有提升身份关联？**

### Protocol B：完整系统比较

分别使用各方法自己的完整检测与跟踪流程。

比较最终端到端性能。

这个协议回答：

**我们的系统整体是否更好？**

不能混用两个协议的指标。

### 强制报告的指标

| 类型 | 指标 |
|---|---|
| 综合跟踪 | HOTA、MOTA、IDF1 |
| HOTA 分解 | DetA、AssA、LocA |
| 检测 | DetRe、DetPr、FP、FN |
| 关联 | AssRe、AssPr |
| 身份 | IDP、IDR、IDSW |
| 轨迹完整性 | Frag、MT、ML |
| 模型成本 | Parameters、FLOPs、GPU 显存 |
| 在线效率 | FPS、单帧 latency p50/p95 |

注意：

- HOTA 使用官方阈值平均结果。
- AssA 和 DetA 必须读取官方输出。
- 不能用最终 HOTA 的平方反推最终 AssA。
- 同时报告总体结果、每视频结果和开发集结果。
- 对有希望的最终模型至少使用三个随机种子。
- 尽可能使用视频级 bootstrap 或其他合理方法报告不确定性。
- 速度测试要包含实际 ReID、预测和关联成本。
- 区分缓存检测后的关联延迟与完整端到端延迟。
- 不使用离线未来帧、视频末尾全局优化或 GT 辅助轨迹恢复。

### 核心消融表

至少覆盖：

1. V3 + ByteTrack。
2. V3 + HybridSORT-ReID。
3. V3 + BoT-SORT-ReID。
4. V3 + RGB ReID。
5. V3 + IR ReID。
6. V3 + RGB-IR ReID。
7. V3 + 普通动态融合。
8. V3 + CMUE。
9. V3 + PLIM。
10. V3 + CMUE + PLIM。
11. V3 + CMUE + PLIM + 安全记忆。
12. 完整模型 + 可选局部交互。

额外进行 leave-one-component-out 消融。

只要实验方法可用，就不能为了让我们的方法看起来领先而省略不利基线。

---

## 11. Phase VII：效果不好时的科研诊断与迭代

这一条非常重要。

**禁止第一次出现 HOTA 或 AssA 下降就直接输出 NO_GO。**

研究的基本流程必须是：

实验<br>
→ 判断性能变化<br>
→ 定位主要错误<br>
→ 提出有证据的修复假设<br>
→ 受控修改<br>
→ 重新训练或评估<br>
→ 决定是否保留修改

至少区分以下情况。

### A. DetA 不变但 AssA 下降

说明更可能是关联器的问题。

排查：

- ReID 特征不能区分不同鱼。
- 相似鱼负样本不足。
- RGB 与 IR 特征分布不一致。
- 候选匹配校准有偏。
- NULL/newborn 决策出错。
- 历史身份记忆污染。
- 轨迹丢失后错误重激活。

优先分析关联错误，不立即重新训练检测器。

### B. DetA、MOTA 和 HOTA 同时下降

排查检测输出、置信度、预处理、NMS、框坐标、输入通道顺序和检测阈值。

确定问题在检测阶段还是关联阶段。

### C. RGB+IR 低于 RGB-only

研究生成模态是否存在负迁移。

分析：

- IR 生成质量差。
- 模态分布不匹配。
- RGB/IR 配准错误。
- IR 只复制了 RGB 的无用背景信息。
- IR 特征在鱼群交叉时更不稳定。
- 门控权重没有学到条件价值。
- 训练样本的正负收益严重不平衡。

尝试模态校准、hard-negative、可靠性约束及 RGB fallback。

不要简单增加网络深度。

### D. PLIM 不如 EMA

分析：

- 潜在特征预测误差。
- 预测表征坍缩。
- 时间窗口不合适。
- 历史身份污染。
- 运动变化太剧烈。
- 训练与在线推理存在状态分布差异。
- 预测表征不具备身份区分能力。

先用真实 ID 轨迹做受控预测，再接入真实在线轨迹。

### E. 离线身份识别表现好，在线跟踪却退化

重点调查：

- GT crop 与预测框 crop 的差异。
- 在线轨迹更新错误。
- 短时间错误匹配造成的记忆污染。
- 不合理的轨迹出生和消亡逻辑。
- 在线因果约束与训练采样不一致。
- 候选召回率不足。
- 不同关联阶段之间的优先级冲突。

### F. 精度提升但 FPS 大幅下降

分析：

- 重复提取 RGB/IR 特征。
- 每帧重新编码历史数据。
- Transformer 复杂度。
- 全局候选矩阵过大。
- 频繁的 CPU/GPU 数据传输。
- 小批量特征提取效率。
- 重复运行编码器。

尝试共享特征、增量更新记忆、局部候选、批处理和轻量预测器。

### 迭代要求

每次失败都必须保存：

- 原始指标。
- 每序列指标。
- 失败样本。
- 错误关联事件。
- 具体根因假设。
- 修改内容。
- 修改后指标。
- 是否支持原假设。

优先通过训练集内部诊断定位问题，再用冻结开发协议比较修复效果。

原则上，每个核心研究假设至少允许三轮有针对性的诊断与修复尝试，但不得为了达到指标无限扩展模型搜索或反复使用测试集。

不要把“更多训练轮数”“更大的网络”“再调一下阈值”默认当成解决方案。

如果证据表明某个假设确实不成立，要明确记录负结果，调整研究设计，而不是伪造收益。

---

## 12. 研究选择与成功判定

FINAL GOAL 的成功判定必须事先定义，不允许根据最终结果随意修改。

以独立开发集上相同检测框协议的最强可复现关联基线为主要比较对象。

建议研究目标：

- HOTA 具备稳定、可复现的正向提升。
- AssA 具备稳定、可复现的正向提升。
- IDF1 有实质性改善。
- IDSW 不增加，优先显著减少。
- MOTA 不出现明显退化。
- 完整模型的在线延迟处于合理范围。
- CMUE 与 PLIM 分别具有独立消融价值。
- 生成 IR 的贡献能够由正确配对、移除及扰动实验支持。

可以把开发集 HOTA +1.0、AssA +2.0、IDF1 +2.0 个百分点作为初步努力目标，但这些不是已经实现的结果，也不是可以强行追逐的测试集阈值。

如果一个方法 HOTA 提升但 MOTA 下降，应报告完整 Pareto trade-off，分析检测和关联变化，而不是只汇报上升指标。

最终模型应该通过开发集预先冻结。

只有冻结后才进行最终测试报告。

如果历史测试集已经被用于参数优化，应明确记录污染风险，不使用“完全独立盲测”措辞。

### 不要过早结束研究

允许状态：

- `IN_PROGRESS`
- `BASELINE_REPRODUCED`
- `HYPOTHESIS_TESTED`
- `DIAGNOSIS_REQUIRED`
- `ITERATION_IN_PROGRESS`
- `VALIDATED_ON_DEV`
- `FINAL_EVALUATED`
- `GOAL_MET`
- `GOAL_NOT_MET_WITH_EVIDENCE`

不要在首次实验失败后直接标记 `NO_GO`。

但是也不能无期限宣称继续优化就一定能成功。

经过充分受控实验、合理修复和资源审计后，如果核心假设仍然没有证据支持，必须如实给出 `GOAL_NOT_MET_WITH_EVIDENCE`，说明哪些问题不可解或当前资源无法解决。

**目标是获得可信的科研成果，而不是制造一个看起来成功的实验。**

---

## 13. 必须交付的代码与研究文档

建议按现有项目风格组织，不要为了遵循文件名称破坏原有架构。

新增以下逻辑模块：

- 双模态目标级身份编码器。
- CMUE 反事实收益估计器。
- PLIM 潜在时序预测器。
- 条件式身份记忆更新器。
- 歧义感知在线关联器。
- 可插拔 RGB-only fallback。
- 跨模态训练样本构建工具。
- 视频级划分与数据泄漏检查工具。
- 统一 TrackEval 包装与完整指标统计。
- 每序列与每事件的错误诊断器。
- 自动化消融实验工具。

必须建立：

`docs/research/FINAL_GOAL.md`

`docs/research/LITERATURE_15_CODE_AUDIT.md`

`docs/research/DATA_AND_EVALUATION_PROTOCOL.md`

`docs/research/ARCHITECTURE_DESIGN.md`

`docs/research/PHASE_I_MODALITY_IDENTITY_DIAGNOSIS.md`

`docs/research/CMUE_EXPERIMENTS.md`

`docs/research/PLIM_EXPERIMENTS.md`

`docs/research/FAILURE_ANALYSIS.md`

`docs/research/ABLATION_RESULTS.md`

`docs/research/FINAL_RESEARCH_REPORT.md`

另创建实验注册表，记录：

- run_id。
- 实验假设。
- Git commit。
- 随机种子。
- 配置哈希。
- 检测权重哈希。
- ReID 权重哈希。
- 数据划分。
- 训练轮次。
- 选择 checkpoint 的开发集指标。
- 全部最终指标。
- 日志及预测输出路径。
- 实验结论。

最终提供经过验证的训练、评估、复现命令。

禁止在文档中填写未实际运行的结果。

---

## 14. 资源与执行纪律

开始时自动检测 GPU 资源、剩余存储、CPU 和内存。

优先安排：

- 不训练即可完成的基线复现。
- 身份信息诊断。
- 冻结特征实验。
- 轻量 ReID。
- CMUE。
- 轻量 PLIM。
- 最后才尝试计算昂贵的联合训练。

不要直接运行 V-JEPA 2 或 DINOv3 的完整大规模预训练。

可以使用可取得且许可允许的预训练权重。

若网络无法下载权重，先调查本地缓存和已有模型，再判断是否需要替代方案。

不能擅自购买算力、使用付费服务或覆盖其他研究项目。

尽量支持断点恢复、训练日志保存及中断后继续执行。

代码完成后要运行：

- 基本单元测试。
- 模块输入输出测试。
- 因果时间顺序测试。
- RGB fallback 一致性测试。
- 轨迹 ID 唯一性测试。
- 多模态配对一致性测试。
- NaN/Inf 检查。
- TrackEval 输出一致性测试。
- 最终推理完整视频测试。

代码和必要文档可按项目现有 Git 规范提交并推送到新研究分支；数据、权重、缓存及其他大文件保留本机。

---

## 15. 最终汇报标准

最终报告必须能够回答：

1. 现有 V3 是否被准确复现？
2. 当前真正的 AssA、DetA 和 HOTA 是多少？
3. 构建 IR 在什么情况下有效？
4. 构建 IR 在什么情况下有害？
5. CMUE 是否比普通质量门控更好？
6. 反事实监督是否真的带来增益？
7. PLIM 是否比普通 EMA、GRU 和运动预测更好？
8. 是否显著降低相似鱼交叉时的 IDSW？
9. 哪个创新点对 HOTA 的贡献最大？
10. 哪个创新点对 AssA 的贡献最大？
11. 完整模型的 MOTA 是否受到影响？
12. 是否出现 IDF1 上升但 HOTA 不上升？
13. 三个随机种子的稳定性如何？
14. 每序列的收益和退化是否一致？
15. 实际端到端 FPS、GPU 显存和延迟是多少？
16. 与 HybridSORT-ReID 等强基线是否公平？
17. 失败实验分析出了哪些真实原因？
18. 模型是否完全在线、没有使用未来帧？
19. 最终贡献是否区别于已有模态门控、预测学习和记忆模块？
20. 当前证据是否足以支持一篇具有独立方法贡献的科研论文？

最终报告必须有完整对比表、消融表、实验设置、误差案例与科研结论。

---

## 16. 现在立即执行

不要只给我一份实施建议，也不要只写任务清单。

按照以下顺序实际推进：

**第一阶段：**
环境审计、15 项文献代码审计、历史 V3 复现、全部 HOTA/AssA 指标恢复、数据划分与固定评测协议。

**第二阶段：**
固定 V3 检测框，完成 RGB/IR ReID 强基线和模态身份价值诊断。

**第三阶段：**
实现并训练 CMUE，进行反事实收益与负迁移实验。

**第四阶段：**
实现并训练 PLIM，验证预测式身份关联以及安全记忆更新。

**第五阶段：**
根据误差证据判断是否需要局部鱼群交互模块，完成在线架构整合。

**第六阶段：**
执行独立开发集比较、完整消融、三种子复现、在线延迟测量和最终冻结评估。

**第七阶段：**
整理完整论文证据、最终报告、复现命令、代码提交及 GitHub 推送记录。

每个阶段都必须有实际代码和机器生成的验证结果。

出现性能下降时，优先诊断并做有针对性的研究迭代，不要因为单次失败提前终止整个项目。

但所有科学判断必须基于真实证据，不能保证或伪造正向结果。

**请从 Phase 0 和 15 个开源项目的代码审计开始执行，并持续推进至 FINAL GOAL 得到可信验证或者经过充分诊断后形成明确、可复现的科学结论。**
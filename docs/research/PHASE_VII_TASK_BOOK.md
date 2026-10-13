# IR-MFT Phase VII：开放式科研探索与算法实现任务书

## FINAL GOAL

**EvidenceFishMOT: Learning Fine-Grained Identity Evidence and Joint Association for Online Multi-Object Fish Tracking**

你的身份是本项目的计算机视觉首席研究员、算法设计负责人、训练工程师和实验评审负责人。

本次任务不是简单运行现有模型，也不是做完几个预设实验就结束。

我要你以已经完成的 Phase I–VI 研究为基础，研究此前模型为什么没有稳定提升，并借鉴 30 篇真实开源论文，进一步构建具有独立科研贡献、稳定关联收益和合理计算成本的新方法。

### 最高目标

解决以下核心问题：

1. 当前鱼类身份特征是否具有足够的实例区分能力？
2. 为什么原始历史特征在部分检索实验中比 GRU 压缩原型更有区分力？
3. 能否从鱼体局部纹理、形状及跨帧变化中获得更稳定的身份信息？
4. Dense JEPA 是否能预测真正具有身份判别力的局部表征？
5. 能否让多条轨迹联合判断候选身份，而不是只做逐轨迹的独立相似度匹配？
6. 如何处理 NULL、新生鱼、丢失鱼、重激活及多条轨迹竞争同一候选？
7. 鱼类运动和遮挡关系是否能在外观证据不足时提供补充？
8. 如何保证上述方法能够改善 HOTA、AssA、IDF1，同时控制 MOTA、IDSW 和实际运行成本？

研究可以调整架构和技术路线，但必须有明确的实验依据。

**不要为了采用新模型而采用新模型，也不要为了保留 JEPA 而保留 JEPA。**

如果普通 Transformer 或普通局部特征已经足够强，应当通过真实消融证明复杂方法的必要性。

---

# 第一部分：项目状态及开发保护

## 1. 项目位置

GitHub：

https://github.com/LYQ1107/IR-MFT

服务器目录：

`/data3/liuyeqiang/IR-MFT`

当前已完成研究分支：

`research/recoverability-candidate-jepa-v1`

已核验提交：

`042608779d936bcd442d8c266734a6d4d47636c1`

首先检查 GitHub 远端最新提交、本地 HEAD 和工作树。

如已有更新，先读取新版本，不要强制回退。

建立新分支：

`research/evidence-fishmot-v1`

如该分支存在，先检查状态并安全继续。

## 2. 已有重要研究结果

历史 Phase V：

普通 RGB GRU + 当前匹配检测框输出，两个历史开发视频的三种子均值：

- HOTA：53.591386
- AssA：39.138938
- DetA：73.456991
- IDF1：62.700856
- MOTA：93.672112
- IDSW：87.333333

强 BoT-SORT + RGB ReID：

- HOTA：52.481818
- AssA：37.498032
- DetA：73.517605
- IDF1：57.476822
- MOTA：93.552298

Phase VI 六视频关联 OOF 参考：

- 原基线种子0：HOTA 33.4884
- A+B 的种子0：HOTA 34.3611
- 更简化的预测+固定 margin：种子1/2仅提升0.0258/0.0613 HOTA，并各增加2次IDSW。

Phase VI 最终决定：

不部署 A/B/C 候选，保留普通 GRU/current-measurement 作为强基线。

原因：

- learned risk 不稳定优于普通 margin。
- EMA teacher 不具备稳定独立收益。
- K2 多假设没有稳定身份改善。
- 直接替换几何关联分数会产生严重的 IDSW 退化。
- 身份坐标、轨迹生成器变化后，原有提升几乎消失。

### 本轮不能重复的低价值研究

不要重新大量训练：

- CMUE 常规模态效用门控。
- 旧的单原型 PLIM。
- 旧 RIS 单纯历史权重。
- 同一512维历史原型上的普通 GRU/Transformer 预测。
- 已经验证不足的 EMA 系数搜索。
- 旧 K2 多假设分数的小幅阈值搜索。
- 固定 10% 记忆拒写比例的重复搜索。

上述代码和权重全部保留，但不是本轮主线。

## 3. 重要的诊断证据

Phase VI 在 SN-015 上发现：

相同实际状态、已知正候选条件下：

- 原始历史特征均值 Rank@1：约62.36%。
- 原普通 GRU 原型 Rank@1：约35.85%。

另外，在已知存在正确历史观测的同一子集上：

- 历史均值 Rank@1：约76.99%。
- 仅选择已核验正确历史线索的事后 oracle：约83.49%。

这些只是离线身份排名结果，不是在线 HOTA，也不是可直接部署的 GT 方案。

但是这说明：

**历史数据中可能已经包含有效身份信息，而我们的时序表示或训练目标可能没有充分利用。**

首先要核验这组诊断结果，并分析为什么 GRU 离线 Rank@1 比简单均值低，但完整跟踪依然较强。

重点区分：

- 候选集合。
- 几何资格。
- 成本权重。
- 特征归一化。
- 置信度校准。
- 轨迹真实在线历史。
- 原型表达能力。
- 训练/推理分布差异。

如果不能解释这个现象，不要直接开始大型 JEPA 训练。

---

# 第二部分：30 篇开源代码深度审计

## 4. 必须调查的参考项目

先对下列30项进行准确 commit 的代码审计。

### 鱼类与跨模态

1. FishMAE
https://github.com/Wynne-cc/FishMAE

2. Towards Visual Re-Identification of Fish
https://github.com/msamdk/Fish_Re_Identification

3. SU-T
https://github.com/vranlee/SU-T

4. TIDE
https://github.com/vranlee/TIDE

5. M4FT
https://github.com/vranlee/M-4-FT

6. DM3T
https://github.com/vranlee/DM-3-T

7. TFMFT
https://github.com/vranlee/TFMFT

8. PFTrack
https://github.com/wqw123wqw/PFTrack

### 细粒度表征及视觉自监督

9. FDTA
https://github.com/Spongebobbbbbbbb/FDTA

10. TransReID
https://github.com/damo-cv/TransReID

11. CLIP-ReID
https://github.com/Syliz517/CLIP-ReID

12. DINOv3
https://github.com/facebookresearch/dinov3

13. DINOv2
https://github.com/facebookresearch/dinov2

14. I-JEPA
https://github.com/facebookresearch/ijepa

15. V-JEPA 2 / 2.1
https://github.com/facebookresearch/vjepa2

16. DINO-Tracker
https://github.com/AssafSinger94/dino-tracker

### 预测与遮挡

17. GOT-JEPA
https://github.com/chenshihfang/GOT

18. CoTracker3
https://github.com/facebookresearch/co-tracker

19. SAM 2
https://github.com/facebookresearch/sam2

20. SAMURAI
https://github.com/yangchris11/samurai

### 联合关联

21. MOTIP
https://github.com/MCG-NJU/MOTIP

22. COVTrack
https://github.com/zekunqian/COVTrack

23. TrackTrack
https://github.com/kamkyu94/TrackTrack

24. TCEI
https://github.com/1941Zpf/TCEI

25. MeMOTR
https://github.com/MCG-NJU/MeMOTR

26. MOTRv2
https://github.com/megvii-research/MOTRv2

### 运动和强基线

27. DiffMOT
https://github.com/ShuCvlab/DiffMOT

28. HybridSORT
https://github.com/ymzis69/HybridSORT

29. BoT-SORT
https://github.com/NirAharon/BoT-SORT

30. ByteTrack
https://github.com/FoundationVision/ByteTrack

### 审计要求

不是只看 README。

每篇至少调查：

- 论文年份、场景和主要贡献。
- 准确代码 commit。
- 开源许可证。
- 实际训练入口。
- 实际推理入口。
- 核心模型文件。
- 输入特征和张量维度。
- 模型输出。
- 损失函数。
- 数据集采样方式。
- 正负样本构造。
- 预训练权重来源。
- 推理时是否需要未来帧。
- 是否有真实在线版本。
- GPU / CUDA 依赖。
- 与 IR-MFT 现有代码的兼容性。
- 可直接借鉴的机制。
- 不应直接复制的内容。
- 与我们研究假设的创新重合。

请建立：

`docs/research/PHASE_VII_30_PAPERS_CODE_AUDIT.md`

`docs/research/PHASE_VII_REPOSITORY_REGISTRY.json`

每篇给出具体的源码文件路径，不接受仅有论文摘要的记录。

### 外部代码处理

参考代码放在独立、被Git忽略的位置。

建议：

`/data3/liuyeqiang/CFP-FishMOT_references/phase7`

先检查原有参考仓库是否已经存在。

优先复用已有仓库。

不要下载全部论文的模型权重和数据集。

对于有价值的关键代码，可 clone 到独立目录，记录 commit。

不允许直接把许可证不兼容的外部代码复制进 IR-MFT。

对仅有不完整训练脚本的仓库，应说明实际限制。

特别注意：

FishMAE 的部分路径和训练配置仍需适配，不能假定README原始示例可以直接运行。

FDTA 的原始Spatial Adapter依赖深度图；不能把生成深度当作真实额外传感器。

DINOv3 权重访问和许可证需要确认；如果无法合法获取可用权重，优先使用已有DINOv2等替代。

CoTracker3 必须区分在线与离线模型。

SAM 2、SAMURAI、DINO-Tracker、GOT-JEPA主要属于单目标、点轨迹或分割任务，不能直接把它们的输出当成已完成的多目标ID分配。

MOTIP、MeMOTR、FDTA是较重的端到端架构，应优先研究可移植的特征和身份关联模块。

**不需要运行全部30个项目的大型训练；需要完成全部30项代码审计，并挑选必要的代表方法实施公平对照。**

---

# 第三部分：数据与身份表征的基础研究

## 5. Phase VII-A0：建立鱼类身份特征诊断平台

冻结原始 V3 检测器及其检测候选。

冻结原普通GRU和目前最佳输出策略。

在训练视频中建立身份表示诊断数据集。

每个鱼的身份应采用：

`(video_id, physical_track_id)`

作为复合键。

不同视频中同名ID不能直接认为是同一条鱼。

### 正样本

同一物理鱼在：

- 相邻帧。
- 中等时间间隔。
- 长时间间隔。
- 遮挡前后。
- 转身前后。
- 不同尺度。
- 不同亮度。
- 非刚性形变。

### 负样本

重点建立困难负样本：

- 同帧的不同鱼。
- 距离很近的不同鱼。
- 外观高度相似的不同鱼。
- 交叉事件中的不同鱼。
- 尺度相近的不同鱼。
- 运动轨迹发生交汇的不同鱼。

不能只使用跨视频随机负样本，这类样本可能太容易。

### 两种 crop 协议

必须同时建立：

A. Training GT crop，用于监督特征诊断。

B. Actual detector crop，用于真实在线分布验证。

最终在线方法只能使用当前真实检测框。

训练时不能只用非常准确的GT框，而从不测试预测框误差。

### 基础分析

比较：

- Raw FishReID feature。
- BN-neck feature。
- Last feature。
- FIFO mean。
- EMA。
- Anchor + Recent。
- 普通 GRU prototype。
- 其他现有可复用历史原型。

指标：

- Same-ID cosine similarity。
- Different-ID cosine similarity。
- Hard-negative margin。
- Pair AUC。
- EER。
- Rank@1。
- mAP。
- 跨时间间隔 Rank@1。
- 遮挡前后 Rank@1。
- 候选缺失率。
- 不同鱼特征混淆率。

绘制按视频、密度和遮挡程度分组的结果。

这一步的最终目标：

确定当前身份信息的主要损失来源。

可能是：

- 视觉编码器无法分辨不同鱼。
- GRU 压缩损失了有效身份信息。
- 训练正负样本不合适。
- 预测框导致背景过多。
- 鱼体形变造成表征不稳定。
- 历史身份污染。
- 匹配器对好特征的使用不合理。

必须提出有证据支持的结论。

---

# 第四部分：创新方向 A——鱼类细粒度身份表征

## 6. Phase VII-A1：建立强视觉特征基线

首先比较：

A0：现有 FishReID。

A1：DINOv2-S冻结特征。

A2：DINOv3-S或适合当前资源的小型DINOv3特征。

A3：FishMAE风格的鱼类自监督编码器。

A4：TransReID局部特征。

A5：FDTA风格的身份细化Adapter。

所有模型需要统一：

- Crop数据。
- 有效候选集合。
- 特征归一化。
- 同一套检索评价。
- 开发数据划分。

首先进行 frozen feature 诊断，不需要全部从头训练。

如果某种基础特征已经更适合鱼类区分，可以在此基础上训练轻量投影层。

### FishMAE 实现任务

参考：

`mae-fish/train_mae.py`

`mae-fish/fish_reid_mae/train.py`

`mae-fish/fish_reid_mae/model.py`

`mae-fish/fish_reid_mae/sampler.py`

`mae-fish/fish_reid_mae/loss.py`

第一阶段：

采用不使用身份标签的合法训练视频鱼体裁剪进行MAE预训练。

探索合理的掩码比例、图像分辨率和训练步数。

不能直接无依据地复制原论文训练轮数。

第二阶段：

加入身份监督。

基础损失：

`L_reid = L_ce + lambda_tri * L_triplet`

可加入监督对比损失，但必须有CE-only和Triplet-only对照。

使用P×K identity-balanced sampler。

同一batch中应有困难负样本。

确保视频ID不泄漏。

第三阶段：

冻结或微调小部分编码层，对比：

- Frozen encoder。
- Linear probe。
- LoRA/Adapter。
- Partial fine-tune。
- Full fine-tune，仅在资源允许时。

### FDTA 相关实现

重点阅读：

`models/fdta/fdta.py`

`models/fdta/Temporal_Adapter.py`

`models/fdta/id_criterion.py`

`models/fdta/id_decoder.py`

借鉴：

- 身份特征细化。
- 时间上下文。
- 身份区分监督。
- 质量感知训练。

先实现适合FishReID的Identity Adapter。

不要直接移植整个Deformable DETR及深度估计分支。

比较：

- 同参数普通MLP Adapter。
- Temporal Adapter。
- Identity Adapter。
- 两种结合。
- 无质量权重。
- 相同训练数据的普通CE+Triplet。

如果普通MLP已经达到相同效果，就不能把FDTA式复杂机制称为新的有效创新。

---

## 7. Phase VII-A2：局部身份特征

目标：

避免鱼的整体特征把局部纹理差异平均掉。

设计：

`FineGrainedFishEncoder`

输入：

当前鱼的RGB crop。

输出：

- Global feature。
- K个Local patch tokens。
- Token有效性/可见性估计。
- 可选的方向和尺度信息。

可以测试K=4、8或更合理的轻量token数。

局部token可以来自ViT Patch embedding、ROI特征图或轻量卷积块。

不要假设数据中已经具有鱼头、鱼尾的精确监督。

如果只是无监督Patch token，报告中就称为局部区域，而不是有标注的解剖部位。

### 局部训练目标

可以尝试：

- Global identity CE。
- Local identity CE。
- Global-local contrastive。
- Same-ID temporal consistency。
- Hard-negative local ranking。
- Visibility-weighted matching。

优先保持损失简洁。

需要验证哪些局部区域真正贡献了身份区分。

例如：

对同一条鱼的历史和当前crop建立局部token相似度矩阵。

测试：

- Global average。
- Local token mean。
- Top-K patch similarity。
- Attention aggregation。
- Optimal matching。
- Temporal local correspondence。

如果局部对应不是稳定的，不能强行使用部位级匹配。

### 训练与推理差异

训练可以使用更丰富的局部增强。

在线推理必须只用当前及历史图像。

不能使用完整视频未来帧对当前局部token进行后验校正。

---

# 第五部分：创新方向 B——真正不同于旧PLIM的Dense JEPA

## 8. Phase VII-B：Object-Centric Dense JEPA

以前我们已经训练过普通潜在身份预测。

但当前计划要预测的是局部身份相关token，而不是一个整鱼全局向量。

参考：

I-JEPA

V-JEPA 2.1

GOT-JEPA

DINO-Tracker

CoTracker3

### B1：局部JEPA训练结构

输入：

同一轨迹的历史局部视觉特征。

例如：

`Z_{t-L}, ..., Z_{t-1}`

每帧包含若干局部token。

Student：

- Local context encoder。
- Temporal context encoder。
- Masked token predictor。
- Candidate-conditioned relation head。

Teacher：

- EMA target encoder。
- 训练期干净当前/未来观测。
- Stop-gradient target。

预测：

- 目标局部token。
- 局部关系的一致性。
- 当前候选之间具有区分性的latent表示。

注意：

不能让学生在预测某个masked token时读取其未掩码的真实值。

如果学生输入包含当前候选，可用于身份决策，但不能把当前目标token原样喂给预测器，然后把重建它算成预测成功。

时间、mask和目标使用范围必须审计。

### B2：建议的损失

初期：

`L = L_dense_pred + lambda_id * L_identity`

分别表示：

局部latent预测损失。

实例身份判别损失。

后续有证据时可以加入：

`L_temporal`

或：

`L_hard_negative`

但不要一次堆叠五六种损失。

### B3：重要的负样本

要针对真实困难事件采样：

- 同帧相似鱼。
- 交叉时的其他鱼。
- 遮挡后的错误重激活候选。
- 形态相似但物理身份不同的鱼。

不能只预测出不同鱼共享的平均身体纹理。

### B4：与普通模型的公平对照

至少比较：

- Global FishReID。
- Global + GRU。
- Local Patch Attention。
- Local Temporal Transformer。
- Local temporal contrastive。
- Dense JEPA without EMA。
- Dense JEPA with EMA。
- Dense JEPA without masked prediction。
- Dense JEPA without identity supervision。
- Dense JEPA full。

关键问题：

Dense JEPA提升的是表示的身份区分力，还是仅仅降低了潜在预测损失？

如果后者成立，但Rank@1、AssA、HOTA没有提升，应重新设计预测目标。

允许继续探索：

- 预测关系而不是原始token。
- 预测局部身份不变特征。
- 基于候选竞争的预测。
- 时序一致性蒸馏。
- 局部可见性预测。
- 只训练轻量adapter。

### B5：与CoTracker3/SAM2的关系

可以研究是否利用在线局部点轨迹或目标mask，减少鱼体转身造成的Patch位置变化。

首先做轻量局部几何对齐。

只有轻量方法无法解释局部错位，并且已有点/分割模型的时序结果确实有效时，才引入额外模型。

CoTracker3、SAM2的离线完整视频推理，不得作为当前在线MOT的隐藏未来输入。

如只在训练阶段使用离线教师，应在论文中明确说明。

---

# 第六部分：创新方向 C——联合多目标身份决策

## 9. Phase VII-C：Joint Identity Association

这是本轮需要认真实现的新研究重点。

原 GRU 主要为每条轨迹独立生成身份原型。

原 Phase VI 候选模型主要进行局部候选关联修正，尚未充分复现多目标ID预测中的完整集合竞争结构。

现在参考：

MOTIP

MeMOTR

COVTrack

TrackTrack

MOTRv2

### C1：候选集合输入

当前共有N条有效或可重激活轨迹。

共有M个检测候选。

构建：

历史轨迹特征：

`T ∈ R^{N×L×D}`

当前检测特征：

`Dets ∈ R^{M×D}`

运动和几何成本：

`C_geo ∈ R^{N×M}`

外观和局部身份成本：

`C_app ∈ R^{N×M}`

模型还应包含：

- 轨迹状态。
- 时间间隔。
- 候选有效掩码。
- 检测置信度。
- NULL / unmatched 信息。
- 新生目标信息。
- 丢失轨迹重激活状态。

### C2：联合身份网络

建议实现：

`JointFishIDDecoder`

可采用：

- Track memory encoder。
- Detection candidate encoder。
- Track-to-detection cross attention。
- Track-to-track competition。
- Detection-to-detection competition。
- Motion/geometry bias。
- NULL prediction。
- Newborn prediction。

输出：

当前检测到历史轨迹的分配分数。

可以采用：

`[M, N+1]`

其中额外类别代表当前检测未匹配旧轨迹。

对于未匹配检测，后续出生逻辑必须与NULL语义一致。

也可以采用其他合理的一对一关联表示，但需明确说明选择。

### C3：避免重复上一轮严重退化

Phase VI曾出现直接替换关联成本使IDSW达到上万次的情况。

因此第一版不要彻底移除现有几何约束。

设计两条实验路线：

路线1：

保留原 GRU / IoU 成本，仅增加有界联合关联残差。

路线2：

完整学习式联合ID解码。

分别测试。

不要把路线2的早期失败当成路线1没有价值，也不能只用几何回退制造表面提升。

### C4：训练目标

首先尝试：

- ID classification。
- NULL / newborn classification。
- Same-frame hard-negative ranking。
- Joint assignment consistency。

根据需要探索：

- Set matching loss。
- One-to-one consistency。
- Structured association loss。
- Risk-sensitive association objective。

避免直接采用没有标签支撑的复杂辅助监督。

历史ID作为训练上下文时，应随机重映射局部字典编号，防止模型只记忆训练集中的固定身份数字。

### C5：NULL训练

这是上一轮的重要缺口。

在训练视频中真实模拟：

- 当前没有正确候选。
- 当前只有错误候选。
- 目标完全被遮挡。
- 目标暂时丢失。
- 新目标进入画面。
- 同一候选被多条轨迹竞争。

NULL标签不能通过随意把unknown设为negative来增加。

需要区分：

- 已知真实缺失。
- 标注不明确。
- 候选未覆盖。
- 未知物理身份。

只有有监督依据的样本进入对应损失。

### C6：必须实现的对照

- 原 GRU + Hungarian。
- GRU + 普通MLP pair scorer。
- GRU + Candidate Transformer。
- 全集合普通 Transformer。
- MOTIP式 ID Decoder。
- 联合Transformer + geometry prior。
- 联合Transformer + local identity tokens。
- JointFishIDDecoder。
- 去除track-to-track competition。
- 去除detection-to-detection competition。
- 去除NULL。
- 去除motion bias。
- 去除temporal memory。

对MOTIP的比较应区分：

A. 移植其ID解码头到固定V3候选上的适配版。

B. 官方完整MOTIP训练系统。

不能因为一个简化适配版效果差就声称超过官方MOTIP。

如果有足够资源和合法数据，可以考虑完整复现官方基线，否则说明边界。

---

# 第七部分：创新方向 D——鱼类运动与遮挡

## 10. Phase VII-D：Motion and Occlusion-Aware Association

这一方向相对独立。

首先借鉴已有无须大规模训练的机制：

SU-T：

- UKF。
- FishIoU。
- 鱼类非线性运动。

TIDE：

- AGCIoU。
- 非刚性几何一致性。

HybridSORT：

- 运动方向。
- 置信度。
- 高度弱线索。

DiffMOT：

- 非线性运动预测。

### D1：先做轻量对照

冻结检测与RGB身份特征。

比较：

- 原Kalman。
- UKF。
- FishIoU。
- AGCIoU。
- 原GRU + FishIoU。
- 原GRU + UKF。
- 原GRU + 遮挡状态特征。

优先实现能够隔离单一因素的版本。

不能把改变检测输出框、轨迹生命周期、外观特征和运动算法的结果混成同一个创新消融。

### D2：非线性运动学习

如果数据表明非线性运动确实是显著错误来源，可以探索：

- 轻量GRU motion predictor。
- State Space Model。
- 运动方向与形变状态。
- Diffusion motion predictor。
- 局部邻居辅助预测。

但 DiffMOT 本身已研究扩散运动，直接移植不是原创。

需要明确鱼类场景特有的难点和独立改进。

### D3：遮挡感知

使用可观测量估计遮挡：

- 当前检测框重叠。
- 候选密度。
- 运动预测分歧。
- 外观冲突。
- 可见局部区域数量。

如果使用分割mask，必须记录来源及计算成本。

不允许在推理时使用未来帧生成遮挡标签。

---

# 第八部分：模块之间如何组合

## 11. 不允许一开始全部堆叠

本轮有四个独立方向：

A：Fine-Grained Fish ReID

B：Dense JEPA

C：Joint Identity Decoder

D：Motion/Occlusion

实验顺序建议：

先验证A。

C可以在已有FishReID上独立开发，不必等待A完全成功。

B在基本局部token接口稳定后开展。

D先做轻量无训练对照。

随后再验证模块互补性。

### 必须测试的组合

基础：

B0：V3 + 现有RGB ReID + 普通GRU/current-measurement。

A：细粒度ReID + 原GRU。

B：Dense JEPA + 对应普通关联器。

C：Joint Decoder + 原FishReID。

D：原GRU + 运动/遮挡增强。

组合：

A+C。

A+B。

B+C。

A+B+C。

A+D。

A+C+D。

其他组合根据实际独立收益决定。

不要机械执行无意义的所有全组合。

只有当组合能够回答明确的互补性问题时，才扩大实验。

如果普通ReID已经很好，Dense JEPA没有额外收益，可以不保留B。

如果Joint Decoder不如普通Hungarian，可以研究原因或替换C。

如果简单FishIoU已经解决部分运动问题，不强制使用复杂扩散模型。

最终架构以真实证据为准。

---

# 第九部分：完整实验协议

## 12. 三层评测

### Protocol A：身份表征检索

使用统一的真实候选。

指标：

- Rank@1。
- Rank@5。
- mAP。
- Pair AUC。
- Hard-negative accuracy。
- 跨时间间隔检索。
- 遮挡前后检索。
- GT crop与detector crop的性能差异。

必须分别记录：

- Clean history。
- Actual accepted history。
- Corrupted history。

### Protocol B：冻结检测框的在线MOT

所有方法使用相同V3冻结检测输出。

保持相同的：

- 视频。
- 帧顺序。
- GT。
- 输入检测。
- 评测代码。
- 输出框协议。
- 合法候选定义。

主要报告：

HOTA、AssA、DetA、LocA。

IDF1、IDP、IDR。

MOTA、IDSW。

FP、FN、Frag。

AssRe、AssPr。

DetRe、DetPr。

MT、ML。

同时提供每个视频的完整指标。

不允许只看HOTA。

### Protocol C：完整严格在线系统

从视频当前帧开始：

RGB decode。

因果IR构建。

V3检测。

NMS。

目标裁剪。

身份特征提取。

时序和联合关联。

生命周期更新。

结果输出。

测量：

- Full FPS。
- p50 latency。
- p95 latency。
- GPU显存。
- 模型参数量。
- Countable FLOPs。
- 新模块额外推理时间。

完整视频必须从头到尾执行。

不能只用缓存关联速度报告端到端FPS。

---

## 13. 数据划分与泛化

当前原始MFT26IR数据包含有限训练视频。

V3检测器已见过部分历史开发视频。

历史七视频测试集曾用于调参。

这些事实必须始终在报告中披露。

本轮新身份编码器和联合关联器，优先采用视频级外层留一验证、内层校准。

重要：

**新的ReID编码器也必须排除外层被评估视频，才能把结果称为该编码器的真正视频级泛化。**

如果冻结了曾经见过该视频的旧编码器，只能称为条件关联验证。

不要把不同训练视频集合的HOTA直接相减。

每个方法必须有对应同种子、同数据、同检测输入基线。

### 外部数据

调查公开鱼类跟踪或ReID数据。

优先检查：

- MFT25。
- MFT-Edge。
- BrackishMOT-M4FT。
- FishMAE使用的数据。
- 其他可访问的鱼类身份数据。

但外部数据必须核验：

- 数据许可。
- 是否有真实跨帧身份。
- 与现有数据是否有视频重叠。
- 训练、验证、测试划分。
- 是否含独立真实红外。
- 标注是否连续。

不能把同一个视频重新裁剪后当成独立测试集。

无合法下载权限时不绕过访问限制。

不使用机密或未经授权的数据。

---

# 第十部分：真实科研探索机制

## 14. 不设置机械NO_GO次数

本轮特别要求：

**不要完成三组实验或者三个epoch不好就结束。**

你应当主动根据证据提出下一步方案。

但不能无限重复没有信息价值的实验。

失败后按照以下逻辑：

### 情况A：Frozen DINOv3/FishMAE没有优于ResNet

分析：

- 鱼体裁剪分辨率。
- Patch尺寸。
- 水下域差异。
- 背景占比。
- 鱼体尺度。
- 归一化与颜色预处理。
- 是否需要轻量身份适配。
- 正负样本难度。

再决定微调、局部特征或更换基础模型。

### 情况B：局部ReID提高Rank@1但HOTA下降

分析：

- 当前候选成本量纲。
- 特征距离分布。
- GRU坐标系不匹配。
- Hungarian排序。
- 几何门控。
- Track更新机制。
- 背景与Patch污染。

不能只继续训练ReID。

### 情况C：Dense JEPA降低latent loss但身份没有提升

分析：

- 目标token是否具有身份区分力。
- Mask任务是否过于容易。
- 是否仅学习姿态而非身份。
- 预测目标是否与候选关联脱节。
- Teacher是否产生共享特征坍缩。
- 训练/推理是否存在可见token泄漏。

然后考虑候选条件化预测或局部身份损失。

### 情况D：Joint Decoder大量错误出生

分析：

- NULL训练比例。
- 不同候选之间的竞争。
- 当前帧一对一约束。
- 历史ID编码。
- 轨迹状态。
- 不存在正确候选时的行为。
- 新生和重激活冲突。

不能只使用当前帧分类准确率判断整个Decoder成功。

### 情况E：联合模型效果不稳定

分析：

- 不同鱼种。
- 不同密度。
- 不同视频背景。
- 摄像机运动。
- 非线性鱼体运动。
- 特征分布差异。
- 在线历史污染。

应进行按场景和错误类别的归因，而不是只选择最有利的视频。

---

## 15. 实验注册表

建立：

`docs/research/PHASE_VII_EXPERIMENT_REGISTRY.json`

`docs/research/PHASE_VII_HYPOTHESIS_LEDGER.md`

每项记录：

- 实验ID。
- 科学假设。
- 改动原因。
- 具体网络结构。
- 输入和输出。
- 训练损失。
- 数据版本。
- 训练视频。
- 评估视频。
- 参数量。
- 随机种子。
- 运行命令。
- Git commit。
- 权重SHA。
- 源码SHA。
- 真实指标。
- 失败类型。
- 下一步实验建议。

同一次执行发生的训练失败、环境故障或中断，也要记录。

不能将重复评测计数伪装成新的独立训练。

### 自主研究要求

每次完成一项有意义的实验后，自动回答：

“这项实验排除了哪些可能的解释？”

“还剩下哪些科学问题？”

“继续训练同样模型是否值得？”

“有没有更小、更明确的实验可以帮助判断？”

然后继续执行最有信息价值的实验。

没有必要每个阶段都等待用户重新授权普通科研任务。

但是涉及额外付费、未经授权的数据下载、危险删除或破坏其他用户环境时，必须遵守相应权限和安全限制。

---

# 第十一部分：计算成本与性能优化

## 16. 不能忽略1.2FPS的瓶颈

此前完整管线约1.2FPS。

本轮先建立性能Profiler。

区分：

- RGB解码。
- 因果IR生成。
- V3检测。
- NMS。
- GPU传输。
- Crop提取。
- RGB编码器。
- Patch token提取。
- GRU。
- Dense JEPA。
- Joint Decoder。
- Hungarian。
- 输出。
- 进程通信。

特别检查历史实现中：

V3检测批次复制到8的真实开销。

FishReID裁剪编码批次补齐128的真实开销。

这些可能是主要系统开销，但不能凭直觉直接删除。

因为原来的cuDNN/CUDA批次形状曾影响检测结果。

优化时要求：

- 保留原始精度协议。
- 建立相同输入输出对照。
- 有数值差异必须记录。
- 测试不同批次引入的误差。
- 新旧方法采用可比的测速方式。

允许探索：

- 动态Batch。
- 共享特征。
- 减少重复编码。
- 更轻量ReID。
- 混合精度。
- 特征缓存的合法增量复用。
- 异步流水线，但不允许改变严格因果结果。
- 模型导出或推理优化。

效率优化不能悄悄修改实验基线，从而让算法性能比较失去公平性。

---

# 第十二部分：算力与数据安全

## 17. 资源规范

首先检查GPU数量、型号、负载和剩余空间。

当前上一次报告显示约163GiB空闲，但必须重新检测实际值。

默认至少保留40GiB空间。

不超过四张可用物理GPU，不能抢占其他用户进程。

优先使用现有：

- V3权重。
- FishReID权重。
- 训练数据。
- 检测缓存。
- TrackEval。
- Conda环境。
- 历史实验注册表。

外部基础模型优先小模型。

不要直接开始DINOv3大型模型或V-JEPA2.1巨型模型的完整预训练。

可以先使用已有或合法下载的冻结权重。

训练支持断点恢复。

所有新模型都保存：

- 网络权重。
- 优化器状态。
- 学习率调度状态。
- 随机种子和RNG。
- 数据协议。
- 配置哈希。

不覆盖旧资产。

不把数据、权重、特征缓存上传GitHub。

---

# 第十三部分：代码与科研交付

## 18. 建议的新模块

根据实际工程结构组织，可调整文件名。

`cfp/finegrained_reid.py`

`cfp/fish_patch_encoder.py`

`cfp/fish_identity_adapter.py`

`cfp/patch_association.py`

`cfp/object_dense_jepa.py`

`cfp/dense_jepa_loss.py`

`cfp/joint_identity_decoder.py`

`cfp/joint_assignment.py`

`cfp/occlusion_motion.py`

`cfp/fish_motion.py`

`tools/cfp_train_fish_reid.py`

`tools/cfp_train_dense_jepa.py`

`tools/cfp_train_joint_id.py`

`tools/cfp_eval_identity.py`

`tools/cfp_eval_phase_vii.py`

`tools/cfp_profile_online.py`

需要增加对应的：

- 数据集。
- 配置。
- 测试。
- 消融。
- 实验注册。
- 复现脚本。

尽量复用现有代码，而不是重新构造完整项目。

### 必须通过的测试

- 关闭新增模块时，复现强GRU结果。
- 数据视频划分无泄漏。
- 同一鱼正样本标签正确。
- Hard negative不是同一物理鱼。
- Patch token mask正确。
- JEPA masked target不泄漏。
- Teacher不参与在线推理。
- NULL/newborn没有标签混淆。
- 不同轨迹不能输出同一候选。
- ID没有跨视频错误复用。
- 历史和当前时间顺序正确。
- 模块可处理空历史/空检测。
- 全部输出有限且数值稳定。
- 同一配置可复现。
- 官方TrackEval指标一致。

---

## 19. 必须生成的文档

`docs/research/PHASE_VII_FINAL_GOAL.md`

`docs/research/PHASE_VII_30_PAPERS_CODE_AUDIT.md`

`docs/research/PHASE_VII_DATA_PROTOCOL.md`

`docs/research/PHASE_VII_BASELINE_AUDIT.md`

`docs/research/PHASE_VII_REID_COMPARISON.md`

`docs/research/PHASE_VII_LOCAL_IDENTITY_EVIDENCE.md`

`docs/research/PHASE_VII_DENSE_JEPA_EXPERIMENTS.md`

`docs/research/PHASE_VII_JOINT_ASSOCIATION_EXPERIMENTS.md`

`docs/research/PHASE_VII_MOTION_EXPERIMENTS.md`

`docs/research/PHASE_VII_ABLATIONS.md`

`docs/research/PHASE_VII_FAILURE_ANALYSIS.md`

`docs/research/PHASE_VII_ONLINE_RUNTIME.md`

`docs/research/PHASE_VII_FINAL_RESEARCH_REPORT.md`

若实验阶段正在执行，文档应标明IN_PROGRESS，不得提前写成完成。

---

# 第十四部分：GitHub提交与持续执行

研究分支：

`research/evidence-fishmot-v1`

每完成重要研究阶段：

1. 运行测试。
2. 更新实验记录。
3. 保存小型正式结果。
4. 提交源代码。
5. 推送GitHub。
6. 核验远端HEAD。
7. 保存提交哈希和日志。

不提交大型数据和模型权重。

如发生网络连接问题，应保护本地成果，恢复后再核验推送。

不能未经核验声称GitHub与本地一致。

---

# 第十五部分：论文创新性与最终科学判断

## 20. 研究目标不等于强制拼装30篇方法

最终论文尽量形成两个有关系的核心方法贡献。

候选贡献一：

**鱼类细粒度、时间稳定的身份信息建模。**

候选贡献二：

**面向多目标竞争的联合身份关联。**

Dense JEPA可以成为贡献一的具体技术机制。

运动与遮挡可以作为贡献二的补充，或者成为独立创新。

但是是否保留它们，必须由消融结果决定。

若最终普通DINOv3 + GRU明显优于其他方法，这属于一个很有价值的强基线，但不应直接包装成原创网络。

若Joint Decoder有独立收益，需要证明不是单纯因为参数更多或候选扩大。

若Dense JEPA有独立收益，需要证明EMA预测、局部时序监督比普通Patch Transformer更好。

若没有模块达到可靠独立改善，不允许写出虚假的方法贡献。

但不能因前几次失败就机械停止所有探索。

应允许根据真实证据调整主线。

### 最终需要回答

1. 当前FishReID的主要缺陷是什么？
2. 新ReID改善了多少身份检索？
3. 新ReID是否改善真实在线HOTA/AssA？
4. 局部Patch是否优于单一全局特征？
5. Dense JEPA是否真正增加身份区分信息？
6. Dense JEPA是否超越普通时序Transformer？
7. 新Joint Decoder能否处理NULL和多轨迹竞争？
8. 是否超过同输入普通Hungarian？
9. 运动/遮挡模块是否有独立贡献？
10. HOTA、AssA、IDF1、MOTA、IDSW是否稳定改善？
11. 三种子变化如何？
12. 跨视频表现是否一致？
13. 对未知场景的泛化证据是什么？
14. 运行速度和实际成本是多少？
15. 哪些模块是真正必要的？
16. 哪些负结果有科学价值？
17. 30篇参考中哪些方法实际迁移成功？
18. 哪些已有工作可能削弱我们的创新主张？
19. 最终是否形成独立方法论文的可靠证据？
20. 最终推荐使用哪套架构，为什么？

---

# 第十六部分：立即执行

请不要只向我输出任务计划。

从现在开始实际执行：

**Step 1**

核验最新GitHub和服务器环境，保护已有Phase I–VI结果，创建Phase VII研究分支。

**Step 2**

完成30篇真实开源论文的源码级审计，记录训练实现和与本项目的可移植性。

**Step 3**

复现并分析当前FishReID与GRU的身份排名差异，找到表征瓶颈。

**Step 4**

实施DINOv2/DINOv3/FishMAE及局部ReID的受控实验，训练必要的小型适配网络。

**Step 5**

将有效身份表征接入原GRU和固定检测关联协议，运行完整TrackEval。

**Step 6**

实现真实的联合多目标身份解码，比较普通Transformer、MOTIP式Decoder、原GRU+Hungarian。

**Step 7**

实施Object-Centric Dense JEPA，验证其与普通局部时序网络的区别。

**Step 8**

独立测试鱼类运动、几何和遮挡机制。

**Step 9**

根据实际结果选择性组合，而不是把所有模块强制拼装。

**Step 10**

完成三种子、完整指标、跨视频分析、消融和严格在线测量。

**Step 11**

根据失败诊断自主设计下一轮研究假设，并继续实施有信息价值的实验。

**Step 12**

持续提交代码和实验记录至GitHub，最终形成完整科学报告。

### 最重要的执行纪律

- 不要机械地做完固定三轮实验就宣布整个项目NO_GO。
- 不要无限重复已经证明没有信息价值的参数搜索。
- 不要先决定JEPA一定成功，再为了证明它而选择数据。
- 不要用历史测试集寻找最佳配置。
- 不要因为单个种子表现好就宣布突破。
- 不要为了保持多模态名称而保留无效的IR身份分支。
- 不要将官方不同数据集的指标直接与本项目数字比较。
- 不要将外部论文的现有机制直接称为我们首次提出。
- 不要牺牲严格在线性来制造离线身份恢复收益。
- 不要只看HOTA，必须完整比较AssA、DetA、IDF1、MOTA和IDSW。
- 不要只做缓存实验，必须完成有代表性的完整视频实际运行。

**最终研究目标是找到一个能够稳定增强鱼类身份信息、改善多目标联合身份决策，并在完整在线MOT中获得可复现收益的方法。**

现在开始实际执行Phase VII。

首先完成环境与30篇代码审计，然后推进身份表征诊断及第一轮受控ReID实验。后续根据科学证据继续开展联合身份推理、Dense JEPA、运动和遮挡研究，直到形成可靠的最终科研结论。
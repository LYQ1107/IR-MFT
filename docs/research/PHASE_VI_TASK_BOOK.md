# IR-MFT Phase VI：自主科研探索任务书

## Recoverability-Aware Candidate Reasoning and Predictive Identity Learning for Online Multi-Object Fish Tracking

### 一、你的角色与最终目标

你现在担任本项目的首席计算机视觉研究员、算法架构师、PyTorch 工程师、实验负责人和论文审稿人。

我授权你在现有 IR-MFT 项目基础上开展一轮具有较高自主性的科研探索。

**本轮最重要的要求：不要再采用“预先规定三轮实验，未超过某个指标就立即 NO_GO”的工作方式。**

我要的是持续、严谨、具有科研探索性的工作，而不是机械完成几个配置表。

你可以根据真实实验结果：

- 主动提出新的研究假设。
- 设计新的损失函数和训练策略。
- 调整模型输入、输出及内部结构。
- 改变模块间的信息交互方式。
- 扩大或缩小候选集合。
- 引入新的合理基线。
- 修改无效的创新设计。
- 对有价值的局部结果继续深入研究。
- 探索新的 JEPA、身份关联与多假设决策方案。
- 发现现有方向不可行时，提出有根据的新技术路径。

但所有探索必须遵守科学实验规范，不能通过反复调测试集、遗漏不利结果或者选择单个幸运种子制造提升。

### FINAL GOAL

**Learning When and How to Recover Identity: Recoverability-Aware Predictive Association for Online Multi-Object Tracking**

最终希望实现一个严格在线的鱼类 MOT 框架，能够：

1. 判断当前身份错误是否具有可观测的恢复条件。
2. 识别哪些历史身份信息仍然可靠。
3. 利用当前候选集合与历史轨迹进行候选条件化身份推理。
4. 学习预测能够区分不同候选身份的潜在表示。
5. 在鱼类交叉、遮挡和重激活时管理多种可能的身份解释。
6. 在不损害轨迹输出质量的前提下，提高长期身份一致性。
7. 保证完整方法符合严格在线和可复现实验要求。

希望实现的结果：

- 在相同检测输入下，HOTA 稳定超过现有最强普通基线。
- AssA 和 IDF1 获得可重复、可解释的改善。
- IDSW 尽量减少。
- MOTA 和 DetA 不发生无法解释的大幅退化。
- 每个核心新机制都有独立消融证据。
- 模型具备明确的计算成本和完整在线运行结果。
- 最终形成有机会支撑 CVPR 等计算机视觉会议投稿的研究成果。

**这些是努力方向，不是保证一定达到的指标，也不是允许通过数据泄漏实现的硬门槛。**

---

## 二、项目现状与代码入口

项目目录：

`/data3/liuyeqiang/IR-MFT`

GitHub：

https://github.com/LYQ1107/IR-MFT

当前研究分支：

`research/identity-repair-jepa-v1`

已验证提交：

`00ae0e8f1d75b7853edfbe42d4324bac2784ff8c`

首先检查远端和本地是否有更新，不得盲目回退最新代码。

创建新研究分支：

`research/recoverability-candidate-jepa-v1`

如果已经存在同名分支，应先检查工作树和提交关系，再安全继续，不得强制覆盖。

### 当前最重要的普通强基线

V3 RGB-IR 检测器 + RGB ReID + 普通 GRU + 当前匹配检测框输出。

两视频开发集三种子结果：

| 指标 | 当前基线 |
|---|---:|
| HOTA | 53.591 |
| AssA | 39.139 |
| DetA | 73.457 |
| IDF1 | 62.701 |
| MOTA | 93.672 |
| IDSW | 87.333 |

另外保留：

BoT-SORT + RGB ReID：

- HOTA 52.482
- AssA 37.498
- DetA 73.518
- IDF1 57.477
- MOTA 93.552

任何新方法都必须与这两类强基线进行比较。

不能重新使用旧的 HOTA 51.885 GRU 作为唯一基线，从而夸大新方法的改善。

### 必须阅读的历史报告

- `docs/research/PHASE_V_FINAL_RESEARCH_REPORT.md`
- `docs/research/LIFECYCLE_CROSSOVER_RESULTS.md`
- `docs/research/IDENTITY_REPAIR_EXPERIMENTS.md`
- `docs/research/JEPA_TRAINING_RESULTS.md`
- `docs/research/JEPA_FAILURE_ANALYSIS.md`
- `docs/research/JEPA_CASE_STUDIES.md`
- `docs/research/PHASE_V_RELATED_WORK.md`
- `docs/research/PHASE_V_STATE.json`

### 现有可复用代码

- `cfp/association.py`
- `cfp/predictive_association.py`
- `cfp/fish_jepa.py`
- `cfp/recoverable_identity.py`
- `cfp/recovery_supervision.py`
- `cfp/recovery_training_data.py`
- `cfp/lifecycle_crossover.py`
- `cfp/bot_lifecycle.py`
- `cfp/history.py`
- `cfp/online.py`
- `cfp/jepa_runtime.py`
- `tools/cfp_run_jepa_online.py`
- `tools/cfp_run_jepa_training.py`
- `tools/cfp_run_recovery_campaign.py`

注意：

现有 `PredictiveAssociationTracker` 主要利用历史轨迹生成单个身份原型，再计算与当前检测特征的距离。

现有 `FishJEPAStudent` 主要预测过去身份表征对应的潜在身份状态。

现有 `RecoverableIdentityState` 主要依据历史 token 的可靠性计算加权身份原型。

**本轮要探索的是从“单个历史原型预测”转向“候选集合条件化的身份推断与恢复决策”。**

不能简单重命名旧类就称为完成新创新。

---

## 三、本轮科研原则：开放探索，而不是固定失败门槛

请严格区分两类规则。

### A. 必须遵守的工程和科学约束

- 禁止 GT 或未来帧进入在线推理。
- 禁止在已经用于调参的历史测试集上继续搜索。
- 禁止覆盖历史有效权重、日志和预测。
- 禁止伪造实验、补写不存在的指标或删除负结果。
- 禁止只展示最优单种子。
- 禁止为不同开发视频设置专属阈值制造提升。
- 禁止在模型训练和验证中混淆同一视频的相邻帧。
- 禁止无依据地宣称合成 IR 带来了独立传感器信息。
- 禁止破坏其他用户的 GPU 任务和服务器环境。

### B. 不再设置机械式科研终止门槛

以下情况都**不是**项目级 NO_GO：

- 第一次训练 HOTA 下降。
- 某种损失函数无效。
- 第一版 JEPA 不如 GRU。
- 某个视频改善而另一个视频退化。
- 模态融合没有收益。
- 单个实验出现表示坍缩。
- 多假设关联第一版发生过多 IDSW。
- 完成三轮针对性训练但尚未找到有效方案。

遇到这些情况，先回答：

**为什么失败？失败揭示了什么新的研究问题？接下来哪项实验可以区分不同解释？**

然后根据证据继续探索。

没有必要把所有可能的实验提前列死，也不限制只能使用本文列出的模型结构。

但是，自主探索必须形成明确的科研循环：

`提出假设 → 实施最小有效实验 → 分析证据 → 保留或修正假设 → 扩展验证`

如果某个研究分支已经有充分负证据，可以关闭该分支，同时转向更有依据的方向。

不能无目的地无限增加网络参数、训练轮数或超参数搜索。

**项目级最终判断应基于整体科学证据，而不是固定次数的失败。**

---

# 四、Phase VI-0：冻结强基线与建立可靠实验平台

首先创建新的研究目标文件：

`docs/research/PHASE_VI_FINAL_GOAL.md`

明确记录：

- 新科研问题。
- 已知负结果。
- 保留的最强基线。
- 新研究假设。
- 数据划分和已有验证集暴露。
- 科学成功条件。
- 可解释的失败状态。
- 资源使用规范。

### 0.1 基线保护

冻结：

- V3 检测器。
- 当前因果构建 IR 流程。
- FishReID 编码器。
- 普通 RGB GRU。
- 当前检测框输出策略。
- BoT-SORT。
- 官方 TrackEval 配置。

通过配置哈希和权重 SHA256 验证原资产。

关闭所有新模块时，必须能恢复相同基线输出。

### 0.2 统一评测接口

新增模块统一输入：

- 历史轨迹状态。
- 当前候选检测框。
- 当前候选特征。
- 运动预测。
- IoU / 匹配距离。
- 检测置信度。
- 历史观测时间。
- 已知在线轨迹状态。

统一输出：

- 关联分数矩阵。
- NULL / unmatched 决策。
- 可选的身份恢复建议。
- 可选的多假设信息。
- 记忆写入建议。
- 下一时刻内部轨迹状态。

在线模块不得返回 GT 信息或使用未来帧。

任何新模块都要支持完全关闭，使代码能精确退回原来的普通 GRU 跟踪系统。

### 0.3 新旧协议区分

保留已冻结 Phase V 协议和历史缓存。

另建 Phase VI 协议，记录其父协议、输入哈希和所有变化。

历史 BT-005 / SN-013 已经多次用于方法分析，不是新的盲测集。

优先使用训练视频内部的视频级交叉验证和校准集探索新结构。

如有合法且未参与模型选择的外部鱼类 MOT 数据集，应调查其授权、数据重叠、身份标注和模态属性，建立独立最终验证。

如无法获得新的独立数据，应如实报告泛化局限，不能把多次开发集对比称为独立测试成功。

---

# 五、创新点 A：可恢复性感知的身份决策

## Recoverability-Aware Identity Association

### 5.1 核心科学问题

此前 RIS 失败说明：

“历史中包含更多正确观测”并不意味着在线跟踪表现一定改善。

因此我们不再仅学习：

“哪个历史 token 是可靠的？”

而是学习：

**给定当前候选集合及过去身份状态，是否存在足够证据做出正确身份决定？**

必须区分：

1. 正确目标根本没有被检测到。
2. 正确目标被检测到了，却被候选门控排除。
3. 正确候选存在，但所有历史身份特征都不可靠。
4. 正确候选存在，历史中也有可靠信息，但模型匹配错误。
5. 候选证据不足，错误地强制匹配。
6. 正确匹配之后，又发生了错误的记忆写入。

这些情况需要不同处理机制。

### 5.2 首先建立可恢复性分析器

建议新增：

`tools/cfp_analyze_recoverability.py`

对已经保存的真实在线错误事件进行离线分析。

GT 仅用于事后分类。

至少统计：

- Detection candidate coverage。
- Geometry candidate coverage。
- Appearance candidate coverage。
- Union candidate coverage。
- Candidate top-K recall。
- 正确候选被拒绝的比例。
- 真实身份历史中的已知正确线索比例。
- 当前候选的身份可分离程度。
- Top1/Top2 margin。
- 高置信错误关联。
- NULL / newborn 错误。
- 重激活错误。
- 可由候选重新排序纠正的错误。
- 必须改变候选集合才能纠正的错误。

重点生成二维及分层分析：

`正确候选是否存在 × 可信身份线索是否存在`

但“可信身份线索”必须明确区分事后 GT 诊断和推理时可观测的可靠性估计。

可以在训练集上使用 GT 进行诊断性 oracle 干预，估计固定检测候选条件下的可纠错空间。

这种 oracle 仅作为分析参考，不允许直接使用其结果作为线上方法，也不能声称是无条件的信息论性能上限。

### 5.3 设计 Recoverability Estimator

建议新增：

`cfp/recoverability.py`

输入至少包括：

- GRU 历史身份状态。
- 当前候选 RGB 表征。
- 轨迹与候选之间的相似度。
- 候选排序 margin。
- 候选分布熵。
- 运动残差。
- IoU。
- 轨迹 age / gap。
- 历史特征一致性。
- 历史多原型之间的冲突。
- 检测置信度。
- 匹配阶段。
- 可观测的遮挡与拥挤程度。

首先训练一个小型风险估计器，而不是重新设计大型网络。

它可以预测：

- 当前候选身份匹配的可信度。
- 接受匹配造成错误的风险。
- 延迟确认身份的风险。
- 当前观测写入记忆的风险。

但所有标签必须来自训练数据或训练集 OOF 结果。

必须把未知身份标签与已知错误标签分开。

不得再次出现上一轮 RIS 的错误：把 unknown 当成 wrong。

### 5.4 不再只输出二分类

允许探索以下在线动作：

- Accept：接受当前身份匹配。
- Keep unmatched：保留旧轨迹但不强行关联。
- Provisional：仅保留暂时解释，不立即污染稳定身份记忆。
- Reactivate：依据可靠证据重新激活旧身份。
- Newborn：建立新轨迹。
- Fallback：使用原普通 GRU / 运动规则。

初始版本可以只实现可靠匹配与保守回退。

如果数据证明二分类不够，再扩展多动作决策。

### 5.5 训练和实验

尝试：

- 普通交叉熵。
- NULL-aware candidate classification。
- Pairwise ranking。
- Selective risk prediction。
- Risk-coverage calibration。
- Confidence-weighted identity association。

训练和评价不应只看 AUC。

必须记录：

- Risk-coverage 曲线。
- 正确接受率。
- 错误接受率。
- 纠错成功率。
- 误拒绝率。
- Calibration error。
- N01：原本错误、现在正确。
- N10：原本正确、现在错误。
- 净纠错 N01−N10。
- HOTA、AssA、IDF1、IDSW。

### 5.6 必须做的对照

- 原普通 GRU。
- GRU + 固定置信阈值。
- GRU + 普通可学习阈值。
- GRU + margin gate。
- GRU + entropy gate。
- 同输入 MLP 风险预测。
- Recoverability Estimator。
- 移除历史证据。
- 移除候选上下文。
- 移除 NULL 决策。
- 不使用风险校准。
- 容量匹配随机初始化控制。

目标是证明模型学到了真正有用的可恢复性信息，而不是重新学习一个普通匹配阈值。

### 5.7 若效果不好，允许继续探索

例如：

如果风险 AUC 很高，但在线 HOTA 不变，应调查风险是否没有改变最终 Hungarian 分配。

如果正确候选频繁被几何门控排除，应研究候选覆盖，而不是只增强身份编码器。

如果拒绝匹配减少 IDSW 但增加 FN，应进一步设计生命周期协同决策。

如果模型不能可靠区分可恢复与不可恢复状态，可以探索集合级不确定性建模、校准后的置信区域或显式候选冲突推理。

不要机械地训练三次就停止。

---

# 六、创新点 B：候选条件化 JEPA

## Candidate-Conditioned Joint-Embedding Predictive Association

### 6.1 为什么重新设计 JEPA？

上一轮 Fish-JEPA 已经真实实现了：

- Student encoder。
- EMA target encoder。
- Latent prediction。
- Masked history。
- History corruption。
- Identity supervision。

但是仍然没有稳定超过普通 RGB GRU。

本轮必须改变核心建模问题。

旧方法主要是：

`History → Predicted Identity Prototype → Candidate Similarity`

新的研究方向是：

**联合考虑历史和当前所有候选，预测候选与历史身份状态之间的一致性。**

例如两条外观相似的鱼 A、B 交叉。

旧 JEPA 预测 A 的身份原型，再分别与候选进行余弦比较。

新 Candidate-JEPA 应学习：

“候选 1 和候选 2 分别与 A 的历史变化规律是否一致？哪些特征支持或反对这两种解释？”

### 6.2 候选集合接口

建议实现：

`cfp/candidate_graph.py`

`cfp/candidate_jepa.py`

输入：

- N 条活跃或可重激活轨迹。
- M 个当前检测候选。
- 每条轨迹的历史身份 token。
- 每个候选的当前 RGB 特征。
- 当前与历史时间差。
- IoU、运动残差。
- 历史身份置信度。
- 当前候选集合及合法匹配 mask。

输出：

`[N, M+1]` 候选身份分数。

最后一列代表 NULL / unmatched。

轨迹出生和重激活可在后续生命周期模块统一处理。

不得默认候选 top-K 内一定存在正确身份。

必须记录：

- 全检测候选覆盖。
- 几何门控覆盖。
- 实际进入 JEPA 的候选覆盖。
- 正确候选是否被排除。
- NULL 类定义。

### 6.3 第一类结构：Candidate Cross-Attention

输入过去身份 token 作为轨迹上下文。

当前候选表征作为 query。

通过 cross-attention 建模候选与历史轨迹之间的关系。

必须同时包含：

- 仅普通 Cross-Attention。
- 同容量普通 Transformer。
- Candidate-conditioned GRU。
- Candidate-conditioned JEPA。

不能把普通候选 Transformer 的收益归因于 JEPA。

### 6.4 第二类结构：Predictive Association JEPA

保留学生和 EMA 教师。

训练阶段：

学生输入：

- 严格过去的身份历史。
- 当前可见候选信息。
- 时间和运动信息。
- 遮挡或污染历史。

教师目标：

- 训练数据中已知物理身份对应的干净当前或未来表征。
- 与学生严格区分的训练期目标分支。
- 仅用于学习身份一致性的监督。

教师目标必须 stop-gradient。

EMA 只在成功优化后更新。

推理时不允许保留教师、GT 标签或未来观测。

学生输出候选条件化表示，形成当前身份匹配分数。

### 6.5 重点探索的训练任务

A. Candidate-aware Latent Prediction

学习历史条件下当前候选应满足的身份潜在表示。

B. Candidate Contrastive Prediction

同一身份候选接近，当前候选集合中其他鱼作为困难负样本。

C. Masked Identity Evidence Modeling

隐藏部分可靠历史观测，学习从剩余过去证据判断当前候选。

D. Corrupted History Recovery

向历史中注入真实风格的身份污染，训练模型仍然区分候选身份。

E. NULL-aware Predictive Association

在正确候选不存在或无法确认时，学习拒绝错误匹配。

F. Candidate Distribution Prediction

不只预测一个 latent vector，而是预测当前候选的相对身份分布，并研究其不确定性和校准。

以上是开放研究候选，并非要求不加区分地全部堆叠。

### 6.6 损失函数

初始可研究：

- Candidate cross entropy。
- Supervised contrastive loss。
- Hard-negative ranking loss。
- Latent prediction loss。
- Masked latent loss。
- NULL-aware classification loss。
- Calibration loss。

先建立合理的少损失基线。

新增损失必须有明确目标和消融依据。

特别关注：

预测 latent 更准，是否真的会提高候选排序？

如果没有，说明 latent target 可能不适合当前关联任务。

允许进一步研究候选分布、关系预测或任务对齐目标，而不是仅继续降低 cosine prediction loss。

### 6.7 JEPA 研究对照

必须包含：

- 普通 RGB GRU。
- 普通预测 GRU。
- 普通 Candidate-MLP。
- 普通 Candidate-Transformer。
- MOTIP 风格候选 ID Decoder 基线。
- Candidate-JEPA without EMA。
- Candidate-JEPA with EMA。
- Candidate-JEPA without masking。
- Candidate-JEPA with masking。
- Candidate-JEPA with corruption。
- Candidate-JEPA with NULL。
- Candidate-JEPA with recoverability evidence。

需要做参数量、训练数据、采样预算和候选信息量匹配。

注意 MOTIP 已研究历史上下文 ID 预测。我们必须通过清晰的目标定义和公平实验，证明新预测机制与普通候选解码器之间的真实区别。

### 6.8 防止训练和评价错位

上一轮出现：

干净历史身份排名接近 99%，实际受污染历史只有约 61%。

因此必须在训练时包含真实跟踪器产生的接受历史。

不能只在干净 GT 轨迹上训练 JEPA，再期望它直接适应在线错误。

必须分别分析：

- Clean history。
- Actual accepted history。
- Synthetic corruption。
- Natural contaminated history。
- Candidate-missing events。
- Long-gap reactivation。

真实轨迹历史应由训练集上的合法在线回放生成，不能由 GT 决策修复后伪装成在线历史。

---

# 七、创新点 C：歧义感知多假设关联

## Ambiguity-Aware Multi-Hypothesis Online Association

此前 Phase V 因前两个模块没有稳定突破，尚未实施真正的多假设关联。

**本轮不允许仅因为 A 或 B 第一版不理想，就直接跳过 C。**

至少完成一个真实可运行、可公平比较的轻量多假设实验。

### 7.1 核心问题

当两条鱼交叉，当前存在多个合理的身份解释时：

普通 Hungarian 会给出一个确定分配。

如果该分配错误，后面的历史记忆可能持续受到污染。

我们希望允许系统内部保留少量合理解释，再利用之后的观测判断哪一种更可信。

严格在线要求：

- 当前帧必须按当前可见信息输出。
- 未来观测只影响未来决策。
- 不允许改写已经输出的历史结果。
- 不允许未来图像参与当时的关联评分。

### 7.2 第一版轻量实现

建议新增：

`cfp/hypothesis_bank.py`

`cfp/ambiguity_association.py`

每条高歧义轨迹或局部冲突组维护有限候选解释。

首先测试：

- K=1。
- K=2。
- K=3或4，只有计算资源与实际歧义支持时再探索。

使用：

- 当前候选得分。
- 运动一致性。
- 历史身份状态。
- 当前置信区间。
- 可恢复性估计。
- 候选条件化 JEPA 分数。

动态决定是否触发多假设。

不要对所有普通高置信匹配都启用复杂分支。

### 7.3 多假设记忆

候选分支应分别维护自身的临时身份状态。

只有获得足够证据时才允许更新稳定身份记忆。

必须防止：

- 同一候选同时分配给两条实际输出轨迹。
- 同一物理目标在单帧生成重复 ID。
- 不同分支共享可变记忆导致污染。
- 历史输出被事后修改。
- 分支数量无界增长。
- 无可靠证据却强制恢复身份。

### 7.4 必须比较

- 原单假设 Hungarian。
- 单假设 + risk gate。
- 双假设，不使用 JEPA。
- 双假设 + 普通 GRU。
- 双假设 + Candidate-JEPA。
- 双假设 + recoverability。
- 双假设 + Candidate-JEPA + recoverability。
- 简单延迟写入但不保留多假设。

如果简单延迟写入就能实现同样收益，那么不能宣称多假设建模是必要的。

### 7.5 重点指标

- 交叉事件 IDSW。
- 恢复成功数。
- 错误假设保留率。
- 正确假设覆盖率。
- 假设最终选择准确性。
- 净纠错。
- 额外延迟。
- 分支扩张次数。
- HOTA。
- AssA。
- IDF1。
- MOTA。
- DetA。

---

# 八、A、B、C 应该怎样组合？

不要从一开始就强制使用完整 A+B+C。

所有模块必须拥有独立开关。

建议测试以下研究矩阵：

| 方案 | Recoverability | Candidate-JEPA | Multi-Hypothesis |
|---|---|---|---|
| B0 | 关闭 | 关闭 | 关闭 |
| A | 开启 | 关闭 | 关闭 |
| B | 关闭 | 开启 | 关闭 |
| C | 关闭 | 关闭 | 开启 |
| A+B | 开启 | 开启 | 关闭 |
| A+C | 开启 | 关闭 | 开启 |
| B+C | 关闭 | 开启 | 开启 |
| A+B+C | 开启 | 开启 | 开启 |

首先独立验证 A、B、C。

然后分析哪个模块之间存在互补性。

如果某个模块单独无收益，但在另一模块加入后有明确互补效果，可以继续研究其交互机制。

必须通过删除该模块的消融来证明它不可替代。

如果 A+B 已经优于 A+B+C，则允许最终架构不保留 C。

如果普通 Candidate-Transformer 优于 Candidate-JEPA，则必须承认 JEPA 不是必要技术，并研究是否应该更换方法主线。

如果一个强普通模型就能够得到最好结果，不能通过复杂模块数量强行包装创新。

**最终论文可以只有两个真正有效的核心创新点，不要求把全部三个模块都保留。**

---

# 九、允许自由扩展的创新探索池

除 A、B、C 外，我授权你在证据支持下探索其他合理方向。

以下是值得关注的潜在研究分支。

### 方向 D：候选覆盖恢复

此前真实错误中存在正确候选被几何门控排除的情况。

如果诊断证明这类问题占比较高，可以探索：

- 动态几何候选扩展。
- 外观辅助候选召回。
- 候选重排序与候选召回解耦。
- 基于可恢复性的不确定候选扩展。
- 置信候选集合而非固定单一阈值。

注意候选召回增加也可能引入更多错误匹配。

必须同步分析 FP、IDSW 和运行成本。

### 方向 E：身份条件化的时序表征

如果冻结 FishReID 特征不足以支持身份恢复，可以探索：

- 轻量身份特征 adapter。
- 局部鱼体 token。
- 多时间尺度身份表示。
- 交叉事件困难负样本训练。
- 强视觉预训练特征的受控对照。
- 目标级 Dense JEPA。

不能直接从头训练大型 V-JEPA。

优先冻结视觉编码器并训练轻量 adapter，确认有收益后再决定是否微调。

### 方向 F：运动与局部交互关系

如果身份错误主要集中在转向、遮挡和相互穿插事件，可以研究：

- 鱼体主轴和朝向。
- 非线性运动状态。
- 邻居间的相对运动。
- 局部交互图。
- 群体交叉事件建模。

必须证明这些信息比简单运动平滑和局部几何规则更有效。

### 方向 G：合成 IR 作为训练期辅助视图

原合成 IR 没有稳定证明身份收益。

可以探索：

- RGB-only。
- RGB + 因果合成 IR。
- RGB + 第二 RGB 增强视图。
- 合成 IR 作为 teacher 辅助输入。
- 推理仅 RGB 的辅助蒸馏。

如果没有稳定收益，允许完全放弃身份阶段的 IR 分支。

但原 V3 检测器输入仍需按照冻结协议保持不变，除非开启明确独立的检测研究。

### 自由探索规则

新方向必须建立独立研究假设、对照和记录。

不能把未验证方向直接添加到完整模型中。

如果某项新假设有清晰证据支持，可以主动扩展实验，不需要因本文没有提前列出具体配置就停止。

---

# 十、实验方式：从便宜的诊断到真实训练

不要一开始就同时训练大量完整模型。

采用循环式科研流程。

## Experiment Loop 1：错误可恢复性

不训练。

通过原有真实轨迹和物理标签，统计候选覆盖、历史可靠性及可纠错事件。

确定最有价值的错误类别。

## Experiment Loop 2：最小模块验证

实现小型 Recoverability Estimator 与候选集评分器。

先做：

- 正确性测试。
- 训练集 OOF 诊断。
- 候选级指标。
- 小规模完整视频在线回放。

对比普通 GRU 和普通 MLP。

## Experiment Loop 3：候选 JEPA 训练

先建立普通候选 Transformer。

再加入 EMA target、masked prediction 和候选级 latent objective。

固定数据采样及容量，验证 JEPA 特有的收益。

## Experiment Loop 4：真实污染与遮挡恢复

使用原始跟踪器产生的真实接受历史。

分析：

- 自然错误历史。
- 无正确历史线索。
- 错误初始身份。
- 重激活。
- 相似鱼交叉。
- 大时间间隔。

记录 N01、N10 和实际在线跟踪变化。

## Experiment Loop 5：多假设实验

至少实现轻量 K=2，并与单假设、普通延迟记忆等基线对照。

如果有有效信号，再扩大或联合其他模块。

## Experiment Loop 6：自主科研迭代

根据已得到的证据自由提出下一轮实验。

例如：

- 扩大可观测候选上下文。
- 改变 NULL 监督。
- 训练候选风险分布。
- 更换预测目标。
- 改变历史 token 采样。
- 针对真实错误分布重新训练。
- 使用其他身份表示。
- 学习多假设分支的置信度。
- 研究真实在线状态下的表征分布漂移。

每次实验都需要有明确原因。

不能仅因为“上一轮没有提升”就无依据地更换更大的模型。

---

# 十一、实验选择与公平比较

采用三级证据标准。

### Level 1：训练与诊断证据

例如：

- Loss。
- Rank@1。
- Pair AUC。
- Margin。
- Risk AUC。
- Calibration。
- N01 / N10。

只证明局部机制是否工作。

不能据此宣称 MOT 性能提升。

### Level 2：固定检测的完整视频评价

必须报告：

- HOTA。
- AssA。
- DetA。
- LocA。
- AssRe。
- AssPr。
- DetRe。
- DetPr。
- IDF1。
- IDP。
- IDR。
- MOTA。
- IDSW。
- FP。
- FN。
- Frag。
- MT。
- ML。

所有方法使用官方 TrackEval。

比较相同冻结检测框、相同 GT 和相同输出框协议。

不能把回退旧平滑框的结果与当前匹配检测框结果直接混合作为同因素消融。

### Level 3：完整严格在线系统

对于有希望的模型，运行真实当前帧推理。

速度必须包含：

- 图像解码。
- 当前帧因果 IR 构建。
- V3 检测。
- NMS。
- ROI 裁剪。
- RGB 身份编码。
- 新预测模块。
- 多假设或风险决策。
- 生命周期更新。
- 最终输出。

报告：

- FPS。
- 单帧 p50。
- 单帧 p95。
- 显存。
- 参数量。
- 可计数 FLOPs。
- 整体输入输出正确性。

上一轮完整模型约 0.79 FPS，应当把性能优化纳入研究，而不能忽视。

但推理速度必须在统一硬件、合理负载下测量；缓存关联速度不能冒充端到端速度。

### 多种子要求

初期可以用一个种子做结构诊断。

当出现有价值的实验信号时，扩展到至少三个种子。

最终选择看均值、方差、每视频结果和事件级证据，不能只看最佳种子。

三种子方差不能当作两个开发视频之外的场景泛化置信度。

---

# 十二、数据隔离与防止反复探索过拟合

这是开放探索必须解决的风险。

现有开发视频只有 BT-005 和 SN-013，而且此前已经进行了大量模型诊断。

如果每次自由探索都根据这两个视频的 HOTA 调参，最终一定存在过拟合风险。

因此：

1. 在现有身份训练视频内部建立按完整视频隔离的探索/校准流程。
2. 尽可能使用多折视频级验证比较研究假设。
3. 将已有两视频开发集作为历史参考和阶段性固定对照，不要每次训练都用它直接挑模型。
4. 重要候选必须先在训练内部的独立视频评估中显示收益。
5. 最终冻结配置后，再报告已有开发集的完整结果。
6. 如果获得新的合适外部鱼类跟踪数据，应建立真正未接触的独立验证。
7. 不允许从历史七个已调参测试视频中反向寻找针对性规则。

V3 检测器此前见过两个开发视频，必须继续披露这一限制。

如果没有新的独立数据，不得宣称完全独立的跨场景验证成功。

---

# 十三、科研迭代记录机制

这是本轮相比以往最重要的改动。

建立：

`docs/research/PHASE_VI_EXPERIMENT_REGISTRY.json`

`docs/research/PHASE_VI_HYPOTHESIS_LEDGER.md`

每个实验必须记录：

- Hypothesis ID。
- 实验要验证的问题。
- 为什么认为它可能有效。
- 依赖哪项实际错误现象。
- 改动代码。
- 模型输入和输出。
- 数据和参数量。
- 训练配置。
- 随机种子。
- Git commit。
- Checkpoint SHA。
- 开发集指标。
- 每视频指标。
- 错误改善和反向改错。
- 推理开销。
- 实验结论。
- 下一步建议。

当某个实验失败时，不允许直接写：

“NO_GO，停止后续全部研究。”

而应该给出：

**Observed failure：** 实际发生了什么。

**Root-cause hypotheses：** 可能有哪些原因。

**Discriminating experiments：** 哪些实验能够区分这些原因。

**Repair or pivot：** 应当修复、替换还是转向其他方向。

**Evidence update：** 新证据是否支持原假设。

如果一种思路反复没有收益，应将其降级，探索其他更有根据的问题。

科研过程不依赖固定三次或五次迭代计数。

但应防止无限无效搜索。

定期评估是否还有值得测试的新假设，以及新增实验提供了多少实际信息。

---

# 十四、论文创新性审查

目标会议可以参考 CVPR 的严格审稿标准。

每次准备保留新模块时，必须回答：

1. 解决了什么具体问题？
2. 为什么普通 GRU/Transformer/MLP 不能同样解决？
3. 为什么需要 JEPA 而不是普通监督关联？
4. 为什么需要可恢复性估计，而不是可学习阈值？
5. 为什么需要多假设，而不是简单延迟更新？
6. 是否改变了实际在线匹配决策？
7. 有多少真正纠正的身份错误？
8. 有多少原本正确的身份被新方法改错？
9. 哪些改进只是候选数量增加？
10. 哪些改进只是输出框策略造成的？
11. 哪些改进来自参数量增加？
12. 是否在两个开发视频中呈现不同趋势？
13. 是否能够跨场景泛化？
14. 方法是否符合严格在线约束？
15. 是否具有合理速度和计算成本？

参考和继续核验：

MOTIP：
https://github.com/MCG-NJU/MOTIP

GOT-JEPA：
https://github.com/chenshihfang/GOT

V-JEPA 2 / 2.1：
https://github.com/facebookresearch/vjepa2

HybridSORT：
https://github.com/ymzis69/HybridSORT

BoT-SORT：
https://github.com/NirAharon/BoT-SORT

TrackEval：
https://github.com/JonathonLuiten/TrackEval

还应主动检索最新可用的身份恢复、多假设关联、拒绝匹配及不确定性学习工作，核验论文年份和具体代码。

不能将常见技术重新组合后未经对照就认定为独立创新。

---

# 十五、项目工程组织

建议根据现有代码实际结构，新增以下模块，允许按需要调整名称：

`cfp/recoverability.py`

`cfp/candidate_graph.py`

`cfp/candidate_jepa.py`

`cfp/candidate_decoder.py`

`cfp/hypothesis_bank.py`

`cfp/ambiguity_association.py`

`cfp/online_recovery.py`

以及对应的：

- 数据构建工具。
- 训练入口。
- 评估入口。
- 事件级诊断工具。
- 实验计划执行器。
- 消融收集器。
- 单元及集成测试。

尽量复用原有 `cfp` 框架。

不要为了实现新功能把全部跟踪系统从头重写。

所有模块必须有类型、数据范围、时间语义和输入输出约束。

特殊情况包括：

- 空候选。
- 空历史。
- 单候选。
- 多轨迹竞争同一检测。
- 错误初始化。
- 轨迹重新激活。
- 长时间 gap。
- 无可靠身份线索。
- 缺失 IR。
- 非有限特征。
- 序列切换。
- 分支记忆隔离。

都必须进行测试。

---

# 十六、算力与存储

当前权重、数据和历史资产必须保留。

先执行硬件与磁盘审计。

沿用不超过四张物理 GPU 的保护规则，不抢占其他用户任务。

保留至少 40 GiB 磁盘安全余量。

每轮可按照磁盘状况规划新的训练预算，但不要把过去任务的 8 GiB 输出预算机械当作整个新研究的次数或创新探索上限。

如果存储接近保护阈值：

- 暂停高存储任务。
- 检查本轮新增产物。
- 优先压缩重复且可重建的中间产物。
- 保留关键权重、指标和证据。
- 不删除旧阶段资产。

尽量复用已有冻结检测缓存和身份编码结果。

大型模型权重、特征库和数据不提交 GitHub。

训练应支持精确的 checkpoint/resume、随机状态恢复及日志一致性验证。

---

# 十七、GitHub 持续交付

每完成一个有实际内容的研究阶段，都要：

1. 更新研究记录。
2. 完成相应测试。
3. 提交新增代码和配置。
4. 将小型正式指标与分析写入 Git。
5. 推送到研究分支。
6. 核验本地和远端 commit 一致。

分支：

`research/recoverability-candidate-jepa-v1`

GitHub：

https://github.com/LYQ1107/IR-MFT

优先提交：

- 源代码。
- 配置。
- 方法设计。
- 复现脚本。
- 正式指标。
- 实验注册表。
- 消融结论。
- 失败分析。
- 阶段报告。

不要提交：

- 大型权重。
- 原始数据。
- 大量图像缓存。
- 特征库。
- 大型推理缓存。
- 服务器秘钥或访问凭证。

GitHub 如果因网络错误无法推送，保存本地提交及错误日志，恢复连接后核验推送。

不能声称推送成功但没有远端提交证据。

---

# 十八、最终科研报告要求

最终形成：

`docs/research/PHASE_VI_FINAL_RESEARCH_REPORT.md`

至少包含：

1. 本轮完整创新探索路线。
2. A/B/C 分别实现了什么。
3. 实际完成的新增探索方向。
4. 可恢复性分析结论。
5. 候选条件化 JEPA 的结构与训练过程。
6. 多假设关联的实现和实际触发行为。
7. 与原 RIS、PLIM、Fish-JEPA 的本质区别。
8. 与普通 GRU/Transformer 的公平比较。
9. JEPA 教师—学生机制的独立贡献。
10. 错误纠正 N01/N10 和净收益。
11. HOTA、AssA、DetA、IDF1、MOTA、IDSW 全指标表。
12. 两个开发视频的分开结果。
13. 三种子均值及波动。
14. A/B/C 单独和组合消融。
15. 模型参数、延迟和完整在线 FPS。
16. 数据污染与泛化限制。
17. 所有重要负结果。
18. 实验过程中主动提出的新假设。
19. 哪些假设被验证，哪些被否定。
20. 最终推荐的有效模型和完整复现方式。
21. GitHub 最新提交和分支。
22. 尚需独立验证或外部数据支持的结论。

不要求最终一定保留 A+B+C。

最终应选择实际科学证据最强、结构合理、计算成本可接受的架构。

如果最后只有 A+B 有贡献，就保留 A+B。

如果 Candidate-JEPA 不如普通 Transformer，但多假设关联有效，可以调整主要科研方向。

如果新的探索揭示了比原计划更重要的科学问题，允许重构研究目标，但必须保留原实验及其负结果。

---

# 十九、现在直接开始执行

不要只输出实施方案。

请真正执行以下工作：

**Step 1：**

检查 IR-MFT 最新代码和现有资产，建立新研究分支，核验强基线和实验协议。

**Step 2：**

完成真实错误可恢复性统计，回答当前身份错误中：

- 多少可以通过候选重新排序纠正？
- 多少必须改变候选覆盖？
- 多少受到不可靠历史影响？
- 多少是生命周期问题？
- 多少缺乏可观测纠错证据？

**Step 3：**

实现 Recoverability-Aware Identity Association，先与强 GRU、普通 MLP 和风险门控对比。

**Step 4：**

实现 Candidate-Conditioned JEPA，并完成与普通 Candidate-Transformer、普通 GRU、无 EMA 和无预测监督模型的受控训练比较。

**Step 5：**

实现轻量多假设在线关联，至少完成独立 K=1/K=2 对照。即使 A/B 初期不成功，也不直接取消 C。

**Step 6：**

根据真实实验自主决定 A/B/C 如何组合，并探索有证据支持的其他技术方向。

**Step 7：**

对有价值的方案进行三种子、每视频、多指标、完整在线验证。

**Step 8：**

针对不足进行新的科学假设分析和迭代，而不是因为完成固定实验次数就停止。

**Step 9：**

持续提交代码、实验记录和研究报告，核验 GitHub 远端同步。

### 最重要的执行要求

**不要将本任务理解为“做完 A/B/C 三个模块就结束”。**

也不要理解为“如果第一个模型没有达到 HOTA 54 就终止”。

这是一个有最终科研目标的自主研究过程。

你必须主动判断下一步最值得做什么。

每次出现负结果，都要尽可能回答：

- 是表征问题？
- 是候选覆盖问题？
- 是身份推断问题？
- 是在线记忆污染问题？
- 是生命周期问题？
- 是损失目标不匹配？
- 是训练数据不足？
- 是新方法并没有超过普通模型的真实优势？

然后根据证据实施有针对性的下一轮研究。

不需要在每次普通实验失败后等待我重新提供任务书。

但如果遇到必须由用户授权的外部费用、数据权限或高风险系统操作，应停止相应操作并明确说明，不得擅自执行。

---

## 最终研究宣言

**The goal is not to build a more complicated tracker, but to develop a tracker that can recognize identity ambiguity, estimate whether recovery is possible, and make reliable online association decisions using predictive and historical evidence.**

中文：

**我们的目标不是设计越来越复杂的跟踪器，而是让跟踪器真正理解身份歧义、判断能否恢复，并利用历史与预测信息做出更加可靠的在线身份决策。**

从现在开始执行 Phase VI。

请先建立研究目标和实验注册表，复核已有强基线，再开展可恢复性分析。随后持续完成代码实现、实际训练、完整验证、科研诊断和必要的自主迭代，直至形成经过充分实验支持的成果或有明确证据支撑的最终科学结论。
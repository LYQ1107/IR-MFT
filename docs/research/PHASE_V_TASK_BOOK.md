
# IR-MFT PHASE V — 长执行科研任务书
# Recoverable Identity Association + Error-Aware JEPA

你现在担任 IR-MFT 项目的首席研究工程师、计算机视觉研究员和实验负责人。

本次任务不是生成研究计划，而是实际检查代码、实施实验、训练模型、验证科学假设、分析失败原因、迭代改进，并将代码和实验记录持续推送到 GitHub。

允许较长的自主执行过程。不要每完成一个实验就停下来等待下一次指令。

但必须保持科研真实性：不能伪造指标，不能为了达到目标反复在测试集上调参，不能把未经验证的模块称为有效创新。

==================================================
一、最终研究目标 FINAL GOAL
==================================================

项目暂定名：

RecoverMOT-JEPA

目标：

Learning to Recover:
Error-Aware Joint-Embedding Prediction for Online Multi-Object Fish Tracking

我们希望解决：

1. 复杂鱼群交叉时，错误身份匹配如何被传播？
2. 当历史记忆已经被错误身份污染，如何在线恢复可信身份状态？
3. 如何联合决定匹配、保持、丢失、重激活以及新生轨迹？
4. JEPA 能否通过遮挡和污染历史下的隐空间预测，提供普通 GRU/Transformer 不具备的身份恢复能力？
5. 如何在严格在线的条件下改善 HOTA、AssA、IDF1、MOTA 和 IDSW？

最终必须形成：

- 一个可运行的完整在线 MOT 方法。
- 至少一个通过强基线和消融实验验证的核心方法贡献。
- 真实的训练、评测和在线运行结果。
- 完整的创新点对照、失败分析和可复现记录。
- GitHub 上的代码和科研报告。

### 数值目标

当前开发集参考：

BoT-SORT：
HOTA 52.482
AssA 37.498
DetA 73.518
IDF1 57.477
MOTA 93.552
IDSW 113.667

普通 RGB GRU：
HOTA 51.885
AssA 37.716
DetA 71.433
IDF1 62.298
MOTA 92.562
IDSW 89

ByteTrack：
HOTA 49.297
IDSW 86

研发挑战目标：

HOTA >= 54.0
AssA >= 40.0
IDF1 >= 64.0
IDSW <= 75

同时争取改善或维持 MOTA、DetA 和运行效率。

这些是研究目标，不是预计结果。

主要成功条件应当是：

在公平同检测输入下，三种子开发集平均 HOTA 超过当前 BoT-SORT 强基线，并在 AssA、IDF1 或 IDSW 中展示具有独立消融支持的身份收益。

如果 MOTA、DetA 退化，必须完整披露并分析。

不能通过降低其他关键指标来掩盖失败。

==================================================
二、现有代码与资产
==================================================

项目根目录：

/data3/liuyeqiang/IR-MFT

GitHub：

https://github.com/LYQ1107/IR-MFT

当前研究分支：

research/cfp-fishmot-v1

已确认提交：

4f1b226343335d219af015d6c2954c0c81d28c7a

先执行 Git 状态和远端同步检查，确认是否存在更新提交或未提交的修改。

不要覆盖现有工作。

基于当前最新有效研究提交创建：

research/identity-repair-jepa-v1

如果该分支已经存在，先检查差异，安全继续，不强制覆盖。

资产：

/data3/liuyeqiang/IR-MFT_assets

请优先阅读：

docs/research/FINAL_RESEARCH_REPORT.md
docs/research/FAILURE_ANALYSIS.md
docs/research/PLIM_EXPERIMENTS.md
docs/research/CMUE_EXPERIMENTS.md
docs/research/ARCHITECTURE_DESIGN.md
docs/research/ABLATION_RESULTS.md

重点检查：

cfp/association.py
cfp/predictive_association.py
cfp/predictive.py
cfp/history.py
cfp/reliable_writes.py
cfp/joint_association.py
cfp/online.py
cfp/detector_runtime.py
tools/cfp_run_online.py

还有已有的训练、TrackEval 和实验注册工具。

不要重复实施已经完成且没有合理新假设的 CMUE 三轮训练、简单 RGB/IR 门控和上一轮时间编码搜索。

保留原有 V3、BoT-SORT、RGB GRU、PLIM 和 CMUE 全部资产。

历史结果已证明：

- V3 检测器可以完整复现。
- 普通 RGB GRU 的身份指标有竞争力。
- RGB+IR 预测模型没有独立稳定领先。
- CMUE 最终没有接受有效 IR。
- 简单拒绝写入记忆会导致退化。
- 历史记忆存在明显污染。
- BoT-SORT 的 DetA 优于我们的 GRU 方法。
- 目前在线速度约1.2FPS。

另外：

V3 曾在全部8个原训练视频上训练。
新增身份模块目前使用6个训练视频和2个开发视频。
开发集是 BT-005 / SN-013，共4834帧。

因此当前开发集并不是整个检测系统的独立盲测。

历史7视频测试集曾被调参，不能重新拿来进行新方法选择。

严格使用已经冻结的数据和 GT 处理协议，尤其是 SN-013 中已审计的歧义标注。

==================================================
三、Phase 0：环境与基线复核
==================================================

先自动检查：

GPU资源
CPU/内存
Conda环境
PyTorch/CUDA/cuDNN
剩余存储
Git状态
权重SHA256
数据集
既有检测缓存
实验注册表
TrackEval版本

沿用上轮已验证的运行栈：

V3检测器的cu117/cuDNN8.5兼容运行环境；
身份编码器原有cu118环境。

必须保留历史批次8和当前帧在线实现之间的数值差异记录。

不能随意修改检测器、NMS或已冻结的检测缓存。

对已有三个最重要的系统进行核验：

A. V3 + ByteTrack
B. V3 + BoT-SORT + RGB ReID
C. V3 + 普通 RGB GRU

优先使用已经存在且经过哈希验证的检测、特征和跟踪记录。

不需要重新训练V3检测器。

确认目前的100项测试仍可通过。

确认两个开发视频上的HOTA、AssA、DetA、IDF1、MOTA、IDSW与历史报告一致。

如果不一致，先定位环境、缓存、输出策略或评测实现原因。

交付：

docs/research/PHASE_V_BASELINE_AUDIT.md

==================================================
四、Phase I：关联器与生命周期四组合实验
==================================================

这是整个下一阶段最重要的第一步。

当前发现：

BoT-SORT：
HOTA较高、DetA较高，但IDF1和IDSW较弱。

RGB GRU：
身份指标较强，但DetA和HOTA不如BoT-SORT。

首先不要训练新模型。

请将关联评分和轨迹生命周期管理尽可能拆分成两个独立组件。

关联器负责：

- 计算当前轨迹与候选检测的成本。
- 使用运动、几何和外观信息。
- 生成候选匹配及身份置信度。

生命周期负责：

- 轨迹初始化。
- 已确认轨迹更新。
- unmatched处理。
- Lost状态。
- Removed状态。
- 重新激活。
- 新生轨迹判断。
- 输出过滤。

完成以下四个受控组合：

A：
BoT关联 + BoT生命周期。

B：
RGB GRU关联 + BoT生命周期。

C：
BoT关联 + CFP生命周期。

D：
RGB GRU关联 + CFP生命周期。

必须共用：

- 相同V3检测候选。
- 相同冻结ReID编码器。
- 相同视频和GT。
- 相同特征坐标及归一化。
- 相同评测程序。
- 公平的开发集参数预算。

特别注意：

BoT-SORT和CFP的内部状态、坐标变换、匹配阶段、出生确认规则和输出策略可能不同。

不要仅靠替换一个Python函数就宣称完成公平交叉实验。

应建立显式适配层。

如果某些逻辑无法独立交换，必须说明耦合原因，并使用更细的事件级干预实验区分影响。

至少报告：

HOTA
AssA
DetA
IDF1
MOTA
IDSW
FP
FN
Frag
MT
ML
AssRe
AssPr
DetRe
DetPr

同时保存每个视频的完整结果。

### 错误归因

建立事件级错误分析器。

分类：

1. 外观相似导致的身份交换。
2. 两条鱼交叉后的关联错误。
3. 遮挡后重新出现却获得新ID。
4. 错误重激活。
5. 重复轨迹输出。
6. 轨迹过早终止。
7. 轨迹错误初始化。
8. 高置信错误匹配。
9. 长期记忆污染。
10. 真实候选存在但被几何门控排除。

对每个类别统计：

事件数量、受影响帧数、对应轨迹、错误前后成本矩阵、候选召回、匹配置信度和官方指标影响。

要区分官方TrackEval匹配事件与真实在线决策。

不要把CLEAR的后验匹配ID当作推理时可获取的身份标签。

特别重新分析：

BT-005 frame 1049附近的身份交叉事件。

但不能根据这个开发集案例专门调规则。

必须在训练视频内寻找同类型事件，进行泛化验证。

最终回答：

“我们HOTA落后，究竟有多少来自关联，有多少来自生命周期？”

交付：

docs/research/LIFECYCLE_CROSSOVER_RESULTS.md

阶段完成后提交和推送GitHub。

==================================================
五、Phase II：创新点1——可恢复身份状态
==================================================

暂定模块：

Recoverable Identity State，RIS。

原有IdentityTrack存储单一身份历史或EMA。

现在需要探索：

身份历史不总是正确的。

如果某条轨迹已经混入别的鱼，继续平均或预测旧特征可能让错误持续存在。

RIS需要学习：

- 当前历史是否存在身份矛盾。
- 哪些历史观测相对可信。
- 是否应该修复当前内部身份状态。
- 是否应该暂时保留不同候选身份解释。

### 结构设计

尝试：

1. 多个身份锚点。
2. 短期身份观测。
3. 历史候选一致性统计。
4. 记忆可靠性估计。
5. 受污染记忆检测头。
6. 身份恢复候选评分。

重要约束：

最早的身份锚点本身也可能错误！

不能把first observation永久视为真值。

锚点可靠性只能根据在线可见证据更新，不使用GT身份。

### 训练数据

从6个训练视频建立真实身份关联样本。

生成两种历史：

A. 真实物理身份对应的相对干净历史。

B. 通过原有在线跟踪器获得的实际接受历史。

在训练阶段额外注入：

- ID swap。
- 邻近鱼错误写入。
- 局部遮挡。
- 长时间未更新。
- 特征漂移。
- 重激活错误。
- 候选身份冲突。

错误注入必须模拟真实跟踪错误分布。

不要使用完全随机、与真实场景无关的噪声作为唯一训练来源。

标签仅在训练阶段用于构造监督。

在线推理绝对不能读取GT。

### 对照

EMA
FIFO
Anchor + Recent
普通GRU
预测GRU
多原型记忆
仅拒绝写入
软权重写入
RIS身份恢复

评价：

身份污染检测AUC
真正错误历史的恢复率
误修复率
纠错数N01
反向改错数N10
净纠错收益
IDSW
AssA
IDF1
HOTA

如果RIS只是增加参数而没有优于容量匹配模型，不认定创新成立。

交付：

docs/research/IDENTITY_REPAIR_EXPERIMENTS.md

==================================================
六、Phase III：深入探索JEPA
==================================================

这是本轮新的重点研究方向。

必须真实阅读以下开源代码：

GOT-JEPA
https://github.com/chenshihfang/GOT

重点文件：

pytracking/ltr/models/tracking/tompnet_JEPA.py
pytracking/ltr/models/tracking/tompnet_JEPAp.py
pytracking/ltr/models/transformer/filter_predictor_JEPA_context.py
pytracking/ltr/train_settings/tomp/GOT_JEPA_378_pretrain.py
pytracking/ltr/train_settings/tomp/GOT_JEPA_378_finetune.py

V-JEPA 2 / V-JEPA 2.1
https://github.com/facebookresearch/vjepa2

重点：

app/vjepa/train.py
app/vjepa_2_1/train.py
app/vjepa_2_1/models/predictor.py
src/models/predictor.py
configs/train_2_1/vitb16/pretrain-256px-16f.yaml

MOTIP
https://github.com/MCG-NJU/MOTIP

重点：

models/motip/id_decoder.py
docs/GET_STARTED.md

记录各仓库的准确commit。

调研目标不是直接复制完整V-JEPA。

必须理解并实现符合我们任务的JEPA机制。

### 关键区别

上一轮PLIM主要是：

历史特征 -> GRU/Transformer -> 下一时刻特征预测。

现在要探索：

被遮挡、受污染的身份上下文
-> JEPA context encoder
-> latent predictor
-> 预测可靠身份表示
-> 当前候选身份关联。

训练阶段：

较干净的目标观测
-> EMA target encoder
-> latent supervision。

其中teacher输出停止梯度。

student预测目标潜在表示，而非恢复RGB像素。

teacher只能在训练阶段读取干净目标观测。

未来帧可以用于训练监督，但绝对不能用于在线推理。

### 设计轻量化Fish-JEPA

建议模块：

FishJEPAContextEncoder

FishJEPATargetEncoder

FishJEPAPredictor

OcclusionAwareMasking

IdentityDiscriminationHead

RecoveryConfidenceHead

优先复用已有FishReID后颈特征和历史轨迹表示。

首版使用轻量预测器。

不要直接训练完整ViT-L/ViT-H/ViT-G。

不重新下载没有必要的大型视频数据集。

训练时探索三种目标：

A. 普通未来潜在特征预测。

B. Masked Temporal Prediction：
遮挡历史中的部分观测，预测对应可靠身份表示。

C. Error-Aware Prediction：
对历史中被替换成其他鱼特征的观测进行扰动，学习恢复与正确身份一致的表征。

模型需要避免学出所有鱼都差不多的平均表征。

可以使用：

- Latent prediction loss。
- Identity contrastive loss。
- Hard negative ranking loss。
- Variance / covariance regularization。
- Association classification loss。

先测试简洁的损失组合。

不要一次加入过多不可解释的损失项。

### 两种JEPA实施路线

路线A：Track-level JEPA

使用已有512维身份特征及历史轨迹。

建立teacher/student/predictor结构。

训练输入：
历史身份token、时间差、观测可靠性。

预测：
当前目标身份相关的latent表示。

优先使用该路线打通实验。

路线B：Object-centric Dense JEPA

如果路线A存在明确正向证据，再进一步探索目标图像块的时空token。

参考V-JEPA 2.1的：

- Dense predictive loss。
- 多层中间特征监督。
- 遮挡token预测。
- 图像与视频token建模。

重点关注鱼头、鱼尾、身体局部纹理和遮挡形变相关的局部表征。

如果现有标注不能支持精确局部监督，可以使用ROI特征块和自监督掩码，但需要证明其优于全局特征。

不能因为使用了patch token就宣称理解了鱼体结构。

### 核心JEPA消融

J0：原普通RGB GRU。
J1：原RGB预测GRU。
J2：同参数量普通Transformer。
J3：EMA teacher + latent prediction。
J4：J3 + temporal masking。
J5：J4 + identity contrastive。
J6：J5 + history corruption。
J7：J6 + RIS身份恢复。
J8：完整JEPA + 生命周期联合决策。

还需要：

无EMA teacher
无掩码预测
无历史污染训练
无身份对比损失
冻结encoder
微调轻量adapter
不同历史窗口
不同遮挡强度

所有模型必须在相同训练数据、训练预算和合理匹配的参数量下进行对照。

不能把普通GRU预测器重新命名为JEPA后宣布创新完成。

### JEPA额外诊断

必须分析：

latent prediction cosine
clean/noisy feature alignment
positive-negative margin
feature variance
collapse指标
遮挡长度与错误率
污染历史恢复率
实际在线identity rank
错误恢复与反向改错
历史置信度校准

特别比较：

Teacher-forced clean histories。

Actual accepted histories。

Artificially corrupted histories。

必须说明各组之间的差异。

如果JEPA只在teacher-forced干净历史上好，在真实在线轨迹上退化，则不能认为研究成功。

如果JEPA预测损失下降而HOTA不升，说明representation objective与身份关联任务可能不一致，需要针对性修改训练目标。

如果JEPA不优于普通GRU，先分析是表示坍缩、对比监督不足、训练数据不够，还是预测目标本身没有身份信息。

至少完成三轮证据驱动的针对性尝试，再给出该方向的科学结论。

交付：

docs/research/JEPA_CODE_AUDIT.md
docs/research/FISH_JEPA_ARCHITECTURE.md
docs/research/JEPA_TRAINING_RESULTS.md
docs/research/JEPA_FAILURE_ANALYSIS.md

训练代码、模型和日志也必须保留。

==================================================
七、Phase IV：受控多假设身份关联
==================================================

如果身份恢复和JEPA表现出有效信号，再探索：

Uncertainty-Aware Multi-Hypothesis Association。

当两条鱼交叉且两个身份解释分数非常接近：

不要立即把全部身份记忆写死。

可以在内部保留最多两个候选身份解释。

随后利用新观测更新候选可信度。

严格限制：

只使用当前和过去帧。

每帧仍然必须输出唯一、有效的在线身份分配。

不能事后修改已经输出的历史结果。

如果没有可用证据，不允许凭空恢复真实ID。

比较：

单假设。
双假设。
仅短期延迟内部写入。
RIS。
JEPA。
RIS + JEPA。
RIS + JEPA + 多假设。
加上生命周期联合决策。

必须分析多假设分支实际执行次数、错误假设保留率、净纠错收益和额外计算成本。

不要引入无界搜索树或离线全局轨迹优化。

如果该模块成本大、收益不稳定，就保留为探索结果而不是强行加入最终系统。

==================================================
八、Phase V：重新评估合成IR的价值
==================================================

合成IR不是本轮首要创新。

已知其来自RGB的规则响应与伪彩映射，而非独立热红外传感器。

必须使用现有的严格因果构建IR。

禁止使用需要未来帧统计的旧生成器完成在线验证。

只有RGB身份恢复和JEPA主线打通之后，才进行以下探索：

A. RGB-only JEPA。

B. RGB + causal IR JEPA。

C. RGB + 第二种RGB增强视图JEPA。

D. 合成IR只作为训练teacher的辅助输入，推理仅用RGB。

E. 有条件才探索IR student分支。

必须进行容量匹配、计算量匹配和正确配对/扰动对照。

重点验证：

合成IR是否提高遮挡后身份恢复能力？

合成IR是否减少历史记忆污染？

是否优于第二种RGB增强视图？

是否存在独立于模型容量的收益？

如果没有稳定收益，就诚实放弃推理时IR分支。

不能为了保持“多模态”名称而保留无效组件。

==================================================
九、Phase VI：统一评测和模型选择
==================================================

所有模型统一报告：

HOTA
AssA
DetA
LocA
AssRe
AssPr
DetRe
DetPr
IDF1
IDP
IDR
MOTA
IDSW
FP
FN
Frag
MT
ML
FPS
latency p50/p95
显存
参数量
有效FLOPs/成本说明

主要协议：

Protocol A：
固定V3检测输入。

Protocol B：
严格在线完整推理。

Protocol C：
身份污染/遮挡事件级诊断。

首先在Protocol A上确定真正有价值的身份创新。

再将通过验证的组合放入Protocol B。

不得把缓存检测后的关联FPS称为端到端FPS。

完整在线速度必须包含：

图像读取
因果IR生成（若使用）
检测
NMS
ROI裁剪
身份编码
JEPA预测
轨迹决策
最终输出

已有约1.2FPS只是上一轮RGB预测完整管线的参考。

新模型必须重新计时。

不要虚报为实时25FPS。

### 模型选择

初始实验可以使用seed0诊断。

正式候选必须使用seed0/1/2完整复现。

模型和参数只能通过开发协议选择。

必须报告BT-005、SN-013各自结果，不能只报告combined。

对可能的收益计算均值、标准差和事件级变化。

开发视频只有两个，不要将三种子标准差解释为跨场景泛化的置信区间。

若条件允许，寻找新的合法外部鱼类视频数据和身份标注，开展真正独立验证。

不能将历史七视频已调参测试结果称作新的盲测。

### 关键消融

最终至少包含：

BoT原始基线
RGB GRU
最佳四组合生命周期基线
RIS
JEPA
RIS + JEPA
RIS + 生命周期
JEPA + 生命周期
RIS + JEPA + 生命周期
完整方法
完整方法去除每个核心模块

必须验证每项创新的独立价值。

只改变模块的实验应固定其余因素。

==================================================
十、失败时的自动诊断和继续研究
==================================================

特别重要：

不要因为第一轮HOTA没有超过52.482就停止。

科研过程：

实验
-> 统计失败类型
-> 提出具体原因
-> 做受控诊断
-> 修改模型或训练目标
-> 重新验证
-> 记录是否支持假设。

如果AssA升而DetA降：

检查输出生命周期、轨迹删除、错误出生、匹配拒绝、框输出和低置信检测利用率。

如果JEPA训练损失降而HOTA不升：

检查目标特征身份信息量、预测表示坍缩、真实轨迹历史污染和候选集合差异。

如果记忆修复损害IDF1：

检查原型被错误覆盖、初始锚点不可靠、置信度校准和误修复率。

如果多假设增加IDSW：

检查错误解释持续时间、身份切换时机和置信度排序。

如果IR分支退化：

分析模态冗余、表示质量和训练分布，不要立即增大模型。

如果BT-005提升而SN-013退化：

分析场景密度、交叉事件、数据分布及身份可靠性。

不允许为两个开发视频分别配置专属阈值来制造总体提升。

对每个核心假设至少允许三次有根据的迭代。

但不是无上限的训练搜索。

如果完整诊断后仍没有支持证据，可以停止该分支并记录可信负结果。

不要为了永远避免NO_GO而牺牲科研真实性。

==================================================
十一、算力、存储和运行规范
==================================================

实际检查GPU资源后安排任务。

不得终止其他用户进程。

优先使用已有检测缓存和编码器特征。

不要重复保存整份数据集。

开始时确认剩余空间，设置至少40GiB安全余量。

优先顺序：

1. 不训练的生命周期交叉实验。
2. 错误事件分析。
3. 轻量RIS训练。
4. 轻量Track-level JEPA。
5. JEPA三种子对照。
6. 有证据后再探索Dense JEPA。
7. 最后运行完整在线系统。

不允许直接开始数亿参数视频模型的大规模预训练。

如果使用外部权重，记录许可证、来源、哈希和精确架构。

训练支持：

resume
AMP
checkpoint
seed
完整日志
异常恢复
配置快照

所有结果必须绑定实际代码版本和输入哈希。

重点测试：

无未来帧泄漏。
无在线GT读取。
身份ID唯一性。
新旧记忆顺序。
跨序列状态重置。
错误轨迹的恢复逻辑。
JEPA teacher不进入在线推理。
关闭RIS/JEPA时原基线结果不变。
在线输出不被事后重写。
数值稳定。
完整TrackEval一致性。

==================================================
十二、GitHub提交与复现
==================================================

研究分支：

research/identity-repair-jepa-v1

每个主要阶段完成后：

1. 运行相关测试。
2. 更新研究日志。
3. 保存实验配置和必要指标。
4. Git提交。
5. 推送GitHub。
6. 核实远端HEAD。
7. 记录本次commit。

必须提交：

代码
配置
文献审计
实验协议
对照结果
错误统计
消融报告
最终研究报告

不要提交：

大型训练权重
原始数据集
图片缓存
完整特征库
大型推理结果
秘钥与账号信息

大文件保留本地，Git中保存路径、SHA256和复现说明。

如GitHub连接失败，保留本地提交和日志，在可安全恢复时重试，不覆盖远端或修改其他账户配置。

==================================================
十三、最终报告
==================================================

最终输出：

docs/research/PHASE_V_FINAL_RESEARCH_REPORT.md

内容必须包括：

1. 本轮实现的创新点。
2. 每个创新点与2025–2026相关工作的区别。
3. 关联器和生命周期四组合的结论。
4. 记忆污染的真实错误类型。
5. RIS是否能发现和纠正错误身份。
6. JEPA是否真正完成教师-学生隐空间训练。
7. JEPA是否超过普通GRU/Transformer。
8. 遮挡预测和历史污染恢复的独立收益。
9. 多假设机制是否有效。
10. 合成IR是否有独立价值。
11. HOTA/AssA/DetA/IDF1/MOTA/IDSW完整对照。
12. 每个视频的指标和失败案例。
13. 三个随机种子的均值与方差。
14. 推理耗时、模型大小和计算成本。
15. 运行环境、数据划分和评测限制。
16. 所有关键失败及修复记录。
17. 最终代码和模型的复现命令。
18. GitHub提交链接。
19. 尚未执行的实验。
20. 是否具备独立方法论文的有效实验依据。

状态允许：

IN_PROGRESS
BASELINE_REPRODUCED
FAILURE_DIAGNOSED
RIS_VALIDATED
JEPA_VALIDATED
DEV_IMPROVED
FINAL_EVALUATED
GOAL_MET
GOAL_NOT_MET_WITH_EVIDENCE

首次训练不理想不能直接结束整个任务。

但已经证明无效的研究分支不得无限重复。

==================================================
十四、现在立即执行
==================================================

请不要只给我任务计划。

直接：

第一步：
审计当前IR-MFT仓库、资产、历史报告和环境。

第二步：
实现并运行BoT-SORT / RGB GRU与两套生命周期的交叉实验，找出HOTA与DetA差距来源。

第三步：
利用真实身份污染日志建立纠错训练集，实现RIS，并与简单记忆策略公平比较。

第四步：
研究GOT-JEPA、V-JEPA2.1和MOTIP源码，完成轻量Track-level JEPA训练，实施Masked / Error-Aware latent prediction。

第五步：
完成普通GRU、预测GRU、Transformer、JEPA、RIS+JEPA等公平对照。

第六步：
在有效性证据支持下实施多假设关联、生命周期联合优化和可选Dense JEPA。

第七步：
完成三种子、完整HOTA/AssA/MOTA/IDF1对比、在线测速和消融。

第八步：
提交代码、实验记录和最终科研报告，并推送GitHub。

任何阶段失败：

先进行根因诊断和有针对性的修复，而不是立即宣布NO_GO。

任何模块成功：

必须通过匹配容量、普通模型、无监督变体和强外部基线验证其独立贡献。

持续推进，直至FINAL GOAL获得可靠证据支持，或者经过充分受控实验确认其不成立。

最终重点回答：

“当鱼类在线跟踪的身份历史已经被污染时，JEPA能否学到比普通GRU更可靠的身份恢复表示，并在不损害轨迹输出质量的情况下改善HOTA和AssA？”

现在开始执行Phase 0，不要停留在规划阶段。

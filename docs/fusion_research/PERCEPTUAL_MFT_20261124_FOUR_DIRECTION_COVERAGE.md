# 新版 November IR：四个方向的实际覆盖

本核查区分代码、正式拟合、完整评估和独立贡献。目前四方向的最终完成
条件仍未满足，**已验证的独立多模态贡献为 0**。当前最高观测 Test
31.504 不是新选出的主模型，也未达到原始 RGB 29.529 加 5 点的目标。

六组全量训练对应13项冻结Test现已全部完成，固定主方案29.834177对
同预算RGB29.467182为+0.366995、对灰度29.421963为+0.412215；对原始
RGB29.529为+0.305177。C自身最佳消融的Dev／Test分别+0.557085／
+0.414747，但尚缺模态×C交互及原创性证据；S不稳定、R为负贡献。
完整47项、原CSV及两种模式每模式13项实测B4成本均已补齐，原十项成本保留。
原控制器和初版九项补测控制器均已实际退出；并不代表四方向全部完成。
详见[FULL_TRAIN_TEST47_ANALYSIS](PERCEPTUAL_MFT_20261125_FULL_TRAIN_TEST47_AUDIT_ANALYSIS_ZH.md)。

初版结构九项完整七段Test也已完成并独立核对47项／原CSV／MOT。
旧Dev所选IR参照复现30.275295，对同家族RGB、灰度分别+1.453967、
+0.637674，但对原始RGB仍仅+0.746295。初版C同轮次消融为正，消融
自身最佳Dev却略高于完整IR。旧1152缓存Train样本、硬循环版本不能与
新27251帧全量版本混算，不增加架构或独立贡献计数；详见
[LEGACY_TEST47_ANALYSIS](PERCEPTUAL_MFT_20261125_STRUCTURAL_LEGACY_TEST47_AUDIT_ANALYSIS_ZH.md)。

六次缺测成本GPU4执行已全部退出0，原生等待器已自动恢复。质量／时序
两个模式×RGB／灰度／IR共六组真实GPU检查实际通过，零／缺失B1精确，
同模式初始参数一致。原正式六遍共同预算队列已运行，当前两张卡拟合
quality_rgb／quality_gray已各完成首遍27251帧／6814更新并继续第二遍，其他
四组待调度；六遍共同预算未完成，尚无新Dev/Test。
这使F1／F3的GPU工程门槛有了真实证据，但不增加方向完成数或创新数。
见[NATIVE_GPU_START](PERCEPTUAL_MFT_20261125_NATIVE_PYRAMID_GPU_START.md)及
[COMPLETE_RESULTS](PERCEPTUAL_MFT_20261125_FULL_TRAIN_COMPLETE_RESULTS.md)。

首遍快照与实际交付另有独立CPU核对，所有权与启动记录中的原PID／启动tick
一致；未另起训练或改变预算。图集补齐旧六组432条训练／36个正常Dev点及
逐段Test的可复核SVG，只是完成已有证据的呈现，不增加方向或创新计数。
见[FIRST_EPOCH](PERCEPTUAL_MFT_20261125_NATIVE_FIRST_EPOCH.md)及
[FIGURES](PERCEPTUAL_MFT_20261125_FULL_TRAIN_FIGURES.md)。

六组已训练融合P3后续完成Train身份区分诊断：原固定72帧／1079个GT框，
18份实际快照均保留。相同宏循环的IR对同预算RGB在三档采样间隔仅多
5／1／0次正确匹配，对灰度多1／4／少2次；没有稳定身份收益。CPU实际
退出0、13项检查通过，无新训练、GPU或缓存。这是GT理想框及已见身份的
长间隔闭集检索，不是HOTA、F4学习融合转移或整个家族的排除证据，
不增加四方向完成数或独立创新数。详见
[TRAIN_FUSED_ROI_IDENTITY](PERCEPTUAL_MFT_20261125_TRAIN_FUSED_ROI_IDENTITY.md)。

TaskUtility后续完整流程已补齐并登记427项来源，真实CPU控制器828842
等待F4及全部前序导出和实际退出，当前phases为空。11项集成检查及
人工数据上的六／九／十二遍共同预算调度分支已核验；这些不是实际
神经训练或Dev/Test准确率。没有该候选的GPU资格、正式拟合或新指标，
不增加四方向完成数或独立贡献。下方早期组件“调度尚缺”等文字保留为
阶段记录，最新状态见
[TASK_UTILITY_WORKFLOW](PERCEPTUAL_MFT_20261125_TASK_UTILITY_WORKFLOW.md)。

六组结构匹配完整Train现已全部独立核验：各72宏循环／三遍27251帧、
20736更新，所有实际地址摘要相同于原日程；四IR字节清单一致。
完整Dev两段共4834帧、50对象现已全部官方评估并按原规则冻结；主IR
HOTA48.764689，对RGB47.878421高0.886267、对更强灰度47.928942高
0.835747点。但S／R消融自身最佳反而更高，只有C出现约0.557点Dev
支持，不能计三项完整创新；原10项独立成本阶段与13项七段Test已结束，
完整结果与成本缺口见本页顶部最新分析。
原始RGB／灰度字节未记录、灰度旧退出码未知等限制保持。
这完成拟合交付核验，不等于该方向完整Dev/Test归因完成，见
[ALL_SIX_FULL_TRAIN_AUDIT](PERCEPTUAL_MFT_20261125_ALL_SIX_FULL_TRAIN_AUDIT.md)。

全部50份原276列CSV和完整47项逐段／合并值已另存，独立审计退出0；
保留审计器R1格式错误及失败成本，不修改原指标或模型选择。详见
[FULL_TRAIN_DEV47_ANALYSIS](PERCEPTUAL_MFT_20261125_FULL_TRAIN_DEV47_AUDIT_ANALYSIS_ZH.md)。

任务效用空间监督后续补齐独立训练步／断点组件，六项人工检查通过；
真实四帧12条件CPU接口检查已完成36更新及12次全缺失复核，实际退出0，
CPU1261.114秒／RSS2.790GiB。另有八项人工Dev规则检查，完整正式控制器
仍未完成，没有新GPU资格、正式拟合、完整循环或准确率。
首次日志冲突失败已保留，修正字段后同预算重跑。此实现不增加独立
贡献或方向完成数，见[T4_CPU_ENGINEERING](PERCEPTUAL_MFT_20261125_TASK_UTILITY_T4_CPU_ENGINEERING.md)及
[STUDY_RULES](PERCEPTUAL_MFT_20261125_TASK_UTILITY_STUDY_RULES.md)。

后续补齐任务监督研究配置、独立worker、只读诊断与CPU加载组件，
10＋4项人工检查通过；一个IR联合条件完成三次T4更新及12帧B1/NMS/
ByteTrack实际CPU检查。原图roll实际改变输入和特征，零／缺失输出及
内部跟踪状态精确；没有正式TaskUtility快照、真实GPU资格、完整调度、
新准确率或F4融合记忆转移，不增加方向或贡献计数。见
[WORKER_COMPONENTS](PERCEPTUAL_MFT_20261125_TASK_UTILITY_WORKER_COMPONENTS.md)。

任务监督的逐帧生产、官方47项评估及完整Dev／Test前冻结组件后续补齐，
27项人工检查通过。真实RGB前12帧与完整4834帧基线包重放已执行；R1因
跨视频ID字节比较假设不当退出1并保留，R2只读逐行复核及官方全部指标／
原CSV精确复现。没有修改轨迹，没有新IR模型准确率、GPU资格、完整正式
调度或独立贡献；独立成本、总控制器／导出和集成资格仍待补齐，见
[OBSERVATION_COMPONENTS](PERCEPTUAL_MFT_20261125_TASK_UTILITY_OBSERVATION_COMPONENTS.md)。

独立成本组件后续已接通并完成17项人工检查及真实Train八帧的B4 CPU
路径核验，六次原RGB前向／24次访问，独立原始head、NMS及MOT精确。
历史RGB的B4与25条新B1路径分开核对、计时，26项准确率配方不变。
这是CPU功能证据，没有IR分支、正式GPU成本或完整研究资格；总控制器、
总导出与集成检查尚待完成，不增加方向／创新计数。见
[RUNTIME_COMPONENTS](PERCEPTUAL_MFT_20261125_TASK_UTILITY_RUNTIME_COMPONENTS.md)。

后续已补充当前IR原生空间门控家族的CPU原型：三种组合×三个cue，
六段Train各一帧，共54次工程更新、10项单元检查通过。新编码器与空间／
冗余分支均有梯度，但空间变化很小，尚无正式拟合、Dev/Test或贡献证明。
见[NATIVE_SPATIAL_ENGINEERING](PERCEPTUAL_MFT_20261124_NATIVE_SPATIAL_ENGINEERING.md)。
下表中的“原空间gate＋冗余抑制完整当前IR验证”仍然未完成。

完整九组空间研究后续已接通并登记，323来源与11依赖锁定，CPU控制器
184707已核实等待前序全部导出和实际退出；实际worker九组×三更新及
18个独立B1帧条件通过。当前仍无GPU工程资格、正式拟合或新准确率，
不能据此把本方向或四方向总体标为验证完成。见
[NATIVE_SPATIAL_QUEUE](PERCEPTUAL_MFT_20261124_NATIVE_SPATIAL_QUEUE.md)。

后续72个Train样本的全量拟合后机制诊断已完成：IR111实际P3修正非零，
但弱RGB框内与背景结构置信接近，逐域只有三段框内均值较高；三遍全Train
后固定稠密锚点支持仍低于原RGB。它不读取Dev/Test，不是完整Train
测量或新跟踪指标；不能替代新空间门控家族的正式比较。详见
[STRUCTURAL_TRAIN_MECHANISM](PERCEPTUAL_MFT_20261125_STRUCTURAL_TRAIN_MECHANISM.md)。

| 方向 | 当前 November IR 证据 | 尚需完成 |
| --- | --- | --- |
| F1 动态质量 | `NativePyramidFusion(mode='quality')` 学习当前图像辅助金字塔，各 FPN 层预测质量 gate；RGB/gray/IR三组真实GPU门槛均已通过。quality_rgb／quality_gray已各完成一遍Train并继续六遍预算，IR待调度 | 完整 Train/Dev/Test、强对照、去质量等归因；尚无该模型的新 HOTA |
| F2 空间互补 | local correlation、结构描述匹配和有界残差已有当前 IR 训练与评估；S/C/R 八组属于同一空间残差家族的消融。旧缓存主模型 Dev 51.329、完整 Test 29.142。六组全量 Train 各三遍、50项完整Dev与13项完整Test及两种成本现已结束；固定主方案Dev48.764689／Test29.834177，对同预算RGB分别+0.886267／+0.366995；初版九项47字段补测也已结束 | 必要模态交互与原创性归因；不能把这些替代路线等同于任务书原空间 gate＋冗余抑制的完整当前 IR 验证，不能计八种架构 |
| F3 因果特征时序 | `NativePyramidFusion(mode='temporal')` 在 F1 基础上加入 RGB 当前到过去光流和正方差递归滤波，连续 T≤4 片段、跨片段状态且过去状态 detach；三输入真实GPU门槛已通过，正式拟合在共同队列中待调度 | 正式拟合、完整 Dev/Test、去历史／去运动等对照，证明收益超过单帧和同容量 RGB。旧描述向量 GRU 不能代替此特征融合实验 |
| F4 融合特征轨迹记忆 | 三路记忆及P3 ROIAlign、在线生产、官方评估、成本与调度完整代码已登记384来源并做CPU核验；完整4834帧零权重与BYTETracker严格一致。原CPU控制器等待空间父模型 | 当前IR父模型资格、真实GPU门槛、实际已学P3/q转移及完整正式比较和成本。已训练P3的Train闭集检索诊断不能替代F4验证；尚无方向完成或独立贡献 |

F4后续已补齐当前RGB／辅助／实际融合P3到ROI与记忆的捕获接口，
15项新增检查及三类未训练模块共24帧合成输入检查通过；时序辅助取当前
滤波状态，逐帧释放引用。注意力只是显式代理，未来仍须相同权重RGB与
固定权重1的强对照。当前没有读取学习权重或真实图像，没有新准确率，
仍不能计为当前IR学习特征已转移或F4正式验证。见
[FUSION_MEMORY_CAPTURE](PERCEPTUAL_MFT_20261125_FUSION_MEMORY_CAPTURE.md)。

F4普通RGB组件后续使用真实缓存P3跑通五段连续Train片段／20帧、268个ROI
向量。两种非零权重均实际更新记忆并添加正外观惩罚，但短片段MOT仍与
ByteTrack相同；没有学习融合特征转移、GPU或新指标。另有完整F4科学
设计与纯选择规则的25项合成记录检查，真实完整Dev与执行器仍未完成，
不能计为本方向验证完成。见
[F4_REAL_RGB](PERCEPTUAL_MFT_20261125_F4_REAL_RGB_ENGINEERING.md)与
[F4_DESIGN](PERCEPTUAL_MFT_20261125_FUSION_MEMORY_DESIGN.md)。

F4实际文件核验代码后续已实现，22项人工文件检查与25项纯规则共47项
通过；真实目录检查为两家族六个完成／冻结标记待生成，资格仍未知。
没有读取实际父模型Dev表或checkpoint，没有学习融合特征转移、GPU、
新HOTA或完整执行器。不能增加本方向完成数或独立创新数；见
[F4_PARENT_IO](PERCEPTUAL_MFT_20261125_F4_PARENT_IO.md)。

F4在线内存组件与MOT写盘已补齐：14项人工／未训练模块集成检查，以及
实际Train RGB前八帧完整在线编码检查通过，109个ROI。零与非零权重
的检测包、ROI／保留记忆／KF／ID及MOT均匹配独立参照；释放已消费
观测引用保持后续决策一致。但两个非零配置的短片段MOT仍与ByteTrack
相同，没有精度收益或学习IR特征转移。正式父模型、加载、GPU、完整
观测／评估／成本／调度仍待完成，不增加方向完成数；见
[F4_ONLINE_ENGINEERING](PERCEPTUAL_MFT_20261125_F4_ONLINE_ENGINEERING.md)。

F4严格CPU张量加载与固定序列源图像交付组件后续已实现，20项张量及
8项交付检查通过。实际父模型仍因六个缺失标记待验证，融合张量读取0；
只加载原RGB并核对八帧Train工程前缀，IR仅做字节来源核对。完整Dev/Test
截短拒绝与失败前缀保护已实现，但正式GPU生产入口、评估／成本／调度
仍未完整接通，无新HOTA或方向完成数；见
[F4_LOADING_DELIVERY](PERCEPTUAL_MFT_20261125_F4_LOADING_DELIVERY.md)。

F4官方47字段评估与Dev文件冻结后续已实现，10项人工文件检查通过；
已有RGB完整Dev4834帧官方复核精确重现39摘要、47项逐段／汇总和276列
原CSV。Dev47.579162属于已有基线，不是新F4/Test结果；尚无真实F4冻结
或IR学习特征转移。IR101／110第一遍完整Train已核验，仍继续三遍预算。
这些不增加独立贡献或四方向完成数；见
[F4_OFFICIAL_EVALUATION](PERCEPTUAL_MFT_20261125_F4_OFFICIAL_EVALUATION.md)。

F2 的直接证据见 [八组完整 Test 分析](PERCEPTUAL_MFT_20261124_FACTORIAL_ASSESSMENT.md)，
F1/F3 的冻结方案见 [native study](../../configs/perceptual_native_pyramid_study_20261124.json)，
F4 的边界和原始检查见 [工程报告](PERCEPTUAL_MFT_20261124_FUSION_MEMORY_ENGINEERING.md)。
当前 F1/F3 使用中心周边对比、质量门控、光流和递归滤波等经典启发；
基础算法、模块命名、代码完成或 loss 下降均不能直接作为创新贡献。
F3 是本轮经典算法启发的替代假说，不冒充原任务书 GRU 实验的复现。

历史 `qra_fusion.py` 确有 DynamicQuality、Spatial、Redundancy 和 Temporal
类，但 Phase VI 结果使用 `causal_cache`，与固定 November DiffV2IR PNG
不同。它们不能直接计入本轮当前 IR 四方向完成数。原 native36/neck72
身份分支读取的候选描述也不等同于把实际学得的视觉融合 P3 与可靠性传给
F4 记忆；同一分支的许多行为组合不增加独立架构数。

前序正式拟合和所有负 Dev 对象仍按已登记方案完成完整七段 Test，
不读 Test 决定主模型或触发 F4。某一个视觉家族若失败，只能否定其自身
假说；不能据此把 F4 或其他当前 IR 架构一概判为已有证据排除。
固定 seed0、物理 GPU4/5 最多两卡、64 GiB 工作目录及 40 GiB 独立空间
余量继续适用。历史 Test 暴露及原 RGB 预训练见过当前 Dev 的限制保留。

F4执行上下文和正式生产/分离CPU评估入口已部分实现，17项组件检查及
实际CPU preflight完成；六个父模型完成标记仍缺，四个后续入口未齐。
无完整登记、新GPU、学习融合转移或新HOTA，不能增加方向完成数。
见[F4_WORKFLOW_CONTEXT](PERCEPTUAL_MFT_20261125_F4_WORKFLOW_CONTEXT.md)。

F4独立测速组件后续已实现，23项人工检查通过。实际Train RGB八帧初次
精确匹配，修订后重跑发生摘要异常；后续零记忆逐帧诊断复现旧参照，但
原因未解决，不能放行资格或据此增加方向完成数。GPU工程/调度/导出仍缺，
无新GPU、学习融合转移或HOTA；见[F4_RUNTIME_COMPONENTS](PERCEPTUAL_MFT_20261125_F4_RUNTIME_COMPONENTS.md)。

F4真实GPU工程入口及独立比较器后续已实现，19项人工检查通过；两份
原RGB CPU模型在固定Train八帧的检测/状态/当前ROI重放完全一致。此前CPU
异常原因仍未知，尚无真实GPU或学习IR资格、完整Dev/Test或HOTA，不能
增加方向完成数；完整调度/导出仍待实现。见[F4_GPU_ENGINEERING_COMPONENTS](PERCEPTUAL_MFT_20261125_F4_GPU_ENGINEERING_COMPONENTS.md)。

F4完整逐配置出版/Dev固定分析组件后续已实现，25项人工文件检查通过；
实际preflight仍缺六个父模型标记和调度器，没有完整登记、真实F4结果、
GPU或新HOTA，不增加方向完成数或创新数。见[F4_RESULT_EXPORT_COMPONENTS](PERCEPTUAL_MFT_20261125_F4_RESULT_EXPORT_COMPONENTS.md)。

F4完整控制器后续已接齐并通过95项CPU检查，384来源锁定登记；实际CPU
控制器488189等待空间前序完整导出和实际退出，phases为空。真实CPU预检
确认五个入口齐全、六个父模型标记仍缺，资格未知；无张量／图像／指标／
GT读取或CUDA调用。实际GPU门槛、学习融合转移与正式Dev/Test仍未执行。
CPU人工完整流程和子进程检查不能证明融合贡献，旧CPU摘要异常原因亦仍
未知；四方向完成数、独立创新0和Goal未达结论不变。见
[F4_FULL_WORKFLOW](PERCEPTUAL_MFT_20261125_F4_FULL_WORKFLOW.md)。

剩余两组IR后续已核验48宏循环／两遍完整Train，仍继续第三遍，未新增
Dev/Test。三个已完成模型的当前／过去输入诊断也完成66个Train帧：固定
0.6支持为RGB680、当前IR671、过去IR682，IR在六域均无更高支持；0.1下
当前IR对过去+2而对RGB为0，0.7对RGB−10。IR前景差值排序AUC0.493933，
不能把特征／语义差异等同于任务互补。这是固定几何结构路线的Train样本
证据，不替尚未完成的原生CNN、时序或空间门控判定成败，不增加方向完成数。
详见[TRAIN_CUE_SPECIFICITY](PERCEPTUAL_MFT_20261125_TRAIN_CUE_SPECIFICITY.md)。

F2后续增加任务相关空间监督原型：相同推理结构和553705参数，四种监督
组合×RGB／灰度／IR，真实六帧、72次CPU工程更新完成，12项核心检查通过。
冻结RGB分配下的逐锚点检测损失差监督效用，前景框监督有已有工作基础。
直接梯度、72项缺失和四项IR图像roll响应均核对，但只证明计算路径生效。
R1零路径数值失败保留，R2未再次复现，根因未知；无正式拟合、GPU资格、
Dev/Test或独立贡献，不把这项训练扩展另计为融合架构。见
[TASK_UTILITY_CPU_ENGINEERING](PERCEPTUAL_MFT_20261125_TASK_UTILITY_CPU_ENGINEERING.md)。

# 新版 November IR：四个方向的实际覆盖

本核查区分代码、正式拟合、完整评估和独立贡献。目前四方向的最终完成
条件仍未满足，**已验证的独立多模态贡献为 0**。当前最高观测 Test
31.504 不是新选出的主模型，也未达到原始 RGB 29.529 加 5 点的目标。

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
| F1 动态质量 | `NativePyramidFusion(mode='quality')` 学习当前图像辅助金字塔，各 FPN 层预测质量 gate；匹配 RGB/gray/IR 和完整流程已登记、工程检查已做，正式拟合排在现有训练后 | 实际 GPU 门槛、完整 Train/Dev/Test、强对照、去质量等归因；尚无该模型的新 HOTA |
| F2 空间互补 | local correlation、结构描述匹配和有界残差已有当前 IR 训练与评估；S/C/R 八组属于同一空间残差家族的消融。缓存主模型 Dev 51.329、完整 Test 29.142，相对原始 RGB Test 为 −0.387。六组全量 Train 方案正在执行 | 全量拟合和对应完整 Dev/Test；不能把这些替代路线等同于任务书原空间 gate＋冗余抑制的完整当前 IR 验证，不能计八种架构 |
| F3 因果特征时序 | `NativePyramidFusion(mode='temporal')` 在 F1 基础上加入 RGB 当前到过去光流和正方差递归滤波，连续 T≤4 片段、跨片段状态且过去状态 detach；匹配流程已登记 | 正式拟合、完整 Dev/Test、去历史／去运动等对照，证明收益超过单帧和同容量 RGB。旧描述向量 GRU 不能代替此特征融合实验 |
| F4 融合特征轨迹记忆 | 本轮新增指定 tracker、三路记忆及 P3 ROIAlign；11 项 CPU 检查通过，完整 4834 帧零权重输出及内部数值状态与原 BYTETracker 严格一致 | 先通过当前 IR 视觉资格，再接入实际已学 P3/q、登记并运行完整正式比较和成本。现阶段只是工程原型，不能作为已验证方向或创新 |

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

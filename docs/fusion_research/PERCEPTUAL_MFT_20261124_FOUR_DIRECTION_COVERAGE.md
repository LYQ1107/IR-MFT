# 新版 November IR：四个方向的实际覆盖

本核查区分代码、正式拟合、完整评估和独立贡献。目前四方向的最终完成
条件仍未满足，**已验证的独立多模态贡献为 0**。当前最高观测 Test
31.504 不是新选出的主模型，也未达到原始 RGB 29.529 加 5 点的目标。

| 方向 | 当前 November IR 证据 | 尚需完成 |
| --- | --- | --- |
| F1 动态质量 | `NativePyramidFusion(mode='quality')` 学习当前图像辅助金字塔，各 FPN 层预测质量 gate；匹配 RGB/gray/IR 和完整流程已登记、工程检查已做，正式拟合排在现有训练后 | 实际 GPU 门槛、完整 Train/Dev/Test、强对照、去质量等归因；尚无该模型的新 HOTA |
| F2 空间互补 | local correlation、结构描述匹配和有界残差已有当前 IR 训练与评估；S/C/R 八组属于同一空间残差家族的消融。缓存主模型 Dev 51.329、完整 Test 29.142，相对原始 RGB Test 为 −0.387。六组全量 Train 方案正在执行 | 全量拟合和对应完整 Dev/Test；不能把这些替代路线等同于任务书原空间 gate＋冗余抑制的完整当前 IR 验证，不能计八种架构 |
| F3 因果特征时序 | `NativePyramidFusion(mode='temporal')` 在 F1 基础上加入 RGB 当前到过去光流和正方差递归滤波，连续 T≤4 片段、跨片段状态且过去状态 detach；匹配流程已登记 | 正式拟合、完整 Dev/Test、去历史／去运动等对照，证明收益超过单帧和同容量 RGB。旧描述向量 GRU 不能代替此特征融合实验 |
| F4 融合特征轨迹记忆 | 本轮新增指定 tracker、三路记忆及 P3 ROIAlign；11 项 CPU 检查通过，完整 4834 帧零权重输出及内部数值状态与原 BYTETracker 严格一致 | 先通过当前 IR 视觉资格，再接入实际已学 P3/q、登记并运行完整正式比较和成本。现阶段只是工程原型，不能作为已验证方向或创新 |

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

# 当前正式候选的 Dev 选模范围与 Test 冻结顺序

本次直接读取现有执行源码，核对 native 与其后七个队列的选模范围。它没有重新选模型、增加跨家族统一选模规则，或把历史最高 Test 行升级为正式主模型。机器可读结果见 [selection scopes JSON](PERCEPTUAL_MFT_20261126_SELECTION_SCOPES.json)。

## 已有规则

| 家族 / 队列 | 主模型与对照如何确定 | Test 之前的源码顺序 |
| --- | --- | --- |
| 原生多尺度 native | quality/temporal × RGB/gray/IR 六组中，按完整 Dev 选正常 IR 主模型；保留全部六组各自 Dev 最佳模型。去除消融同时保留自身 Dev 最佳与原主模型轮次。沿用原来的 39 指标选模精度与并列规则。 | `launch_native_pyramid_study.py:291` 选模型，292 冻结正常选择，324 完整 Dev，329 冻结 Test 角色，343 才执行 Test 图像推理。 |
| CGR 回补 | 继承六组训练及输出策略原有完整 Dev 选择；13 个角色对应 12 个唯一条件。回补覆盖不改变原主模型。 | `launch_correlation_recovery_workflow.py:242` 完整 Dev 证明，248 冻结原有角色，256 执行完整 Test 图像推理。 |
| neck72 | 主模型头 `N72_robust_new_ir` 与 consensus 读出在新头 Dev 前已固定。每条头/读出曲线只按 Dev HOTA、并列时较早轮次选权重。 | `launch_neck72_study.py:253` 选权重，255 补齐所有所选条件 Dev，260 冻结，261–262 才处理 Test。 |
| neck72 测速 | 继承已冻结准确率实验的条件，没有新的准确率选模。 | `launch_neck72_runtime.py:193` 等待原任务退出；197 验证 Dev 输入/输出等价；201 才执行 Test 成本检查。 |
| 2×2 融合消融 | 继承 native 已冻结的主模型模式、轮次与共同预算；四个 IR 消融单元同时保留配对轮次和各自 Dev 最佳。 | `launch_fusion_factorial.py:215` 完整 Dev 冻结，216 计划固定结果复用，217 才产生缺失 Test 结果。 |
| 空间融合 spatial | 九组结构/模态正常训练中按完整 Dev 选 IR 主模型；保留九组各自最佳、匹配及本家族最强 RGB/灰度。使用官方 CSV 全精度 HOTA、AssA、IDF1 与既定并列规则。 | `launch_spatial_native.py:275` 选模型，277 冻结正常选择，296 完整 Dev，301 冻结 Test 角色，306 执行 Test。 |
| F4 融合记忆 | 只在 native 与 spatial 原有冻结 IR 主模型之间，根据完整 Dev 和严格对照选择合格父模型，不重新选父模型轮次。然后按 F4 完整 Dev 选择记忆方案；普通 RGB 记忆胜出时如实保留。 | `fusion_memory_pipeline.py:268` 冻结正常选择，270 冻结额外 Dev 后的全部 Test 条件，271 执行 Test。父模型均不合格时只报告该拒绝，不能宣称已完成 F4 实验。 |
| TaskUtility | 十二组正常模型按完整 Dev 选择 IR 主模型；保留十二组各自最佳、匹配及本家族最强 RGB/灰度。额外 Dev 去除与几何干预不能替换主模型。 | `task_utility_controller.py:138` 冻结正常选择，142 完整额外 Dev，144 冻结全部条件，145 执行 Test；`task_utility_selection.py:119` 强制保留 26 个完整 Test 对象。 |

这里 spatial、TaskUtility 中的“最强 RGB/灰度”均有明确家族范围；F4 的跨家族父模型选择也只覆盖预先登记的 native/spatial 两个来源。这些规则不能解释成“看全部 Test 后选一个最好结果”。所有正式结果需要保留负结果，并把正式 Dev 冻结角色与描述性的历史最高 Test 行分别标明。

## 实际完成范围

核查时 native 处于 `NORMAL_TRAIN_09`。原控制器与后七个控制器的 PID、内核启动时刻、原始命令 SHA256 均直接核对；后七个控制器仍为 CPU 等待状态，`CUDA_VISIBLE_DEVICES=''`，且阶段作业为空。[当前进程与源码证明](perceptual_20261126_selection_scopes/actual_controller_and_source_verification.json)保存了实际读取结果。

原 native 登记的 102 个源码/配置文件和最终 TaskUtility 登记的 427 个文件逐项重算 SHA256，均未变化。这是两个登记集合的核查，不声称后者覆盖全部其它家族配置。

[18 段原始源码与各自文件 SHA256](perceptual_20261126_selection_scopes/static_source_call_order_evidence.json)保存了上述顺序的静态依据。读取源码不等于排队实验已经通过 GPU 工程检查、完成训练或产生 Dev/Test 指标。以后仍需核对真实退出码、冻结文件、全帧结果和原输入字节。

## 仍需报告的限制

- 历史 Test 已被多轮实验观察，最终结果属于历史集上的探索性比较；RGB 预训练也见过当前 Dev 视频。
- 伪红外由当前 RGB 生成，不构成独立物理热传感器的信息源。
- 仅在 IR 条件下完成的四单元消融，尚不能单独证明某项交互为 IR 独有贡献；与 RGB/灰度的公平归因仍需单列。
- 现有 native 测速按七个 Test 序列各取前 32 帧、前 8 帧预热，合计只计时 168 帧，并明确排除离线 DiffV2IR 生成。不能称为完整 Test 全系统 FPS。原生成记录的成本正在另行核查，完整生成与跟踪一体计时仍未完成。

本次没有读取 Test GT 或准确率，没有新 GPU 工作，没有新增创新点证明；数值目标与独立贡献目标仍未完成。

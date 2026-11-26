# Causal auxiliary modality audit

当前固定11月DiffV2IR版本的总审计状态为**部分完成**。本轮研究继续使用固定新版伪红外PNG，没有切换到EMA差分表示。伪红外由RGB生成，不是独立物理热传感器输入；旧版`infrared`、EMA辅助表示和新版DiffV2IR结果分别保留来源。

## 当前固定11月DiffV2IR

原始M3FD/EMA生成器与SAM ViT-B权重已逐字节校验，实际GPU4完成8次Train生成调用，覆盖5个不同RGB帧。模型代码、实际载入模块、固定配置、环境、RGB／伪红外／SAM哈希和每次调用结果见[真实检查及原件](PERCEPTUAL_MFT_20261126_NOVEMBER_GENERATOR_CAUSALITY.md)与[机器报告](PERCEPTUAL_MFT_20261126_NOVEMBER_GENERATOR_CAUSALITY.json)。固定caption、BLIP关闭，每帧沿用原生成种子20261101；没有新增训练种子或替换固定PNG。

| 要求 | 当前实际证据 | 覆盖范围与状态 |
| --- | --- | --- |
| 来源与固定权重 | 原生成脚本、配置及运行时模块SHA；实际M3FD与SAM权重完整SHA | 本次固定资产核验通过 |
| 改变未来不改变前缀 | 两套完整三帧目录在生成前均已建立；BT-003前两帧相同，第三帧改为BT-001内容；故意读取未来的CPU负对照能够区分 | 所测两帧的伪红外与SAM均逐像素相等；小样本通过 |
| 切换序列后重置 | BT-003首帧、另一序列、返回BT-003首帧 | 所测伪红外与SAM逐像素相等；小样本通过 |
| 与原固定缓存比较 | 全部8次调用对应IR／SAM像素比较，包含重复输入 | 8/8相等；实际为5个不同Train RGB帧 |
| 全部Test缓存与逐帧在线生成一致 | 当前没有覆盖全部7段15985帧的完整在线一致性证明 | **未完成**；上述Train样本不能替代此项 |
| 整系统速度包含生成开销 | 原Test生成组件记录已审计；当前小样本也记录了加载、推理、写入和资源成本 | **未完成完整生成＋融合检测＋跟踪系统计时**；不是端到端FPS |

两次早期辅助尝试在任何GPU生成前分别因夹具修正、挂载NAS大权重读取较慢而被停止，实际生成均为0；原始非零退出、源代码、配置和终止事件保留在上述证据目录。最终辅助生成进程实际退出0，7.524GiB私有匿名RAM权重缓存已释放；不复制大权重或原RGB到磁盘，不修改原数据与全局缓存链接，不向MFT训练发送信号。当前实验仍只使用物理GPU4／5。

[历史完整Test生成组件成本](PERCEPTUAL_MFT_20261126_NOVEMBER_AUXILIARY_COST.md)核对了15985帧的31970份原生成／SAM记录，原两项选定计时区域之和平均4.102096秒／帧，并对231个Test样本的693份现存RGB／IR／SAM文件核对字节。它不等于全Test重新在线生成，也不含完整模型加载、读写及MFT系统开销。当前8次Train生成检查也不能证明全部48070帧缓存复现或跟踪精度改善。

当前MFT Protocol V2的角色如下，以[冻结协议](../../configs/cfp/protocol_v2.json)为准：

| 角色 | 序列 | 帧数 |
| --- | --- | ---: |
| Train | BT-001、BT-003、MSK-002、PF-001、SN-001、SN-015 | 27251 |
| Dev | BT-005、SN-013 | 4834 |
| 历史Test | BT-002、BT-004、MSK-003、PF-002、SN-009、SN-011、SN-014 | 15985 |

## 保留的历史EMA样本审计

[同名旧JSON](CAUSAL_AUXILIARY_MODALITY_AUDIT.json)保持原字节，其`PASS`只属于`CausalAuxiliaryGenerator`：当前亮度减去前缀EMA背景，先输出再更新背景。该文件15段视频各检查8个样本，不能用来证明固定11月DiffV2IR生成器或全帧缓存通过。旧JSON把BT-005、SN-013的`split`字段写为`train`，与当前协议的Dev角色不同；不以该旧字段重定义本轮划分。旧JSON的SHA256为`e15096e47a6536cfde93bfa98140911f2d84d0385c201517ea47f8417e6b76e8`。

本页不新增独立创新或HOTA结果。当前训练与评估进度见[共同九轮证明](PERCEPTUAL_MFT_20261126_NATIVE_ALL_SIX_NINE_EPOCHS.md)和[持续更新的研究状态](PERCEPTUAL_MFT_20261125_NATIVE_TRAINING_PROGRESS.md)。

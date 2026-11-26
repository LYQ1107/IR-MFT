# 固定11月伪红外：小样本生成因果性与缓存一致性

真实 GPU4 检查已正常结束：8次生成、5个不同 Train RGB 帧，辅助子进程与工具会话均退出0。所测两个相同前缀帧在改变后续帧后，伪红外和SAM输出均逐像素不变；切换到另一序列再返回，输出也完全一致。全部8次调用均与对应原始固定缓存的伪红外、SAM像素一致。

该结果补充生成流程的因果性与缓存一致性证据。它不增加跟踪精度结果，不计作独立创新，不代表48070帧或全部Test缓存已复现；完整生成加检测跟踪的系统运行成本仍待实测。

| 检查 | 实际结果 |
|---|---|
| 原序列 | Train BT-003，第1/2/3帧 |
| 改变未来 | 第1/2帧相同，第3帧换为Train BT-001第3帧 |
| 未来预置 | 两套完整三帧目录均在生成任一前缀前建立 |
| 故意读取未来的CPU负对照 | 两个前缀均检测出不同后续帧 |
| 前缀输出比较 | 两帧的伪红外与SAM均逐像素相等 |
| 序列重置 | BT-003第1帧→另一序列→返回，第1帧输出相等 |
| 固定缓存对照 | 8/8伪红外、8/8SAM相等；实际为5个不同RGB帧 |
| 原始数据与模型训练 | 固定PNG不替换；原MFT训练不发送信号、不新增训练或随机种子 |

复用原始DiffV2IR M3FD/EMA与SAM ViT-B，BLIP关闭、固定caption，每帧采用原生成种子20261101，512 letterbox、25步、CFG 7.5/1.5/1.5。实际权重逐字节SHA、原始生成代码及运行时模块SHA、环境版本、逐帧RGB/IR/SAM哈希见原始证据。两个未来目录使用原RGB软链接，无RGB数据副本。

权重通过原始SSH传输工具加载到私有匿名内存并完整校验：共7.524210GiB，未写入磁盘。检查完成后，两个内存持有进程均退出0，临时匿名内存已释放；原始全局缓存链接不修改。主训练继续用GPU5，辅助检查用GPU4，实际观察的项目GPU始终限于4/5。原native102项及后续TaskUtility427项冻结源文件均核对不变。

八次调用的平均SAM选定区域为1.108218秒，DiffV2IR为3.840489秒；含本地预处理与写入、排除额外缓存对比的平均区域为5.348726秒。选定模型加载区域为33.787124秒，实际生成子进程墙钟133.954198秒；另有760.476089秒RAM预载，包含预载与清理的辅助总墙钟900.669755秒。辅助进程峰值已分配GPU内存8.488523GiB，NVML采样峰值17548MiB，峰值进程RSS15.201431GiB。两种GPU口径不同；RAM缓存与进程RSS也不能直接相加。该小样本检查含重复输入、额外哈希和缓存读取，输出写本地，不能将这些计时换算为正式FPS或完整MFT系统速度。

保留了两个早期中断尝试：r1在完成大权重读取和任何GPU生成之前，为加强未来预置夹具而停止；r2同样在任何GPU生成之前，因挂载NAS大权重读取较慢而停止，随后改用既有SSH匿名RAM预载。两个被停止的自有子进程实际退出−15，工具会话退出1，实际生成帧均为0；不能当作通过或跟踪模型失败。r2旧回执中“夹具已准备”的启动字段是意图，实际夹具尚未产生，以保留的原始文件、生成数和终止事件为准。

已验证的CPU夹具检查只检查目录与故意泄漏的负对照；真实生成结论来自随后GPU8次调用。可执行检查为 `tools/audit_fixed_November_generator_causality.py`，冻结配置为 `configs/perceptual_fixed_November_causality_probe_20261126.json`。该工具绑定本次原始进程身份及固定本地生成资产；进程结束后不能直接复用旧PID配置。原始RAM监督程序与各次源代码均保留，匿名FD路径在清理后失效是预期行为。公开内容只有代码、配置、JSON、CSV、日志，不含生成PNG、SAM、RGB或权重。

[机器报告](PERCEPTUAL_MFT_20261126_NOVEMBER_GENERATOR_CAUSALITY.json) · [真实GPU完整结果](perceptual_20261126_November_generator_causality/actual_GPU_complete.json) · [全部8次记录](perceptual_20261126_November_generator_causality/all_eight_generation_calls.csv) · [实际退出与RAM释放](perceptual_20261126_November_generator_causality/actual_tool_and_RAM_release_verification.json) · [原始监督回执](perceptual_20261126_November_generator_causality/actual_GPU_supervisor_execution.json) · [SHA清单](perceptual_20261126_November_generator_causality/publication_manifest.json) · [已有全Test生成组件成本](PERCEPTUAL_MFT_20261126_NOVEMBER_AUXILIARY_COST.md)

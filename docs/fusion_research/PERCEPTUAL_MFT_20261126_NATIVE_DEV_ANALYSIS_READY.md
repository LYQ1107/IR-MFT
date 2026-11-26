# 完整原生Dev分析入口已通过CPU检查

新增 `tools/analyze_mft_native_complete_dev.py`，读取原独立审计完成的六／九／十二轮Dev报告。它要求准确阶段、全部正常／错位／缺失条件、两段完整视频4834帧、原始worker实际数值退出0、原CSV哈希及原六组rounded-official最佳选择；扩展阶段另要求原输入字节摘要检查通过。不改变预算或选择，不运行模型、官方评估、Test、图像或权重读取。

真实62项六轮报告回归已通过：六组各自最佳、16项汇总比较、48项逐视频比较、六条正常曲线36个点、同轮架构与输入交互、原共同预算决定，均与已公开分析完全一致。新增全部47字段有符号差值，同时保留原字段单位与来源列。新的文字只按实际数值生成表格，不复用前六轮写死的正负结论。六轮输入被要求作为九轮分析时，工具明确拒绝并且没有生成输出目录。此拒绝检查没有制造任何九轮结果。

CPU父工具实际退出0；参考六轮子进程退出0，预期拒绝的阶段错误子进程退出1，PID／启动tick／完整argv SHA、日志及原输入SHA均保留。第一次工程执行因把字段元数据对象当作字符串而退出1，尚未创建输出目录；失败源码、实际退出与日志保留，随后按真实字段结构修正。冻结native102项及TaskUtility427项源码均核对不变。

这只是分析入口工程完成，不是九轮或十二轮真实指标已完成，不增加HOTA或独立创新计数。当前完整九轮评估仍由原两卡流程执行；必须先等原完整Dev独立导出实际退出0，再绑定真实导出文件的SHA后调用：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 \
  tools/analyze_mft_native_complete_dev.py \
  --dev-report docs/fusion_research/PERCEPTUAL_MFT_20261126_NATIVE_STAGE9_FULL_DEV.json \
  --expect-report-sha256 <真实完整九轮导出文件的SHA256> \
  --stage 9 \
  --output-directory <新的独立输出目录>
```

命令中的SHA与目录需在真实阶段完成后确定；这里没有预设九轮结果或自动授权额外训练。输出为`complete.json`和`analysis.md`，目录必须全新。十二轮使用真实十二轮父报告及`--stage 12`。保留原六轮分析与全部失败结果，不新增全局Dev选模策略。独立创新仍0，Test至少+5点及2–3项贡献目标尚未达到。

[机器证明](PERCEPTUAL_MFT_20261126_NATIVE_DEV_ANALYSIS_READY.json) · [实际CPU工程与退出](perceptual_20261126_native_complete_Dev_analysis_engineering/actual_CPU_engineering_complete.json) · [真实六轮参考输出](perceptual_20261126_native_complete_Dev_analysis_engineering/actual_reference6_output.json) · [原件SHA清单](perceptual_20261126_native_complete_Dev_analysis_engineering/publication_manifest.json) · [当前训练与评估进度](PERCEPTUAL_MFT_20261125_NATIVE_TRAINING_PROGRESS.md)

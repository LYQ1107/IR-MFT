# 固定 11 月伪红外：完整七段 Test 的历史生成成本核查

这次从原全量生成队列和逐帧元数据恢复 SAM 与 DiffV2IR 的真实历史计时，没有重生成图像、加载模型或启动 GPU。七个 Test 序列全部 15,985 帧、31,970 份生成/掩码记录均通过检查。实际 CPU 工具与子进程均退出 0。

按所有帧加权，DiffV2IR 推理计时平均 **2.967860 秒/帧**，SAM 平均 **1.134236 秒/帧**，两段计时之和 **4.102096 秒/帧**。总计 65572.004523 秒是分开执行的历史同步计时累加，不能当成单卡独占 GPU 计算时长、真实并行流水线墙钟，或完整系统 FPS。

| Test 序列 | 全部帧 | DiffV2IR 秒/帧 | SAM 秒/帧 | 分项之和 秒/帧 | 记录中任一模型加载后首帧数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BT-002 | 493 | 2.919608 | 0.904092 | 3.823700 | 1 |
| BT-004 | 763 | 3.859432 | 1.361938 | 5.221370 | 2 |
| MSK-003 | 629 | 2.928375 | 1.058045 | 3.986420 | 2 |
| PF-002 | 3000 | 2.910728 | 1.076734 | 3.987461 | 6 |
| SN-009 | 4500 | 2.949874 | 1.263838 | 4.213712 | 9 |
| SN-011 | 4500 | 2.930231 | 1.119572 | 4.049803 | 9 |
| SN-014 | 2100 | 2.867872 | 0.964202 | 3.832074 | 5 |

## 数据与版本对应

每一帧都校验：原队列已完成、帧号连续且无重复、RGB/输出/掩码路径对应、记录的输出 SHA256 与队列一致、配置指纹一致、固定字幕、25 步 Euler ancestral、512×512 保持比例填充、M3FD/EMA 权重标识、固定生成种子 20261101、SAM ViT-B 参数和 RGB/掩码摘要对应。全部 Test 记录的 SAM 复用标记为 0，固定字幕不运行 BLIP。这是旧生成任务的固定种子，不是新增训练随机种子。

每段前 32 帧及最后一帧，共 231 帧，对原 RGB、伪 IR、SAM 掩码重新读取当前文件字节并计算 SHA256，共 693 次图像检查，全部匹配元数据。前 32 帧覆盖现有神经模型测速的全部输入前缀。其余原生成输出只核对原队列/元数据记录，本次没有重读全部 15,985 张 PNG；不能把抽查写成全量当前像素复核。原 48,070 帧全生成审计原件也随附保留，时间与范围分别说明。

全部原始元数据有按序 SHA256 汇总，每段还保存元数据声明的 RGB/IR 输入字节对摘要。这些摘要用于后续输入核对，不代表尚未运行的 native Test 已消费对应图像。原 `configs/cfp/protocol_v2.json` 中旧的 provenance 占位字段保持不变，因为该配置被在跑实验固定；新增证据独立保存在此报告。

权重 SHA256 是原配置、原生成身份与原脚本保存的来源标识；本次没有读取或重新校验大权重。原生成配置曾使用历史 Test 序列抽样探查，不能把这些历史集结果称为全新盲测。伪红外仍由 RGB 生成，并非独立热传感器。

## 计时包含与缺失

- DiffV2IR：`config_worker.py:31` 中 CUDA 同步后的 `run` 区域，含该函数内部预处理；排除计时前的 RGB 解码，以及计时后的灰度转换、PNG/元数据保存、模型加载与调度等待。
- SAM：`final_aux_worker.py:69-75` 的 `generator.generate(rgb)` 和同步；排除 RGB 解码、掩码并集、保存、模型加载与调度等待。
- 原生成发生于 11 月初共享的 A100-SXM4-40GB。记录中的物理 GPU0–3 属于该历史生产，不是当前新 GPU 作业；现在仍只用 GPU4/5 训练。没有把不同硬件占用下的旧抽样速度当成当前优化收益。
- 11 月 2 日逐帧 BLIP 探针的 8.714986 秒/帧来自不同字幕配置，不能当作当前固定字幕全量生成成本。本报告直接使用当前实际生成记录，不扣减后假冒新测速。
- 原 native 逐模型测速只计七段前缀的 168 帧，并排除离线生成。上述生成分项应单独列入成本表，不能把原模型 FPS 改名为生成在内的端到端 FPS。读写、加载、模型共存以及生成与跟踪一体化计时仍未完成。

因此，**完整生成在内的系统速度仍未测得**；本次补齐的是有实际记录支撑的生成推理成本分项。没有新增 Dev/Test 准确率，没有增加独立创新点计数。

## 复现与原件

实际 CPU 审计墙钟 608.297055 秒，峰值 RSS 0.049778 GiB；原任务、数据和活动优化器均未修改。命令依赖当前固定的生成归档、原只读队列数据库及数据集；这些大资产留在本地/NAS，不上传 Git。原小型配置、脚本、全量生成审计和本次输出均按字节复制到下列目录。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -u tools/audit_diffv2ir_generation_cost.py --output-directory /var/tmp/a_new_exclusive_generation_cost_audit_directory
```

[机器可读报告](PERCEPTUAL_MFT_20261126_NOVEMBER_AUXILIARY_COST.json) · [逐序列分项 CSV](perceptual_20261126_November_auxiliary_cost/per_sequence_component_costs.csv) · [当前抽查图像摘要](perceptual_20261126_November_auxiliary_cost/current_sampled_image_bytes.json) · [实际执行及退出码](perceptual_20261126_November_auxiliary_cost/actual_CPU_execution.json) · [原始日志](perceptual_20261126_November_auxiliary_cost/actual_CPU_log.txt) · [原件清单](perceptual_20261126_November_auxiliary_cost/publication_manifest.json) · [各家族 Dev 选择范围](PERCEPTUAL_MFT_20261126_SELECTION_SCOPES.md)

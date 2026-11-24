# F4：官方47字段评估与完整Dev文件冻结组件

评估和Dev文件冻结组件已实现，10项人工文件检查通过。既有RGB完整Dev
4834帧重新运行原官方TrackEval，39项摘要、逐序列／汇总47字段和原276列
CSV均精确复现，已有Dev HOTA47.579162182711485。**这是已有RGB Dev结果
的复核，不是F4、新候选或Test成绩，不计提升。**

## 评估与冻结实现

[评估组件](../../tools/fusion_memory_evaluation.py)在GT评估前核对两段Dev或
七段Test的原完整帧数、顺序MOT／检测包摘要、源图像锚点、冻结spec与
实际代码版本；拒绝Train工程片段、不完整或已变化的文件。官方评估
独立进行，保留全部原CSV，按原工具单位读取47字段，不平均逐序列HOTA。
全部276列必须为有限数值，原39摘要须匹配CSV，完整精度HOTA／AssA／IDF1
用于Dev选择，其余官方字段仍全部保留。

[冻结组件](../../tools/fusion_memory_dev_freeze.py)读取全部正常Dev文件后
先冻结正常选择，再依据该文件摘要登记干预／必要去阈值。所有负结果
必须齐全，干预不能回改正常主选择。零权重须精确复现原视觉模型的
原276列CSV及每段MOT／检测包；缺失IR须匹配相同权重、相同可靠性阈值
的新鲜RGB控制，核对完整表与轨迹字节，不能只比三个选择指标。

[10项检查日志](PERCEPTUAL_MFT_20261125_F4_OFFICIAL_EVALUATION_PYTEST.txt)
使用临时人工完整长度元数据／CSV，并不实际产生78个研究配置的结果。
覆盖所有78个条件的文件冻结、非AUC列NaN、截短／缺段、摘要不同、GT
评估前拒绝不完整输入、遗漏负条件、正常选择被替换、MOTP_sum变化但
三项选择值不变时仍拒绝缺失等价、Test准确率路径禁止读取。
最初一次夹具按错权重编号筛选，9通过／1失败；改为读取明确配置字段
后10项通过，原失败记录保留在证明文件中，不是科研失败或准确率。

## 实际官方链复核

[预登记](../../configs/perceptual_f4_official_evaluation_20261125.json)在执行前
写入。[实际CPU证明](PERCEPTUAL_MFT_20261125_F4_OFFICIAL_EVALUATION_CPU_PROOF.json)
由[probe](../../tools/probe_f4_official_evaluation.py)执行，只读取原canonical
RGB完整Dev已保存检测包与轨迹，没有新图像推理或权重加载。先核对
4834帧输入，写推理完成来源记录，再读两个Dev GT文件并校验冻结SHA。
GT只用于此独立评估，推理阶段读取0、Test读取0、模型张量加载0。

CPU7.698秒、峰值RSS0.611GiB；39摘要、全部逐序列／汇总47值和整个
[276列原表](perceptual_f4_engineering_20261125/existing_RGB_Dev_official.csv)
精确一致。该时间是已有轨迹的评估成本，不是推理FPS。原源文件未变，
仅删除本次评估器自动生成的冗余MOT副本，保留CSV与所有证明。

本次没有真实F4 Dev冻结、合格IR父模型或学习融合特征转移。实际父模型
仍因六个未来标记缺失而待验证；组件不能单独授权Test。正式GPU生产
入口、完整执行／独立成本／调度与最终分析仍需接通，独立创新0、目标
34.529未达、Goal保持active。历史Test暴露和原RGB训练见过当前Dev的
限制不变。

[来源审计](PERCEPTUAL_MFT_20261125_F4_EVALUATION_SOURCE_AUDIT.json)确认九份
既有登记、空间323来源／11依赖及测速137动态来源未改。4／5卡IR101／110
已到27/72宏循环；两者第一遍完整Train均已核验，见
[剩余IR第一遍交付](PERCEPTUAL_MFT_20261125_IR_REMAINING_FIRST_FULL_TRAIN_AUDIT.md)。
工作58.236GiB、独立盘空闲2744.820GiB，没有新GPU、NN拟合或图像／FPN盘缓存。

复现：CPU环境运行`tests/test_fusion_memory_evaluation.py`并保存源码摘要
记录，再传给`tools/probe_f4_official_evaluation.py --pytest-receipt`。已有
输出拒绝覆盖；正式生产调用仍须完整上下文和父模型资格。

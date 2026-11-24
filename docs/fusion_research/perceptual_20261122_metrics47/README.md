# 已发布结果的47项指标视图

当前快照覆盖 **1,747份完整官方CSV、5,726个逐序列/COMBINED行、69个实验阶段**。每个输出行包含47个指标，以及candidate/split/seq三个标识，共50列。源表、SHA256、字段/单位映射和阶段文件列表见[index.json](index.json)。原始276列官方CSV全部保留。

这47个名字直接来自当前固定TrackEval实现：HOTA类15项、CLEAR类22项、Identity类6项、Count类4项。官方摘要只显示其中39项，另外8项在明细中：

| 摘要未显示的字段 | 此视图单位 | 原始CSV字段 |
| --- | --- | --- |
| HOTA_TP | 19个HOTA阈值上的计数均值，可为小数 | HOTA_TP___AUC |
| HOTA_FN | 同上 | HOTA_FN___AUC |
| HOTA_FP | 同上 | HOTA_FP___AUC |
| CLR_F1 | 百分数 | CLR_F1 |
| FP_per_frame | 每帧误检数，不乘100 | FP_per_frame |
| MOTAL | 百分数 | MOTAL |
| MOTP_sum | 已匹配IoU之和，不乘100 | MOTP_sum |
| CLR_Frames | 帧数整数 | CLR_Frames |

全部比例字段转为百分数，保持HOTA等常用指标的展示口径；其余计数保留原单位。框架名为`___AUC`的列在实际`_detailed_row`中按阈值算术均值生成，本工具直接读取该值，不另行定义积分。HOTA_TP/FN/FP不能当作某个单独阈值的整数计数。

COMBINED行直接来自官方的序列合并结果，没有对各序列分数再求平均。导出对全部5,726行检查GT/检测计数守恒、身份计数关系和每帧FP单位。直接RGB基线的未舍入HOTA为29.5289706874，对应原摘要29.529；CLR_Frames仍为15985、IDSW仍为450、FP_per_frame仍为1.6494213325。此处没有产生新的准确率。

现有表中包含历史已完成的其他种子记录，它们只是原结果归档；本次没有追加训练、推理、种子复验或Test选模。覆盖范围仅是当前已发布官方表，不证明所有正式候选都已训练或完成Test。未来新完成的整组结果发布后，可重新运行`tools/export_perceptual_metrics47.py`更新视图；工具拒绝已有源表被修改或删除。

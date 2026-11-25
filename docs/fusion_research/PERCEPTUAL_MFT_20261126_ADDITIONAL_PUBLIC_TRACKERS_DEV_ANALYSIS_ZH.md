# 追加公开后端完整Dev：收益随关联器变化

完整2段4834帧Dev已全部结束。8个新增worker均实际退出0，独立CSV／MOT核对也实际退出0；加上原ByteTrack／SORT／OC-SORT，共20个方法×输入组合，全部官方47字段及逐段／COMBINED保留。新增8份原276列CSV、8份实际worker日志按字节发布。完整Test已经按相同冻结参数开始，包含所有负Dev条件。

| 新增固定后端 | IR HOTA | 同预算RGB | 灰度 | IR−RGB | IR−灰度 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BoT-SORT无ReID＋ORB | 47.950983 | 47.962779 | 47.411648 | −0.011796 | +0.539335 |
| HybridSORT无ReID官方预设 | 42.582257 | 42.066430 | 43.408960 | +0.515827 | −0.826703 |

BoT-SORT中，IR的DetA比同预算RGB高约0.171点、AssA低约0.097点，IDF1高约0.464点；HOTA接近持平，IDSW均112。HybridSORT中，IR相对RGB的AssA／IDF1提高约0.579／1.748点、IDSW少3；但灰度的HOTA、IDF1与IDSW更好。因此当前不能以“对RGB为正”替代对RGB／灰度同时成立的独立多模态证据。

这些结果延续了已发布的后端依赖现象：原ByteTrack的Dev IR−RGB约+0.886，SORT约+0.528、OC-SORT约+1.015；新增BoT-SORT转为接近零，HybridSORT虽为正却不超过灰度。不同后端／序列的增量不能相加，也不能把Dev差值换成七段Test提升。各后端的COMBINED来自官方合并，不取逐段简单平均。

本次固定的是原全量Train几何家族的Dev选定快照与priority／raw策略，所有后端共用同一组检测包。没有为BoT／Hybrid重选检测器，也没有调参搜索；这些是公开代码的已披露无ReID预设与MFT输出过滤，不能当成其完整论文person-ReID最优值。旧V3＋鱼类RGB ReID成绩继续分开。原生CNN／时序家族仍在GPU4／5训练，尚无完整Dev，当前表格不替它判断成败。

BoT-SORT全部帧使用实际RGB进行原ORB补偿，同一视频的RGB内容滚动SHA在四路检测流中一致。HybridSORT的单元素分数布局适配只改变返回形状，原状态、协方差与数值已核对；源码／输入包及原生产记录SHA保持一致。三次工程失败均在正式评测前，原代码、配置、执行与日志保留在登记包。新Test冻结记录绑定了全部8份完整Dev的metrics SHA，原主方案未重选。

各项CPU update／replay成本及实际RSS、wall time在JSON中。BoT的update包含ORB，replay另含RGB读图／解码；它们均排除检测、融合与IR生成，不能和原GPU端到端B4吞吐混用。没有新增训练、种子、GPU或权重／图像／FPN磁盘副本。

原七段Test主方案仍HOTA29.834177，相对固定RGB29.529为+0.305177，目标34.529未达。C只有条件消融支持，S不稳定、R有负贡献；本次公共关联器与布局兼容不增加创新点，独立计数仍0。历史Test暴露、RGB预训练见Dev及RGB派生伪IR限制保持。全部固定Test及原生训练／完整Dev继续，Goal保持active。

[完整五后端Dev表](PERCEPTUAL_MFT_20261126_ADDITIONAL_PUBLIC_TRACKERS_DEV.md) · [全部47字段JSON](PERCEPTUAL_MFT_20261126_ADDITIONAL_PUBLIC_TRACKERS_DEV.json) · [CSV](PERCEPTUAL_MFT_20261126_ADDITIONAL_PUBLIC_TRACKERS_DEV.csv)

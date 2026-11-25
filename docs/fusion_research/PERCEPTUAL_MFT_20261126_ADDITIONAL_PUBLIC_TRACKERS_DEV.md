# 五种公开跟踪后端：完整Dev比较

使用同一组冻结的源RGB／同预算RGB／灰度／IR检测包。原ByteTrack主方案保持；公开后端不计为原创贡献。

完整Dev：2段4834帧。

| 后端 | 源RGB HOTA | 同预算RGB | 灰度 | IR | IR−RGB | IR−灰度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bytetrack | 47.579162 | 47.878421 | 47.928942 | 48.764689 | +0.886267 | +0.835747 |
| sort | 36.809230 | 35.993662 | 36.114354 | 36.521420 | +0.527758 | +0.407066 |
| ocsort | 43.519676 | 42.373929 | 41.204479 | 43.388788 | +1.014859 | +2.184309 |
| botsort | 47.281767 | 47.962779 | 47.411648 | 47.950983 | -0.011796 | +0.539335 |
| hybridsort | 42.375034 | 42.066430 | 43.408960 | 42.582257 | +0.515827 | -0.826703 |

BoT-SORT保留ORB、无ReID；HybridSORT采用官方MOT17无ReID的HMIoU／TCM／低分关联预设。MFT统一输出面积>100，BoT不应用person宽高比过滤；其25fps参数及分数布局兼容均按登记保留。

表中快照／priority或raw策略沿用原ByteTrack完整Dev选择，没有针对其他后端重选检测器或做参数搜索。官方47字段、逐序列及COMBINED均在JSON，新增原276列CSV及实际worker日志按字节保存。

CPU update包含BoT的ORB，replay另含RGB读图／解码；均不含检测器、融合或IR生成，不能作为端到端FPS。较早的BoT／HybridSORT V3＋鱼类RGB ReID结果使用不同输入，继续单独保留。

本页仅完成Dev审计。全部8项固定Test继续运行，包含负Dev条件；原7段Test主方案仍29.834177，目标34.529未达。

历史Test已暴露、RGB预训练见Dev及RGB派生伪IR限制保持。独立创新计数仍0，Goal保持active。

[完整JSON](PERCEPTUAL_MFT_20261126_ADDITIONAL_PUBLIC_TRACKERS_DEV.json) · [47字段CSV](PERCEPTUAL_MFT_20261126_ADDITIONAL_PUBLIC_TRACKERS_DEV.csv)

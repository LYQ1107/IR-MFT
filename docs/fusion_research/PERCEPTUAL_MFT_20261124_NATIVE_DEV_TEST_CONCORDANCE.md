# 原生身份分支：Dev/Test与自身关闭对照的一致性

使用原始276列官方CSV的COMBINED HOTA___AUC×100，保留完整精度；没有重新推理或评估。

23组配对中，Dev胜过RGB有7组，Test有14组，两者同时胜过仅3组。再要求同时胜过自身增量关闭对照，满足四项必要增益的有0组。

这些是必要数值检查，不是创新性判定。缺失Dev关闭对照保留未知；不能用弱RGB对照替代自身模块贡献。

| IR配置 | Dev相对RGB | Test相对RGB | Dev相对自身关闭 | Test相对自身关闭 |
| --- | ---: | ---: | ---: | ---: |
| NR01_consensus_query_rgb_ir | -0.021230 | +0.013189 | 未知 | 未知 |
| NR02_consensus_query_rgb_ir | -0.033670 | +0.002111 | 未知 | +0.000000 |
| NR07_ordered_query_rgb_ir | +0.842075 | -0.075104 | -1.253401 | -0.018056 |
| NR08_ordered_query_rgb_ir | -0.255108 | -0.285648 | 未知 | -0.001563 |
| NR12_reachable_query_rgb_ir | -0.043937 | -0.070436 | 未知 | 未知 |
| NR13_reachable_query_rgb_ir | -0.255108 | -0.146285 | -0.824323 | +0.145285 |
| NR17_consensus_query_new_ir | -0.018990 | +0.063453 | 未知 | 未知 |
| NR18_consensus_query_new_ir | -0.020722 | -0.000481 | 未知 | +0.000042 |
| NR23_ordered_query_new_ir | +0.049232 | +0.133726 | +0.361997 | -0.387788 |
| NR24_ordered_query_new_ir | +0.320813 | +0.090856 | 未知 | -0.332438 |
| NR28_reachable_query_new_ir | -0.213854 | +0.281952 | 未知 | 未知 |
| NR29_reachable_query_new_ir | +0.303563 | +0.055260 | +0.634483 | -0.343920 |
| NR33_counterfactual_query_rgb_ir | -0.168930 | +0.416768 | -1.905551 | -0.102529 |
| NR36_native_query_rgb_ir | -1.799879 | +1.021975 | +0.413578 | +0.802660 |
| NR38_counterfactual_query_new_ir | +3.777965 | -0.196239 | 未知 | 未知 |
| NR40_counterfactual_query_new_ir | -0.003684 | +0.293060 | 未知 | 未知 |
| NR41_counterfactual_query_new_ir | +3.777965 | -0.196812 | 未知 | 未知 |
| NR44_counterfactual_query_new_ir | +0.347385 | -0.183634 | 未知 | 未知 |
| NR45_counterfactual_query_new_ir | -0.173526 | +0.635084 | 未知 | 未知 |
| NR49_counterfactual_query_new_ir | -0.071943 | -0.193529 | 未知 | 未知 |
| NR50_counterfactual_query_new_ir | -1.813311 | +1.004435 | 未知 | 未知 |
| NR51_counterfactual_query_new_ir | -3.682220 | +0.011169 | -2.908692 | -1.271720 |
| NR56_native_query_new_ir | -1.039249 | +1.385048 | -3.320575 | -0.805049 |

所有数值都是HOTA绝对百分点。未据Test选择新主方法；历史Test暴露、RGB预训练见过Dev的限制继续适用。

关闭new前端的native query增量不等于关闭检测器全部IR路径。本表不代表四种独立融合架构、2–3项独立创新或达到+5目标。

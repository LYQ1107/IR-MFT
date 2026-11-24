# 原58种native行为的完整Test核对

全部58个冻结对象的HOTA范围为28.408–31.504，观测最大值相对原RGB29.529为+1.975点，尚无对象达到34.529。这只是既有结果范围，未据Test改选主方法。

23组严格匹配的RGB/IR对照中，14组IR较高、9组较低。最大配对差值+1.386对应IR29.794与RGB28.408；该IR相对原始29.529仅+0.265点，不能把较弱对照造成的差值当成+5或多项创新。

All58 pre-existing behaviors now have complete7Test coverage. 23 cue pairs match every inference field except cue and independently Dev-selected head snapshot; all12unmatched original behaviors retained. Positive pair differences are not automatically 2-3independent innovations, four architectures, or a>=5point result. No Test winner is promoted.

| 原IR组 | 原RGB组 | IR Dev轮次 | RGB Dev轮次 | IR Test HOTA | RGB Test HOTA | IR−RGB | IR−关闭增量 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NR01_consensus_query_rgb_ir | NR03_consensus_query_rgb_rgb | 30 | 18 | 30.169 | 30.156 | +0.013 | 未严格配对 |
| NR02_consensus_query_rgb_ir | NR04_consensus_query_rgb_rgb | 6 | 18 | 30.174 | 30.171 | +0.003 | +0.000 |
| NR07_ordered_query_rgb_ir | NR09_ordered_query_rgb_rgb | 18 | 30 | 30.156 | 30.231 | -0.075 | -0.018 |
| NR08_ordered_query_rgb_ir | NR10_ordered_query_rgb_rgb | 6 | 18 | 30.172 | 30.458 | -0.286 | -0.002 |
| NR12_reachable_query_rgb_ir | NR14_reachable_query_rgb_rgb | 18 | 18 | 30.860 | 30.931 | -0.071 | 未严格配对 |
| NR13_reachable_query_rgb_ir | NR15_reachable_query_rgb_rgb | 18 | 18 | 30.319 | 30.465 | -0.146 | +0.145 |
| NR17_consensus_query_new_ir | NR19_consensus_query_new_rgb | 18 | 18 | 31.504 | 31.440 | +0.064 | 未严格配对 |
| NR18_consensus_query_new_ir | NR20_consensus_query_new_rgb | 30 | 6 | 31.442 | 31.443 | -0.001 | +0.000 |
| NR23_ordered_query_new_ir | NR25_ordered_query_new_rgb | 18 | 30 | 31.054 | 30.921 | +0.133 | -0.388 |
| NR24_ordered_query_new_ir | NR26_ordered_query_new_rgb | 30 | 18 | 31.110 | 31.019 | +0.091 | -0.332 |
| NR28_reachable_query_new_ir | NR30_reachable_query_new_rgb | 18 | 18 | 31.046 | 30.764 | +0.282 | 未严格配对 |
| NR29_reachable_query_new_ir | NR31_reachable_query_new_rgb | 18 | 18 | 31.106 | 31.051 | +0.055 | -0.344 |
| NR33_counterfactual_query_rgb_ir | NR34_counterfactual_query_rgb_rgb | 18 | 6 | 30.683 | 30.266 | +0.417 | -0.103 |
| NR36_native_query_rgb_ir | NR37_native_query_rgb_rgb | 18 | 12 | 30.849 | 29.827 | +1.022 | +0.803 |
| NR38_counterfactual_query_new_ir | NR39_counterfactual_query_new_rgb | 18 | 18 | 30.407 | 30.603 | -0.196 | 未严格配对 |
| NR40_counterfactual_query_new_ir | NR42_counterfactual_query_new_rgb | 18 | 18 | 30.964 | 30.671 | +0.293 | 未严格配对 |
| NR41_counterfactual_query_new_ir | NR43_counterfactual_query_new_rgb | 18 | 18 | 30.408 | 30.604 | -0.196 | 未严格配对 |
| NR44_counterfactual_query_new_ir | NR46_counterfactual_query_new_rgb | 18 | 18 | 30.675 | 30.859 | -0.184 | 未严格配对 |
| NR45_counterfactual_query_new_ir | NR47_counterfactual_query_new_rgb | 18 | 18 | 29.891 | 29.256 | +0.635 | 未严格配对 |
| NR49_counterfactual_query_new_ir | NR52_counterfactual_query_new_rgb | 18 | 18 | 30.680 | 30.874 | -0.194 | 未严格配对 |
| NR50_counterfactual_query_new_ir | NR53_counterfactual_query_new_rgb | 18 | 18 | 30.659 | 29.655 | +1.004 | 未严格配对 |
| NR51_counterfactual_query_new_ir | NR54_counterfactual_query_new_rgb | 12 | 6 | 30.042 | 30.031 | +0.011 | -1.272 |
| NR56_native_query_new_ir | NR57_native_query_new_rgb | 12 | 12 | 29.794 | 28.408 | +1.386 | -0.805 |

HOTA/IDF1 differences here use the original39 headline rounded summaries. Full precision and every threshold/sequence remain in the unchanged276-columnCSV/47fieldviews. Equal HOTA alone is not exact track equivalence.

这58种是同一家族不同路线/记忆/读出配置，不是58种融合架构。所有旧配置由完整Dev选快照，Test只统一评估，固定主方法未改。12种IR专属去除/干预没有强行配成RGB对照，原结果仍完整保留。相同固定前端下比较新增向量，关闭增量不等于关闭new前端全部IR路径。

Cached CPU replay only; full detector/encoder/point/IR-generation costs excluded.

旧状态未保存数值退出码；这些值保留未知。核验依据已完成产物、原始来源、逐帧覆盖、实际进程退出及控制器完成前的零退出检查，没有补造exit_code=0。

历史Test已暴露、RGB预训练见过Dev、伪IR从RGB派生的限制保留。Goal未完成。

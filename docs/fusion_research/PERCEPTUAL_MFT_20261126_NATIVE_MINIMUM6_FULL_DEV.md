# 原生辅助金字塔：最低六轮完整 Dev

两段完整Dev共4834帧，保留36个正常检查点、12个实际IR错位、12个缺失模态与两种RGB基线。原控制器的全部65个生产/规范源/官方评估worker均已实际退出0；本审计不要求整个研究控制器已退出。

| 同模式各自Dev最佳 | RGB HOTA | 灰度 HOTA | IR HOTA | IR−RGB | IR−灰度 |
| --- | ---: | ---: | ---: | ---: | ---: |
| quality | 49.918478 | 49.413533 | 49.703387 | -0.215090 | +0.289855 |
| temporal | 48.097061 | 48.941466 | 50.115210 | +2.018149 | +1.173744 |

最佳检查点沿用原官方39字段的舍入值与既定排序；上表显示对应CSV的未舍入值。原统一训练预算决定：{"completed_stage": 6, "next_stage": 9, "extend_all_six": true, "comparisons": [{"mode": "quality", "cue": "rgb", "earlier_best_HOTA": 49.918, "current_best_HOTA": 49.918, "improvement_pp": 0.0}, {"mode": "quality", "cue": "gray", "earlier_best_HOTA": 48.467, "current_best_HOTA": 49.414, "improvement_pp": 0.9470000000000027}, {"mode": "quality", "cue": "ir", "earlier_best_HOTA": 48.061, "current_best_HOTA": 49.703, "improvement_pp": 1.642000000000003}, {"mode": "temporal", "cue": "rgb", "earlier_best_HOTA": 47.932, "current_best_HOTA": 48.097, "improvement_pp": 0.16499999999999915}, {"mode": "temporal", "cue": "gray", "earlier_best_HOTA": 48.941, "current_best_HOTA": 48.941, "improvement_pp": 0.0}, {"mode": "temporal", "cue": "ir", "earlier_best_HOTA": 47.408, "current_best_HOTA": 50.115, "improvement_pp": 2.7070000000000007}], "numerical_convergence_claim": false, "selection_input_split": "dev"}。

JSON保留全部47字段、逐序列及COMBINED、同轮对照和IR错位/缺失差值；原276列CSV、原metrics、完整生产清单、冻结Dev决定及worker日志按字节保存。未发布图像、权重、检测包或轨迹。缺失模态全部47字段、逐序列、完整CSV及MOT均精确等于新B1 RGB。

这是完整Dev阶段证据，尚不代表完整Test、收敛、独立创新或+5目标完成。后续严格沿原6/9/12规则推进，负结果同样进入必要Test。历史Test暴露、RGB预训练见Dev及伪IR由RGB生成等限制保持。

[全部62路指标](PERCEPTUAL_MFT_20261126_NATIVE_MINIMUM6_FULL_DEV.json) · [47字段总表](PERCEPTUAL_MFT_20261126_NATIVE_MINIMUM6_FULL_DEV.csv)

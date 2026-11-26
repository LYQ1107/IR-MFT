# 原生融合：完整9轮 Dev 独立核对

两段完整Dev共4834帧，累计92项；包括所有正常、错位与缺失条件及两种RGB。

| 同模式各自Dev最佳 | RGB HOTA | 灰度 HOTA | IR HOTA | IR−RGB | IR−灰度 |
| --- | ---: | ---: | ---: | ---: | ---: |
| quality | 49.918478 | 49.413533 | 49.703387 | -0.215090 | +0.289855 |
| temporal | 48.097061 | 48.941466 | 50.115210 | +2.018149 | +1.173744 |

跨两种结构的最强IR与最强RGB差为+0.196732点，必须同时保留。各自检查点排序沿用原官方39字段舍入规则；CSV未舍入值不用于重选。

全部阶段记录的RGB／原始伪IR图像字节消费摘要均与九轮Dev前冻结的六轮参考一致。这是生产者消费摘要核对，不是独立图像重放或逐图清单。全部缺失条件的47字段、逐序列、完整CSV与MOT精确等于新鲜B1 RGB。

原共同预算决定：{"completed_stage": 9, "next_stage": null, "extend_all_six": false, "comparisons": [{"mode": "quality", "cue": "rgb", "earlier_best_HOTA": 49.918, "current_best_HOTA": 49.918, "improvement_pp": 0.0}, {"mode": "quality", "cue": "gray", "earlier_best_HOTA": 49.414, "current_best_HOTA": 49.414, "improvement_pp": 0.0}, {"mode": "quality", "cue": "ir", "earlier_best_HOTA": 49.703, "current_best_HOTA": 49.703, "improvement_pp": 0.0}, {"mode": "temporal", "cue": "rgb", "earlier_best_HOTA": 48.097, "current_best_HOTA": 48.097, "improvement_pp": 0.0}, {"mode": "temporal", "cue": "gray", "earlier_best_HOTA": 48.941, "current_best_HOTA": 48.941, "improvement_pp": 0.0}, {"mode": "temporal", "cue": "ir", "earlier_best_HOTA": 50.115, "current_best_HOTA": 50.115, "improvement_pp": 0.0}], "numerical_convergence_claim": false, "selection_input_split": "dev"}。

该文件仅为完整Dev阶段核对；没有运行新推理、训练、TrackEval或读取Test，不重新选择最终方案，不代表收敛、独立模块贡献或Test提升五点。原研究控制器仍须独立核对最终实际退出码。

[全部指标](PERCEPTUAL_MFT_20261126_NATIVE_STAGE9_FULL_DEV.json) · [47字段总表](PERCEPTUAL_MFT_20261126_NATIVE_STAGE9_FULL_DEV.csv)

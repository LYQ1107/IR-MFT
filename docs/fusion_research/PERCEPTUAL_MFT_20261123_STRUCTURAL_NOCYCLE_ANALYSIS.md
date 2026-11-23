# No-cycle structural refinement: matched controls and clean removals

Complete4834-frame Dev; original RGB pretraining included these scenes. Primary IR epoch6/raw was frozen before new control accuracy. All4snapshots and bothpolicies retained equally; no new seed runs.

| Input | Dev choice | HOTA | DetA | AssA | IDF1 | BT HOTA | SN HOTA |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original RGB | CG3_source_RGB | 47.579 | 68.864 | 32.908 | 55.865 | 55.174 | 41.901 |
| Matched RGB repeat | NC_rgb_no_cycle_s0_e06 | 47.918 | 68.918 | 33.351 | 56.256 | 55.802 | 42.016 |
| Matched gray | NC_gray_no_cycle_s0_e06_raw | 49.132 | 68.506 | 35.280 | 56.570 | 56.898 | 43.349 |
| Frozen primary IR | SR_ir_no_cycle_s0_e06_raw | 51.329 | 68.411 | 38.548 | 60.610 | 59.350 | 45.384 |

Primary IR minus original RGB: +3.750; minus strongest matched control: +2.197 HOTA.

| Removed mechanism | Primary − same6/raw | Primary − best removal curve |
| --- | ---: | ---: |
| structural_weights_validity_cost_confidence | +3.667 | +0.391 |
| semantic_centering | +4.480 | +3.358 |
| teacher_feature_risk_placement | +3.483 | +2.700 |

| IR intervention | HOTA | Change from correct IR at same policy |
| --- | ---: | ---: |
| NC_IR_image_roll_s0_e06 | 47.693 | -0.179 |
| NC_IR_unavailable_s0_e06 | 47.579 | -0.293 |
| NC_IR_image_roll_s0_e06_raw | 47.376 | -3.953 |
| NC_IR_unavailable_s0_e06_raw | 47.579 | -3.750 |

Hard cycle consistency was harmful and receives no contribution credit. The structure removal jointly removes local structural weights, validity and absolute-cost confidence. The risk removal removes only (1-p)*(p<.6), retaining structural confidence and feature/logit bounds. These task-specific mechanism groups still require fair modality controls, complete Test and prior-art analysis.

36 new evaluations and17 reused references, not53 new runs. No >=+5 Test gain or2–3 supported innovation claim is established by this Dev report.

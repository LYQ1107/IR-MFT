# Structural validation and relative semantic correction

Full4834-frame Dev, official combined metrics; not Test. All4 checkpoints and both readout policies retained equally for each of6 seed0 fits. No seed-robustness claim. Original RGB pretraining included current Dev scenes.

| Cue | Dev-selected configuration | HOTA | DetA | AssA | IDF1 | BT HOTA | SN HOTA |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original RGB | CG3_source_RGB | 47.579 | 68.864 | 32.908 | 55.865 | 55.174 | 41.901 |
| rgb | SR_rgb_full_s0_e06_raw | 49.455 | 68.373 | 35.801 | 57.586 | 59.344 | 41.823 |
| gray | SR_gray_full_s0_e24 | 47.679 | 68.761 | 33.088 | 55.093 | 58.891 | 38.740 |
| ir | SR_ir_full_s0_e06_raw | 49.599 | 68.451 | 35.967 | 57.606 | 59.455 | 42.009 |

IR minus original RGB: +2.020 HOTA; IR minus strongest matched control: +0.144 HOTA.

| Removal | Paired epoch/policy | Full − paired | Best removal | Full − best |
| --- | --- | ---: | --- | ---: |
| no_structure | SR_ir_no_structure_s0_e06_raw | +1.937 | SR_ir_no_structure_s0_e12_raw | -1.339 |
| no_cycle | SR_ir_no_cycle_s0_e06_raw | -1.730 | SR_ir_no_cycle_s0_e06_raw | -1.730 |
| no_centering | SR_ir_no_centering_s0_e06_raw | +0.330 | SR_ir_no_centering_s0_e24_raw | -0.015 |

| Intervention | HOTA | Change from same-epoch/policy correct IR |
| --- | ---: | ---: |
| SR_IR_image_roll_s0_e06 | 48.693 | +1.125 |
| SR_IR_image_roll_s0_e06_raw | 48.818 | -0.781 |
| SR_IR_image_roll_s0_e12 | 48.422 | +0.852 |
| SR_IR_image_roll_s0_e12_raw | 48.695 | -0.153 |
| SR_IR_image_roll_s0_e18 | 48.651 | +1.973 |
| SR_IR_image_roll_s0_e18_raw | 48.602 | +0.608 |
| SR_IR_image_roll_s0_e24 | 48.651 | +1.972 |
| SR_IR_image_roll_s0_e24_raw | 48.604 | +2.141 |
| SR_IR_unavailable_s0_e24 | 47.579 | +0.900 |
| SR_IR_unavailable_s0_e24_raw | 47.579 | +1.116 |

Two provisional mechanism groups: structural validation (no_structure; no_cycle is its nested diagnostic) and relative semantic centering. A nested cycle removal is not a third independent innovation. Classical self-similarity, RGB geometry copying and response subtraction are prior art. No novelty credit or Test gain is inferred from Dev alone.

The original >=+5 Test HOTA and2–3 contribution goal remains unestablished.

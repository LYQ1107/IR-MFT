# 完整开发集：去掉区域弱化训练

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

Same actual November pseudo-IR, frozen RGB source and 24-epoch adapter budget. Eight fully completed Dev policies. Removing the region weakening is compared against both the complete CGR repair and the earlier stronger CGA controls. No new Test selection.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CGR_ir_no_curriculum_s0_e06 | 48.618 | 68.777 | 34.399 | 56.544 | 90.203 | 91 | 0 | 0 |
| CGR_ir_no_curriculum_s0_e06_raw | 49.186 | 68.757 | 35.216 | 57.172 | 90.132 | 86 | 0 | 0 |
| CGR_ir_no_curriculum_s0_e12 | 47.385 | 68.717 | 32.715 | 53.793 | 90.041 | 96 | 0 | 0 |
| CGR_ir_no_curriculum_s0_e12_raw | 47.514 | 68.518 | 32.990 | 53.166 | 89.811 | 85 | 0 | 0 |
| CGR_ir_no_curriculum_s0_e18 | 45.655 | 68.718 | 30.376 | 51.506 | 90.028 | 96 | 0 | 0 |
| CGR_ir_no_curriculum_s0_e18_raw | 47.570 | 68.346 | 33.142 | 54.648 | 89.931 | 93 | 0 | 0 |
| CGR_ir_no_curriculum_s0_e24 | 45.651 | 68.705 | 30.376 | 51.502 | 90.011 | 96 | 0 | 0 |
| CGR_ir_no_curriculum_s0_e24_raw | 46.461 | 68.259 | 31.664 | 52.325 | 89.824 | 97 | 0 | 0 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

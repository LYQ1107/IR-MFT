# Ordinary pair calibration: complete common30-epoch Dev audit

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

All six ordinary association heads executed30 epochs. The frozen12-epoch-lookback rule forced a final common36-epoch stage. No Test outcomes chose the budget or model.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G_new_ir_s0_e30 | 46.637 | 68.631 | 31.741 | 53.294 | 90.480 | 113 | 0 | 0 |
| G_new_ir_s1_e30 | 44.843 | 68.519 | 29.400 | 51.193 | 90.493 | 121 | 0 | 0 |
| G_new_ir_s2_e30 | 46.103 | 68.487 | 31.084 | 52.765 | 90.508 | 119 | 0 | 0 |
| G_new_rgb_s0_e30 | 49.385 | 68.492 | 35.656 | 58.012 | 90.519 | 112 | 0 | 0 |
| G_new_rgb_s1_e30 | 46.543 | 68.182 | 31.814 | 54.492 | 90.506 | 114 | 0 | 0 |
| G_new_rgb_s2_e30 | 47.219 | 68.228 | 32.722 | 54.648 | 90.493 | 116 | 0 | 0 |
| G_rgb_ir_s0_e30 | 46.958 | 68.367 | 32.311 | 54.550 | 89.860 | 135 | 0 | 0 |
| G_rgb_ir_s1_e30 | 44.275 | 68.648 | 28.604 | 51.674 | 89.798 | 140 | 0 | 0 |
| G_rgb_ir_s2_e30 | 44.679 | 68.390 | 29.235 | 51.145 | 89.921 | 128 | 0 | 0 |
| G_rgb_rgb_s0_e30 | 44.660 | 68.426 | 29.197 | 50.021 | 89.895 | 124 | 0 | 0 |
| G_rgb_rgb_s1_e30 | 45.795 | 68.427 | 30.698 | 52.688 | 89.933 | 129 | 0 | 0 |
| G_rgb_rgb_s2_e30 | 43.796 | 68.435 | 28.076 | 50.378 | 89.936 | 129 | 0 | 0 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

# Strong RGB residual: common epoch24

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

Matched RGB/IR residuals all execute the registered common budget. Dev only; generic fitted RGB controls receive no multimodal novelty credit.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F_new_ir_s0_e24 | 46.775 | 68.943 | 31.776 | 54.098 | 90.349 | 120 | 0 | 0 |
| F_new_ir_s1_e24 | 44.583 | 68.352 | 29.120 | 51.212 | 90.392 | 119 | 0 | 0 |
| F_new_ir_s2_e24 | 46.495 | 68.389 | 31.657 | 53.784 | 90.362 | 127 | 0 | 0 |
| F_new_rgb_s0_e24 | 49.895 | 69.016 | 36.126 | 58.570 | 90.512 | 121 | 0 | 0 |
| F_new_rgb_s1_e24 | 45.424 | 68.824 | 30.034 | 52.287 | 90.420 | 120 | 0 | 0 |
| F_new_rgb_s2_e24 | 49.593 | 68.479 | 35.957 | 58.450 | 90.452 | 115 | 0 | 0 |
| F_rgb_ir_s0_e24 | 44.863 | 68.774 | 29.328 | 49.535 | 89.834 | 135 | 0 | 0 |
| F_rgb_ir_s1_e24 | 44.242 | 68.514 | 28.611 | 49.195 | 89.929 | 135 | 0 | 0 |
| F_rgb_ir_s2_e24 | 47.388 | 68.741 | 32.718 | 55.007 | 89.860 | 135 | 0 | 0 |
| F_rgb_rgb_s0_e24 | 46.883 | 68.419 | 32.185 | 55.567 | 89.901 | 129 | 0 | 0 |
| F_rgb_rgb_s1_e24 | 46.197 | 68.099 | 31.377 | 53.911 | 89.923 | 128 | 0 | 0 |
| F_rgb_rgb_s2_e24 | 48.870 | 68.439 | 34.944 | 56.442 | 89.929 | 124 | 0 | 0 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

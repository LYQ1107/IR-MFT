# Strong RGB residual: common epoch30

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

Full registered12 Dev configurations after the forced common extension; preserve every seed and frontend.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F_new_ir_s0_e30 | 50.409 | 68.509 | 37.123 | 58.304 | 90.394 | 114 | 0 | 0 |
| F_new_ir_s1_e30 | 46.370 | 68.482 | 31.436 | 53.790 | 90.394 | 120 | 0 | 0 |
| F_new_ir_s2_e30 | 46.589 | 68.425 | 31.767 | 53.445 | 90.459 | 124 | 0 | 0 |
| F_new_rgb_s0_e30 | 47.341 | 68.985 | 32.548 | 55.138 | 90.489 | 122 | 0 | 0 |
| F_new_rgb_s1_e30 | 45.802 | 68.769 | 30.558 | 52.825 | 90.394 | 117 | 0 | 0 |
| F_new_rgb_s2_e30 | 49.349 | 68.523 | 35.586 | 57.880 | 90.560 | 119 | 0 | 0 |
| F_rgb_ir_s0_e30 | 44.083 | 68.578 | 28.388 | 49.481 | 89.871 | 140 | 0 | 0 |
| F_rgb_ir_s1_e30 | 44.648 | 68.698 | 29.069 | 50.835 | 89.927 | 139 | 0 | 0 |
| F_rgb_ir_s2_e30 | 46.956 | 68.749 | 32.126 | 54.004 | 89.897 | 135 | 0 | 0 |
| F_rgb_rgb_s0_e30 | 45.849 | 68.271 | 30.842 | 53.782 | 89.867 | 136 | 0 | 0 |
| F_rgb_rgb_s1_e30 | 44.968 | 68.132 | 29.728 | 51.446 | 89.933 | 132 | 0 | 0 |
| F_rgb_rgb_s2_e30 | 48.486 | 68.376 | 34.428 | 55.935 | 89.955 | 125 | 0 | 0 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

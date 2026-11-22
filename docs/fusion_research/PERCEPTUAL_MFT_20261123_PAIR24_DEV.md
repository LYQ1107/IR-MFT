# Ordinary association calibration: complete12-row common24-epoch Dev audit

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

All six ordinary-control heads executed24 epochs with identical matched data and three head seeds. IR increments over raw point odds remain unstable. The frozen lookback rule forced all six onward to30 and36; common24 is retained as an intermediate complete curve point. No Test outcome selected this extension.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G_new_ir_s0_e24 | 45.485 | 68.467 | 30.267 | 52.827 | 90.502 | 122 | 0 | 0 |
| G_new_ir_s1_e24 | 44.528 | 68.484 | 29.004 | 51.231 | 90.506 | 118 | 0 | 0 |
| G_new_ir_s2_e24 | 45.697 | 68.684 | 30.458 | 52.484 | 90.484 | 117 | 0 | 0 |
| G_new_rgb_s0_e24 | 46.165 | 68.487 | 31.159 | 52.885 | 90.502 | 115 | 0 | 0 |
| G_new_rgb_s1_e24 | 49.382 | 68.235 | 35.789 | 58.791 | 90.543 | 111 | 0 | 0 |
| G_new_rgb_s2_e24 | 47.972 | 68.060 | 33.855 | 56.331 | 90.553 | 110 | 0 | 0 |
| G_rgb_ir_s0_e24 | 42.988 | 68.415 | 27.065 | 48.434 | 89.897 | 129 | 0 | 0 |
| G_rgb_ir_s1_e24 | 44.509 | 68.411 | 29.007 | 51.666 | 89.813 | 139 | 0 | 0 |
| G_rgb_ir_s2_e24 | 45.396 | 68.831 | 29.979 | 52.271 | 89.974 | 123 | 0 | 0 |
| G_rgb_rgb_s0_e24 | 43.667 | 68.434 | 27.915 | 49.198 | 89.880 | 135 | 0 | 0 |
| G_rgb_rgb_s1_e24 | 46.904 | 68.446 | 32.184 | 54.436 | 89.949 | 124 | 0 | 0 |
| G_rgb_rgb_s2_e24 | 44.881 | 68.448 | 29.474 | 51.470 | 89.955 | 131 | 0 | 0 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

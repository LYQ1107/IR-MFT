# Mismatch-robust native queries at final common36 budget

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

All matched fits completed the registered36-epoch cap, retaining every earlier checkpoint and stage history. Full two-video Dev; no Test selection. Retain the stronger point-only RGB controls; ordinary identity-memory heads do not receive novelty credit.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births | Prebirth ID recoveries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| U_new_ir_s0_e36 | 50.096 | 68.593 | 36.634 | 58.518 | 90.611 | 111 | 0 | 0 | 31 |
| U_new_ir_s1_e36 | 47.602 | 68.280 | 33.227 | 56.291 | 90.553 | 109 | 0 | 0 | 24 |
| U_new_ir_s2_e36 | 47.909 | 68.303 | 33.653 | 56.558 | 90.564 | 111 | 0 | 0 | 22 |
| U_new_rgb_s0_e36 | 49.123 | 68.897 | 35.077 | 57.479 | 90.558 | 113 | 0 | 0 | 33 |
| U_new_rgb_s1_e36 | 46.108 | 68.838 | 30.922 | 52.829 | 90.525 | 107 | 0 | 0 | 34 |
| U_new_rgb_s2_e36 | 47.607 | 68.317 | 33.222 | 55.498 | 90.450 | 116 | 0 | 0 | 31 |
| U_rgb_ir_s0_e36 | 44.366 | 68.670 | 28.711 | 49.908 | 89.839 | 133 | 0 | 0 | 39 |
| U_rgb_ir_s1_e36 | 46.962 | 68.756 | 32.119 | 54.183 | 89.944 | 120 | 0 | 0 | 33 |
| U_rgb_ir_s2_e36 | 44.685 | 68.529 | 29.186 | 50.852 | 90.000 | 129 | 0 | 0 | 32 |
| U_rgb_rgb_s0_e36 | 43.950 | 68.663 | 28.180 | 49.180 | 89.787 | 137 | 0 | 0 | 47 |
| U_rgb_rgb_s1_e36 | 45.731 | 68.783 | 30.451 | 52.444 | 89.927 | 123 | 0 | 0 | 45 |
| U_rgb_rgb_s2_e36 | 44.743 | 68.520 | 29.265 | 50.829 | 89.912 | 132 | 0 | 0 | 45 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

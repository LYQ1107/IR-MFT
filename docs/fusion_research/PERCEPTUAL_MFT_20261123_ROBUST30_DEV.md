# Mismatch-robust native queries, common30-epoch Dev

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

Full two-video Dev, all matched frontends/cues/seeds; common30 epochs. No Test selection. Native identity is weaker than the RGB reference on held-out videos. Preserve all rows; no established multimodal innovation.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births | Prebirth ID recoveries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| U_new_ir_s0_e30 | 50.096 | 68.593 | 36.634 | 58.518 | 90.611 | 111 | 0 | 0 | 31 |
| U_new_ir_s1_e30 | 47.602 | 68.280 | 33.227 | 56.291 | 90.553 | 109 | 0 | 0 | 24 |
| U_new_ir_s2_e30 | 47.909 | 68.303 | 33.653 | 56.558 | 90.564 | 111 | 0 | 0 | 22 |
| U_new_rgb_s0_e30 | 49.117 | 68.902 | 35.065 | 57.475 | 90.564 | 113 | 0 | 0 | 36 |
| U_new_rgb_s1_e30 | 46.086 | 68.835 | 30.894 | 52.830 | 90.519 | 108 | 0 | 0 | 32 |
| U_new_rgb_s2_e30 | 47.839 | 68.336 | 33.534 | 55.812 | 90.519 | 115 | 0 | 0 | 32 |
| U_rgb_ir_s0_e30 | 44.366 | 68.670 | 28.711 | 49.908 | 89.839 | 133 | 0 | 0 | 39 |
| U_rgb_ir_s1_e30 | 46.962 | 68.756 | 32.119 | 54.183 | 89.944 | 120 | 0 | 0 | 33 |
| U_rgb_ir_s2_e30 | 44.684 | 68.525 | 29.187 | 50.846 | 89.994 | 129 | 0 | 0 | 31 |
| U_rgb_rgb_s0_e30 | 43.959 | 68.666 | 28.192 | 49.172 | 89.763 | 137 | 0 | 0 | 48 |
| U_rgb_rgb_s1_e30 | 45.731 | 68.781 | 30.452 | 52.446 | 89.923 | 123 | 0 | 0 | 44 |
| U_rgb_rgb_s2_e30 | 44.743 | 68.520 | 29.265 | 50.829 | 89.912 | 132 | 0 | 0 | 45 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

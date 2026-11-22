# Clean-only-head consensus loss controls

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

Complete12 epoch18 clean-only matched loss-removal controls. Encoder/data/thresholds preserved.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births | Prebirth ID recoveries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T_new_ir_s0 | 49.446 | 68.563 | 35.707 | 58.067 | 90.568 | 113 | 0 | 0 | 26 |
| T_new_ir_s1 | 47.599 | 68.278 | 33.224 | 56.290 | 90.553 | 109 | 0 | 0 | 23 |
| T_new_ir_s2 | 48.071 | 68.840 | 33.617 | 56.533 | 90.556 | 110 | 0 | 0 | 23 |
| T_new_rgb_s0 | 49.465 | 68.563 | 35.734 | 58.058 | 90.547 | 112 | 0 | 0 | 28 |
| T_new_rgb_s1 | 47.599 | 68.278 | 33.224 | 56.290 | 90.553 | 109 | 0 | 0 | 23 |
| T_new_rgb_s2 | 47.912 | 68.320 | 33.649 | 56.564 | 90.515 | 115 | 0 | 0 | 27 |
| T_rgb_ir_s0 | 45.192 | 68.326 | 29.936 | 50.955 | 89.869 | 136 | 0 | 0 | 36 |
| T_rgb_ir_s1 | 46.463 | 68.441 | 31.585 | 53.605 | 89.944 | 122 | 0 | 0 | 30 |
| T_rgb_ir_s2 | 44.684 | 68.525 | 29.187 | 50.846 | 89.994 | 129 | 0 | 0 | 31 |
| T_rgb_rgb_s0 | 45.216 | 68.329 | 29.966 | 50.958 | 89.873 | 135 | 0 | 0 | 38 |
| T_rgb_rgb_s1 | 46.465 | 68.449 | 31.585 | 53.602 | 89.957 | 122 | 0 | 0 | 31 |
| T_rgb_rgb_s2 | 44.683 | 68.522 | 29.186 | 50.842 | 89.972 | 129 | 0 | 0 | 33 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

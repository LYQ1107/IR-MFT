# Clean-only consensus at final common36 budget

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

All matched fits completed the registered36-epoch cap, retaining every earlier checkpoint and stage history. Full two-video Dev; no Test selection. Retain the stronger point-only RGB controls; ordinary identity-memory heads do not receive novelty credit.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births | Prebirth ID recoveries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T_new_ir_s0_e36 | 49.445 | 68.560 | 35.707 | 58.061 | 90.558 | 113 | 0 | 0 | 26 |
| T_new_ir_s1_e36 | 47.599 | 68.278 | 33.224 | 56.290 | 90.553 | 109 | 0 | 0 | 23 |
| T_new_ir_s2_e36 | 48.071 | 68.840 | 33.617 | 56.533 | 90.556 | 110 | 0 | 0 | 23 |
| T_new_rgb_s0_e36 | 49.465 | 68.563 | 35.734 | 58.058 | 90.547 | 112 | 0 | 0 | 28 |
| T_new_rgb_s1_e36 | 47.599 | 68.278 | 33.224 | 56.290 | 90.553 | 109 | 0 | 0 | 23 |
| T_new_rgb_s2_e36 | 47.908 | 68.312 | 33.647 | 56.552 | 90.543 | 113 | 0 | 0 | 23 |
| T_rgb_ir_s0_e36 | 45.195 | 68.329 | 29.938 | 50.958 | 89.871 | 136 | 0 | 0 | 37 |
| T_rgb_ir_s1_e36 | 46.463 | 68.441 | 31.585 | 53.605 | 89.944 | 122 | 0 | 0 | 30 |
| T_rgb_ir_s2_e36 | 44.684 | 68.525 | 29.187 | 50.846 | 89.994 | 129 | 0 | 0 | 31 |
| T_rgb_rgb_s0_e36 | 45.213 | 68.326 | 29.964 | 50.955 | 89.871 | 135 | 0 | 0 | 37 |
| T_rgb_rgb_s1_e36 | 46.463 | 68.441 | 31.585 | 53.605 | 89.944 | 122 | 0 | 0 | 30 |
| T_rgb_rgb_s2_e36 | 44.699 | 68.523 | 29.207 | 50.841 | 89.979 | 128 | 0 | 0 | 34 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

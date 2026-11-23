# Clean-only native queries at final common36 budget

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

All matched fits completed the registered36-epoch cap, retaining every earlier checkpoint and stage history. Full two-video Dev; no Test selection. Retain the stronger point-only RGB controls; ordinary identity-memory heads do not receive novelty credit.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births | Prebirth ID recoveries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O_new_ir_s0_e36 | 49.157 | 69.095 | 35.021 | 57.582 | 90.553 | 114 | 0 | 0 | 33 |
| O_new_ir_s1_e36 | 47.843 | 68.264 | 33.574 | 55.756 | 90.540 | 110 | 0 | 0 | 30 |
| O_new_ir_s2_e36 | 47.761 | 68.875 | 33.167 | 55.440 | 90.512 | 111 | 0 | 0 | 26 |
| O_new_rgb_s0_e36 | 49.757 | 68.916 | 35.974 | 57.902 | 90.568 | 111 | 0 | 0 | 38 |
| O_new_rgb_s1_e36 | 45.954 | 68.232 | 30.991 | 52.969 | 90.506 | 108 | 0 | 0 | 27 |
| O_new_rgb_s2_e36 | 47.903 | 68.302 | 33.646 | 56.559 | 90.519 | 115 | 0 | 0 | 29 |
| O_rgb_ir_s0_e36 | 43.937 | 68.687 | 28.154 | 49.162 | 89.785 | 133 | 0 | 0 | 47 |
| O_rgb_ir_s1_e36 | 46.810 | 68.727 | 31.927 | 53.769 | 89.944 | 119 | 0 | 0 | 41 |
| O_rgb_ir_s2_e36 | 44.688 | 68.534 | 29.187 | 50.848 | 89.974 | 125 | 0 | 0 | 34 |
| O_rgb_rgb_s0_e36 | 43.097 | 68.512 | 27.152 | 47.408 | 89.858 | 138 | 0 | 0 | 49 |
| O_rgb_rgb_s1_e36 | 45.707 | 68.742 | 30.438 | 52.456 | 89.878 | 125 | 0 | 0 | 44 |
| O_rgb_rgb_s2_e36 | 44.558 | 68.524 | 29.021 | 50.394 | 89.944 | 136 | 0 | 0 | 45 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

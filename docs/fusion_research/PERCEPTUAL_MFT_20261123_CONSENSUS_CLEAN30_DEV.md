# Clean-only consensus, common30-epoch Dev

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

Full two-video Dev, all matched frontends/cues/seeds; common30 epochs. No Test selection. Native identity is weaker than the RGB reference on held-out videos. Preserve all rows; no established multimodal innovation.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births | Prebirth ID recoveries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T_new_ir_s0_e30 | 49.445 | 68.560 | 35.707 | 58.061 | 90.558 | 113 | 0 | 0 | 26 |
| T_new_ir_s1_e30 | 47.599 | 68.278 | 33.224 | 56.290 | 90.553 | 109 | 0 | 0 | 23 |
| T_new_ir_s2_e30 | 48.071 | 68.840 | 33.617 | 56.533 | 90.556 | 110 | 0 | 0 | 23 |
| T_new_rgb_s0_e30 | 49.465 | 68.563 | 35.734 | 58.058 | 90.547 | 112 | 0 | 0 | 28 |
| T_new_rgb_s1_e30 | 47.599 | 68.278 | 33.224 | 56.290 | 90.553 | 109 | 0 | 0 | 23 |
| T_new_rgb_s2_e30 | 47.914 | 68.325 | 33.649 | 56.568 | 90.540 | 115 | 0 | 0 | 25 |
| T_rgb_ir_s0_e30 | 45.195 | 68.329 | 29.938 | 50.958 | 89.871 | 136 | 0 | 0 | 37 |
| T_rgb_ir_s1_e30 | 46.463 | 68.441 | 31.585 | 53.605 | 89.944 | 122 | 0 | 0 | 30 |
| T_rgb_ir_s2_e30 | 44.684 | 68.525 | 29.187 | 50.846 | 89.994 | 129 | 0 | 0 | 31 |
| T_rgb_rgb_s0_e30 | 45.216 | 68.329 | 29.966 | 50.958 | 89.873 | 135 | 0 | 0 | 38 |
| T_rgb_rgb_s1_e30 | 46.463 | 68.441 | 31.585 | 53.605 | 89.944 | 122 | 0 | 0 | 30 |
| T_rgb_rgb_s2_e30 | 44.683 | 68.522 | 29.186 | 50.842 | 89.972 | 129 | 0 | 0 | 33 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

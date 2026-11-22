# Two-observation consensus, common30-epoch Dev

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

Full two-video Dev, all matched frontends/cues/seeds; common30 epochs. No Test selection. Native identity is weaker than the RGB reference on held-out videos. Preserve all rows; no established multimodal innovation.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births | Prebirth ID recoveries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S_new_ir_s0_e30 | 49.446 | 68.563 | 35.707 | 58.067 | 90.568 | 113 | 0 | 0 | 26 |
| S_new_ir_s1_e30 | 47.599 | 68.278 | 33.224 | 56.290 | 90.553 | 109 | 0 | 0 | 23 |
| S_new_ir_s2_e30 | 47.909 | 68.303 | 33.653 | 56.558 | 90.564 | 111 | 0 | 0 | 22 |
| S_new_rgb_s0_e30 | 49.465 | 68.563 | 35.735 | 58.061 | 90.562 | 112 | 0 | 0 | 27 |
| S_new_rgb_s1_e30 | 47.600 | 68.310 | 33.208 | 56.182 | 90.556 | 109 | 0 | 0 | 25 |
| S_new_rgb_s2_e30 | 47.925 | 68.326 | 33.665 | 56.588 | 90.519 | 115 | 0 | 0 | 27 |
| S_rgb_ir_s0_e30 | 45.192 | 68.326 | 29.936 | 50.955 | 89.869 | 136 | 0 | 0 | 36 |
| S_rgb_ir_s1_e30 | 46.463 | 68.441 | 31.585 | 53.605 | 89.944 | 122 | 0 | 0 | 30 |
| S_rgb_ir_s2_e30 | 44.684 | 68.525 | 29.187 | 50.846 | 89.994 | 129 | 0 | 0 | 31 |
| S_rgb_rgb_s0_e30 | 45.226 | 68.335 | 29.977 | 50.959 | 89.878 | 135 | 0 | 0 | 38 |
| S_rgb_rgb_s1_e30 | 46.463 | 68.441 | 31.585 | 53.605 | 89.944 | 122 | 0 | 0 | 30 |
| S_rgb_rgb_s2_e30 | 44.700 | 68.524 | 29.207 | 50.843 | 89.985 | 128 | 0 | 0 | 33 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

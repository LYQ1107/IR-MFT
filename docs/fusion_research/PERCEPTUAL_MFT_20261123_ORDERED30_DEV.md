# Clean-only native queries, common30-epoch Dev

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

Full two-video Dev, all matched frontends/cues/seeds; common30 epochs. No Test selection. Native identity is weaker than the RGB reference on held-out videos. Preserve all rows; no established multimodal innovation.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births | Prebirth ID recoveries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O_new_ir_s0_e30 | 49.804 | 69.116 | 35.933 | 58.016 | 90.571 | 112 | 0 | 0 | 36 |
| O_new_ir_s1_e30 | 46.491 | 68.258 | 31.707 | 54.038 | 90.528 | 113 | 0 | 0 | 30 |
| O_new_ir_s2_e30 | 47.761 | 68.875 | 33.167 | 55.440 | 90.512 | 111 | 0 | 0 | 26 |
| O_new_rgb_s0_e30 | 49.757 | 68.917 | 35.974 | 57.902 | 90.571 | 111 | 0 | 0 | 37 |
| O_new_rgb_s1_e30 | 45.945 | 68.233 | 30.978 | 52.966 | 90.512 | 108 | 0 | 0 | 28 |
| O_new_rgb_s2_e30 | 47.827 | 68.336 | 33.519 | 55.791 | 90.551 | 115 | 0 | 0 | 31 |
| O_rgb_ir_s0_e30 | 43.937 | 68.687 | 28.154 | 49.162 | 89.785 | 133 | 0 | 0 | 47 |
| O_rgb_ir_s1_e30 | 46.805 | 68.801 | 31.886 | 53.769 | 89.946 | 119 | 0 | 0 | 40 |
| O_rgb_ir_s2_e30 | 44.688 | 68.534 | 29.187 | 50.848 | 89.974 | 125 | 0 | 0 | 34 |
| O_rgb_rgb_s0_e30 | 43.097 | 68.512 | 27.152 | 47.408 | 89.858 | 138 | 0 | 0 | 49 |
| O_rgb_rgb_s1_e30 | 45.708 | 68.744 | 30.438 | 52.456 | 89.908 | 125 | 0 | 0 | 43 |
| O_rgb_rgb_s2_e30 | 44.365 | 68.554 | 28.759 | 49.944 | 89.938 | 134 | 0 | 0 | 45 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

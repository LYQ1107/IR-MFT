# Train-fitted physical reachability memory controls

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

All30 full/reference/auxiliary-only controls. Bounds fit only on six Train trajectories; do not delete visible proposals or renormalize hypotheses.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births | Prebirth ID recoveries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G_new_ir_s0_auxiliary_only | 50.024 | 68.517 | 36.568 | 58.455 | 90.502 | 114 | 0 | 0 | 5 |
| G_new_ir_s0_full | 50.079 | 68.578 | 36.617 | 58.498 | 90.568 | 111 | 0 | 0 | 29 |
| G_new_ir_s1_auxiliary_only | 49.382 | 68.235 | 35.789 | 58.791 | 90.543 | 111 | 0 | 0 | 0 |
| G_new_ir_s1_full | 47.599 | 68.278 | 33.224 | 56.290 | 90.553 | 109 | 0 | 0 | 23 |
| G_new_ir_s2_auxiliary_only | 49.499 | 68.266 | 35.936 | 58.560 | 90.573 | 114 | 0 | 0 | 0 |
| G_new_ir_s2_full | 47.909 | 68.303 | 33.653 | 56.558 | 90.564 | 111 | 0 | 0 | 22 |
| G_new_reference_s0 | 49.445 | 68.560 | 35.707 | 58.061 | 90.558 | 113 | 0 | 0 | 26 |
| G_new_reference_s1 | 47.599 | 68.278 | 33.224 | 56.290 | 90.553 | 109 | 0 | 0 | 23 |
| G_new_reference_s2 | 47.909 | 68.303 | 33.653 | 56.558 | 90.564 | 111 | 0 | 0 | 22 |
| G_new_rgb_s0_auxiliary_only | 50.238 | 69.091 | 36.575 | 58.437 | 90.575 | 113 | 0 | 0 | 12 |
| G_new_rgb_s0_full | 49.776 | 68.919 | 35.999 | 57.923 | 90.611 | 111 | 0 | 0 | 32 |
| G_new_rgb_s1_auxiliary_only | 48.171 | 68.819 | 33.763 | 55.632 | 90.579 | 109 | 0 | 0 | 12 |
| G_new_rgb_s1_full | 46.086 | 68.835 | 30.894 | 52.830 | 90.519 | 108 | 0 | 0 | 32 |
| G_new_rgb_s2_auxiliary_only | 49.213 | 68.281 | 35.514 | 57.518 | 90.476 | 119 | 0 | 0 | 12 |
| G_new_rgb_s2_full | 47.607 | 68.317 | 33.222 | 55.498 | 90.450 | 116 | 0 | 0 | 31 |
| G_rgb_ir_s0_auxiliary_only | 44.642 | 68.480 | 29.149 | 50.746 | 89.852 | 132 | 0 | 0 | 4 |
| G_rgb_ir_s0_full | 44.368 | 68.674 | 28.712 | 49.914 | 89.854 | 133 | 0 | 0 | 39 |
| G_rgb_ir_s1_auxiliary_only | 45.744 | 68.449 | 30.613 | 52.117 | 89.968 | 122 | 0 | 0 | 4 |
| G_rgb_ir_s1_full | 45.568 | 68.773 | 30.235 | 51.283 | 89.953 | 119 | 0 | 0 | 33 |
| G_rgb_ir_s2_auxiliary_only | 45.867 | 68.581 | 30.722 | 53.175 | 89.955 | 129 | 0 | 0 | 3 |
| G_rgb_ir_s2_full | 44.684 | 68.525 | 29.187 | 50.846 | 89.992 | 129 | 0 | 0 | 32 |
| G_rgb_reference_s0 | 45.192 | 68.326 | 29.936 | 50.955 | 89.869 | 136 | 0 | 0 | 36 |
| G_rgb_reference_s1 | 46.463 | 68.441 | 31.585 | 53.605 | 89.944 | 122 | 0 | 0 | 30 |
| G_rgb_reference_s2 | 44.684 | 68.525 | 29.187 | 50.846 | 89.994 | 129 | 0 | 0 | 31 |
| G_rgb_rgb_s0_auxiliary_only | 44.686 | 68.495 | 29.200 | 50.770 | 89.869 | 132 | 0 | 0 | 12 |
| G_rgb_rgb_s0_full | 44.623 | 68.682 | 29.038 | 50.226 | 89.865 | 135 | 0 | 0 | 48 |
| G_rgb_rgb_s1_auxiliary_only | 46.090 | 68.467 | 31.073 | 53.274 | 89.925 | 123 | 0 | 0 | 17 |
| G_rgb_rgb_s1_full | 45.721 | 68.768 | 30.446 | 52.430 | 89.901 | 123 | 0 | 0 | 45 |
| G_rgb_rgb_s2_auxiliary_only | 45.908 | 68.577 | 30.778 | 53.189 | 89.940 | 134 | 0 | 0 | 16 |
| G_rgb_rgb_s2_full | 44.729 | 68.525 | 29.245 | 50.845 | 89.944 | 133 | 0 | 0 | 44 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

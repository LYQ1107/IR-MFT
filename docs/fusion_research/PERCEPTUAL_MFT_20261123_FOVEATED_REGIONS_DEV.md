# Full working-region foveated observation Dev controls

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

All38 registered full-Dev controls. Unchanged global checkpoints, actual new IR pixels, equal crop models/counts/scales and original ByteTrack. Crop-induced localization changes can harm association. No established IR attention advantage over the strongest RGB control; no foveated Test.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Y_baseline_new | 46.236 | 68.766 | 31.137 | 53.119 | 90.224 | 108 | 0 | 0 |
| Y_baseline_rgb | 47.579 | 68.864 | 32.908 | 55.865 | 90.200 | 94 | 0 | 0 |
| Y_new_ir_half_k2_nms | 36.831 | 62.013 | 21.943 | 39.419 | 78.931 | 618 | 0 | 0 |
| Y_new_ir_half_k2_refine | 40.997 | 64.712 | 26.022 | 44.665 | 85.773 | 234 | 0 | 0 |
| Y_new_ir_half_k4_nms | 29.858 | 59.413 | 15.081 | 28.753 | 73.550 | 849 | 0 | 0 |
| Y_new_ir_half_k4_refine | 34.042 | 62.713 | 18.562 | 36.896 | 83.091 | 339 | 0 | 0 |
| Y_new_ir_quarter_k2_nms | 45.848 | 68.220 | 30.861 | 52.093 | 89.404 | 112 | 0 | 0 |
| Y_new_ir_quarter_k2_refine | 44.670 | 67.855 | 29.458 | 50.755 | 88.831 | 114 | 0 | 0 |
| Y_new_ir_quarter_k4_nms | 44.650 | 67.977 | 29.374 | 49.773 | 88.967 | 111 | 0 | 0 |
| Y_new_ir_quarter_k4_refine | 42.667 | 67.342 | 27.087 | 47.335 | 87.829 | 142 | 0 | 0 |
| Y_new_rgb_half_k2_nms | 31.353 | 63.087 | 15.664 | 31.018 | 80.409 | 563 | 0 | 0 |
| Y_new_rgb_half_k2_refine | 40.966 | 64.993 | 25.877 | 46.511 | 86.018 | 214 | 0 | 0 |
| Y_new_rgb_half_k4_nms | 31.274 | 59.708 | 16.464 | 30.612 | 74.060 | 830 | 0 | 0 |
| Y_new_rgb_half_k4_refine | 39.438 | 63.310 | 24.630 | 44.587 | 83.300 | 259 | 0 | 0 |
| Y_new_rgb_quarter_k2_nms | 44.585 | 68.131 | 29.226 | 50.099 | 89.139 | 128 | 0 | 0 |
| Y_new_rgb_quarter_k2_refine | 44.394 | 67.557 | 29.226 | 49.941 | 88.407 | 138 | 0 | 0 |
| Y_new_rgb_quarter_k4_nms | 44.240 | 67.766 | 28.929 | 49.125 | 88.423 | 125 | 0 | 0 |
| Y_new_rgb_quarter_k4_refine | 41.770 | 67.041 | 26.075 | 46.827 | 87.413 | 154 | 0 | 0 |
| Y_new_uniform_k4_nms | 32.973 | 60.813 | 17.971 | 33.062 | 76.860 | 664 | 0 | 0 |
| Y_new_uniform_k4_refine | 35.459 | 63.452 | 19.916 | 37.493 | 83.608 | 273 | 0 | 0 |
| Y_rgb_ir_half_k2_nms | 35.740 | 62.802 | 20.411 | 36.832 | 79.940 | 486 | 0 | 0 |
| Y_rgb_ir_half_k2_refine | 42.966 | 64.862 | 28.488 | 50.604 | 85.405 | 258 | 0 | 0 |
| Y_rgb_ir_half_k4_nms | 31.483 | 60.313 | 16.521 | 29.768 | 75.009 | 661 | 0 | 0 |
| Y_rgb_ir_half_k4_refine | 35.708 | 63.220 | 20.234 | 38.972 | 82.887 | 323 | 0 | 0 |
| Y_rgb_ir_quarter_k2_nms | 45.170 | 68.363 | 29.888 | 52.194 | 89.382 | 106 | 0 | 0 |
| Y_rgb_ir_quarter_k2_refine | 44.124 | 67.740 | 28.794 | 50.634 | 88.610 | 110 | 0 | 0 |
| Y_rgb_ir_quarter_k4_nms | 43.549 | 67.985 | 27.938 | 48.817 | 88.810 | 120 | 0 | 0 |
| Y_rgb_ir_quarter_k4_refine | 44.349 | 66.906 | 29.442 | 49.641 | 87.531 | 128 | 0 | 0 |
| Y_rgb_rgb_half_k2_nms | 37.014 | 63.648 | 21.590 | 37.417 | 81.107 | 485 | 0 | 0 |
| Y_rgb_rgb_half_k2_refine | 39.605 | 65.384 | 24.041 | 42.755 | 85.846 | 228 | 0 | 0 |
| Y_rgb_rgb_half_k4_nms | 31.175 | 60.507 | 16.151 | 29.979 | 75.115 | 680 | 0 | 0 |
| Y_rgb_rgb_half_k4_refine | 37.812 | 63.474 | 22.590 | 41.379 | 83.145 | 288 | 0 | 0 |
| Y_rgb_rgb_quarter_k2_nms | 46.549 | 68.344 | 31.740 | 54.179 | 89.208 | 114 | 0 | 0 |
| Y_rgb_rgb_quarter_k2_refine | 47.848 | 67.871 | 33.766 | 55.853 | 88.302 | 136 | 0 | 0 |
| Y_rgb_rgb_quarter_k4_nms | 46.124 | 67.858 | 31.385 | 53.652 | 88.674 | 115 | 0 | 0 |
| Y_rgb_rgb_quarter_k4_refine | 44.949 | 67.185 | 30.113 | 52.492 | 87.346 | 147 | 0 | 0 |
| Y_rgb_uniform_k4_nms | 34.722 | 60.989 | 19.838 | 36.037 | 77.161 | 521 | 0 | 0 |
| Y_rgb_uniform_k4_refine | 38.211 | 63.700 | 22.998 | 42.912 | 83.754 | 291 | 0 | 0 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

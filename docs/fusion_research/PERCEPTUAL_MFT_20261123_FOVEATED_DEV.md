# Full original foveated observation Dev controls

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

All38 registered full-Dev controls. Actual new IR, unchanged crop model and matched RGB/uniform compute controls, all scales/budgets/fusion rules and global detector frontends. Compare every row against the strongest RGB control; ordinary crop augmentation alone is not a multimodal contribution. No foveated Test.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Z_baseline_new | 46.236 | 68.766 | 31.137 | 53.119 | 90.224 | 108 | 0 | 0 |
| Z_baseline_rgb | 47.579 | 68.864 | 32.908 | 55.865 | 90.200 | 94 | 0 | 0 |
| Z_new_ir_half_k2_nms | 33.794 | 63.065 | 18.170 | 34.730 | 80.681 | 587 | 0 | 0 |
| Z_new_ir_half_k2_refine | 38.492 | 65.576 | 22.652 | 40.551 | 86.231 | 229 | 0 | 0 |
| Z_new_ir_half_k4_nms | 29.370 | 60.248 | 14.390 | 28.455 | 75.308 | 788 | 0 | 0 |
| Z_new_ir_half_k4_refine | 37.486 | 63.530 | 22.184 | 40.237 | 83.838 | 294 | 0 | 0 |
| Z_new_ir_quarter_k2_nms | 44.402 | 68.151 | 28.981 | 49.729 | 89.393 | 120 | 0 | 0 |
| Z_new_ir_quarter_k2_refine | 43.690 | 67.873 | 28.172 | 50.337 | 88.924 | 141 | 0 | 0 |
| Z_new_ir_quarter_k4_nms | 44.381 | 67.787 | 29.106 | 49.644 | 88.754 | 122 | 0 | 0 |
| Z_new_ir_quarter_k4_refine | 42.612 | 67.388 | 26.993 | 48.706 | 88.065 | 143 | 0 | 0 |
| Z_new_rgb_half_k2_nms | 35.817 | 62.968 | 20.444 | 36.241 | 80.685 | 549 | 0 | 0 |
| Z_new_rgb_half_k2_refine | 38.820 | 64.953 | 23.259 | 43.110 | 85.773 | 208 | 0 | 0 |
| Z_new_rgb_half_k4_nms | 32.175 | 60.239 | 17.244 | 31.177 | 75.838 | 734 | 0 | 0 |
| Z_new_rgb_half_k4_refine | 35.864 | 62.991 | 20.499 | 38.736 | 83.250 | 279 | 0 | 0 |
| Z_new_rgb_quarter_k2_nms | 45.552 | 68.226 | 30.457 | 51.883 | 89.361 | 112 | 0 | 0 |
| Z_new_rgb_quarter_k2_refine | 45.021 | 67.902 | 29.899 | 51.058 | 88.896 | 120 | 0 | 0 |
| Z_new_rgb_quarter_k4_nms | 45.283 | 67.978 | 30.211 | 51.343 | 88.943 | 121 | 0 | 0 |
| Z_new_rgb_quarter_k4_refine | 44.684 | 67.532 | 29.618 | 50.038 | 88.181 | 138 | 0 | 0 |
| Z_new_uniform_k4_nms | 33.021 | 60.831 | 18.017 | 33.104 | 76.847 | 667 | 0 | 0 |
| Z_new_uniform_k4_refine | 35.443 | 63.398 | 19.915 | 37.491 | 83.593 | 273 | 0 | 0 |
| Z_rgb_ir_half_k2_nms | 34.042 | 63.347 | 18.399 | 35.401 | 81.229 | 458 | 0 | 0 |
| Z_rgb_ir_half_k2_refine | 39.465 | 65.048 | 24.017 | 44.329 | 85.668 | 243 | 0 | 0 |
| Z_rgb_ir_half_k4_nms | 30.176 | 60.603 | 15.109 | 28.303 | 76.204 | 619 | 0 | 0 |
| Z_rgb_ir_half_k4_refine | 37.619 | 63.138 | 22.486 | 43.085 | 83.078 | 340 | 0 | 0 |
| Z_rgb_ir_quarter_k2_nms | 46.744 | 68.320 | 32.022 | 53.208 | 89.350 | 106 | 0 | 0 |
| Z_rgb_ir_quarter_k2_refine | 46.703 | 67.712 | 32.254 | 54.180 | 88.939 | 122 | 0 | 0 |
| Z_rgb_ir_quarter_k4_nms | 46.429 | 68.062 | 31.711 | 52.870 | 88.857 | 111 | 0 | 0 |
| Z_rgb_ir_quarter_k4_refine | 46.831 | 67.694 | 32.438 | 54.637 | 88.205 | 123 | 0 | 0 |
| Z_rgb_rgb_half_k2_nms | 32.514 | 63.162 | 16.811 | 30.310 | 81.154 | 443 | 0 | 0 |
| Z_rgb_rgb_half_k2_refine | 35.940 | 65.002 | 19.922 | 38.575 | 85.213 | 260 | 0 | 0 |
| Z_rgb_rgb_half_k4_nms | 31.419 | 60.586 | 16.361 | 30.282 | 76.344 | 589 | 0 | 0 |
| Z_rgb_rgb_half_k4_refine | 36.082 | 63.225 | 20.660 | 38.986 | 83.014 | 324 | 0 | 0 |
| Z_rgb_rgb_quarter_k2_nms | 47.566 | 68.354 | 33.135 | 55.720 | 89.217 | 106 | 0 | 0 |
| Z_rgb_rgb_quarter_k2_refine | 46.374 | 68.100 | 31.619 | 54.180 | 88.730 | 130 | 0 | 0 |
| Z_rgb_rgb_quarter_k4_nms | 47.285 | 67.801 | 33.012 | 55.440 | 88.844 | 114 | 0 | 0 |
| Z_rgb_rgb_quarter_k4_refine | 47.066 | 67.209 | 32.999 | 55.775 | 88.037 | 155 | 0 | 0 |
| Z_rgb_uniform_k4_nms | 34.764 | 61.054 | 19.862 | 36.271 | 77.226 | 518 | 0 | 0 |
| Z_rgb_uniform_k4_refine | 38.209 | 63.708 | 22.994 | 42.914 | 83.776 | 291 | 0 | 0 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

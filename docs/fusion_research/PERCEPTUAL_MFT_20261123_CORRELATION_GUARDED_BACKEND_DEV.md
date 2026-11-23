# 完整开发集：出生阈值与清晰运动保护的集成修复

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

All36 matched fullDev cases complete. Restoring0.7 birth recovers much of naive composition loss, but all12 three-association-seed group means remain below their corresponding NN+ByteTrack parents. Clear-motion protection does not establish a consistent IR benefit. Individual best seeds are retained without cherry-picking. Detector adapter seed remains0. No new Test.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CG_gray_ir_s0_birth07 | 44.836 | 67.947 | 29.632 | 50.616 | 89.025 | 132 | 0 | 0 |
| CG_gray_ir_s0_birth07_clear | 45.267 | 67.955 | 30.196 | 51.607 | 89.105 | 122 | 0 | 0 |
| CG_gray_ir_s1_birth07 | 49.465 | 68.180 | 35.920 | 58.688 | 89.068 | 128 | 0 | 0 |
| CG_gray_ir_s1_birth07_clear | 49.794 | 68.275 | 36.345 | 58.969 | 89.189 | 118 | 0 | 0 |
| CG_gray_ir_s2_birth07 | 49.945 | 68.043 | 36.698 | 57.489 | 89.165 | 127 | 0 | 0 |
| CG_gray_ir_s2_birth07_clear | 50.963 | 68.170 | 38.130 | 59.257 | 89.249 | 113 | 0 | 0 |
| CG_gray_rgb_s0_birth07 | 44.595 | 68.149 | 29.224 | 49.705 | 89.195 | 122 | 0 | 0 |
| CG_gray_rgb_s0_birth07_clear | 46.198 | 68.041 | 31.402 | 52.968 | 89.156 | 122 | 0 | 0 |
| CG_gray_rgb_s1_birth07 | 48.007 | 68.154 | 33.850 | 55.323 | 89.150 | 119 | 0 | 0 |
| CG_gray_rgb_s1_birth07_clear | 47.049 | 68.054 | 32.558 | 53.625 | 89.148 | 122 | 0 | 0 |
| CG_gray_rgb_s2_birth07 | 48.655 | 68.003 | 34.867 | 56.920 | 88.950 | 131 | 0 | 0 |
| CG_gray_rgb_s2_birth07_clear | 47.092 | 67.869 | 32.716 | 54.058 | 89.030 | 133 | 0 | 0 |
| CG_ir_ir_s0_birth07 | 47.842 | 67.892 | 33.738 | 56.074 | 89.854 | 113 | 0 | 0 |
| CG_ir_ir_s0_birth07_clear | 45.781 | 68.350 | 30.690 | 53.464 | 89.807 | 117 | 0 | 0 |
| CG_ir_ir_s1_birth07 | 48.120 | 67.943 | 34.114 | 56.612 | 89.817 | 101 | 0 | 0 |
| CG_ir_ir_s1_birth07_clear | 48.661 | 68.503 | 34.598 | 57.358 | 89.856 | 99 | 0 | 0 |
| CG_ir_ir_s2_birth07 | 50.219 | 68.533 | 36.830 | 60.066 | 89.856 | 102 | 0 | 0 |
| CG_ir_ir_s2_birth07_clear | 48.921 | 68.476 | 34.977 | 58.849 | 89.824 | 102 | 0 | 0 |
| CG_ir_rgb_s0_birth07 | 50.882 | 68.374 | 37.905 | 60.564 | 89.867 | 94 | 0 | 0 |
| CG_ir_rgb_s0_birth07_clear | 50.482 | 68.361 | 37.312 | 60.349 | 89.865 | 102 | 0 | 0 |
| CG_ir_rgb_s1_birth07 | 47.467 | 67.868 | 33.237 | 55.077 | 89.766 | 98 | 0 | 0 |
| CG_ir_rgb_s1_birth07_clear | 48.714 | 68.365 | 34.745 | 56.760 | 89.830 | 101 | 0 | 0 |
| CG_ir_rgb_s2_birth07 | 47.965 | 67.760 | 33.985 | 55.905 | 89.884 | 89 | 0 | 0 |
| CG_ir_rgb_s2_birth07_clear | 48.554 | 68.387 | 34.503 | 56.758 | 89.830 | 98 | 0 | 0 |
| CG_rgb_ir_s0_birth07 | 46.413 | 68.412 | 31.531 | 52.620 | 89.998 | 103 | 0 | 0 |
| CG_rgb_ir_s0_birth07_clear | 46.827 | 68.588 | 32.008 | 53.571 | 89.964 | 114 | 0 | 0 |
| CG_rgb_ir_s1_birth07 | 48.277 | 68.451 | 34.083 | 56.417 | 89.921 | 109 | 0 | 0 |
| CG_rgb_ir_s1_birth07_clear | 48.225 | 68.571 | 33.950 | 56.155 | 90.005 | 108 | 0 | 0 |
| CG_rgb_ir_s2_birth07 | 50.294 | 68.286 | 37.080 | 59.590 | 89.910 | 110 | 0 | 0 |
| CG_rgb_ir_s2_birth07_clear | 50.621 | 68.510 | 37.436 | 60.335 | 89.912 | 116 | 0 | 0 |
| CG_rgb_rgb_s0_birth07 | 49.114 | 68.359 | 35.331 | 57.913 | 89.938 | 105 | 0 | 0 |
| CG_rgb_rgb_s0_birth07_clear | 49.787 | 68.591 | 36.177 | 59.191 | 89.964 | 107 | 0 | 0 |
| CG_rgb_rgb_s1_birth07 | 45.423 | 68.507 | 30.160 | 51.553 | 90.002 | 100 | 0 | 0 |
| CG_rgb_rgb_s1_birth07_clear | 45.209 | 68.640 | 29.813 | 51.509 | 90.028 | 103 | 0 | 0 |
| CG_rgb_rgb_s2_birth07 | 47.028 | 68.290 | 32.424 | 54.784 | 89.910 | 103 | 0 | 0 |
| CG_rgb_rgb_s2_birth07_clear | 46.139 | 68.493 | 31.127 | 53.746 | 89.860 | 108 | 0 | 0 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

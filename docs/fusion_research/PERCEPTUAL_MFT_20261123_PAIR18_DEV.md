# Train-only pair calibration: complete36-row first18-epoch Dev audit

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

All six heads actually executed18 epochs with checkpoints6/12/18, identical batches and three seeds. New-front RGB calibration reaches49.499 on Dev; IR increments over raw RGB point odds are unstable. No Test selection or Test improvement is inferred. A common-budget extension is still governed by the frozen12-epoch-lookback rule; this initial18-epoch family is preserved intact.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G_new_ir_s0_e06 | 47.197 | 68.538 | 32.558 | 54.930 | 90.596 | 118 | 0 | 0 |
| G_new_ir_s0_e12 | 43.463 | 68.567 | 27.606 | 49.478 | 90.459 | 131 | 0 | 0 |
| G_new_ir_s0_e18 | 45.934 | 68.421 | 30.889 | 52.858 | 90.472 | 118 | 0 | 0 |
| G_new_ir_s1_e06 | 47.509 | 68.586 | 32.966 | 54.936 | 90.642 | 121 | 0 | 0 |
| G_new_ir_s1_e12 | 46.686 | 68.508 | 31.856 | 53.298 | 90.422 | 125 | 0 | 0 |
| G_new_ir_s1_e18 | 46.576 | 68.473 | 31.739 | 54.011 | 90.508 | 121 | 0 | 0 |
| G_new_ir_s2_e06 | 46.260 | 68.560 | 31.267 | 52.445 | 90.474 | 116 | 0 | 0 |
| G_new_ir_s2_e12 | 47.855 | 68.686 | 33.400 | 54.962 | 90.478 | 123 | 0 | 0 |
| G_new_ir_s2_e18 | 47.442 | 68.670 | 32.832 | 55.549 | 90.532 | 115 | 0 | 0 |
| G_new_rgb_s0_e06 | 46.137 | 68.429 | 31.146 | 51.754 | 90.487 | 115 | 0 | 0 |
| G_new_rgb_s0_e12 | 49.122 | 68.489 | 35.276 | 57.304 | 90.517 | 112 | 0 | 0 |
| G_new_rgb_s0_e18 | 46.430 | 68.464 | 31.524 | 52.745 | 90.478 | 112 | 0 | 0 |
| G_new_rgb_s1_e06 | 46.121 | 68.176 | 31.253 | 53.876 | 90.519 | 109 | 0 | 0 |
| G_new_rgb_s1_e12 | 45.384 | 68.196 | 30.245 | 52.171 | 90.560 | 112 | 0 | 0 |
| G_new_rgb_s1_e18 | 48.262 | 68.297 | 34.152 | 56.551 | 90.508 | 108 | 0 | 0 |
| G_new_rgb_s2_e06 | 48.034 | 68.307 | 33.822 | 55.537 | 90.487 | 108 | 0 | 0 |
| G_new_rgb_s2_e12 | 49.499 | 68.266 | 35.936 | 58.560 | 90.573 | 114 | 0 | 0 |
| G_new_rgb_s2_e18 | 46.032 | 68.061 | 31.170 | 53.293 | 90.549 | 111 | 0 | 0 |
| G_rgb_ir_s0_e06 | 47.792 | 68.743 | 33.278 | 55.452 | 89.944 | 142 | 0 | 0 |
| G_rgb_ir_s0_e12 | 44.119 | 68.469 | 28.480 | 51.282 | 89.858 | 145 | 0 | 0 |
| G_rgb_ir_s0_e18 | 46.195 | 68.419 | 31.232 | 53.441 | 89.918 | 136 | 0 | 0 |
| G_rgb_ir_s1_e06 | 45.177 | 68.473 | 29.848 | 51.407 | 89.899 | 145 | 0 | 0 |
| G_rgb_ir_s1_e12 | 44.234 | 68.401 | 28.655 | 51.331 | 89.852 | 144 | 0 | 0 |
| G_rgb_ir_s1_e18 | 46.224 | 68.743 | 31.125 | 53.480 | 89.843 | 148 | 0 | 0 |
| G_rgb_ir_s2_e06 | 44.006 | 68.373 | 28.373 | 49.150 | 89.921 | 132 | 0 | 0 |
| G_rgb_ir_s2_e12 | 44.650 | 68.831 | 29.015 | 50.134 | 89.923 | 141 | 0 | 0 |
| G_rgb_ir_s2_e18 | 43.262 | 68.635 | 27.319 | 48.216 | 89.977 | 125 | 0 | 0 |
| G_rgb_rgb_s0_e06 | 43.682 | 68.327 | 27.974 | 49.791 | 89.916 | 133 | 0 | 0 |
| G_rgb_rgb_s0_e12 | 44.939 | 68.352 | 29.590 | 51.343 | 89.893 | 132 | 0 | 0 |
| G_rgb_rgb_s0_e18 | 45.579 | 68.448 | 30.395 | 51.777 | 89.865 | 134 | 0 | 0 |
| G_rgb_rgb_s1_e06 | 44.599 | 68.423 | 29.117 | 50.784 | 89.884 | 130 | 0 | 0 |
| G_rgb_rgb_s1_e12 | 45.308 | 68.469 | 30.028 | 52.626 | 89.895 | 132 | 0 | 0 |
| G_rgb_rgb_s1_e18 | 46.104 | 68.406 | 31.117 | 53.763 | 89.916 | 126 | 0 | 0 |
| G_rgb_rgb_s2_e06 | 44.932 | 68.469 | 29.529 | 51.718 | 89.964 | 124 | 0 | 0 |
| G_rgb_rgb_s2_e12 | 44.825 | 68.354 | 29.441 | 52.622 | 89.959 | 125 | 0 | 0 |
| G_rgb_rgb_s2_e18 | 45.231 | 68.519 | 29.900 | 51.780 | 90.005 | 126 | 0 | 0 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

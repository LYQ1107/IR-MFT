# Fitted-RGB cue correction: complete36-row initial18-epoch Dev audit

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

All six residual heads executed18 epochs. Seed0 new-IR head reaches50.933 but improvements do not survive all three seeds. Fixed RGB inheritance and equal-capacity RGB controls are retained. The common stopping rule requires extending all six to24; single-seed gains are not independent contribution proof.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F_new_ir_s0_e06 | 49.322 | 68.373 | 35.628 | 58.343 | 90.347 | 119 | 0 | 0 |
| F_new_ir_s0_e12 | 48.128 | 68.562 | 33.833 | 56.258 | 90.321 | 123 | 0 | 0 |
| F_new_ir_s0_e18 | 50.933 | 68.460 | 37.928 | 60.029 | 90.368 | 116 | 0 | 0 |
| F_new_ir_s1_e06 | 46.114 | 68.582 | 31.052 | 52.822 | 90.435 | 121 | 0 | 0 |
| F_new_ir_s1_e12 | 44.463 | 68.305 | 28.980 | 51.114 | 90.355 | 133 | 0 | 0 |
| F_new_ir_s1_e18 | 44.590 | 68.411 | 29.102 | 51.328 | 90.446 | 124 | 0 | 0 |
| F_new_ir_s2_e06 | 46.264 | 68.929 | 31.090 | 53.099 | 90.416 | 118 | 0 | 0 |
| F_new_ir_s2_e12 | 45.150 | 68.349 | 29.869 | 52.174 | 90.403 | 128 | 0 | 0 |
| F_new_ir_s2_e18 | 46.917 | 68.443 | 32.200 | 53.746 | 90.452 | 128 | 0 | 0 |
| F_new_rgb_s0_e06 | 48.123 | 69.020 | 33.614 | 56.342 | 90.482 | 120 | 0 | 0 |
| F_new_rgb_s0_e12 | 49.219 | 68.967 | 35.177 | 57.482 | 90.435 | 129 | 0 | 0 |
| F_new_rgb_s0_e18 | 48.761 | 69.085 | 34.470 | 57.195 | 90.545 | 120 | 0 | 0 |
| F_new_rgb_s1_e06 | 45.567 | 69.000 | 30.144 | 52.522 | 90.405 | 119 | 0 | 0 |
| F_new_rgb_s1_e12 | 46.470 | 68.535 | 31.554 | 53.664 | 90.366 | 119 | 0 | 0 |
| F_new_rgb_s1_e18 | 45.548 | 68.250 | 30.446 | 52.156 | 90.383 | 121 | 0 | 0 |
| F_new_rgb_s2_e06 | 50.004 | 68.322 | 36.646 | 58.987 | 90.504 | 113 | 0 | 0 |
| F_new_rgb_s2_e12 | 49.920 | 68.476 | 36.437 | 58.537 | 90.422 | 116 | 0 | 0 |
| F_new_rgb_s2_e18 | 48.907 | 68.441 | 34.995 | 57.404 | 90.543 | 115 | 0 | 0 |
| F_rgb_ir_s0_e06 | 44.169 | 68.294 | 28.629 | 49.508 | 89.740 | 139 | 0 | 0 |
| F_rgb_ir_s0_e12 | 44.077 | 68.720 | 28.321 | 48.651 | 89.822 | 133 | 0 | 0 |
| F_rgb_ir_s0_e18 | 43.977 | 68.298 | 28.374 | 49.506 | 89.828 | 135 | 0 | 0 |
| F_rgb_ir_s1_e06 | 45.094 | 68.467 | 29.752 | 51.161 | 89.819 | 147 | 0 | 0 |
| F_rgb_ir_s1_e12 | 43.745 | 68.225 | 28.094 | 48.215 | 89.720 | 141 | 0 | 0 |
| F_rgb_ir_s1_e18 | 44.333 | 68.626 | 28.681 | 48.958 | 89.847 | 134 | 0 | 0 |
| F_rgb_ir_s2_e06 | 43.585 | 68.615 | 27.736 | 50.089 | 89.822 | 133 | 0 | 0 |
| F_rgb_ir_s2_e12 | 45.929 | 68.439 | 30.864 | 53.239 | 89.839 | 136 | 0 | 0 |
| F_rgb_ir_s2_e18 | 45.256 | 68.737 | 29.842 | 52.169 | 89.910 | 135 | 0 | 0 |
| F_rgb_rgb_s0_e06 | 46.719 | 68.199 | 32.057 | 54.099 | 89.940 | 126 | 0 | 0 |
| F_rgb_rgb_s0_e12 | 47.770 | 68.190 | 33.516 | 55.471 | 89.912 | 134 | 0 | 0 |
| F_rgb_rgb_s0_e18 | 47.088 | 68.678 | 32.342 | 53.582 | 89.933 | 133 | 0 | 0 |
| F_rgb_rgb_s1_e06 | 46.850 | 68.540 | 32.066 | 53.687 | 89.985 | 127 | 0 | 0 |
| F_rgb_rgb_s1_e12 | 44.189 | 68.287 | 28.645 | 50.186 | 89.921 | 142 | 0 | 0 |
| F_rgb_rgb_s1_e18 | 45.749 | 68.289 | 30.694 | 52.175 | 89.895 | 137 | 0 | 0 |
| F_rgb_rgb_s2_e06 | 48.934 | 68.371 | 35.071 | 56.626 | 89.955 | 122 | 0 | 0 |
| F_rgb_rgb_s2_e12 | 48.180 | 68.383 | 33.996 | 56.230 | 89.949 | 124 | 0 | 0 |
| F_rgb_rgb_s2_e18 | 48.113 | 68.375 | 33.907 | 55.718 | 89.985 | 129 | 0 | 0 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

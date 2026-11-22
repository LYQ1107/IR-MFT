# Native-cue real/null query Dev controls

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

Complete42 controls. Direct normalized-query replacement hurt the strong fitted RGB reference; preserve all seeds and exact-zero query controls.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births | Prebirth ID recoveries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N_new_ir_s0_e06 | 43.685 | 68.984 | 27.714 | 51.153 | 90.364 | 133 | 0 | 0 | 55 |
| N_new_ir_s0_e12 | 44.576 | 68.869 | 28.920 | 51.242 | 90.213 | 154 | 0 | 0 | 59 |
| N_new_ir_s0_e18 | 43.792 | 68.409 | 28.080 | 50.274 | 89.843 | 166 | 0 | 0 | 86 |
| N_new_ir_s0_e18_zero | 47.896 | 68.451 | 33.572 | 53.048 | 90.454 | 120 | 0 | 0 | 38 |
| N_new_ir_s1_e06 | 43.950 | 68.477 | 28.263 | 50.226 | 89.959 | 139 | 0 | 0 | 60 |
| N_new_ir_s1_e12 | 44.459 | 68.705 | 28.824 | 49.576 | 89.860 | 140 | 0 | 0 | 86 |
| N_new_ir_s1_e18 | 44.286 | 68.372 | 28.740 | 49.808 | 89.712 | 142 | 0 | 0 | 88 |
| N_new_ir_s1_e18_zero | 45.044 | 68.389 | 29.723 | 49.333 | 90.379 | 127 | 0 | 0 | 33 |
| N_new_ir_s2_e06 | 44.649 | 68.752 | 29.055 | 51.595 | 90.340 | 139 | 0 | 0 | 42 |
| N_new_ir_s2_e12 | 43.620 | 68.856 | 27.689 | 48.817 | 90.327 | 139 | 0 | 0 | 43 |
| N_new_ir_s2_e18 | 43.379 | 68.816 | 27.402 | 48.752 | 90.082 | 147 | 0 | 0 | 65 |
| N_new_ir_s2_e18_zero | 43.682 | 68.416 | 27.943 | 47.976 | 90.495 | 132 | 0 | 0 | 29 |
| N_new_rgb_s0_e06 | 45.499 | 68.879 | 30.111 | 51.089 | 90.005 | 141 | 0 | 0 | 90 |
| N_new_rgb_s0_e12 | 45.615 | 68.731 | 30.324 | 52.842 | 90.020 | 154 | 0 | 0 | 101 |
| N_new_rgb_s0_e18 | 44.436 | 68.868 | 28.723 | 49.329 | 90.088 | 151 | 0 | 0 | 85 |
| N_new_rgb_s1_e06 | 43.619 | 68.990 | 27.628 | 48.833 | 90.017 | 130 | 0 | 0 | 70 |
| N_new_rgb_s1_e12 | 46.125 | 69.031 | 30.866 | 52.616 | 90.192 | 119 | 0 | 0 | 73 |
| N_new_rgb_s1_e18 | 46.225 | 68.946 | 31.037 | 52.641 | 89.914 | 131 | 0 | 0 | 71 |
| N_new_rgb_s2_e06 | 46.395 | 69.053 | 31.218 | 52.457 | 90.185 | 143 | 0 | 0 | 59 |
| N_new_rgb_s2_e12 | 44.181 | 68.708 | 28.467 | 50.275 | 90.125 | 147 | 0 | 0 | 72 |
| N_new_rgb_s2_e18 | 44.890 | 68.770 | 29.353 | 50.391 | 89.946 | 146 | 0 | 0 | 86 |
| N_rgb_ir_s0_e06 | 40.516 | 68.221 | 24.107 | 44.374 | 89.574 | 160 | 0 | 0 | 70 |
| N_rgb_ir_s0_e12 | 40.947 | 68.600 | 24.499 | 44.907 | 89.656 | 151 | 0 | 0 | 67 |
| N_rgb_ir_s0_e18 | 41.136 | 68.498 | 24.759 | 44.253 | 89.370 | 168 | 0 | 0 | 89 |
| N_rgb_ir_s0_e18_zero | 40.723 | 68.342 | 24.325 | 44.251 | 89.811 | 147 | 0 | 0 | 39 |
| N_rgb_ir_s1_e06 | 44.505 | 68.043 | 29.157 | 49.699 | 89.856 | 150 | 0 | 0 | 47 |
| N_rgb_ir_s1_e12 | 43.277 | 68.537 | 27.390 | 46.337 | 89.770 | 123 | 0 | 0 | 47 |
| N_rgb_ir_s1_e18 | 42.785 | 68.195 | 26.889 | 46.401 | 89.415 | 161 | 0 | 0 | 83 |
| N_rgb_ir_s1_e18_zero | 44.392 | 68.529 | 28.827 | 48.246 | 89.714 | 150 | 0 | 0 | 44 |
| N_rgb_ir_s2_e06 | 41.461 | 68.366 | 25.198 | 44.992 | 89.746 | 152 | 0 | 0 | 36 |
| N_rgb_ir_s2_e12 | 44.634 | 68.627 | 29.081 | 50.436 | 89.753 | 153 | 0 | 0 | 48 |
| N_rgb_ir_s2_e18 | 42.754 | 68.721 | 26.655 | 47.895 | 89.751 | 155 | 0 | 0 | 60 |
| N_rgb_ir_s2_e18_zero | 43.356 | 68.461 | 27.533 | 46.858 | 89.772 | 159 | 0 | 0 | 40 |
| N_rgb_rgb_s0_e06 | 40.916 | 68.172 | 24.612 | 44.956 | 89.451 | 158 | 0 | 0 | 91 |
| N_rgb_rgb_s0_e12 | 42.936 | 68.215 | 27.071 | 45.253 | 89.337 | 167 | 0 | 0 | 110 |
| N_rgb_rgb_s0_e18 | 40.398 | 68.157 | 23.984 | 44.115 | 89.460 | 169 | 0 | 0 | 81 |
| N_rgb_rgb_s1_e06 | 43.990 | 68.062 | 28.475 | 50.003 | 89.516 | 156 | 0 | 0 | 88 |
| N_rgb_rgb_s1_e12 | 41.787 | 67.957 | 25.750 | 46.693 | 89.443 | 161 | 0 | 0 | 84 |
| N_rgb_rgb_s1_e18 | 41.679 | 68.174 | 25.527 | 45.826 | 89.382 | 154 | 0 | 0 | 102 |
| N_rgb_rgb_s2_e06 | 40.648 | 68.452 | 24.197 | 43.645 | 89.561 | 144 | 0 | 0 | 82 |
| N_rgb_rgb_s2_e12 | 40.459 | 68.313 | 24.022 | 43.740 | 89.475 | 151 | 0 | 0 | 93 |
| N_rgb_rgb_s2_e18 | 41.758 | 68.186 | 25.625 | 45.846 | 89.434 | 154 | 0 | 0 | 95 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

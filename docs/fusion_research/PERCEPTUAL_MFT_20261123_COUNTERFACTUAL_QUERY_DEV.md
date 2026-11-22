# Counterfactual query correction Dev controls

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

Complete42 controls. Positive posterior gain and clear-RGB protection alone do not establish a robust new-IR advantage.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births | Prebirth ID recoveries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C_new_ir_s0_e06 | 46.000 | 68.956 | 30.738 | 54.036 | 90.431 | 122 | 0 | 0 | 32 |
| C_new_ir_s0_e12 | 46.476 | 68.938 | 31.375 | 53.974 | 90.327 | 126 | 0 | 0 | 36 |
| C_new_ir_s0_e18 | 43.992 | 68.729 | 28.203 | 51.855 | 90.125 | 142 | 0 | 0 | 47 |
| C_new_ir_s0_e18_zero | 49.385 | 68.492 | 35.656 | 58.012 | 90.519 | 112 | 0 | 0 | 0 |
| C_new_ir_s1_e06 | 48.778 | 68.452 | 34.812 | 57.981 | 90.256 | 126 | 0 | 0 | 36 |
| C_new_ir_s1_e12 | 46.086 | 68.526 | 31.039 | 52.844 | 90.125 | 130 | 0 | 0 | 55 |
| C_new_ir_s1_e18 | 45.920 | 68.507 | 30.827 | 52.718 | 90.097 | 139 | 0 | 0 | 56 |
| C_new_ir_s1_e18_zero | 49.382 | 68.235 | 35.789 | 58.791 | 90.543 | 111 | 0 | 0 | 0 |
| C_new_ir_s2_e06 | 47.793 | 68.663 | 33.313 | 56.632 | 90.441 | 121 | 0 | 0 | 24 |
| C_new_ir_s2_e12 | 45.408 | 68.780 | 30.029 | 52.187 | 90.282 | 137 | 0 | 0 | 33 |
| C_new_ir_s2_e18 | 41.992 | 68.804 | 25.674 | 47.009 | 90.076 | 148 | 0 | 0 | 49 |
| C_new_ir_s2_e18_zero | 49.499 | 68.266 | 35.936 | 58.560 | 90.573 | 114 | 0 | 0 | 0 |
| C_new_rgb_s0_e06 | 50.159 | 68.826 | 36.594 | 57.843 | 90.157 | 121 | 0 | 0 | 64 |
| C_new_rgb_s0_e12 | 48.036 | 69.068 | 33.444 | 54.867 | 90.226 | 122 | 0 | 0 | 65 |
| C_new_rgb_s0_e18 | 44.293 | 68.946 | 28.493 | 48.620 | 90.254 | 123 | 0 | 0 | 61 |
| C_new_rgb_s1_e06 | 45.339 | 68.818 | 29.920 | 51.184 | 90.175 | 119 | 0 | 0 | 54 |
| C_new_rgb_s1_e12 | 48.199 | 68.799 | 33.812 | 56.775 | 90.114 | 118 | 0 | 0 | 55 |
| C_new_rgb_s1_e18 | 44.705 | 68.823 | 29.080 | 51.650 | 90.002 | 129 | 0 | 0 | 52 |
| C_new_rgb_s2_e06 | 46.183 | 68.751 | 31.080 | 54.062 | 90.136 | 131 | 0 | 0 | 45 |
| C_new_rgb_s2_e12 | 45.232 | 68.488 | 29.929 | 52.838 | 90.043 | 133 | 0 | 0 | 56 |
| C_new_rgb_s2_e18 | 44.956 | 68.767 | 29.437 | 51.463 | 89.854 | 132 | 0 | 0 | 63 |
| C_rgb_ir_s0_e06 | 43.319 | 68.592 | 27.394 | 49.417 | 89.563 | 142 | 0 | 0 | 46 |
| C_rgb_ir_s0_e12 | 43.240 | 68.613 | 27.291 | 49.539 | 89.443 | 151 | 0 | 0 | 66 |
| C_rgb_ir_s0_e18 | 43.674 | 68.606 | 27.864 | 48.649 | 89.553 | 164 | 0 | 0 | 66 |
| C_rgb_ir_s0_e18_zero | 45.579 | 68.448 | 30.395 | 51.777 | 89.865 | 134 | 0 | 0 | 0 |
| C_rgb_ir_s1_e06 | 46.155 | 68.284 | 31.243 | 53.745 | 89.718 | 128 | 0 | 0 | 27 |
| C_rgb_ir_s1_e12 | 41.147 | 68.468 | 24.784 | 44.807 | 89.621 | 136 | 0 | 0 | 33 |
| C_rgb_ir_s1_e18 | 41.811 | 68.202 | 25.683 | 46.981 | 89.393 | 157 | 0 | 0 | 74 |
| C_rgb_ir_s1_e18_zero | 46.904 | 68.446 | 32.184 | 54.436 | 89.949 | 124 | 0 | 0 | 0 |
| C_rgb_ir_s2_e06 | 45.929 | 68.839 | 30.689 | 53.403 | 89.675 | 139 | 0 | 0 | 26 |
| C_rgb_ir_s2_e12 | 43.741 | 68.738 | 27.887 | 50.599 | 89.742 | 153 | 0 | 0 | 32 |
| C_rgb_ir_s2_e18 | 43.842 | 68.849 | 27.972 | 50.629 | 89.688 | 155 | 0 | 0 | 45 |
| C_rgb_ir_s2_e18_zero | 45.867 | 68.580 | 30.722 | 53.177 | 89.959 | 129 | 0 | 0 | 0 |
| C_rgb_rgb_s0_e06 | 43.843 | 68.486 | 28.110 | 49.522 | 89.393 | 154 | 0 | 0 | 80 |
| C_rgb_rgb_s0_e12 | 43.364 | 68.014 | 27.684 | 49.050 | 89.327 | 157 | 0 | 0 | 85 |
| C_rgb_rgb_s0_e18 | 39.598 | 68.365 | 22.984 | 42.036 | 89.359 | 198 | 0 | 0 | 69 |
| C_rgb_rgb_s1_e06 | 45.889 | 68.110 | 30.952 | 54.235 | 89.516 | 149 | 0 | 0 | 67 |
| C_rgb_rgb_s1_e12 | 41.995 | 68.014 | 25.969 | 48.777 | 89.557 | 161 | 0 | 0 | 72 |
| C_rgb_rgb_s1_e18 | 42.130 | 68.362 | 26.000 | 49.002 | 89.447 | 156 | 0 | 0 | 79 |
| C_rgb_rgb_s2_e06 | 41.266 | 68.669 | 24.853 | 47.296 | 89.507 | 146 | 0 | 0 | 72 |
| C_rgb_rgb_s2_e12 | 42.133 | 68.372 | 26.010 | 47.534 | 89.423 | 146 | 0 | 0 | 80 |
| C_rgb_rgb_s2_e18 | 43.020 | 68.442 | 27.082 | 47.879 | 89.288 | 153 | 0 | 0 | 85 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

# Complete RGB-preserving IR and supported-candidate controls

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

All56 Dev configurations completed. Largest new-frontend HOTA48.448 is mostly explained by the ordinary global birth0.6 control48.443; the IR increment is0.005pp. IR-supported births improve over the harmful RGB-only supported-birth variant by0.550pp, but do not meaningfully beat omitting that added birth route. This is useful conditional filtering evidence, not an established independent multimodal contribution.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D_new_00_RGB | 47.608 | 68.423 | 33.172 | 56.288 | 90.289 | 98 | 0 | 0 |
| D_new_01_spatial_IR | 47.612 | 68.429 | 33.175 | 56.305 | 90.323 | 98 | 0 | 0 |
| D_new_02_validated_IR | 47.612 | 68.429 | 33.175 | 56.305 | 90.323 | 98 | 0 | 0 |
| D_new_03_three_anchors | 47.608 | 68.423 | 33.172 | 56.288 | 90.289 | 98 | 0 | 0 |
| D_new_04_weak_IR | 47.612 | 68.429 | 33.175 | 56.305 | 90.323 | 98 | 0 | 0 |
| D_new_05_strong_IR | 46.334 | 68.460 | 31.405 | 54.902 | 90.394 | 97 | 0 | 0 |
| D_new_06_IR_target | 47.608 | 68.423 | 33.172 | 56.288 | 90.289 | 98 | 0 | 0 |
| D_new_07_majority_IR | 47.612 | 68.429 | 33.175 | 56.305 | 90.323 | 98 | 0 | 0 |
| D_new_08_less_conservative_IR | 47.612 | 68.430 | 33.175 | 56.305 | 90.323 | 98 | 0 | 0 |
| D_new_09_short_validation | 47.612 | 68.429 | 33.175 | 56.305 | 90.323 | 98 | 0 | 0 |
| D_new_10_RGB_birth | 47.067 | 68.691 | 32.294 | 56.198 | 90.338 | 100 | 0 | 20 |
| D_new_11_RGB_global_birth | 48.443 | 68.847 | 34.131 | 55.844 | 90.519 | 108 | 0 | 0 |
| D_new_12_IR_birth | 47.617 | 68.501 | 33.148 | 56.243 | 90.368 | 98 | 0 | 19 |
| D_new_13_IR_RGBbirth | 47.072 | 68.698 | 32.297 | 56.215 | 90.373 | 100 | 0 | 20 |
| D_new_14_IR_distinct | 47.612 | 68.429 | 33.175 | 56.305 | 90.323 | 98 | 0 | 0 |
| D_new_15_IR_distinct_birth | 47.617 | 68.501 | 33.148 | 56.243 | 90.368 | 98 | 0 | 19 |
| D_new_16_RGB_distinct_birth | 47.067 | 68.691 | 32.294 | 56.198 | 90.338 | 100 | 0 | 20 |
| D_new_17_IR_global_birth | 48.448 | 68.853 | 34.134 | 55.862 | 90.553 | 108 | 0 | 0 |
| D_new_18_RGB_confident | 46.481 | 68.073 | 31.783 | 53.986 | 90.211 | 108 | 0 | 0 |
| D_new_19_IR_confident | 46.481 | 68.073 | 31.783 | 53.986 | 90.211 | 108 | 0 | 0 |
| D_new_20_RGB_middle | 46.757 | 68.151 | 32.122 | 54.366 | 90.246 | 104 | 0 | 0 |
| D_new_21_IR_middle | 46.757 | 68.151 | 32.122 | 54.366 | 90.246 | 104 | 0 | 0 |
| D_new_22_RGB_unopposed | 47.671 | 68.510 | 33.218 | 56.295 | 90.302 | 99 | 0 | 0 |
| D_new_23_IR_unopposed | 47.676 | 68.517 | 33.221 | 56.313 | 90.336 | 99 | 0 | 0 |
| D_new_24_RGB_motion | 47.209 | 68.412 | 32.626 | 55.720 | 90.293 | 98 | 0 | 0 |
| D_new_25_IR_motion | 47.209 | 68.412 | 32.626 | 55.720 | 90.293 | 98 | 0 | 0 |
| D_new_26_RGB_strict_motion | 46.485 | 68.080 | 31.784 | 53.994 | 90.218 | 108 | 0 | 0 |
| D_new_27_IR_strict_motion | 46.485 | 68.080 | 31.784 | 53.994 | 90.218 | 108 | 0 | 0 |
| D_rgb_00_RGB | 44.661 | 68.771 | 29.042 | 52.234 | 90.144 | 97 | 0 | 0 |
| D_rgb_01_spatial_IR | 44.666 | 68.773 | 29.048 | 52.263 | 90.175 | 95 | 0 | 0 |
| D_rgb_02_validated_IR | 44.667 | 68.779 | 29.046 | 52.260 | 90.187 | 95 | 0 | 0 |
| D_rgb_03_three_anchors | 44.661 | 68.771 | 29.042 | 52.234 | 90.144 | 97 | 0 | 0 |
| D_rgb_04_weak_IR | 44.661 | 68.771 | 29.042 | 52.234 | 90.144 | 97 | 0 | 0 |
| D_rgb_05_strong_IR | 44.914 | 68.780 | 29.369 | 52.276 | 90.239 | 94 | 0 | 0 |
| D_rgb_06_IR_target | 44.661 | 68.770 | 29.042 | 52.234 | 90.149 | 97 | 0 | 0 |
| D_rgb_07_majority_IR | 44.667 | 68.779 | 29.046 | 52.260 | 90.187 | 95 | 0 | 0 |
| D_rgb_08_less_conservative_IR | 44.667 | 68.779 | 29.046 | 52.260 | 90.187 | 95 | 0 | 0 |
| D_rgb_09_short_validation | 44.667 | 68.779 | 29.046 | 52.260 | 90.187 | 95 | 0 | 0 |
| D_rgb_10_RGB_birth | 43.570 | 68.945 | 27.581 | 50.914 | 90.286 | 103 | 0 | 28 |
| D_rgb_11_RGB_global_birth | 42.272 | 68.428 | 26.167 | 48.523 | 89.955 | 125 | 0 | 0 |
| D_rgb_12_IR_birth | 43.574 | 68.947 | 27.586 | 50.947 | 90.345 | 101 | 0 | 19 |
| D_rgb_13_IR_RGBbirth | 43.576 | 68.954 | 27.585 | 50.940 | 90.330 | 101 | 0 | 28 |
| D_rgb_14_IR_distinct | 45.198 | 68.771 | 29.744 | 53.015 | 90.183 | 94 | 0 | 0 |
| D_rgb_15_IR_distinct_birth | 44.872 | 68.945 | 29.248 | 52.463 | 90.345 | 99 | 0 | 19 |
| D_rgb_16_RGB_distinct_birth | 44.868 | 68.943 | 29.243 | 52.429 | 90.286 | 101 | 0 | 28 |
| D_rgb_17_IR_global_birth | 42.278 | 68.437 | 26.171 | 48.549 | 89.998 | 123 | 0 | 0 |
| D_rgb_18_RGB_confident | 46.858 | 68.824 | 31.941 | 55.044 | 90.177 | 96 | 0 | 0 |
| D_rgb_19_IR_confident | 46.858 | 68.824 | 31.941 | 55.044 | 90.177 | 96 | 0 | 0 |
| D_rgb_20_RGB_middle | 46.586 | 68.585 | 31.682 | 55.001 | 90.151 | 99 | 0 | 0 |
| D_rgb_21_IR_middle | 46.586 | 68.585 | 31.682 | 55.001 | 90.151 | 99 | 0 | 0 |
| D_rgb_22_RGB_unopposed | 45.761 | 68.745 | 30.497 | 53.796 | 90.190 | 96 | 0 | 0 |
| D_rgb_23_IR_unopposed | 45.767 | 68.753 | 30.501 | 53.823 | 90.233 | 94 | 0 | 0 |
| D_rgb_24_RGB_motion | 46.604 | 68.906 | 31.555 | 54.008 | 90.187 | 92 | 0 | 0 |
| D_rgb_25_IR_motion | 46.604 | 68.906 | 31.555 | 54.008 | 90.187 | 92 | 0 | 0 |
| D_rgb_26_RGB_strict_motion | 47.579 | 68.862 | 32.908 | 55.863 | 90.172 | 94 | 0 | 0 |
| D_rgb_27_IR_strict_motion | 47.579 | 68.862 | 32.908 | 55.863 | 90.172 | 94 | 0 | 0 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

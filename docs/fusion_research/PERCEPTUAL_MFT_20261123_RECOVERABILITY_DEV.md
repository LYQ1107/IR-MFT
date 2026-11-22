# Weak identity recovery and causal candidate cycle: complete48-row Dev audit

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

A few retained old identities can be recovered, but long-gap same-identity weak observations rarely retain reliable body-point transport. No new-IR advantage exceeding measurement precision; global low-score threshold reduction damages HOTA. All rows, removals and both detector frontends preserved.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q_new_00_RGB_control | 48.443 | 68.847 | 34.131 | 55.844 | 90.519 | 108 | 0 | 0 |
| Q_new_01_IR_control | 48.448 | 68.853 | 34.134 | 55.862 | 90.553 | 108 | 0 | 0 |
| Q_new_02_RGB_two_observation_birth | 47.777 | 68.660 | 33.295 | 54.309 | 90.209 | 128 | 0 | 53 |
| Q_new_03_IR_two_observation_birth | 48.216 | 68.818 | 33.828 | 55.124 | 90.441 | 115 | 0 | 33 |
| Q_new_04_RGB_three_no_cycle | 48.407 | 68.869 | 34.070 | 55.758 | 90.484 | 112 | 0 | 17 |
| Q_new_05_IR_three_no_cycle | 48.433 | 68.841 | 34.119 | 55.831 | 90.521 | 109 | 0 | 3 |
| Q_new_06_RGB_three_cycle | 48.416 | 68.825 | 34.104 | 55.784 | 90.454 | 110 | 0 | 10 |
| Q_new_07_IR_three_cycle | 48.433 | 68.842 | 34.119 | 55.832 | 90.523 | 109 | 0 | 3 |
| Q_new_08_RGB_weak_06 | 48.446 | 68.853 | 34.133 | 55.852 | 90.530 | 108 | 2 | 0 |
| Q_new_09_IR_weak_06 | 48.450 | 68.859 | 34.135 | 55.868 | 90.562 | 108 | 1 | 0 |
| Q_new_10_RGB_weak_08 | 48.445 | 68.852 | 34.132 | 55.851 | 90.528 | 108 | 1 | 0 |
| Q_new_11_IR_weak_08 | 48.450 | 68.859 | 34.135 | 55.868 | 90.562 | 108 | 1 | 0 |
| Q_new_12_RGB_weak_09 | 48.445 | 68.852 | 34.132 | 55.851 | 90.528 | 108 | 1 | 0 |
| Q_new_13_IR_weak_09 | 48.450 | 68.859 | 34.135 | 55.868 | 90.562 | 108 | 1 | 0 |
| Q_new_14_RGB_full | 48.418 | 68.831 | 34.105 | 55.790 | 90.463 | 110 | 1 | 10 |
| Q_new_15_IR_full | 48.435 | 68.848 | 34.120 | 55.838 | 90.532 | 109 | 1 | 3 |
| Q_new_16_RGB_full_latest_only | 48.418 | 68.831 | 34.105 | 55.790 | 90.463 | 110 | 1 | 10 |
| Q_new_17_IR_full_latest_only | 48.435 | 68.848 | 34.120 | 55.838 | 90.532 | 109 | 1 | 3 |
| Q_new_18_RGB_full_no_distinct | 48.418 | 68.831 | 34.105 | 55.790 | 90.463 | 110 | 1 | 10 |
| Q_new_19_IR_full_no_distinct | 48.435 | 68.848 | 34.120 | 55.838 | 90.532 | 109 | 1 | 3 |
| Q_new_20_ordinary_low03 | 41.601 | 67.590 | 25.684 | 43.471 | 87.538 | 215 | 0 | 0 |
| Q_new_21_RGB_low03_points | 41.758 | 67.223 | 26.002 | 44.628 | 87.471 | 208 | 0 | 0 |
| Q_new_22_IR_low03_points | 41.757 | 67.222 | 26.002 | 44.627 | 87.469 | 208 | 0 | 0 |
| Q_new_23_IR_full_no_point_increment | 48.431 | 68.841 | 34.117 | 55.821 | 90.497 | 109 | 1 | 3 |
| Q_rgb_00_RGB_control | 42.272 | 68.428 | 26.167 | 48.523 | 89.955 | 125 | 0 | 0 |
| Q_rgb_01_IR_control | 42.278 | 68.437 | 26.171 | 48.549 | 89.998 | 123 | 0 | 0 |
| Q_rgb_02_RGB_two_observation_birth | 41.045 | 68.426 | 24.681 | 44.912 | 89.509 | 135 | 0 | 55 |
| Q_rgb_03_IR_two_observation_birth | 41.553 | 68.537 | 25.251 | 46.283 | 89.886 | 132 | 0 | 39 |
| Q_rgb_04_RGB_three_no_cycle | 42.117 | 68.534 | 25.939 | 47.822 | 89.852 | 127 | 0 | 15 |
| Q_rgb_05_IR_three_no_cycle | 42.276 | 68.424 | 26.174 | 48.512 | 89.940 | 125 | 0 | 6 |
| Q_rgb_06_RGB_three_cycle | 42.109 | 68.502 | 25.940 | 47.844 | 89.834 | 125 | 0 | 8 |
| Q_rgb_07_IR_three_cycle | 42.283 | 68.425 | 26.182 | 48.528 | 89.966 | 123 | 0 | 3 |
| Q_rgb_08_RGB_weak_06 | 43.314 | 68.424 | 27.471 | 50.011 | 89.933 | 123 | 1 | 0 |
| Q_rgb_09_IR_weak_06 | 43.321 | 68.432 | 27.476 | 50.038 | 89.989 | 121 | 0 | 0 |
| Q_rgb_10_RGB_weak_08 | 43.315 | 68.424 | 27.472 | 50.012 | 89.946 | 123 | 0 | 0 |
| Q_rgb_11_IR_weak_08 | 43.321 | 68.432 | 27.476 | 50.038 | 89.989 | 121 | 0 | 0 |
| Q_rgb_12_RGB_weak_09 | 43.315 | 68.424 | 27.472 | 50.012 | 89.946 | 123 | 0 | 0 |
| Q_rgb_13_IR_weak_09 | 43.321 | 68.432 | 27.476 | 50.038 | 89.989 | 121 | 0 | 0 |
| Q_rgb_14_RGB_full | 43.152 | 68.492 | 27.240 | 49.330 | 89.822 | 124 | 0 | 8 |
| Q_rgb_15_IR_full | 43.325 | 68.421 | 27.486 | 50.016 | 89.957 | 121 | 0 | 3 |
| Q_rgb_16_RGB_full_latest_only | 43.152 | 68.492 | 27.240 | 49.330 | 89.822 | 124 | 0 | 8 |
| Q_rgb_17_IR_full_latest_only | 43.325 | 68.421 | 27.486 | 50.016 | 89.957 | 121 | 0 | 3 |
| Q_rgb_18_RGB_full_no_distinct | 42.109 | 68.502 | 25.940 | 47.844 | 89.834 | 125 | 0 | 8 |
| Q_rgb_19_IR_full_no_distinct | 42.283 | 68.425 | 26.182 | 48.528 | 89.966 | 123 | 0 | 3 |
| Q_rgb_20_ordinary_low03 | 39.136 | 66.306 | 23.179 | 41.660 | 86.419 | 251 | 0 | 0 |
| Q_rgb_21_RGB_low03_points | 39.166 | 66.582 | 23.118 | 40.732 | 86.354 | 256 | 0 | 0 |
| Q_rgb_22_IR_low03_points | 39.370 | 66.572 | 23.358 | 41.201 | 86.337 | 255 | 0 | 0 |
| Q_rgb_23_IR_full_no_point_increment | 43.319 | 68.412 | 27.482 | 49.990 | 89.914 | 123 | 0 | 3 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

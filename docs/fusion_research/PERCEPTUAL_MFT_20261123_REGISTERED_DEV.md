# Confidence-gated crossmodal ROI registration: complete32-row Dev audit

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

RGB-only paired raw/registered controls are identical across all39 parsed aggregate metrics and all276 detailed CSV columns on both frontends. Alignment repairs some IR-heavy variants but none beats the matched strong RGB control. New-front strong conditional IR improves47.019 to48.263, still below RGB48.443 and conservative IR48.448. Correlation and IR-mask repairs alone do not establish a useful multimodal contribution.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A_new_00_RGB_raw | 48.443 | 68.847 | 34.131 | 55.844 | 90.519 | 108 | 0 | 0 |
| A_new_01_RGB_registered | 48.443 | 68.847 | 34.131 | 55.844 | 90.519 | 108 | 0 | 0 |
| A_new_02_joint_raw | 44.431 | 68.366 | 28.924 | 51.273 | 90.596 | 119 | 0 | 0 |
| A_new_03_joint_registered | 45.217 | 68.549 | 29.870 | 52.312 | 90.594 | 116 | 0 | 0 |
| A_new_04_IR_raw | 45.534 | 68.472 | 30.334 | 52.136 | 90.562 | 117 | 0 | 0 |
| A_new_05_IR_registered | 45.641 | 68.492 | 30.469 | 52.426 | 90.588 | 116 | 0 | 0 |
| A_new_06_RGB_to_joint_raw | 46.008 | 68.501 | 30.944 | 52.338 | 90.594 | 114 | 0 | 0 |
| A_new_07_RGB_to_joint_registered | 46.303 | 68.535 | 31.324 | 52.834 | 90.603 | 111 | 0 | 0 |
| A_new_08_joint_to_RGB_raw | 44.743 | 68.307 | 29.351 | 50.691 | 90.568 | 115 | 0 | 0 |
| A_new_09_joint_to_RGB_registered | 45.127 | 68.449 | 29.792 | 51.231 | 90.556 | 114 | 0 | 0 |
| A_new_10_conditional_raw | 48.448 | 68.853 | 34.134 | 55.862 | 90.553 | 108 | 0 | 0 |
| A_new_11_conditional_registered | 48.448 | 68.853 | 34.134 | 55.862 | 90.553 | 108 | 0 | 0 |
| A_new_12_conditional_strong_raw | 47.019 | 68.828 | 32.166 | 54.472 | 90.607 | 107 | 0 | 0 |
| A_new_13_conditional_strong_registered | 48.263 | 68.804 | 33.900 | 55.841 | 90.594 | 107 | 0 | 0 |
| A_new_14_recovery_cycle_raw | 48.435 | 68.848 | 34.120 | 55.838 | 90.532 | 109 | 1 | 3 |
| A_new_15_recovery_cycle_registered | 48.435 | 68.848 | 34.120 | 55.838 | 90.532 | 109 | 1 | 3 |
| A_rgb_00_RGB_raw | 42.272 | 68.428 | 26.167 | 48.523 | 89.955 | 125 | 0 | 0 |
| A_rgb_01_RGB_registered | 42.272 | 68.428 | 26.167 | 48.523 | 89.955 | 125 | 0 | 0 |
| A_rgb_02_joint_raw | 42.090 | 68.531 | 25.903 | 48.300 | 90.011 | 123 | 0 | 0 |
| A_rgb_03_joint_registered | 42.909 | 68.531 | 26.919 | 48.870 | 90.024 | 120 | 0 | 0 |
| A_rgb_04_IR_raw | 41.744 | 68.495 | 25.492 | 48.335 | 89.908 | 126 | 0 | 0 |
| A_rgb_05_IR_registered | 43.174 | 68.499 | 27.263 | 49.843 | 89.977 | 121 | 0 | 0 |
| A_rgb_06_RGB_to_joint_raw | 43.676 | 68.319 | 27.976 | 49.824 | 89.910 | 121 | 0 | 0 |
| A_rgb_07_RGB_to_joint_registered | 43.644 | 68.324 | 27.931 | 49.825 | 89.910 | 118 | 0 | 0 |
| A_rgb_08_joint_to_RGB_raw | 44.176 | 68.453 | 28.561 | 51.818 | 89.970 | 127 | 0 | 0 |
| A_rgb_09_joint_to_RGB_registered | 44.472 | 68.476 | 28.932 | 52.184 | 89.970 | 127 | 0 | 0 |
| A_rgb_10_conditional_raw | 42.278 | 68.437 | 26.171 | 48.549 | 89.998 | 123 | 0 | 0 |
| A_rgb_11_conditional_registered | 42.278 | 68.437 | 26.171 | 48.549 | 89.994 | 123 | 0 | 0 |
| A_rgb_12_conditional_strong_raw | 42.537 | 68.429 | 26.496 | 48.565 | 90.041 | 122 | 0 | 0 |
| A_rgb_13_conditional_strong_registered | 42.538 | 68.431 | 26.497 | 48.570 | 90.041 | 122 | 0 | 0 |
| A_rgb_14_recovery_cycle_raw | 43.325 | 68.421 | 27.486 | 50.016 | 89.957 | 121 | 0 | 3 |
| A_rgb_15_recovery_cycle_registered | 43.325 | 68.421 | 27.486 | 50.016 | 89.953 | 121 | 0 | 3 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

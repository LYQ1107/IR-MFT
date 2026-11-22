# Positive-only and short-memory Dev controls

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

All30 controls. Remove hard null-based track rejection and test short memory, no clear-motion guard and memory-only alternatives; retain every negative seed.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births | Prebirth ID recoveries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P_new_ir_s0_00_no_hard_rejection | 44.572 | 68.644 | 28.993 | 51.979 | 90.114 | 143 | 0 | 0 | 47 |
| P_new_ir_s0_01_matching_only | 46.142 | 68.841 | 30.970 | 52.859 | 90.431 | 131 | 0 | 0 | 0 |
| P_new_ir_s0_02_short_memory | 47.349 | 69.009 | 32.525 | 54.412 | 90.452 | 129 | 0 | 0 | 14 |
| P_new_ir_s0_03_no_clear_guard | 47.349 | 69.009 | 32.525 | 54.412 | 90.452 | 129 | 0 | 0 | 14 |
| P_new_ir_s0_04_short_memory_only | 49.752 | 69.064 | 35.884 | 57.986 | 90.523 | 115 | 0 | 0 | 11 |
| P_new_ir_s1_00_no_hard_rejection | 46.345 | 68.534 | 31.388 | 53.412 | 90.140 | 136 | 0 | 0 | 56 |
| P_new_ir_s1_01_matching_only | 49.385 | 68.599 | 35.602 | 58.162 | 90.478 | 120 | 0 | 0 | 0 |
| P_new_ir_s1_02_short_memory | 47.691 | 68.032 | 33.485 | 55.545 | 90.491 | 128 | 0 | 0 | 9 |
| P_new_ir_s1_03_no_clear_guard | 47.691 | 68.032 | 33.485 | 55.545 | 90.491 | 128 | 0 | 0 | 9 |
| P_new_ir_s1_04_short_memory_only | 48.556 | 68.249 | 34.594 | 56.861 | 90.571 | 114 | 0 | 0 | 9 |
| P_new_ir_s2_00_no_hard_rejection | 41.976 | 68.697 | 25.693 | 47.131 | 90.073 | 143 | 0 | 0 | 48 |
| P_new_ir_s2_01_matching_only | 45.031 | 69.057 | 29.408 | 51.362 | 90.401 | 127 | 0 | 0 | 0 |
| P_new_ir_s2_02_short_memory | 45.273 | 68.833 | 29.822 | 52.099 | 90.383 | 127 | 0 | 0 | 7 |
| P_new_ir_s2_03_no_clear_guard | 45.273 | 68.833 | 29.822 | 52.099 | 90.383 | 127 | 0 | 0 | 7 |
| P_new_ir_s2_04_short_memory_only | 49.164 | 68.859 | 35.148 | 57.151 | 90.553 | 114 | 0 | 0 | 5 |
| P_new_rgb_s0_00_no_hard_rejection | 44.746 | 69.030 | 29.046 | 49.161 | 90.263 | 121 | 0 | 0 | 63 |
| P_new_rgb_s0_01_matching_only | 45.795 | 69.003 | 30.432 | 51.223 | 90.504 | 113 | 0 | 0 | 0 |
| P_new_rgb_s0_02_short_memory | 43.571 | 68.729 | 27.666 | 48.090 | 90.504 | 117 | 0 | 0 | 16 |
| P_new_rgb_s0_03_no_clear_guard | 43.571 | 68.729 | 27.666 | 48.090 | 90.504 | 117 | 0 | 0 | 16 |
| P_new_rgb_s0_04_short_memory_only | 49.755 | 69.064 | 35.890 | 57.922 | 90.512 | 113 | 0 | 0 | 15 |
| P_new_rgb_s1_00_no_hard_rejection | 44.613 | 68.780 | 28.978 | 51.645 | 90.011 | 128 | 0 | 0 | 54 |
| P_new_rgb_s1_01_matching_only | 48.930 | 68.938 | 34.784 | 57.549 | 90.377 | 119 | 0 | 0 | 0 |
| P_new_rgb_s1_02_short_memory | 47.211 | 68.505 | 32.592 | 55.417 | 90.461 | 119 | 0 | 0 | 10 |
| P_new_rgb_s1_03_no_clear_guard | 47.211 | 68.505 | 32.592 | 55.417 | 90.461 | 119 | 0 | 0 | 10 |
| P_new_rgb_s1_04_short_memory_only | 48.814 | 68.224 | 34.973 | 57.504 | 90.564 | 111 | 0 | 0 | 9 |
| P_new_rgb_s2_00_no_hard_rejection | 44.956 | 68.767 | 29.437 | 51.463 | 89.854 | 132 | 0 | 0 | 63 |
| P_new_rgb_s2_01_matching_only | 47.417 | 68.976 | 32.653 | 54.828 | 90.405 | 132 | 0 | 0 | 0 |
| P_new_rgb_s2_02_short_memory | 45.689 | 68.330 | 30.595 | 52.356 | 90.396 | 126 | 0 | 0 | 11 |
| P_new_rgb_s2_03_no_clear_guard | 45.689 | 68.330 | 30.595 | 52.356 | 90.396 | 126 | 0 | 0 | 11 |
| P_new_rgb_s2_04_short_memory_only | 49.203 | 68.279 | 35.500 | 57.496 | 90.495 | 119 | 0 | 0 | 11 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

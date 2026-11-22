# Mismatch-calibrated identity-query Dev controls

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

All42 first18-epoch snapshots/zero controls. A single seed reaches50.079 Dev HOTA; other seeds do not establish a stable multimodal benefit. Common lookback rule forces continuation.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births | Prebirth ID recoveries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| U_new_ir_s0_e06 | 50.079 | 68.578 | 36.617 | 58.498 | 90.568 | 111 | 0 | 0 | 29 |
| U_new_ir_s0_e12 | 49.443 | 68.556 | 35.708 | 58.060 | 90.558 | 113 | 0 | 0 | 27 |
| U_new_ir_s0_e18 | 50.079 | 68.578 | 36.617 | 58.498 | 90.568 | 111 | 0 | 0 | 29 |
| U_new_ir_s1_e06 | 47.599 | 68.278 | 33.224 | 56.290 | 90.553 | 109 | 0 | 0 | 23 |
| U_new_ir_s1_e12 | 47.590 | 68.279 | 33.211 | 56.287 | 90.560 | 109 | 0 | 0 | 24 |
| U_new_ir_s1_e18 | 47.599 | 68.278 | 33.224 | 56.290 | 90.553 | 109 | 0 | 0 | 23 |
| U_new_ir_s2_e06 | 47.909 | 68.303 | 33.653 | 56.558 | 90.564 | 111 | 0 | 0 | 22 |
| U_new_ir_s2_e12 | 47.909 | 68.303 | 33.653 | 56.558 | 90.564 | 111 | 0 | 0 | 22 |
| U_new_ir_s2_e18 | 47.909 | 68.303 | 33.653 | 56.558 | 90.564 | 111 | 0 | 0 | 22 |
| U_new_reference_s0_e18 | 49.445 | 68.560 | 35.707 | 58.061 | 90.558 | 113 | 0 | 0 | 26 |
| U_new_reference_s1_e18 | 47.599 | 68.278 | 33.224 | 56.290 | 90.553 | 109 | 0 | 0 | 23 |
| U_new_reference_s2_e18 | 47.909 | 68.303 | 33.653 | 56.558 | 90.564 | 111 | 0 | 0 | 22 |
| U_new_rgb_s0_e06 | 49.083 | 68.937 | 34.999 | 58.128 | 90.614 | 107 | 0 | 0 | 33 |
| U_new_rgb_s0_e12 | 49.076 | 68.935 | 34.989 | 58.122 | 90.590 | 107 | 0 | 0 | 35 |
| U_new_rgb_s0_e18 | 49.776 | 68.919 | 35.999 | 57.923 | 90.611 | 111 | 0 | 0 | 32 |
| U_new_rgb_s1_e06 | 46.080 | 68.803 | 30.900 | 52.827 | 90.480 | 108 | 0 | 0 | 34 |
| U_new_rgb_s1_e12 | 45.951 | 68.292 | 30.957 | 52.858 | 90.528 | 108 | 0 | 0 | 30 |
| U_new_rgb_s1_e18 | 46.086 | 68.835 | 30.894 | 52.830 | 90.519 | 108 | 0 | 0 | 32 |
| U_new_rgb_s2_e06 | 47.605 | 68.339 | 33.210 | 55.484 | 90.510 | 116 | 0 | 0 | 29 |
| U_new_rgb_s2_e12 | 47.607 | 68.341 | 33.211 | 55.491 | 90.540 | 116 | 0 | 0 | 29 |
| U_new_rgb_s2_e18 | 47.607 | 68.317 | 33.222 | 55.498 | 90.450 | 116 | 0 | 0 | 31 |
| U_rgb_ir_s0_e06 | 44.368 | 68.674 | 28.712 | 49.914 | 89.854 | 133 | 0 | 0 | 39 |
| U_rgb_ir_s0_e12 | 44.365 | 68.671 | 28.709 | 49.905 | 89.843 | 133 | 0 | 0 | 38 |
| U_rgb_ir_s0_e18 | 44.368 | 68.674 | 28.712 | 49.914 | 89.854 | 133 | 0 | 0 | 39 |
| U_rgb_ir_s1_e06 | 46.585 | 68.776 | 31.597 | 53.600 | 89.949 | 121 | 0 | 0 | 31 |
| U_rgb_ir_s1_e12 | 46.585 | 68.776 | 31.597 | 53.600 | 89.949 | 121 | 0 | 0 | 31 |
| U_rgb_ir_s1_e18 | 45.568 | 68.773 | 30.235 | 51.283 | 89.953 | 119 | 0 | 0 | 33 |
| U_rgb_ir_s2_e06 | 44.684 | 68.525 | 29.187 | 50.846 | 89.994 | 129 | 0 | 0 | 31 |
| U_rgb_ir_s2_e12 | 44.684 | 68.525 | 29.187 | 50.846 | 89.994 | 129 | 0 | 0 | 31 |
| U_rgb_ir_s2_e18 | 44.684 | 68.525 | 29.187 | 50.846 | 89.992 | 129 | 0 | 0 | 32 |
| U_rgb_reference_s0_e18 | 45.192 | 68.326 | 29.936 | 50.955 | 89.869 | 136 | 0 | 0 | 36 |
| U_rgb_reference_s1_e18 | 46.463 | 68.441 | 31.585 | 53.605 | 89.944 | 122 | 0 | 0 | 30 |
| U_rgb_reference_s2_e18 | 44.684 | 68.525 | 29.187 | 50.846 | 89.994 | 129 | 0 | 0 | 31 |
| U_rgb_rgb_s0_e06 | 43.144 | 68.664 | 27.155 | 47.884 | 89.772 | 143 | 0 | 0 | 47 |
| U_rgb_rgb_s0_e12 | 43.947 | 68.664 | 28.176 | 49.177 | 89.785 | 137 | 0 | 0 | 46 |
| U_rgb_rgb_s0_e18 | 44.623 | 68.682 | 29.038 | 50.226 | 89.865 | 135 | 0 | 0 | 48 |
| U_rgb_rgb_s1_e06 | 45.721 | 68.764 | 30.446 | 52.442 | 89.908 | 123 | 0 | 0 | 43 |
| U_rgb_rgb_s1_e12 | 45.716 | 68.760 | 30.441 | 52.446 | 89.931 | 123 | 0 | 0 | 42 |
| U_rgb_rgb_s1_e18 | 45.721 | 68.768 | 30.446 | 52.430 | 89.901 | 123 | 0 | 0 | 45 |
| U_rgb_rgb_s2_e06 | 44.722 | 68.513 | 29.241 | 50.844 | 89.957 | 132 | 0 | 0 | 40 |
| U_rgb_rgb_s2_e12 | 44.734 | 68.552 | 29.240 | 50.842 | 89.968 | 130 | 0 | 0 | 41 |
| U_rgb_rgb_s2_e18 | 44.729 | 68.525 | 29.245 | 50.845 | 89.944 | 133 | 0 | 0 | 44 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

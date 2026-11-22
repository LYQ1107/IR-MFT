# Confidence-guarded residual Dev controls

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

All frozen48 guard configurations. Disabling only the new increment must retain the fitted RGB core.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H_new_ir_s0_00_zero | 49.122 | 68.489 | 35.276 | 57.304 | 90.517 | 112 | 0 | 0 |
| H_new_ir_s0_01_quarter | 47.731 | 68.348 | 33.374 | 56.170 | 90.510 | 116 | 0 | 0 |
| H_new_ir_s0_02_half | 50.858 | 69.061 | 37.485 | 59.596 | 90.450 | 112 | 0 | 0 |
| H_new_ir_s0_03_uncertain | 48.427 | 68.419 | 34.315 | 55.820 | 90.463 | 118 | 0 | 0 |
| H_new_ir_s1_00_zero | 48.262 | 68.297 | 34.152 | 56.551 | 90.508 | 108 | 0 | 0 |
| H_new_ir_s1_01_quarter | 47.066 | 68.385 | 32.438 | 54.753 | 90.364 | 117 | 0 | 0 |
| H_new_ir_s1_02_half | 45.052 | 68.394 | 29.715 | 51.242 | 90.299 | 122 | 0 | 0 |
| H_new_ir_s1_03_uncertain | 44.924 | 68.232 | 29.620 | 51.290 | 90.454 | 119 | 0 | 0 |
| H_new_ir_s2_00_zero | 49.499 | 68.266 | 35.936 | 58.560 | 90.573 | 114 | 0 | 0 |
| H_new_ir_s2_01_quarter | 47.557 | 68.401 | 33.105 | 54.698 | 90.528 | 116 | 0 | 0 |
| H_new_ir_s2_02_half | 48.629 | 68.316 | 34.652 | 56.390 | 90.491 | 119 | 0 | 0 |
| H_new_ir_s2_03_uncertain | 45.899 | 68.473 | 30.807 | 52.993 | 90.482 | 124 | 0 | 0 |
| H_new_rgb_s0_00_zero | 49.122 | 68.489 | 35.276 | 57.304 | 90.517 | 112 | 0 | 0 |
| H_new_rgb_s0_01_quarter | 47.591 | 68.422 | 33.151 | 55.431 | 90.551 | 111 | 0 | 0 |
| H_new_rgb_s0_02_half | 48.865 | 68.266 | 35.023 | 57.042 | 90.463 | 117 | 0 | 0 |
| H_new_rgb_s0_03_uncertain | 46.410 | 69.062 | 31.230 | 52.685 | 90.551 | 122 | 0 | 0 |
| H_new_rgb_s1_00_zero | 48.262 | 68.297 | 34.152 | 56.551 | 90.508 | 108 | 0 | 0 |
| H_new_rgb_s1_01_quarter | 47.420 | 68.236 | 32.997 | 55.473 | 90.472 | 114 | 0 | 0 |
| H_new_rgb_s1_02_half | 46.528 | 68.309 | 31.733 | 54.028 | 90.437 | 116 | 0 | 0 |
| H_new_rgb_s1_03_uncertain | 44.081 | 68.236 | 28.535 | 50.712 | 90.480 | 121 | 0 | 0 |
| H_new_rgb_s2_00_zero | 49.499 | 68.266 | 35.936 | 58.560 | 90.573 | 114 | 0 | 0 |
| H_new_rgb_s2_01_quarter | 47.140 | 68.358 | 32.551 | 54.144 | 90.497 | 117 | 0 | 0 |
| H_new_rgb_s2_02_half | 47.395 | 68.286 | 32.935 | 55.528 | 90.452 | 119 | 0 | 0 |
| H_new_rgb_s2_03_uncertain | 46.651 | 68.190 | 31.964 | 55.106 | 90.521 | 119 | 0 | 0 |
| H_rgb_ir_s0_00_zero | 44.939 | 68.352 | 29.590 | 51.343 | 89.893 | 132 | 0 | 0 |
| H_rgb_ir_s0_01_quarter | 43.726 | 68.454 | 27.978 | 49.396 | 89.906 | 127 | 0 | 0 |
| H_rgb_ir_s0_02_half | 45.492 | 68.373 | 30.322 | 51.473 | 89.867 | 127 | 0 | 0 |
| H_rgb_ir_s0_03_uncertain | 43.954 | 68.441 | 28.288 | 49.712 | 89.796 | 136 | 0 | 0 |
| H_rgb_ir_s1_00_zero | 46.104 | 68.406 | 31.117 | 53.763 | 89.916 | 126 | 0 | 0 |
| H_rgb_ir_s1_01_quarter | 44.214 | 68.397 | 28.626 | 50.635 | 89.895 | 130 | 0 | 0 |
| H_rgb_ir_s1_02_half | 44.499 | 68.390 | 29.004 | 50.920 | 89.903 | 130 | 0 | 0 |
| H_rgb_ir_s1_03_uncertain | 47.117 | 68.617 | 32.403 | 54.066 | 89.869 | 133 | 0 | 0 |
| H_rgb_ir_s2_00_zero | 44.825 | 68.354 | 29.441 | 52.622 | 89.959 | 125 | 0 | 0 |
| H_rgb_ir_s2_01_quarter | 47.038 | 68.329 | 32.433 | 54.337 | 89.931 | 126 | 0 | 0 |
| H_rgb_ir_s2_02_half | 47.202 | 68.411 | 32.618 | 53.533 | 89.910 | 130 | 0 | 0 |
| H_rgb_ir_s2_03_uncertain | 45.052 | 68.709 | 29.585 | 52.139 | 89.845 | 137 | 0 | 0 |
| H_rgb_rgb_s0_00_zero | 44.939 | 68.352 | 29.590 | 51.343 | 89.893 | 132 | 0 | 0 |
| H_rgb_rgb_s0_01_quarter | 45.556 | 68.377 | 30.396 | 52.300 | 89.912 | 128 | 0 | 0 |
| H_rgb_rgb_s0_02_half | 44.994 | 68.384 | 29.657 | 51.906 | 89.910 | 131 | 0 | 0 |
| H_rgb_rgb_s0_03_uncertain | 45.343 | 68.669 | 29.995 | 51.280 | 89.828 | 137 | 0 | 0 |
| H_rgb_rgb_s1_00_zero | 46.104 | 68.406 | 31.117 | 53.763 | 89.916 | 126 | 0 | 0 |
| H_rgb_rgb_s1_01_quarter | 44.145 | 68.700 | 28.416 | 49.826 | 89.910 | 123 | 0 | 0 |
| H_rgb_rgb_s1_02_half | 44.278 | 68.469 | 28.688 | 50.679 | 89.914 | 131 | 0 | 0 |
| H_rgb_rgb_s1_03_uncertain | 45.396 | 68.558 | 30.106 | 51.633 | 89.880 | 138 | 0 | 0 |
| H_rgb_rgb_s2_00_zero | 44.825 | 68.354 | 29.441 | 52.622 | 89.959 | 125 | 0 | 0 |
| H_rgb_rgb_s2_01_quarter | 44.472 | 68.203 | 29.045 | 51.175 | 89.927 | 128 | 0 | 0 |
| H_rgb_rgb_s2_02_half | 47.998 | 68.438 | 33.716 | 55.394 | 89.944 | 131 | 0 | 0 |
| H_rgb_rgb_s2_03_uncertain | 47.956 | 68.395 | 33.675 | 55.704 | 89.944 | 136 | 0 | 0 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

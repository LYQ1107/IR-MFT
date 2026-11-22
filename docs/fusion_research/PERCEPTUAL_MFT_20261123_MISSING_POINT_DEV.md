# Missing-point appearance Dev controls

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

Complete20 Dev controls, including full appearance fallback and explicit missing-point lag policies.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births | Missing-point appearance rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| J_new_ir_00_parent | 50.933 | 68.460 | 37.928 | 60.029 | 90.368 | 116 | 0 | 0 | 0 |
| J_new_ir_01_ambiguous_only | 50.933 | 68.460 | 37.928 | 60.029 | 90.368 | 116 | 0 | 0 | 25 |
| J_new_ir_02_strict_opening | 50.933 | 68.462 | 37.927 | 60.028 | 90.373 | 116 | 0 | 0 | 25 |
| J_new_ir_03_strict_strong | 50.933 | 68.462 | 37.927 | 60.028 | 90.373 | 116 | 0 | 0 | 25 |
| J_new_ir_04_motion_reconsideration | 50.930 | 68.458 | 37.926 | 60.027 | 90.379 | 115 | 0 | 0 | 178 |
| J_new_rgb_00_parent | 48.761 | 69.085 | 34.470 | 57.195 | 90.545 | 120 | 0 | 0 | 0 |
| J_new_rgb_01_ambiguous_only | 48.761 | 69.085 | 34.470 | 57.195 | 90.545 | 120 | 0 | 0 | 13 |
| J_new_rgb_02_strict_opening | 48.761 | 69.085 | 34.470 | 57.195 | 90.545 | 120 | 0 | 0 | 13 |
| J_new_rgb_03_strict_strong | 48.760 | 69.084 | 34.470 | 57.193 | 90.540 | 120 | 0 | 0 | 13 |
| J_new_rgb_04_motion_reconsideration | 48.671 | 69.079 | 34.348 | 57.201 | 90.545 | 121 | 0 | 0 | 151 |
| J_rgb_ir_00_parent | 43.977 | 68.298 | 28.374 | 49.506 | 89.828 | 135 | 0 | 0 | 0 |
| J_rgb_ir_01_ambiguous_only | 43.977 | 68.299 | 28.374 | 49.506 | 89.828 | 135 | 0 | 0 | 22 |
| J_rgb_ir_02_strict_opening | 43.977 | 68.299 | 28.374 | 49.506 | 89.826 | 135 | 0 | 0 | 26 |
| J_rgb_ir_03_strict_strong | 43.977 | 68.299 | 28.374 | 49.507 | 89.822 | 135 | 0 | 0 | 26 |
| J_rgb_ir_04_motion_reconsideration | 43.982 | 68.295 | 28.383 | 49.511 | 89.815 | 136 | 0 | 0 | 224 |
| J_rgb_rgb_00_parent | 47.088 | 68.678 | 32.342 | 53.582 | 89.933 | 133 | 0 | 0 | 0 |
| J_rgb_rgb_01_ambiguous_only | 47.089 | 68.679 | 32.342 | 53.582 | 89.933 | 133 | 0 | 0 | 15 |
| J_rgb_rgb_02_strict_opening | 47.088 | 68.679 | 32.342 | 53.581 | 89.931 | 133 | 0 | 0 | 15 |
| J_rgb_rgb_03_strict_strong | 47.088 | 68.679 | 32.342 | 53.581 | 89.931 | 133 | 0 | 0 | 15 |
| J_rgb_rgb_04_motion_reconsideration | 47.098 | 68.693 | 32.348 | 53.589 | 89.946 | 134 | 0 | 0 | 222 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

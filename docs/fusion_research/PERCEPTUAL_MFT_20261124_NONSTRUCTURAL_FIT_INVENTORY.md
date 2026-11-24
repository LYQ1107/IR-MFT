# Existing nonstructural seed0 fits: direct dependency inventory

These are existing trained candidates or representation controls. Negative Dev does not retroactively make a formal candidate diagnostic. This inventory is incomplete for indirect dependencies and other registered families; it is not a global Test-coverage pass.

| Family | Fit | Executed epochs | Complete Dev references | Complete Test references |
| --- | --- | ---: | ---: | ---: |
| metric | ir_s0 | 24 | 4 | 1 |
| metric | rgb_s0 | 24 | 8 | 2 |
| association_metric | ir_s0 | 36 | 33 | 2 |
| association_metric | rgb_s0 | 36 | 177 | 2 |
| native_cue | ir_s0 | 36 | 0 | 0 |
| native_cue | rgb_s0 | 36 | 0 | 0 |
| native_neck_metric | ir_s0 | 36 | 0 | 0 |
| native_neck_metric | rgb_s0 | 36 | 0 | 0 |
| native_query | new_ir_s0 | 36 | 22 | 0 |
| native_query | new_rgb_s0 | 36 | 19 | 0 |
| native_query | rgb_ir_s0 | 36 | 15 | 0 |
| native_query | rgb_rgb_s0 | 36 | 12 | 0 |
| native_robust_query | new_ir_s0 | 36 | 17 | 0 |
| native_robust_query | new_rgb_s0 | 36 | 14 | 0 |
| native_robust_query | rgb_ir_s0 | 36 | 17 | 0 |
| native_robust_query | rgb_rgb_s0 | 36 | 14 | 0 |

Fixed NovemberDiffV2IR campaign. Read only completed artifacts; no fitting, GPU work, new evaluation, threshold choice, Dev/Test primary selection or seed replication.

Trace feature/cache provenance and each preregistered formal role, freeze original complete-Dev choices, then backfill engineering-passed seed0 roles irrespective of Dev sign. Do not add seed1/2 replications. Do not count generic association calibration as visual fusion novelty.

Counts describe direct checkpoint paths in complete replay configurations. A zero is not proof that an encoder was never used indirectly, and a positive count is not a complete formal-role audit.

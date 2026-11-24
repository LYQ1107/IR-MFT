# Complete S/C/R component Dev evidence

Full4834frame Dev. Four missing IRcomponent fits,all4snapshots/bothpolicies,seed0. Previously completed IRfull/pair curves and RGB/gray controls reused. Primary1116/raw unchanged. Original RGB pretraining included currentDev. No Test-based choice.

S:structure weights/validity/cost confidence; C:semantic centering; R:feature confidence placement.
000 is a trained static local residual, not RGB identity. None of the scalar removal deltas are added.

| S C R | Same6/raw HOTA | Best4/policy HOTA | Best case | Same6/raw IDF1 |
| --- | ---: | ---: | --- | ---: |
| 000 | 46.298 | 48.140 | FCT_ir_000_s0_e18_raw | 51.236 |
| 001 | 45.468 | 48.642 | FCT_ir_001_s0_e18_raw | 51.089 |
| 010 | 46.756 | 47.643 | FCT_ir_010_s0_e06 | 52.809 |
| 011 | 47.662 | 50.938 | SR_ir_no_structure_s0_e12_raw | 53.795 |
| 100 | 46.913 | 47.382 | FCT_ir_100_s0_e24 | 52.107 |
| 101 | 46.849 | 47.971 | NC_ir_no_cycle_no_centering_s0_e12_raw | 52.697 |
| 110 | 47.846 | 48.629 | NC_ir_no_cycle_no_risk_s0_e18 | 55.022 |
| 111 | 51.329 | 51.329 | SR_ir_no_cycle_s0_e06_raw | 60.610 |

Fixed full minus original RGB: +3.750; minus trained static at6/raw: +5.031; minus strongest proper subset curve: +0.391 HOTA.

Algebraic decomposition of paired6/raw HOTA, not independent additive removal gains or a uniquely identified physical causal effect.

S/C/R are provisional task-specific mechanism groups. The zero-bit condition is a trained uniform local/static residual with bounds and original losses, not the untrained RGB identity. Classical primitives are prior art. Do not add removal deltas or count this factorial design/training coverage as new innovations.

These Dev results do not establish the >=+5Test or2–3original-innovation goal.

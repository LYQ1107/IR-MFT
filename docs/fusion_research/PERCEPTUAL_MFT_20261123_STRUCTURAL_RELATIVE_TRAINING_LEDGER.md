# Fixed-seed structural-relative training ledger

Each fit:24 passes over1152 cached Train frames,6912 updates; not24 passes over all27251 Train frames. Train480x864, Dev800x1440.

Matched145600 adapter parameters. RGB-repeat reuses one encoder; gray/actualIR use two shared-weight encodings. Shared producer timing is not independent per-method end-to-end FPS.

| Fit | Epochs | Updates | Parameters | Physical GPU | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rgb_full_s0 | 24 | 6912 | 145600 | 2 | 1583.6 |
| gray_full_s0 | 24 | 6912 | 145600 | 3 | 1722.2 |
| ir_full_s0 | 24 | 6912 | 145600 | 2 | 1718.1 |
| ir_no_structure_s0 | 24 | 6912 | 145600 | 3 | 1651.6 |
| ir_no_cycle_s0 | 24 | 6912 | 145600 | 2 | 1758.7 |
| ir_no_centering_s0 | 24 | 6912 | 145600 | 3 | 1680.6 |

All six fits use seed0 and exactly the same24 epoch sample digests as the original seed0 fit. Every checkpoint hash, training loss curve, cache manifest and source version is in the adjacent JSON. No accuracy claim.

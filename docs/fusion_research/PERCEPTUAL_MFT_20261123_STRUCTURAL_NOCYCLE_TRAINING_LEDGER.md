# Four fixed-seed no-cycle control fits

Four architecture/control fits, not four random seeds. Each24passes over1152 cached Trainframes (192 per6videos),6912updates at480x864; not24passes over the full27251frameTrain. No Dev/Test rows used for fitting. Existing FPN/descriptors reused, no full image/feature cache duplication.

| Fit | Seed | Parameters | Cache passes | Updates | Checkpoints |
| --- | ---: | ---: | ---: | ---: | ---: |
| rgb_no_cycle_s0 | 0 | 145600 | 24 | 6912 | 4 |
| gray_no_cycle_s0 | 0 | 145600 | 24 | 6912 | 4 |
| ir_no_cycle_no_centering_s0 | 0 | 145600 | 24 | 6912 | 4 |
| ir_no_cycle_no_risk_s0 | 0 | 145600 | 24 | 6912 | 4 |

All24 sampling digests match the completed original IR no-cycle fit. The primary IR model was reused, not retrained. All16 checkpoints and original runtime sources are hashed in the adjacent JSON; large checkpoint files remain outside Git.

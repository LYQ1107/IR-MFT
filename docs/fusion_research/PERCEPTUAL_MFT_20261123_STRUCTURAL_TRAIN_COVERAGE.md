# Existing Train-cache coverage

Only existing Train GT, frame counts and cache manifests are read. No images/FPNs copied, no fitting, no Dev/Test labels or accuracy. The cache was chosen before this audit.

| Video | Full frames | Cached frames | Cached IDs / all IDs | Median frame gap | Full-frame share | Cache share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BT-001 | 3000 | 192 | 10 / 10 | 16.0 | 11.009% | 16.667% |
| BT-003 | 243 | 192 | 10 / 10 | 1.0 | 0.892% | 16.667% |
| MSK-002 | 1754 | 192 | 49 / 49 | 9.0 | 6.436% | 16.667% |
| PF-001 | 15000 | 192 | 2 / 2 | 79.0 | 55.044% | 16.667% |
| SN-001 | 684 | 192 | 13 / 13 | 4.0 | 2.510% | 16.667% |
| SN-015 | 6570 | 192 | 13 / 14 | 34.0 | 24.109% | 16.667% |

1152/27251 unique frames (4.227%);24cachepasses = 27648sampleexposures, 1.015times the fullTrainframe count, but still only1152unique frames.

Repeated cache passes do not create new temporal/appearance observations. Coverage differences motivate a controlled sampling study, not a claim that longer training or more unique frames improve HOTA. No new training or selection policy is activated.

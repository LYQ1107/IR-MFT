# Complete three-seed correlation replication

All4834 Dev frames and all registered84 new cases. Seed means average three official combined-video scores, not the two per-sequence scores. Sample standard deviation over3 seeds is descriptive, not a significance test.

Common fullcue epoch/policy chosen by3-seed mean Dev. Head-role evaluation uses the preregistered prior seed0 epochs across every seed, with no new per-seed role epoch selection. Retain every failed control. NMS/emission factorial and original-parent removals are still separate pending evidence.

| Common full-cue selection | Epoch | Policy | HOTA mean ± sample SD | DetA | AssA | IDF1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| rgb | 12 | raw | 47.486 ± 0.590 | 68.604 | 32.907 | 54.765 |
| gray | 12 | raw | 47.976 ± 1.182 | 68.318 | 33.742 | 55.771 |
| ir | 6 | raw | 48.529 ± 0.772 | 68.610 | 34.365 | 56.400 |

| Fixed-epoch role/policy | RGB mean HOTA | Gray | Actual IR | IR − RGB | IR − gray |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_priority | 45.897 | 46.355 | 46.668 | +0.771 | +0.313 |
| full_raw | 46.891 | 46.412 | 48.197 | +1.306 | +1.785 |
| rgb_boxes_priority | 47.721 | 47.716 | 48.607 | +0.887 | +0.891 |
| rgb_boxes_raw | 48.331 | 47.115 | 48.567 | +0.236 | +1.451 |
| rgb_scores_priority | 47.060 | 47.211 | 47.547 | +0.487 | +0.336 |
| rgb_scores_raw | 45.917 | 49.006 | 48.194 | +2.277 | -0.812 |

| Seed0 intervention | Role | Policy | HOTA | Change from correct IR |
| --- | --- | --- | ---: | ---: |
| unavailable | full | priority | 47.579 | +0.191 |
| unavailable | full | raw | 47.579 | -2.192 |
| unavailable | rgb_boxes | priority | 47.579 | -1.337 |
| unavailable | rgb_boxes | raw | 47.579 | -2.936 |
| unavailable | rgb_scores | priority | 47.579 | -0.164 |
| unavailable | rgb_scores | raw | 47.579 | -0.528 |
| spatial_roll | full | priority | 47.600 | +0.212 |
| spatial_roll | full | raw | 47.612 | -2.159 |
| spatial_roll | rgb_boxes | priority | 48.462 | -0.454 |
| spatial_roll | rgb_boxes | raw | 47.963 | -2.552 |
| spatial_roll | rgb_scores | priority | 46.770 | -0.973 |
| spatial_roll | rgb_scores | raw | 47.871 | -0.236 |

These Dev values do not establish the >=+5 Test HOTA target or2–3 independent contributions. Full per-seed metrics, all common curves and sequence-level means are retained in the adjacent JSON. Historical Test exposure and previous RGB pretraining on present Dev scenes remain limitations.

# All25 matched structural-relative controls

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

All3cues x4snapshots xraw/priority plus original RGB. Fixedseed0,145600params, 24passes over1152Trainframes/6912updates. Bothfullsource8/32/full59proofs passed beforeaccuracy. Sharedproducer timing is not per-method end-to-end FPS.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CG3_source_RGB | 47.579 | 68.864 | 32.908 | 55.865 | 90.200 | 94 | 0 | 0 |
| SR_gray_full_s0_e06 | 47.472 | 68.795 | 32.791 | 54.022 | 90.172 | 90 | 0 | 0 |
| SR_gray_full_s0_e06_raw | 47.672 | 68.437 | 33.243 | 54.474 | 89.959 | 103 | 0 | 0 |
| SR_gray_full_s0_e12 | 47.553 | 68.802 | 32.893 | 55.067 | 90.159 | 97 | 0 | 0 |
| SR_gray_full_s0_e12_raw | 46.412 | 68.598 | 31.433 | 52.986 | 89.985 | 106 | 0 | 0 |
| SR_gray_full_s0_e18 | 47.676 | 68.746 | 33.091 | 55.074 | 90.147 | 98 | 0 | 0 |
| SR_gray_full_s0_e18_raw | 46.433 | 68.442 | 31.533 | 52.352 | 89.979 | 105 | 0 | 0 |
| SR_gray_full_s0_e24 | 47.679 | 68.761 | 33.088 | 55.093 | 90.168 | 98 | 0 | 0 |
| SR_gray_full_s0_e24_raw | 46.436 | 68.509 | 31.506 | 52.369 | 90.017 | 108 | 0 | 0 |
| SR_ir_full_s0_e06 | 47.568 | 68.825 | 32.910 | 55.844 | 90.132 | 93 | 0 | 0 |
| SR_ir_full_s0_e06_raw | 49.599 | 68.451 | 35.967 | 57.606 | 90.172 | 91 | 0 | 0 |
| SR_ir_full_s0_e12 | 47.570 | 68.824 | 32.914 | 55.849 | 90.138 | 93 | 0 | 0 |
| SR_ir_full_s0_e12_raw | 48.848 | 68.404 | 34.911 | 56.079 | 90.078 | 89 | 0 | 0 |
| SR_ir_full_s0_e18 | 46.678 | 68.884 | 31.666 | 53.975 | 90.129 | 89 | 0 | 0 |
| SR_ir_full_s0_e18_raw | 47.994 | 68.457 | 33.675 | 54.227 | 90.095 | 86 | 0 | 0 |
| SR_ir_full_s0_e24 | 46.679 | 68.884 | 31.666 | 53.975 | 90.129 | 89 | 0 | 0 |
| SR_ir_full_s0_e24_raw | 46.463 | 68.179 | 31.695 | 52.202 | 90.138 | 88 | 0 | 0 |
| SR_rgb_full_s0_e06 | 48.207 | 68.825 | 33.797 | 56.272 | 90.110 | 91 | 0 | 0 |
| SR_rgb_full_s0_e06_raw | 49.455 | 68.373 | 35.801 | 57.586 | 90.149 | 89 | 0 | 0 |
| SR_rgb_full_s0_e12 | 46.381 | 68.786 | 31.304 | 52.928 | 90.149 | 93 | 0 | 0 |
| SR_rgb_full_s0_e12_raw | 48.298 | 68.119 | 34.273 | 55.264 | 89.966 | 97 | 0 | 0 |
| SR_rgb_full_s0_e18 | 47.751 | 68.797 | 33.172 | 55.484 | 90.155 | 88 | 0 | 0 |
| SR_rgb_full_s0_e18_raw | 47.065 | 68.529 | 32.353 | 52.303 | 90.035 | 101 | 0 | 0 |
| SR_rgb_full_s0_e24 | 47.754 | 68.813 | 33.169 | 55.507 | 90.187 | 88 | 0 | 0 |
| SR_rgb_full_s0_e24_raw | 47.379 | 68.149 | 32.966 | 53.197 | 90.005 | 101 | 0 | 0 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

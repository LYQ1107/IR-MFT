# Global-priority observation integration on complete Dev

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

All20 registered controls use the exact parent ROI predictions without new GPU forwards. All original high-confidence coordinates/scores and distinct original observations are preserved. Report all RGB/IR guides, scales, budgets and global detectors. No demonstrated Test gain or multimodal innovation.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V_regions_baseline_new | 46.236 | 68.766 | 31.137 | 53.119 | 90.224 | 108 | 0 | 0 |
| V_regions_baseline_rgb | 47.579 | 68.864 | 32.908 | 55.865 | 90.200 | 94 | 0 | 0 |
| V_regions_new_ir_half_k2_priority | 43.742 | 68.502 | 27.981 | 49.533 | 89.822 | 127 | 0 | 0 |
| V_regions_new_ir_half_k4_priority | 43.144 | 68.373 | 27.278 | 48.420 | 89.550 | 140 | 0 | 0 |
| V_regions_new_ir_quarter_k2_priority | 46.137 | 68.757 | 31.007 | 52.810 | 90.159 | 110 | 0 | 0 |
| V_regions_new_ir_quarter_k4_priority | 46.115 | 68.748 | 30.982 | 52.802 | 90.140 | 109 | 0 | 0 |
| V_regions_new_rgb_half_k2_priority | 46.166 | 68.443 | 31.195 | 54.851 | 89.779 | 122 | 0 | 0 |
| V_regions_new_rgb_half_k4_priority | 48.461 | 68.535 | 34.321 | 57.716 | 89.667 | 126 | 0 | 0 |
| V_regions_new_rgb_quarter_k2_priority | 45.523 | 68.772 | 30.182 | 51.817 | 90.196 | 110 | 0 | 0 |
| V_regions_new_rgb_quarter_k4_priority | 45.521 | 68.788 | 30.172 | 51.801 | 90.213 | 110 | 0 | 0 |
| V_regions_new_uniform_k4_priority | 43.856 | 68.490 | 28.140 | 49.425 | 89.716 | 141 | 0 | 0 |
| V_regions_rgb_ir_half_k2_priority | 40.656 | 67.952 | 24.384 | 44.069 | 89.611 | 121 | 0 | 0 |
| V_regions_rgb_ir_half_k4_priority | 40.453 | 67.883 | 24.166 | 43.407 | 89.256 | 151 | 0 | 0 |
| V_regions_rgb_ir_quarter_k2_priority | 47.484 | 68.845 | 32.785 | 55.550 | 90.159 | 96 | 0 | 0 |
| V_regions_rgb_ir_quarter_k4_priority | 47.483 | 68.843 | 32.785 | 55.541 | 90.138 | 96 | 0 | 0 |
| V_regions_rgb_rgb_half_k2_priority | 42.685 | 68.176 | 26.778 | 48.093 | 89.703 | 123 | 0 | 0 |
| V_regions_rgb_rgb_half_k4_priority | 42.902 | 68.065 | 27.113 | 48.507 | 89.264 | 143 | 0 | 0 |
| V_regions_rgb_rgb_quarter_k2_priority | 47.482 | 68.592 | 32.904 | 55.856 | 90.190 | 96 | 0 | 0 |
| V_regions_rgb_rgb_quarter_k4_priority | 47.479 | 68.589 | 32.901 | 55.845 | 90.168 | 96 | 0 | 0 |
| V_regions_rgb_uniform_k4_priority | 41.628 | 67.876 | 25.583 | 46.350 | 89.505 | 142 | 0 | 0 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

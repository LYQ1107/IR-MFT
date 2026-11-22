# Ordinary pair calibration: complete common36-epoch Dev audit

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

All six ordinary association heads actually completed the registered36-epoch cap, with all six checkpoint intervals and three head seeds. Final ordinary-control selection uses only the complete72-row Dev curve, then every selected frontend/cue/seed receives all7 Test sequences.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G_new_ir_s0_e36 | 44.913 | 68.609 | 29.454 | 51.677 | 90.467 | 127 | 0 | 0 |
| G_new_ir_s1_e36 | 44.914 | 68.519 | 29.493 | 51.199 | 90.502 | 119 | 0 | 0 |
| G_new_ir_s2_e36 | 45.745 | 68.513 | 30.593 | 52.761 | 90.534 | 112 | 0 | 0 |
| G_new_rgb_s0_e36 | 49.227 | 68.476 | 35.433 | 57.724 | 90.506 | 111 | 0 | 0 |
| G_new_rgb_s1_e36 | 48.974 | 68.189 | 35.222 | 58.318 | 90.506 | 112 | 0 | 0 |
| G_new_rgb_s2_e36 | 47.692 | 68.084 | 33.448 | 55.253 | 90.543 | 110 | 0 | 0 |
| G_rgb_ir_s0_e36 | 46.870 | 68.363 | 32.191 | 54.547 | 89.903 | 134 | 0 | 0 |
| G_rgb_ir_s1_e36 | 44.524 | 68.667 | 28.917 | 51.970 | 89.811 | 138 | 0 | 0 |
| G_rgb_ir_s2_e36 | 44.366 | 68.383 | 28.832 | 50.307 | 89.910 | 132 | 0 | 0 |
| G_rgb_rgb_s0_e36 | 44.602 | 68.428 | 29.119 | 49.916 | 89.895 | 126 | 0 | 0 |
| G_rgb_rgb_s1_e36 | 45.100 | 68.486 | 29.747 | 51.868 | 89.899 | 127 | 0 | 0 |
| G_rgb_rgb_s2_e36 | 45.867 | 68.580 | 30.722 | 53.177 | 89.959 | 129 | 0 | 0 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

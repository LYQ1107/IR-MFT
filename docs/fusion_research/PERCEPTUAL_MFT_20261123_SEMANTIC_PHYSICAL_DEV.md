# Semantic/physical confidence separation: all 24 Dev rows

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

All12 off controls reproduce every39-field metric, both track files and detailedCSV exactly. Geometry-matched current-frame RGBconfidence in primary/pending association costs leaves main IR+RGBboxes HOTA50.515 unchanged; strongest graycontrol becomes50.117. Detection membership, births, emitted confidence and boxes remain fixed. This correction earns no independent contribution; exact dense-anchor score separation is a distinct untested intervention. No Test or parameter fitting.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SP_RGB_identity_off | 47.579 | 68.864 | 32.908 | 55.865 | 90.200 | 94 | 0 | 0 |
| SP_RGB_identity_on | 47.579 | 68.864 | 32.908 | 55.865 | 90.200 | 94 | 0 | 0 |
| SP_RGB_platt_off | 33.043 | 61.986 | 17.709 | 31.735 | 78.220 | 469 | 0 | 0 |
| SP_RGB_platt_on | 34.506 | 61.451 | 19.461 | 34.620 | 78.145 | 491 | 0 | 0 |
| SP_RGB_temperature_off | 44.186 | 68.881 | 28.390 | 49.968 | 90.157 | 130 | 0 | 0 |
| SP_RGB_temperature_on | 44.146 | 68.896 | 28.332 | 49.966 | 90.127 | 133 | 0 | 0 |
| SP_gray_none_off | 49.965 | 68.097 | 36.707 | 58.417 | 89.311 | 113 | 0 | 0 |
| SP_gray_none_on | 50.117 | 68.102 | 36.928 | 58.417 | 89.327 | 110 | 0 | 0 |
| SP_gray_rgb_boxes_off | 48.111 | 68.445 | 33.856 | 55.609 | 89.966 | 103 | 0 | 0 |
| SP_gray_rgb_boxes_on | 48.272 | 68.455 | 34.077 | 55.613 | 89.992 | 100 | 0 | 0 |
| SP_gray_rgb_scores_off | 50.044 | 68.363 | 36.668 | 59.629 | 89.695 | 104 | 0 | 0 |
| SP_gray_rgb_scores_on | 50.044 | 68.363 | 36.668 | 59.629 | 89.695 | 104 | 0 | 0 |
| SP_ir_none_off | 49.771 | 68.456 | 36.217 | 59.972 | 89.884 | 88 | 0 | 0 |
| SP_ir_none_on | 49.770 | 68.454 | 36.217 | 59.969 | 89.882 | 87 | 0 | 0 |
| SP_ir_rgb_boxes_off | 50.515 | 68.697 | 37.170 | 59.907 | 90.235 | 86 | 0 | 0 |
| SP_ir_rgb_boxes_on | 50.515 | 68.695 | 37.170 | 59.906 | 90.233 | 86 | 0 | 0 |
| SP_ir_rgb_scores_off | 48.107 | 68.247 | 33.943 | 56.798 | 89.845 | 100 | 0 | 0 |
| SP_ir_rgb_scores_on | 48.107 | 68.247 | 33.943 | 56.798 | 89.845 | 100 | 0 | 0 |
| SP_rgb_none_off | 49.957 | 68.620 | 36.397 | 58.252 | 90.080 | 96 | 0 | 0 |
| SP_rgb_none_on | 49.928 | 68.667 | 36.330 | 58.032 | 90.071 | 97 | 0 | 0 |
| SP_rgb_rgb_boxes_off | 48.945 | 68.884 | 34.812 | 57.526 | 90.237 | 93 | 0 | 0 |
| SP_rgb_rgb_boxes_on | 49.008 | 68.928 | 34.879 | 57.307 | 90.246 | 91 | 0 | 0 |
| SP_rgb_rgb_scores_off | 45.784 | 68.567 | 30.611 | 53.308 | 89.946 | 98 | 0 | 0 |
| SP_rgb_rgb_scores_on | 45.784 | 68.567 | 30.611 | 53.308 | 89.946 | 98 | 0 | 0 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.

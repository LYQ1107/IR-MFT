# Frozen no-cycle IR: complete7-sequence Test

Full7historicalTest sequences/15985frames, official combined TrackEval. Primary IR6/raw was fixed before earlierCGA14Test accuracy; controls/removal choices use complete Dev only. Identical paired/Dev-best cases are evaluated once and referenced by multiple declared roles. No bestTest method/checkpoint/seed selection. This is exploratory, not a newly blind Test.

| Frozen case | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NT_primary_IR | 29.142 | 45.775 | 18.632 | 32.057 | 50.749 | 455 |
| NT_cycle_restored_Dev_selected | 30.275 | 46.074 | 19.989 | 33.970 | 50.622 | 436 |
| NT_matched_rgb | 29.438 | 46.190 | 18.850 | 33.039 | 50.847 | 466 |
| NT_matched_gray | 29.534 | 45.714 | 19.170 | 32.530 | 50.420 | 471 |
| NT_no_structure_paired6 | 29.048 | 46.382 | 18.274 | 31.461 | 50.941 | 449 |
| NT_no_structure_Dev_best4 | 29.480 | 46.327 | 18.843 | 31.814 | 50.922 | 446 |
| NT_no_centering_paired6 | 29.904 | 45.829 | 19.605 | 33.086 | 50.520 | 432 |
| NT_no_centering_Dev_best4 | 30.378 | 46.024 | 20.148 | 34.110 | 50.487 | 453 |
| NT_no_feature_risk_paired6 | 30.576 | 46.238 | 20.309 | 34.362 | 50.919 | 443 |
| NT_no_feature_risk_Dev_best4 | 29.564 | 46.371 | 18.940 | 32.720 | 50.927 | 463 |
| NT_IR_image_roll | 29.523 | 46.217 | 18.937 | 32.723 | 50.948 | 462 |
| NT_IR_unavailable | 29.529 | 46.187 | 18.968 | 33.072 | 50.897 | 450 |
| NT_source_RGB_fresh | 29.529 | 46.187 | 18.968 | 33.072 | 50.897 | 450 |
| NT_source_RGB_canonical | 29.529 | 46.187 | 18.968 | 33.072 | 50.898 | 450 |

Frozen primary versus canonical RGB: -0.387; versus fresh RGB: -0.387; versus strongest matched RGB/gray: -0.392 HOTA.

| Removal | Primary − same6/raw | Primary − Dev-selected best4 |
| --- | ---: | ---: |
| structural_weights_validity_cost_confidence | +0.094 | -0.338 |
| semantic_centering | -0.762 | -1.236 |
| teacher_feature_risk_placement | -1.434 | -0.422 |

Actual-image roll change: +0.381 HOTA. Unavailable IR reproduces fresh RGB tracks, all39metrics and official detailedCSV exactly.

Hard cycle consistency was harmful and receives no contribution credit. The structure removal jointly removes local structural weights, validity and absolute-cost confidence. The risk removal removes only (1-p)*(p<.6), retaining structural confidence and feature/logit bounds. These task-specific mechanism groups still require fair modality controls, complete Test and prior-art analysis.

A numerical threshold or positive removals alone do not establish originality. The goal remains unmarked until the complete evidence and prior-art boundaries are assessed.

Cached proposal replay FPS excludes encoder/fusion/image preparation. Shared producer timings are not individual-method end-to-end performance. IR generation is also excluded.

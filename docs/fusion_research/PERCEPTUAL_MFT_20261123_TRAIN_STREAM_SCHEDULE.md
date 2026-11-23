# Prepared Train-frame coverage control

This is an engineering preparation, not a trained model or accuracy result.
It does not change the currently running Dev/Test family.

The address schedule keeps the same six Train videos,192slots per video per
pass,24passes and27,648sample exposures /6,912batch4updates. Pass1 exactly
matches each existing cached192-frame selection. Subsequent passes advance
through a fixed cyclic video permutation; all192frames within a pass are
distinct. The two-video/two-frame batch ordering can remain the original
seed0 ordering. All modality/control fits must share the same actual addresses.

| Video | Available frames | Unique scheduled frames | Sample exposures |
| --- | ---: | ---: | ---: |
| BT-001 | 3000 | 3000 | 4608 |
| BT-003 | 243 | 243 | 4608 |
| MSK-002 | 1754 | 1754 | 4608 |
| PF-001 | 15000 | 4608 | 4608 |
| SN-001 | 684 | 684 | 4608 |
| SN-015 | 6570 | 4608 | 4608 |

This schedules14,897unique Trainframes instead of1,152 while retaining the
original exposure count and equal-video allocation. It does not test
frame-proportional sampling or longer optimization. The compressed address
array occupies36,007bytes on the independent work volume; no image or FPN
copy is created. Train frame-count metadata and cache manifests were read;
GT label rows and Dev/Test data were not read by this preparation.

Three targeted tests passed: first-pass equality and complete-cycle coverage,
cue-independent/dictionary-order-independent addresses, and rejection of
extra model seeds or invalid cached addresses. This does not establish live
image/FPN or training parity. Before any fitting, verify original B1 encoder
features, scalar descriptors, GT targets, adapter initialization and the loss
path on existing cached frames, and register equal-budget modality controls.

Additional temporal/appearance observations might help generalization; the
current audit does not prove that they improve HOTA. This is a training
coverage control, not an independent multimodal innovation.

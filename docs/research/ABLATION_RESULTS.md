# Experiment and ablation results

Status: Phase 0 reproduced. All 27 pre-neck and 27 post-neck ordinary association runs, and 18 external runs per representation, are complete. All nine independent RGB post-neck capacity controls are complete; auxiliary-view training is in progress. No trained CMUE/PLIM result exists.

| Evaluation | Detections | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |
|---|---|---:|---:|---:|---:|---:|---:|
| Historical V3-E3 tracks, official metric replay | Historical | 31.0911 | 19.6192 | 49.5748 | 35.0443 | 52.3249 | 560 |
| Unchanged V3-E3 full inference, cu118 runtime | Newly inferred | 30.907 | 19.371 | 49.580 | 34.684 | 52.289 | 568 |
| Unchanged V3-E3 full inference, restored cu117 | Exactly historical arrays | 31.091 | 19.619 | 49.575 | 35.044 | 52.325 | 560 |

These rows test baseline reproducibility, not method improvements. Historical test was previously tuned. Old-cache BYTETracker replay produces byte-identical historical tracks. Historical HybridSORT-ReID screenshot and visible baseline numbers remain context only because shared detections/evaluator/provenance have not been established.

Frozen/trained feature interventions are representation diagnosis in PHASE_I_MODALITY_IDENTITY_DIAGNOSIS.md; GT/oracle-gallery retrieval is not IDF1. Three-seed RGB/causal-IR fish encoders and detector-crop feature banks are complete. Simple appearance tracking baselines use declared geometry/appearance mixing without utility/predictive modules. Disabled mode calls original BYTETracker and full historical replay is exact; enabled zero-weight lifecycle and post-association memory writes pass tests.

Required Protocol A baseline/fusion/CMUE/PLIM/write-policy and optional interaction rows will be added only from registered actual runs. Three final seeds, cost measurements, per-video details and leave-one-component-out evidence remain outstanding.

## Canonical causal association development baseline

4,834 frames from BT-005/SN-013, one common bank of 51,286 V3 proposals. All association methods use this exact bank and versioned derived GT. Detector trained on these images historically; new appearance/association models hold these videos out. Development videos differ from the historical test set above.

| Method | HOTA | AssA | DetA | IDF1 | MOTA | IDSW | FP | FN | Frag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original ByteTrack, causal IR inputs, original fixed settings | 49.2967 | 34.0115 | 71.5087 | 58.8771 | 92.5292 | 86 | 1,353 | 2,032 | 349 |

Source: `/data3/liuyeqiang/IR-MFT_assets/cfp/phase0/causal_dev_bytetrack_off_evaluation/metrics.json`.

## Provisional measured development configurations

Same 51,286 proposals and audited GT as above. These first-run rows are retained as measured configurations. Completed three-setting/all-seed selections are reported below; none authorizes final test evaluation. Shared boxes permit different tracking output coordinates/lifecycle, so DetA can change.

| Configuration | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |
|---|---:|---:|---:|---:|---:|---:|
| RGB / IR / fixed fusion, pre-neck seed 0, appearance weight .3 (identical tracks) | 47.0802 | 31.3048 | 70.8400 | 55.9773 | 92.4926 | 98 |
| Constant identical identity vector, weight .3 (same tracks as preceding row) | 47.0802 | 31.3048 | 70.8400 | 55.9773 | 92.4926 | 98 |
| RGB / IR / fixed fusion, pre-neck seed 0, appearance weight .1 (identical tracks) | 50.7595 | 36.0480 | 71.5208 | 61.2101 | 92.5421 | 84 |
| Constant identical identity vector, weight .1 | 50.7595 | 36.0480 | 71.5208 | 61.2101 | 92.5421 | 84 |
| Constant identical identity vector, weight .5 | 46.7502 | 30.9313 | 70.6879 | 55.9797 | 92.4754 | 100 |
| BoT-SORT + same fish RGB seed 0, appearance threshold .25, GMC none | 49.7996 | 33.8771 | 73.2813 | 54.0137 | 93.4719 | 112 |
| HybridSORT-ReID + same fish RGB seed 0, high/low appearance weights 4/4.4 | 46.3243 | 29.3761 | 73.1633 | 49.1164 | 92.9425 | 141 |

The .1 apparent gain does not establish identity or constructed-IR value: the identity-blind control completes all three weights and reproduces the corresponding appearance trajectories byte-for-byte in completed comparisons. This is positive evidence that geometry-gated acceptance alone explains these outcomes. Checks are retained in `phase1/identity_blind_control_parity.json`, refreshed after the full grid. The default .3 result is below original ByteTrack; evidence-directed metric-space repair is frozen in `reid_repair_v1.json` before its outcomes. Do not select or omit adverse baselines.

External association uses unchanged pinned upstream motion/assignment/lifecycle code with common fish RGB features, not native person-ReID weights or Protocol B. BoT-SORT's provider is replaced by an exact box/order-checked precomputed interface; HybridSORT needs a process-local singleton-array score conversion for NumPy 1.24. Vendor sources remain unchanged. Full configuration budgets are frozen in `external_association_v1.json`; isolated full-video runs and official metrics are under `phase1/causal_dev_{family}_seed*_c*`. CPU-shared cache replay latency is not a dedicated online benchmark. All rate/decomposition/count metrics are retained in registry and metric JSONs; these compact tables are not the entire evaluation.

## Completed three-seed development selections

Every family receives its three declared settings and all seeds 0/1/2. Select the highest mean combined dev HOTA, then AssA/IDF1; never the winning seed. Rates are percentages, ± is sample standard deviation. Machine-readable trials: `configs/cfp/baseline_selection_{pre,post}_neck_v1.json`. All per-video/full metrics remain in official evaluation files and registry.

| Family / representation / selected setting | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |
|---|---:|---:|---:|---:|---:|---:|
| ByteTrack | 49.297 | 34.012 | 71.509 | 58.877 | 92.529 | 86 |
| Constant identity vector, weight .1 | 50.760 | 36.048 | 71.521 | 61.210 | 92.542 | 84 |
| RGB / IR / fixed fusion, pre-neck .1, all seeds identical to constant | 50.760 | 36.048 | 71.521 | 61.210 | 92.542 | 84 |
| BoT-SORT, pre-neck .15 | 49.890±1.547 | 33.961±2.232 | 73.427 | 53.863±3.098 | 93.437 | 111.3±1.2 |
| HybridSORT, pre-neck high/low 1/1.1 | 46.548±0.0002 | 29.648 | 73.194 | 49.117 | 92.948 | 142 |
| RGB, post-neck w0.1 | 49.292±0.694 | 34.013±0.956 | 71.499±0.043 | 59.011±1.174 | 92.564±0.002 | 87.333±2.082 |
| Causal IR, post-neck w0.1 | 48.788±0.021 | 33.324±0.025 | 71.491±0.046 | 58.386±0.010 | 92.508±0.049 | 89.000±1.000 |
| Fixed RGB+IR, post-neck w0.3 | 49.089±0.545 | 33.864±0.779 | 71.229±0.424 | 59.118±0.551 | 92.556±0.061 | 92.000±2.646 |
| BoT-SORT, post-neck c0 | 52.482±0.134 | 37.498±0.165 | 73.518±0.061 | 57.477±0.593 | 93.552±0.024 | 113.667±4.509 |
| HybridSORT, post-neck c2 | 46.657±1.669 | 29.934±2.115 | 72.883±0.135 | 49.375±2.870 | 92.805±0.060 | 161.000±9.539 |

The best measured mean HOTA is post-neck BoT-SORT (52.482±0.134), but its IDF1 is lower than ByteTrack and IDSW is higher. This is a Pareto tradeoff, not a solved identity problem. Ordinary post-neck fusion remains below the identity-blind geometry control. Post-neck export changes only the metric representation of frozen weights, not the training budget; it fixes the narrow cone without establishing a constructed-IR gain. Native association rules can change DetA despite identical input proposals. Native person-weight/full-system Protocol B remains unperformed.

All 27 pre-neck modality/seed/weight runs are byte-identical to corresponding constant-vector controls. The nine independent two-RGB-encoder pre-neck capacity runs also give the same measured results. Exact official CLEAR traces show the .1 geometry control preserves BT-005 GT identities 3 and 7 at crossing frame 1049, while original ByteTrack swaps them. Both remain matched at frame 1049 and 1074; this is not a disappearance of the predictions. The original change requires no identity evidence. Source cases and hashes: `phase1/clear_events_*_with_matches.json` and `constant_gate_switch_figures_v2/`.

Post-neck independent RGB capacity control completes all nine runs: best mean HOTA 48.9835 at .1 (AssA 33.6044, IDF1 58.6921, IDSW 88). Fixed fusion selects .3 with HOTA 49.0889, but this small difference and adverse identity-blind comparator do not establish an IR advantage. Full three-setting outputs: `phase1/causal_dev_rgb_repeat_postneck_campaign_summary.json`. Both long-axis crop caches are complete, with no trained aligned result yet.

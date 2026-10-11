# Experiment and ablation results

Status: Phase 0 reproduced; canonical causal appearance/external association development comparisons are in progress. No CMUE/PLIM ablation has run.

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

Same 51,286 proposals and audited GT as above. These are actual completed configurations, not final model selection; three-setting/all-seed grids remain in progress. Shared boxes permit different tracking output coordinates/lifecycle, so DetA can change.

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

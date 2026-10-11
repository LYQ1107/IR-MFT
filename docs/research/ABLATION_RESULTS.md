# Experiment and ablation results

Status: Phase 0 evidence only. No new-method comparison or ablation has run.

| Evaluation | Detections | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |
|---|---|---:|---:|---:|---:|---:|---:|
| Historical V3-E3 tracks, official metric replay | Historical | 31.0911 | 19.6192 | 49.5748 | 35.0443 | 52.3249 | 560 |
| Unchanged V3-E3 full inference, cu118 runtime | Newly inferred | 30.907 | 19.371 | 49.580 | 34.684 | 52.289 | 568 |
| Unchanged V3-E3 full inference, restored cu117 | Exactly historical arrays | 31.091 | 19.619 | 49.575 | 35.044 | 52.325 | 560 |

These rows test baseline reproducibility, not method improvements. Historical test was previously tuned. Old-cache BYTETracker replay produces byte-identical historical tracks. Historical HybridSORT-ReID screenshot and visible baseline numbers remain context only because shared detections/evaluator/provenance have not been established.

Frozen-feature interventions are actual representation diagnosis results in PHASE_I_MODALITY_IDENTITY_DIAGNOSIS.md, not tracking ablations. A fish-trained RGB ReID baseline is running under fixed last-epoch selection; its GT/oracle-gallery retrieval must not be relabeled as IDF1. Simple appearance tracking baselines are implemented in `cfp/association.py` with declared geometry/appearance mixing and no trained utility/predictive module. Disabled mode calls original BYTETracker; enabled zero-weight lifecycle and post-association memory writes pass tests. Full-video replay and final causal-detection comparisons are still required.

Required Protocol A baseline/fusion/CMUE/PLIM/write-policy and optional interaction rows will be added only from registered actual runs. Three final seeds, cost measurements, per-video details and leave-one-component-out evidence remain outstanding.

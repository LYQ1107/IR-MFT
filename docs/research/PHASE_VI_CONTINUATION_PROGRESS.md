# Phase VI continuation: write quotas and bounded predictive corrections

Status: IN_PROGRESS. Collected on 2026-10-13. All numbers below use complete official evaluations. Six-video OOF means association-module video exclusion; detector and ReID pretraining exposure remains. This is development evidence, not independent generalization.

| Six-video method | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |
|---|---:|---:|---:|---:|---:|---:|
| Corresponding excluded-video ordinary GRU | 33.4884 | 19.6046 | 57.8709 | 35.7681 | 69.6640 | 676.0000 |
| quota/confidence | 33.7004 | 19.8739 | 57.8065 | 36.8130 | 69.6447 | 684.0000 |
| quota/context | 33.8316 | 20.0267 | 57.8213 | 36.6804 | 69.6719 | 678.0000 |
| quota/entropy | 33.3632 | 19.4611 | 57.8661 | 36.0021 | 69.6535 | 683.0000 |
| quota/learned_threshold | 33.6285 | 19.7661 | 57.9126 | 36.4076 | 69.6691 | 685.0000 |
| quota/margin | 33.8863 | 20.0765 | 57.8570 | 36.6479 | 69.6627 | 684.0000 |
| quota/mlp | 33.9488 | 20.1617 | 57.8175 | 36.8006 | 69.6686 | 680.0000 |
| quota/random | 33.3056 | 19.3830 | 57.8916 | 35.6384 | 69.6650 | 684.0000 |
| residual/ema_residual_utility | 33.6817 | 19.8254 | 57.8796 | 35.9775 | 69.6530 | 679.0000 |
| residual/future_residual_utility | 33.5538 | 19.6800 | 57.8848 | 35.8727 | 69.6525 | 681.0000 |
| residual/noema_residual_utility | 33.6817 | 19.8254 | 57.8796 | 35.9775 | 69.6539 | 677.0000 |
| residual/ordinary_residual_utility | 32.9522 | 18.9832 | 57.9165 | 35.2963 | 69.6502 | 676.0000 |
| relation/ordinary_relation_utility | 26.8804 | 13.0598 | 56.7512 | 26.9739 | 61.7990 | 15796.0000 |
| relation/raw_mean | 29.4273 | 15.3583 | 57.3991 | 30.2944 | 68.6051 | 1949.0000 |
| relation/raw_mean_original_eligibility | 32.3529 | 18.2561 | 57.9154 | 34.5715 | 69.6392 | 675.0000 |
| relation/relation_jepa_utility | 27.6981 | 13.9505 | 56.3597 | 26.6053 | 62.7004 | 13777.0000 |
| relation/relation_jepa_utility_future | 27.6938 | 13.9399 | 56.3957 | 26.6236 | 62.3054 | 14617.0000 |
| relation_extra/geometry_only_continuous | 32.9312 | 18.9141 | 58.0254 | 34.2947 | 69.6659 | 660.0000 |
| relation_extra/raw_mean_native | 29.3126 | 15.2372 | 57.4336 | 29.8825 | 68.5784 | 2071.0000 |
| relation_extra/raw_mean_native_eligibility | 32.3576 | 18.2476 | 57.9592 | 34.5588 | 69.6594 | 676.0000 |
| relation_extra/relation_noema_utility | 27.4045 | 13.6368 | 56.5036 | 26.3904 | 62.6857 | 13836.0000 |

## Interpretation and next discriminating experiments

At a fixed 10% matched-edge write budget, context HOTA across three initialization/query seeds is 33.7378±0.0665; same-input MLP is 33.7456±0.2203. The fixed margin rule is 33.8863. Candidate-context complexity has no demonstrated independent advantage. These repetitions keep ordinary producer and encoder seed0 fixed and must not be called three full pipeline seeds.

Replacing ordinary geometry costs causes large IDSW. Original eligibility repairs raw means substantially. Restricting learned corrections to ±0.05 within native appearance eligibility preserves the ordinary prior. EMA and noEMA bounded models have almost identical HOTA; the EMA mechanism has no demonstrated independent benefit. The ordinary residual control is worse than the unchanged producer, so auxiliary prediction deserves further controlled investigation. IDSW is still above baseline, and a small HOTA increase is not sufficient scientific success.

Registered continuation: 36 ordinary/noEMA/EMA fits with initialization/query seeds1/2 and fixed backbone/producer0; 18 corresponding seed0 fits with cue utility disabled. All use frozen epoch8, residual scale0.05, full outer-video replay, and identical protocol. No outer-video tuning or lucky-seed selection. These distinguish prediction regularization, auxiliary utility supervision and initialization variability.

Complete per-video metric vectors, official prediction hashes and paired statistics are in the adjacent measured JSON. Full risk deployment online evaluation is still running and its partial video evidence is not reported as a complete result.

Validation: 182 project tests passed with the migrated dataset path. Unscoped pytest also collected an unrelated legacy ctracker tutorial and failed because skimage is absent; the project tests/ suite is the verified scope. No historical model, cache or prediction was overwritten. Large training artifacts remain local.

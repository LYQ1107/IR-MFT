# Phase VI strong baseline audit

**BASELINE_REPRODUCED**. 88 inherited input/weight hashes checked; all three strong GRU full two-video outputs reproduced byte-for-byte; six independent official TrackEval evaluations exactly match preserved metric dictionaries and metric implementation hashes. No fitting, detector cache changes, or historical seven-video test use.

| System | HOTA mean | AssA | DetA | IDF1 | MOTA | IDSW |
|---|---:|---:|---:|---:|---:|---:|
| BoT_SORT_RGB_native | 52.481818 | 37.498032 | 73.517605 | 57.476822 | 93.552298 | 113.666667 |
| ordinary_gru_current_measurement | 53.591386 | 39.138938 | 73.456991 | 62.700856 | 93.672112 | 87.333333 |

Per-video, all 18 requested metrics, seed results, both standard deviation conventions, inputs, source snapshots and exact commands are in PHASE_VI_BASELINE_AUDIT.json. Current measurement output is frozen for all new GRU/module comparisons; preserved native BoT is an additional strong reference, not a same-output-strategy mechanism ablation.

Initial audit v1 used a wrongly assumed seed1/2 output-evaluation path and stopped after seed0 parity. Its exception and completed outputs are preserved under phase6/baseline_audit_v1; corrected pathrepair_v2 reran the audit in a new directory and completed all six official checks. No prior output was overwritten.

Mandatory reports were read; local/remote original head 00ae0e8 was verified before creating research/recoverability-candidate-jepa-v1. Full current suite: 140 passed, 18 existing warnings (pytest_vi0_vi1_v2.log). Available storage ~168 GiB, 40 GiB reserve, historical assets retained.

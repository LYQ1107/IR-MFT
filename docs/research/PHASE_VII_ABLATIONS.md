# Phase VII ablations

Status: IN_PROGRESS. Results below are the first conditional association-head probe at seed0; all encoder-level folds and three-seed confirmation remain pending.

All rows use the same complete SN-001/SN-015 V3 proposals, inherited RGB512 coordinate, ordinary GRU, native later association stages and measured confirmed output. Disabled output matches the historical baseline byte-for-byte on all684/6,570frames. New heads fit four other videos; SN-001 is inner calibration and SN-015 is an already exposed conditional new-head outer evaluation. These are not independent new-encoder generalization results.

| Route | SN-001 HOTA | SN-001 IDF1 | SN-001 IDSW | SN-015 HOTA | SN-015 IDF1 | SN-015 IDSW |
|---|---:|---:|---:|---:|---:|---:|
| Original GRU |46.2383|65.2514|26|31.9832|29.6359|151|
| Joint bounded0.03 |45.9268|63.7059|26|33.1384|31.4161|152|
| Pair bounded0.03 |45.7638|63.7059|28|33.1384|31.4161|152|
| Untrained uniform+0.03 |45.7638|63.7059|28|33.1384|31.4161|152|
| Joint full ID |33.4164|38.6327|78|18.4664|12.5882|859|
| Pair full ID |34.6024|40.7443|92|18.5699|12.9085|911|

Pinned official TrackEval includes all available per-video HOTA/AssA/DetA/IDF1/MOTA/IDSW/Frag/MT/PT/ML/FP/FN fields in `PHASE_VII_JOINT_REPLAY_MEASURED.json`; table rounding is display only. The combined two-video aggregate mixes calibration and outer roles and cannot be called an outer-only result.

The uniform positive control exactly reproduces pair-bounded prediction bytes on both videos and joint-bounded on SN-015. No attention, learned weight or ID-specific decision is needed to explain the claimed bounded gain on that outer video. A global high-stage cost increase changes effective acceptance from0.97toward0.94; it does not establish improved identity representation. The joint-vs-constant difference on SN-001 is a small exploratory same-seed effect and is not a retained contribution.

Required next controls include global/local fixed-input retrieval, independently trained heldout visual adapters, full-set ordinary Transformer, MOTIP-head adaptation, no track/detection attention, no geometry/memory/NULL, DenseJEPA matched local temporal/contrastive/noEMA/noPrediction/noID, isolated motion and selective module combinations. On-policy trajectories, all video folds, three seeds and complete raw online latency are also pending. The first full-ID failure is diagnosed separately from bounded-route identifiability; it does not close either direction.

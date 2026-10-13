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

## Executed visual/local controls

Four stopped-ImageNet adapters (global, equal-two-projection dualglobal, simpleK8, learnedK8) and two actual backbone fits(global/learnedK8) complete the first outerSN015/innerBT001 seed0 pilot. They share current crops and physical known-ID/same-frame/temporal supervision. Exact rows appear in REID_COMPARISON and VISUAL_PILOT_MEASURED. Simple geometric strip mean and concatenation plus inherited raw/preBN/BN diagnose representation choices; the dualglobal control tests projection capacity without local pixels. None establishes an outer detector retrieval gain above frozen ImageNet. K4train, transformer/fish-pretrain controls, noCE/nohard-negative, raw-versus-projection and full online/all-fold/three-seed remain pending.

## Registered three-seed first-fold visual confirmation

All global/local ImageNet-initialized backbone runs now complete for seeds0/1/2 at the same registered fixed epoch8 and split. Statistics below are mean ± sample SD across training seeds, not query-level independent uncertainty. Crop/query/gallery counts remain the same for each corresponding view.

| Model | Inner BT001 GT | Inner BT001 detector | Outer SN015 GT | Outer SN015 detector |
|---|---:|---:|---:|---:|
| global | 20.000 ± 1.198 | 12.927 ± 1.857 | 45.163 ± 0.364 | 42.804 ± 2.224 |
| learned | 18.995 ± 1.770 | 12.553 ± 1.203 | 44.111 ± 1.516 | 43.757 ± 3.308 |

Frozen ImageNet detector R1 is14.744% oninnerBT001 and53.651% onouterSN015. All new CNNs remain below the outer baseline; the seed0local-vs-global contrast is not stable across seeds. This confirms the negative first recipe, without declaring all local identity learning ineffective or selecting a lucky seed. Only this first fold is confirmed; all six outer/inner folds, stronger pretrained controls, actual localJEPA, repairedC/D and native/rawonline remain unfinished. Complete gap1/25/100R1/R5/mAP, sample denominators, AMP skips, losses, checkpoint/source/data hashes are in PHASE_VII_VISUAL_THREE_SEED_MEASURED.json.

A real globalseed1epoch2early exit and new-process resume onphysicalGPU0 reproduces continuous8epochtraining exactly:818tensor leaves,1NumPyarray and996scalar leaves across model/optimizer/scheduler/AMP/allRNG; training losses/update counts and retrieval are equal. The verification excludes walltime and serialized container bytes and does not certify crosshardware determinism. Stop/resume logs and both complete manifests are hash-linked in PHASE_VII_VISUAL_RESTART_MEASURED.json. Historical seed0source snapshots are preserved; the early-stop CLI added for seed1/2 does not pretend source equality to the previous run.

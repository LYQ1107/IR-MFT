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

## Frozen pretrained diagnostic outcome

DINOv2 official licensed S14 current-crop inference and47,751observation extraction are complete. Same actual/paired pools, query/gap masks and oldbaseline metrics are exactly preserved; paired-detector CLS beatsImageNet on3of6videos, while fixedlocalK4/K8concats are not consistently stronger. This is stronger frozen evidence, not supervisedlocal/JEPA/nativeMOT gain. See REID_COMPARISON table and PHASE_VII_DINOV2_FROZEN_MEASURED.json for allsix R1/R5/mAP/AUC/EER/supports and immutable proofs. Actual raw allframe/runtime and allfold/three-seed trained comparisons remain unfinished.

DINO supervised A controls are now actual three-seed firstfold fits: same current crops/masks/physical-ID sampler/CE+temporal+same-frame losses/1200steps, final-two-block adaptation and384matching coordinates. Global, simple patch pooling, real patch attention and identical-active-capacity CLS-only attention isolate spatial information from added head capacity. Actual sampled coverage is checked across modes; missing regions remain explicit. Allfold/K4K8/three-seed and strict execution-only restart are registered separately. These controls do not replace TransReID-/FDTA-style, masked fish self-supervision, stopped localJEPA or native association baselines.

## Executed object-local Dense JEPA controls

A genuine new local model is now implemented and fitted, distinct from inherited global512PLIM. Immutable licensed DINO observations provide real K4spatial384tokens and currentCLS. An identity-initialized local encoder learns observed-region attention/reliability; a temporal context processes only strictly past object observations. A regional masked-target decoder predictsKlocal tokens. Independent stopped deepcopy target encoder follows EMAafter successfuloptimizer updates, with clean currentGT targets only in training. No currentcandidate/GTtarget/teacher feature enters the prediction path; currentcandidates enter a separate cross-region relation scorer. CurrentGT classification is a separate ordinary training-supervision branch, never a predictor or online input. No copied I-JEPA/V-JEPA/FishMAE code is executed.

All nine first-seed fits complete at fixed8epochs/1200CPUupdates, excluding outerSN015/innerBT001 from gradients/classifier state: currentCLS global, ordinary384GRU, localattention, localTemporalTransformer, localtemporalcontrastive, noEMA, no masked-prediction objective, no explicitIDloss, fullJEPA. The raw backbone is immutable; this is the task-book-allowed learned adapter route. EMA/instant-target methods have1200real stopped-target updates, other models0; actual visual-context/temporal/predictor gradients and changedparameters are recorded. Independent teacher storage and no target gradients are verified. No-ID still uses annotated temporal grouping, so it is not annotation-free learning. NoPrediction keeps the regional query architecture trained through IDrelations but removes masked latent supervision. True active parameters/budgets are reported, including inactive frozen modules.

Training uses exact same per-epoch sampled current/past/target rows and target masks across all nine modes, verified by input digests. Same-frame known different-fish proposals are negatives; unknowns remain in the current inference pool and are excluded from labeled negative softmax denominators. Training past views mix uniqueGT and verified unique-detector observations50/50, gaps1/25/100. Evaluation separates GT/verifieddetector histories, GT/actualdetector current pools and all gaps. It retains currently absent pastknown inventory and empty sampled scenes. These are clean physical offline diagnostic groups, not endogenous native tracker states. `target_absent` denotes verified-positive absence, not proof that an unknown proposal cannot match or a valid NULL training label. Ground-truth identifiers/teacher/target never enter student forward. Natural occlusion/anatomical labels are absent; observed fractions are crop/padding support only.

| Seed0 method | InnerBT001 GT-history→detector R1 | OuterSN015 GT-history→detector R1 |
|---|---:|---:|
| Frozen CLS FIFO | 31.544 |64.894|
| Learned global currentCLS |37.584|72.695|
| Global+ordinary GRU384 |32.886|51.241|
| Local patch attention |36.465|66.312|
| Local temporal Transformer |36.913|64.716|
| Local temporal contrastive |29.306|47.518|
| Dense without EMA |27.293|44.858|
| Without masked-prediction loss |27.740|41.844|
| Without explicitIDloss |15.213|26.241|
| Full Dense JEPA |26.846|44.504|

All entries use the same current pools/unknowns/support: inner447 and outer564 queries with both verifiedpositive and observedpast at gap25. Raw query totals640/793, with verified-target-absent184/217 and history-absent10/17 separately recorded. FullJEPA reduces latent loss yet remains below global/ordinarylocalTemporal/frozen controls. It beats its query-decoder IDonly ablation onouterSN15, which does not establish superiority over the strong controls. No EMA/localJEPA identity or native improvement/novelty claim is retained. Three-seed confirmation remains running. Numeric fullallprotocol R1/R5/mAP/AUC/EER/margins/variance/unknown/support/source/checkpoint evidence is PHASE_VII_LOCAL_DENSE_FIRST_SEED_MEASURED.json. Physical perquery annotation traces and model arrays remain outsideGit.

A read-only fixed-full-checkpoint diagnostic reads innerBT001 only and rejects outer query requests. Keeping the same learned currentencoder/relation scorer and supports, encoder-only FIFO37.584%, temporalcontextpool37.584%, regionalprediction26.846%. Thus a measured inference decoder bottleneck is present; a blanket localencoder-information-loss explanation is unsupported on this inner pool. This motivates a separately registered representation-learning control: keep masked prediction as a training-only auxiliary objective and use encoder/context evidence for identity decisions, as ordinary JEPA downstream evaluation does. No new recipe is selected from outerSN015; the original model/results remain preserved. K8/allfold/onpolicy/native/rawonline, strongsource and finalnovelty work remain outstanding.

## Completed three-seed and actual native controls

All144 supervisedK4/K8 fits and27originalBfirstfold fits are complete. K8patchattention macroR136.013±0.352% versus matchedcapacityCLS36.371±0.087%; original fullJEPA45.095±1.181% versus ordinarylocalTemporal63.948±0.800% on the same offline supported scene. Separate contextidentityV2 excludes predictor/teacher online and its no_prediction/local_temporal are exact aliases, verified by activeparameters/finalstudentstate. Firstfullnative6570frame V2seed0 HOTA30.694/IDF125.298 does not surpassfrozenFIFO32.225/30.445. Full native last/FIFO/EMA three-seed/sixvideo matrix and allnineoriginalB/V2full onpolicy controls are active; no seed0necessity or novelty claim. Actual512frame exactFIFOadapter lifecycle/history parity bounds maximumcosterror3.632e-8. See hash-linked full metrics/proofs in PHASE_VII_NATIVE_CURRENT_IDENTITY_MEASURED.json and PHASE_VII_NATIVE_CURRENT_IDENTITY_METRICS_MEASURED.json.

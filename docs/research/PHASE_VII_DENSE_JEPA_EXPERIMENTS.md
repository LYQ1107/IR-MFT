# Phase VII object-centric Dense JEPA experiments

Status: IN_PROGRESS. Actual independent local-token JEPA and eight controlled alternatives have now been trained; current identity superiority is not established.

A0 shows why prediction-loss optimization alone is insufficient: frozen GRU pure retrieval loses to retained global features on clean SN-015 histories, while fixed native geometry scores hide much of that difference. Joint probes also show that apparent bounded gains can be reproduced by a constant cost edit. A future DenseJEPA result must add identity evidence under matched current candidate support, not merely lower latent error or move the acceptance threshold.

The required implementation uses current object crop local/spatial tokens from verified prior/current observations, visible past-only online context and an EMA target encoder used only during training. Future training target tokens may be stop-gradient labels but cannot enter history, online memory, masks, reliability or decisions. Masked target tokens cannot leak through teacher pooling or context construction. Target masks must correspond to annotated/observable validity; missing physical identity is not a negative.

Matched local TemporalTransformer, contrastive temporal learning, noEMA/noPrediction/noID and simple local aggregation controls remain required. Input coordinate, producer, crop budget, candidate pool, training video split and parameter/compute budget must be recorded. Paired crop features/retrieval and I-JEPA/V-JEPA2/GOT source-field audits are complete. All30reference audits and licensed DINOv2 observations are complete; the implemented controlled localBfirst-seed study is reported below. No dense predictive novelty or performance claim is retained at this stage.

## I-JEPA source audit completed

Exactcommit52c1ae95d05f743e000e8f10a1f3a79b10cff048 has16verifiedsource/config/license files andall11fields inREPOSITORY_REGISTRY. Code is [CCBYNC4](https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/LICENSE), notMIT/Apache; no officialcodecopied/executed orhugeweightsdownloaded. Staticimagecontext/maskedregionSmoothL1/stoppedEMA doesnot providefishidentitysupervision. Actualmasker relaxesdisjointness afterretries andrectangularpositioninterpolation isunfit for128x256; independentlyimplemented DenseJEPA musttesttargetexclusion/current-futurecausality andrectangularlocalshapes explicitly. This audit preparesB; no actuallocalDenseJEPAtraining resultis claimed.

## Source audit completion: thirty available-source field audits

All30 references now have all11 available source-level semantic fields and exact commit/blob/SHA reading proofs. The final completed set includes MOTRv2, TrackTrack and COVTrack;599listed source/config/license paths have exact provenance and targeted reading records. TFMFT missing model/criterion/sampler and TCEI missing train-entry modules remain explicit coverage exclusions, rather than verified complete training; this is not an exhaustive every-line reading claim or official pretrained reproduction. Weight delivery, unverified derivative/data terms and execution exclusions remain explicit in REPOSITORY_REGISTRY.

GOT-JEPA's registered head5f382645 also contains newerGOT-Edit; the README's paper release84e9324317 was independently materialized from existing local Git objects into a new isolated snapshot. Both actor versions define the same callable twice and the active last body ends with an unassigned tuple expression instead of the earlier loss/statistics return. Target/context feature-extractor parameters are shared; noEMA teacher update is implemented. Its latent target is a256-dimensional tracking filter, not Kfish identity tokens. Static AST and canonical source proofs are external/hash-linked; no foreign code was executed.

V-JEPA2/2.1 ordinary encoders attend bidirectional video clips. AC predictor's frame-causal mask does not establish causal backbone features.2.1Base's published train config selects a model name omitted from the train entry's embed-dimension branch; pretrained hub inference is a different path. CoTracker3's online wrapper initializes query/state separately and advances16frame windows by8; earlier window outputs can use later frames in the same window. Current-frame MOT needs explicit prefix boundaries, delay/emission policy and runtime accounting. Original full-video offline caches are excluded.

TransReID and CLIP-ReID static evaluators remove same-ID/same-camera positives, which would remove within-video fish positives if camera=video. Fish adapters must use physical(video,fish) identities and the established past-only unknown-preserving retrieval masks. Original JPM four branches are hardcoded, and CLIP prompt labels are training-class anchors rather than necessary onlineGT. CLIP stage1 actual contrastive temperature is1.0, not assumed.07; the final cache minibatch requires an empty-batch guard when exactly divisible.

The complete task book explicitly requires first completing environment/all30 source audits before further method development. Earlier executed visual/joint controls remain evidence, with their negative/confounded results preserved. All30available-source audits are now complete with missing-module/license/weight exclusions; this prerequisite gate is satisfied. The next work is controlled strong/local visual training, followed by genuine localJEPA and same-input C/D. Full goal remains IN_PROGRESS; actual localDenseJEPA training, stronger lawful pretrained baselines, repairedsetcompetition, independentmotion and all-fold/three-seed/fullrawonline remain required. See PHASE_VII_SOURCE_AUDIT_MEASURED.json for exact counts and artifact hashes.

## Additional eight source audit constraints

DINOv3 uses stopped EMA DINO/iBOT teachers and optional patch-relation Gram anchoring; default source config disables Gram and is not the full published7B recipe. DINO-Tracker full-test-video optimization has no causal JEPA teacher path. These source findings strengthen the requirement for matched local-token temporal prediction controls and forbid presenting pretrained patch distillation or offline video fitting as new online JEPA evidence.

## Additional three source audit constraints

MeMOTR feature memory EMA0.01 with detached reads and confidence gating is not an EMA teacher network or local temporal prediction loss. TCEI calibration is no_grad/cache based; DiffMOT supervision predicts geometry via diffusion rather than local visual latent evidence. These strengthen required same-input temporal/memory/prediction controls and prevent attributing ordinary memory/gating or nonlinear box prediction to new DenseJEPA novelty.

## Final three source audit constraints

All30source audits are complete. COVTrack already combines current/past roundtrip consistency confidence with learned residual appearance/position/semantic fusion; its CycleLoss helper is used even when cycle training loss is disabled. This is not a stopped EMAteacher/local prediction mechanism. DINOv2 genuine spatial162tokens now extracted from official pretrained source; B can use these as frozen observations, but must train a local context/temporal predictor and stopped EMA target under strict outer/inner exclusions, with matched noEMA/noPrediction/noID/contrastive temporal controls.

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

The genuine localJEPA fullseed1 real epoch2exit/fresh-process resume now exactly matches continuous8epochCPUtraining:250tensor/1array/1626scalar leaves acrossstudent, independentEMAteacher+1200updatecounter, AdamW, scheduler, all RNG, per-epoch targetmask/inputdigest/losses and allGT/verified-detector scene metrics/supports. Teachergradientsremainabsent; noCUDAallocation. Walltime/filepaths/containerbytes excluded; no crosshardware guarantee. Exact proof is PHASE_VII_LOCAL_DENSE_RESTART_MEASURED.json.

## Completed first-fold three-seed confirmation

All27 original V1 fits are COMPLETE at the fixed final epoch8, 1200 successful optimizer updates each. Same sampled scene/mask digests and frozen exact query supports are verified; no outer checkpoint or threshold selection. These are offline clean physical groups, not native track histories. SN-015 GT-history/current-detector gap25 supported564 of793 queries, target-present576, verified-positive-absent217 and history-absent17. Mean ± sampleSD over seeds0/1/2:

| Model | Rank1 (%) |
|---|---:|
| global | 72.813 ±0.369 |
| global_gru | 51.537 ±0.369 |
| local_attention | 65.307 ±1.589 |
| local_temporal | 63.948 ±0.800 |
| local_contrastive | 49.527 ±2.888 |
| no_ema | 44.740 ±1.245 |
| no_prediction | 43.203 ±1.647 |
| no_id | 25.591 ±1.609 |
| full | 45.095 ±1.181 |

Frozen CLS FIFO is64.894%. The fullJEPA45.095±1.181% result is below global72.813±0.369% and ordinary local-temporal63.948±0.800%. All12 history/current/gap protocols, R1/R5/mAP/AUC/EER and immutable source/checkpoint/support proofs are preserved in PHASE_VII_LOCAL_DENSE_THREE_SEED_MEASURED.json. NoID removes explicit ID losses but retains supervised GT temporal grouping; verified-positive absence does not certify physical absence or NULL.

## Registered context-identity V2 repair

The innerBT001-only frozen-model stage diagnosis motivated a separately owned V2 class/config/trainer: past local-temporal context supplies identity scores; regional masked prediction remains an auxiliary training objective. Deployment uses `.online()` and never executes the prediction decoder or teacher. Decoder perturbation cannot change identity scores; local/temporal/predictor gradient paths and stoppedteacher behavior are tested. V1 files and negative evidence remain unchanged.

V2fullseed0 has1200 successful optimizer/EMA updates. Offline supported SN015 detector R1 is63.652%, innerBT00136.242%, while the matched V2no_prediction seed0 is64.716%/36.913%. It restores the decoder bottleneck without establishing an extra identity benefit. The V2 no_prediction and local_temporal are exact control aliases; V2no_predictionseed0 student tensors equal the completed original local_temporalseed0, verified outsideGit in local_dense_context_no_prediction_alias_v2.json. Do not count these labels as independent evidence. V2three-seed completion and allfold/K8/native positive effects remain pending; the actual V2restart proof below is now verified.

## Actual native history integration

New features are encoded from each current RGB frame and original causal V3 proposals, including empty states. Native scoring uses only each track’s accepted proposal references with exact strictly-past timestamps, last4 slots/max age1500, plus current candidate observations. No physicalID, GT history, future frame or teacher enters replay. Frozen384DINO B coordinates are checkpoint-guarded; supervised384A features are a different producer and cannot silently feed these B models. The common learned logit scale is explicitly inverted into bounded cosine-cost units, preserving the small relation residual. Geometry/lifecycle/low/unconfirmed/writes remain the registered ordinary tracker.

The actual first512SN015frames parity audit replaces the learned cue with exact FIFO and reproduces accepted boxes/localIDs/lifecycle/history exactly, maximum association-cost difference3.6322e-8 across1024calls. It certifies this native prefix, not full-video learned parity. The first V2full native6570frame replay is complete; official evaluation and matched three-seed V1global/GRU/localAttention/localTemporal/full/ablations plus V2full are running. Offline crop R1 is not reported as nativeMOT improvement.

The first full6570SN015frame contextV2seed0 official result is HOTA30.694/AssA13.405/IDF125.298/MOTA91.032/IDSW146. Same-current frozenFIFO is32.225/14.786/30.445/91.019/152. Original Bglobal model gives31.017/13.664/26.362/91.003/145. Fewer switches accompanies weaker identity continuity; clean physical-history retrieval is not endogenous native performance. No recipe is selected from outerSN015; matched native allseed controls continue.

The separate V2contextfullseed1 real epoch2exit/fresh-process resume exactly matches its completed uninterrupted8epoch execution:250tensor/1array/1626scalar leaves across student, independent stoppedEMAteacher and1200updatecount, optimizer/scheduler, all RNG, observed-input/mask digests, losses and all heldout protocol identity/discrimination/support metrics. Both use the same guarded V2class/config/trainer bytes. Walltime, containerbytes and per-run outputpaths are excluded; no crosshardware guarantee. PHASE_VII_LOCAL_DENSE_CONTEXT_RESTART_MEASURED.json preserves the actual proof. This is execution correctness, not identity or tracking superiority.

## Completed V2 context-identity three-seed confirmation

All27 V2 firstfold fits are COMPLETE. Same observed scene/mask digests, frozen exact query support and1200successful optimizer updates each are verified. Identity reads pastcontext and never executes decoder/teacher online. SN015cleanGT-history/currentdetector gap25R1, mean±sampleSD:

| Mode | Rank1 (%) |
|---|---:|
| global | 72.813 ±0.369 |
| global_gru | 51.537 ±0.369 |
| local_attention | 65.307 ±1.589 |
| local_temporal | 63.948 ±0.800 |
| local_contrastive | 65.366 ±1.599 |
| no_ema | 63.889 ±0.892 |
| no_prediction | 63.948 ±0.800 |
| no_id | 64.657 ±1.083 |
| full | 63.652 ±1.064 |

Fullcontext63.652±1.064 does not exceed ordinarylocalTemporal63.948±0.800 or global72.813±0.369; localcontrastive65.366±1.599 is an established same-input control, notJEPA novelty. NoID64.657±1.083 removes explicitIDloss but retains annotated temporal grouping. NoPrediction and localTemporal are exactaliases, not two independent confirmations. Separate originalnegativeV1, exact V2restart and all12 protocols/R1R5mAP/AUCEER/supports remain preserved. The scientific PNG/SVG/PDF context_identity_three_seed_controls_v2 is generated from the exact reports and visually inspected.

Allsix K4 outer/inner-fold confirmation is now running on four CPUworkers: eight distinct modes ×sixfolds×three seeds=144unique fits.24 unique firstfold fits reused,120new fits planned; three firstfold noPrediction duplicates retained but never counted as independent. Same source/config, fixedepoch8, no outerselected recipe. Explicit outsideGit storage plan budgets all remaining checkpoints and12GiBnativefeature margin, estimated minimumfree45.12GiB; runtime stops before40GiBreserve. K8, truepixelmask/candidate-target identity diagnosis, strongerbaselines, allnative and finalnovelty remain pending.

## Inner-only teacher identity and target-difficulty diagnosis

SamefixedcontextV2fullseed0, BT001inner only; outerrequest actually rejects before weights/banks/outputcreation. StoppedEMAteacher flexiblelocalFIFO R139.150% onexact447supported currentdetector queries vsfullcontext36.242%; fit-onlyconstanttargetmean8.949%. Targetidentitysignal isretained, and a trivialmean isnotcompetitive. CurrentGTtarget tokencosine loss atpastgap1/25/100: regionaldecoder0.522/0.627/0.694, teacherpastFIFO0.379/0.591/0.702, fitconstantmean0.991/0.991/0.992. A smalllonggap reconstruction difference is not identitybenefit; simplerhistory already beatsdecoder atgap1/25. Currenttarget is whollyexcludedfrompredictioninputs; masking merely selects lossqueries and doesnot create image-level visiblecontext masking in this forecastversion. No teacher forward isusedonline. Fullsame-support/absence/unknown/target-count proofs are in PHASE_VII_LOCAL_TARGET_IDENTITY_DIAGNOSIS_MEASURED.json.

A separate ownedDINO/I-JEPA-style primitive now masks patch embeddings before ANYbackbone attention, withindependentstoppedEMAbackbone andteacher/decoder-free `.online()` encoder. ActualApacheDINO22M/2GTfitcrops fromBT003 passeshiddenprepared-pixel perturbation invariance whileunmaskedteacher changes. OneactualCPUoptimizer/EMAupdate hasfinitegradient paths into the final2visualblocks/maskpredictor, frozenprefix remains identical, teachergradients absent. This is144observedmaskedpatches andoneupdate, not a completed8epochmaskedfishfit ornewidentity/native/noveltyclaim. Genuine controlled training remainsrequired; this closes the cached-global-token masking leak hazard. See PHASE_VII_PIXEL_MASKED_IDENTITY_SMOKE_MEASURED.json.

## Completed first-fold B native three-seed controls

All 30 native SN-015 runs (ten configurations × three seeds × 6,570 frames) and pinned official evaluations are complete. The collector verifies fixed excluded outer/inner folds, exact training/checkpoint/current-feature/source hashes, all frame counts, and no GT/teacher/future online input. Results are mean ± sample standard deviation; all per-seed and official metrics remain in [PHASE_VII_LOCAL_NATIVE_THREE_SEED_MEASURED.json](PHASE_VII_LOCAL_NATIVE_THREE_SEED_MEASURED.json).

| Mode | HOTA | AssA | IDF1 | MOTA | IDSW |
|---|---:|---:|---:|---:|---:|
| Frozen CLS FIFO | 32.225 | 14.786 | 30.445 | 91.019 | 152.000 |
| global | 31.031 ± 0.023 | 13.675 ± 0.020 | 26.364 ± 0.002 | 91.006 ± 0.005 | 145.000 ± 0.000 |
| global_gru | 33.008 ± 1.528 | 15.490 ± 1.413 | 29.753 ± 2.118 | 91.011 ± 0.003 | 141.667 ± 1.155 |
| local_attention | 30.964 ± 0.066 | 13.619 ± 0.061 | 26.954 ± 0.621 | 91.016 ± 0.010 | 144.333 ± 1.528 |
| local_temporal | 30.858 ± 0.148 | 13.533 ± 0.116 | 25.679 ± 0.692 | 91.025 ± 0.008 | 145.333 ± 1.155 |
| local_contrastive | 31.701 ± 0.999 | 14.295 ± 0.944 | 28.303 ± 1.820 | 91.015 ± 0.016 | 145.000 ± 1.732 |
| no_ema | 31.370 ± 0.346 | 13.975 ± 0.319 | 28.094 ± 1.817 | 90.998 ± 0.015 | 149.667 ± 1.528 |
| no_prediction | 30.973 ± 0.337 | 13.626 ± 0.285 | 27.355 ± 0.778 | 91.015 ± 0.013 | 144.333 ± 1.528 |
| no_id | 32.710 ± 0.116 | 15.238 ± 0.121 | 30.732 ± 0.124 | 90.998 ± 0.003 | 159.000 ± 0.000 |
| full | 31.201 ± 0.474 | 13.829 ± 0.430 | 27.727 ± 1.199 | 90.998 ± 0.011 | 149.333 ± 0.577 |
| context_full | 30.630 ± 0.105 | 13.338 ± 0.099 | 25.679 ± 0.692 | 91.020 ± 0.012 | 146.667 ± 0.577 |

Original full and context full both reduce HOTA and IDF1 against frozen FIFO. Ordinary global GRU has higher mean HOTA but lower mean IDF1 and substantial seed variation. No-ID has a small HOTA/IDF1 advantage with more switches; it still uses annotated temporal grouping during training and is not a JEPA identity-evidence gain. These single outer-video results do not establish all-six native generalization or independent novelty. All-six context training, ordinary native matrix, on-policy mixed-history diagnosis and controlled true pixel-mask training remain required.

## Registered actual pixel-mask training control

The new owned trainer uses the pinned Apache DINOv2-S backbone, four fitting videos and the original same-frame/temporal sampler, 8 × 150 attempted updates, identical augmentation, last two blocks plus final norm and global384 identity head. Outer SN-015 and inner BT-001 never update fitting or EMA state. Three modes and three seeds are registered in `phase_vii_pixel_masked_identity_v1.json`: supervised-only, visible-target distillation, and true pre-attention pixel masking. All modes run three forward branches. Visible-target and masked modes have the same active predictor and independently stopped EMA target; only masked input differs. The supervised-only zero auxiliary coefficient is an explicit lower active-loss-capacity reference. No anatomy or true occlusion labels are claimed from content fractions.

Half of observable patches are queried using a separate captured RNG, retaining visible content where possible. Official frozen mask embeddings replace selected image embeddings before any backbone attention. Current unmasked teacher targets are training-only; online evaluation calls the student alone. EMA advances only after successful AMP optimizer updates, and the update count must match successful training steps. These are established masked distillation mechanisms, not novelty or official FishMAE reproduction.

Lossless checkpoints omit only bitexact immutable pretrained prefix tensors. The actual licensed model CPU smoke restores all377model keys exactly from87delta keys plus290base keys after a real optimizer/EMA update, with36finite gradient fields and144masked observed patches; the model delta is37,076,321bytes. Student and teacher tails, classifier, predictor, all buffers and optimizer/scheduler/AMP states are saved; all Python/NumPy/CPU/CUDA/sampler/augmentation/mask RNG are captured. The one-update smoke proves serialization/masking/gradient correctness, not an8epoch fit or nativeMOT gain. See PHASE_VII_PIXEL_MASKED_CHECKPOINT_MEASURED.json.

A durable first-fold queue is waiting for the assigned owned GPU1 native producer to fully complete and its process to exit, followed by an actual <=64MiB/zero-utilization check. It registers nine mode/seed fits plus a distinct seed0masked epoch2exit/fresh-process resume versus continuous8epoch fit. Exact model/optimizer/EMA/RNG/loss/retrieval comparison runs before subsequent controls. The initial storage plan reserves42GiB plus5GiB for other jobs and250MiB per fit. Sources/config are hash guarded; no active native/raw source helper was modified. All-six masked folds, genuine MAE pixel-reconstruction/strong source baselines, native pipelines and scientific benefit still require actual experiments.

## Actual RGB reconstruction control, separate from EMA latent targets

An owned masked pixel autoencoding control now predicts588values (14×14×3RGB) per patch directly, with weighted normalized current-pixel targets excluding padding/augmentation-invalid pixels. The official pinned DINO encoder replaces selected patch embeddings before allattention. Two registered modes share the same384->768->588predictor: truepixelmask and visible-pixel-target control. Both use the same four-fit-video sampler/labels/augmentation/8×150attemptedupdates/lasttwoblocks/global384head. There is no EMA teacher; current supervisedidentity and maskedRGBtarget branches are separate. Relative to latent distillation the targetspace, predictorcapacity and forwardcompute differ explicitly. This is an established full-token masked reconstruction control, not official MAE visible-token-only encoder/decoder, FishMAE reproduction, novelmechanism or validatedidentitygain.

Actual licensed DINO/current BT003fit-crop CPU evidence verifies144masked observedpatches,36finitegradient fields, oneoptimizer update, frozenprefix equality, targetpixel perturbation cannot affect maskedpredictions, and all202model-state keys restore exactly from57delta keys plus145immutableofficialbase keys. Modelcheckpoint23,488,085bytes; source/input/checkpoint/gradient/protocol proof in PHASE_VII_PIXEL_RECONSTRUCTION_SMOKE_MEASURED.json. The old first smoke's accidental EMA wording is corrected in the separately re-executed v2 proof; neither smoke establishes8epoch/MOT benefit.

A durable GPU3queue registers sixmode/seed fits plus an actualseed0masked epoch2exit/freshprocessresume duplicate. It waits for the owned nativeproducer to complete, process teardown and actual<=64MiB/zeroGPUutilization, with42GiBreserve/5GiBother-job margin and250MiBperfit. The GPU1EMA-latent-mask queue remains separate and guarded. New fits are pending; all-six/newproducer/native/strongsource comparisons and reliablecontribution remain required.

## Complete all-six K4 context-identity confirmation

All144unique fits are complete: six outer/next-inner/fourfit-video folds ×eight distinct controls ×three seeds,172,800successful optimizer updates. Exact checkpoint/source hashes, heldout classifier exclusion, eight epoch sampling digests across every matched mode, no teacher gradient/decoder online/CUDA allocation, and all12 known/unknown/target/candidate/query support protocols are verified. Four frozen controls are now measured on allsix videos; firsttwo identical helper/data measurements reused with provenance, remainingfour executed. The first allsix collector preparation failure is retained separately, with no successful measurements attributed to it. See PHASE_VII_CONTEXT_IDENTITY_ALLFOLDS_MEASURED.json.

Macrovideo GT-history/currentdetector gap25R1, eachseed averaged acrosssix outer videos then mean ± sampleSD overthree seeds:

| Mode | R1 (%) |
|---|---:|
| frozen_cls_last | 42.326 |
| frozen_cls_fifo | 40.111 |
| frozen_local_corresponding | 42.193 |
| frozen_local_flexible | 41.754 |
| global | 43.618 ± 0.258 |
| global_gru | 32.127 ± 0.511 |
| local_attention | 45.281 ± 0.366 |
| local_temporal | 45.322 ± 0.433 |
| local_contrastive | 45.774 ± 0.591 |
| no_ema | 45.446 ± 0.432 |
| no_id | 40.924 ± 0.415 |
| full | 45.288 ± 0.427 |

Full minus ordinarylocalTemporal paired-seed macroR1 difference is -0.0338 ± 0.0221percentage points. The regionalprediction auxiliary does not establish additionalidentity information; localAttention/contrastive improvements overfrozen are established mechanisms, notJEPA novelty. NoPrediction is an exactlocalTemporal alias and contributes zero additionalindependent fits. All inner/outer protocols/R1/R5/mAP/AUC/EER/supports are retained. These are cleanphysical histories; firstfoldnativeBfull remains worse thanfrozenFIFO. K8/context, onpolicyactualhistories, truepixelmask/RGBreconstruction/strongsource, nativeall-six and finalvalidatednovelty remain required.

## First full pixel-masked fit and actual CUDA restart

The genuine pre-attention pixel-masked seed0firstfold now completes8epochs/1200attempts,1197successful optimizer/EMAupdates (three AMPoverflow skips),153.507training seconds, unchanged immutable prefix and student-only heldout retrieval. Its separately executed epoch2exit/fresh-process resume reaches the identical final result:197tensor/1array/1269scalar leaves across losslessmodeldelta/teacher tail/optimizer/scheduler/scaler/allRNG, everyepoch loss/update/coverage and heldoutretrieval/feature hashes match exactly. The two runs share strict deterministic algorithms and declared cuBLAS workspace on physicalGPU1. PHASE_VII_PIXEL_MASKED_RESTART_MEASURED.json preserves the exact proof and three real process-launch/log records.

This verifies actual fulltraining/restart correctness, not identity or nativeMOT improvement. Other mode/seed controls are running; nine unique supervised/visible-target/truepixel-mask fits must be compared on identical source/pools/support before attributing any gain to masking. Do not compare this whole-video cropgallery evaluation with B's single-current-scene candidate support or claim a luckyseed0 improvement. The separate teacher-free RGBreconstruction queue waits on GPU3. All-six masked/reconstruction folds and newproducer/endogenous/native/fullraw/strongsource/novelty work remain incomplete.

## Completed genuine pixel-target three-seed controls

All15unique firstfoldfits (nine supervised/visiblelatent/pixelmaskedlatent andsixvisibleRGB/maskedRGB), plus two actualrestart duplicates, are complete. Both restart proofs are exact. Separate sampler/augmentation/mask final RNG states, immutable inputs/record/query/gallery support, fixed8epochs/1200attempts, pretrained/source/checkpoint/prefix guards andbackbone changes verified. AMPsuccessful update counts are reported explicitly, not assumed equal. All GT/detector views/gaps1/25/100/R1/R5/mAP/unknown selections retained in PHASE_VII_PIXEL_IDENTITY_THREE_SEED_MEASURED.json.

OuterSN015 currentdetector wholevideo past-only gallery gap25 (630supported/644queries), mean ±sampleSD:

| Mode | R1 (%) | mAP (%) |
|---|---:|---:|
| frozenCLS | 57.937 | 23.857 |
| supervised_only | 63.016 ± 1.514 | 29.775 ± 0.416 |
| visible_target | 63.386 ± 1.283 | 29.780 ± 0.428 |
| pixel_masked | 63.386 ± 1.191 | 29.803 ± 0.401 |
| pixel_masked_reconstruction | 63.810 ± 1.041 | 29.785 ± 0.367 |
| pixel_visible_target | 63.016 ± 0.884 | 29.897 ± 0.408 |

Masked andvisiblelatent target modes haveidenticalmeanR1; pixelRGBreconstruction raisesR1slightly versusvisibleRGB butmAPis lower. No robust masking-specific identitybenefit, nativeMOT gain orJEPA novelty is established. Wholepast-gallery results cannot be compared directly with Bsingle-current-scene candidate retrieval. Source-matched strongerlocal identity heads, actualnative adaptedcoordinates/allfold/newmethod remain required.

## Authoritative outer-fold native context-V2 completion (2026-10-14)

The registered context-V2 models now have a complete same-input native replay for all six videos, three seeds, and three modes: `global` capacity control, ordinary `local_temporal`, and `full` with the masked-prediction branch kept training-only. Each model excludes its evaluated outer video and its declared inner video from fitting; replay uses the frozen DINO current proposal bank, accepted strictly-past native histories, the unchanged lifecycle and the official evaluator. There are 54 complete rows with prediction, replay-manifest, checkpoint and official-metric hashes in `PHASE_VII_CONTEXT_IDENTITY_NATIVE_ALLFOLDS_MEASURED.json`; the raw driver state and full metric vectors remain outside Git.

| Mode | HOTA mean | AssA mean | IDF1 mean | MOTA mean | IDSW mean |
|---|---:|---:|---:|---:|---:|
| Same-input frozen FIFO | 38.784 | 26.686 | 47.426 | 73.459 | 114.5 |
| Context-V2 full | 38.495 | 26.341 | 46.138 | 73.459 | 113.7 |
| Context-V2 local temporal | 38.557 | 26.402 | 46.183 | 73.459 | 113.6 |
| Context-V2 global capacity | 38.374 | 26.195 | 46.226 | 73.449 | 115.3 |

The outer-fold native aggregate loses HOTA, AssA and IDF1 for all three learned modes relative to the same-input frozen FIFO reference; MOTA is effectively unchanged and IDSW does not improve consistently. Video effects are heterogeneous: BT-003 and MSK-002 move slightly upward, while BT-001 and SN-015 decline and PF-001 is byte-equivalent. This is complete negative causal evidence for the registered context-V2 route, not a Dense-JEPA contribution or a whole-raw FPS claim. It closes the all-fold native validation for these controls while leaving pixel-masked final combinations, raw full-pipeline profiling, external generalization and final novelty review open.


### Visible-token reconstruction comparison


The previous pixel reconstruction controls trained the last two encoder blocks. The new owned MAE control removes queried patch embeddings before attention and trains all 12 blocks, the patch embedding, CLS token, and final norm. Pretrained positions and the unused original mask token remain fixed. A two-layer, 192-dimensional decoder reconstructs normalized current RGB patches; deployment uses the encoder alone. The visible-target control retains query pixels and uses the same learned query marker additively, so active parameter counts match. Encoder token counts and compute differ and must be reported separately.

The unlabeled pool contains 15,217 valid detector crops from four fitting videos: BT-003 324, MSK-002 11,002, PF-001 3,113, and SN-001 778. Sampling never uses identity values or GT crop rows. Current RGB image headers and detector boxes supply letterbox geometry. BT-001 and SN-015 are excluded. These are sparse pilot crops, and inherited detector and foundation exposure remains explicit.

On two real detector crops, the CPU check verifies gradients in every encoder block, patch embedding, and CLS token; deleting 129 queried patches leaves 65 and 130 encoder patches. Perturbing hidden pixels after the registered resize leaves predictions unchanged. All 208 model state entries restore exactly, and decoder calls are excluded from inference. Mixed precision, variable token counts, and active query-marker tests pass.

Six fixed eight-epoch pretrains (150 steps per epoch, two modes, three seeds) and a distinct epoch-two exit/fresh-process resume run are registered. The queue waits for the owned GPU 1 native job to finish and exit, then checks actual idleness. Its storage forecast includes remaining native banks, raw traces, restart states, atomic checkpoint copies, and the 40 GiB reserve. Only explicitly temporary optimizer files created by this new queue can be retired after final encoders are retained; historical assets remain protected.

This is an independently implemented DINO initialization for visible-token RGB reconstruction. Identity CE/Triplet supervision, the ordinary supervised encoder control, all folds, native MOT evaluation, and novelty verification remain required. It establishes neither completed pretraining nor identity gains, and does not reproduce the complete FishMAE method. Evidence: PHASE_VII_VISIBLE_TOKEN_FISH_MAE_CPU_MEASURED.json.

RGB reconstruction is a control for the latent prediction experiments. Object-centric JEPA still needs its own identity evidence, matched temporal controls, and native tracking validation.

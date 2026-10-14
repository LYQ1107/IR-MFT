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

## Centered NULL repair and independent linear filtering

All27contextV2 and9centeredCthree-seed controlled fits are complete; no extraJEPAidentitybenefit established. Centeredconstantcost control restores exactbaselinebytes, so no threshold-only residualgain maycount as learnedidentity. Parameter-matched crossTransformer usesgradient-active per-tokenFFNs instead ofset-self-attention; pair remains loweractivecapacity. C privateNULLbalanced objective and knownexisting-only ranking are separate; actual197tensor/1array/931scalar restart verified. Allsix27251nativeframes original8DlinearUKF/KF outputs are exactbyteequal withunchangedMOTmetrics, isolatingfilter implementation fromnewstate/noise/nonlinearity. FishIoU/AGCIoU/occlusion/rawprofiling and finalallmodulecombinednecessity remain pending.

## Complete geometry, target and raw-pipeline controls

All16geometrycontrols×6videos andall18centeredC model/routes×2videos complete. SourceFishcentralbug/repair/continuous-vs-inclusiveIoU/DIoU/finiteAGCv1/affineoffset controls isolateeffects; no stableFishIoU/jointbenefit. Innerteacher39.150R1 versusfitmean8.949 rejects simplecollapse explanation, andpastFIFOtargetloss beatsregionaldecoder atgap1/25. Verifiedrealpixelmask primitive isoneactualCPUupdate, fullcontrolledtrainingpending. Both684frame rawSN001pipelines reproducecachedpredictions exactly withallcurrentIR/proposals equal; algorithmFPSlegacy1.2447/frozenDINO1.1083 andobservedvalidationwallFPSseparate. Fullsource/unknown/metric/runtimecoverage limits remain explicit; overallgoal active/unachieved.

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

## Complete inner endogenous identity diagnosis

Nine actual SN-001 full-route observations (pair/cross Transformer/joint ×three seeds ×684frames) reproduce every original native prediction byte. GT/physical labels are loaded only after all nine outputs and parity checks are fixed. The observer records actual high-stage candidate sets, accepted strictly-past proposal histories and model logits. An actual outerSN015 request is rejected before weights or banks are loaded. Offline independent proposal labels, >=3past-observation frozen EarlyReference and unknown/nonunique-preserving supervision then diagnose the recorded decisions. All scalar counts, source/model/input/output hashes, fixed sparse inputs and trace artifacts are preserved in PHASE_VII_JOINT_ENDOGENOUS_INNER_MEASURED.json and its outsideGit linked sources.

| Architecture | Same-set model existing R1 | Ordinary FIFO R1 | Geometry R1 | Alternative dictionary assignment changes |
|---|---:|---:|---:|---:|
| Pair |93.657 ±1.016|92.633 ±0.822|93.774 ±1.078|3.789 ±0.484%|
| Cross Transformer |82.297 ±4.647|83.202 ±5.345|83.809 ±5.438|15.853 ±5.661%|
| Joint |88.070 ±3.127|87.793 ±2.584|88.701 ±2.632|17.209 ±4.253%|

Mean ± sampleSD overthree seeds. Within each seed, cues use the same actual track/candidate sets and known-existing target support. Each architecture produces different endogenous sets, so cross-row R1 comparisons are descriptive, not a matched independent accuracy experiment. Dictionary interventions keep all images/history/geometry/confidence/track states fixed on136stride5frames/1,168detections per seed; four predeclared alternative distinct-code draws alone change joint assignments17.209±4.253%. This is nuisance sensitivity, not a fulltracker performance gain. Zeroing the dictionary changesNULL tradeoffs as well; it is not promoted to a chosen online recipe.

Joint full averages252.667±66.010 falseNULL decisions on knownexisting targets,166.333±10.066 wrongexisting decisions, and217.000±156.464 falseexisting decisions on knownno-old-match targets. Its track/detection set has2,467.333±350.009 nonuniquephysical assignments ignored by the original categorical loss. The local model hardly exceeds FIFO and is below geometry; NULL, set coverage and nuisance embedding effects remain separate from raw visual identity evidence.

The same offline labeling on the preserved baseline trace reveals an important reference confound: baseline frozen-EarlyReference history mismatch is23,925/42,357=56.484% of repeated known history observations, versus lower mismatch in fragmented learned trajectories, despite the baseline's stronger official IDF1/HOTA. Warm inner data has876/1,175supervisedknownexisting-or-no-old-match queries, while joint endogenous support is substantially smaller and more nonunique. EarlyReference is a noisy initial-history convention; short fragment references can look cleaner without maintaining physical identity continuity. These counts cannot certify better tracking or isolate pollution as a cause. Fit-video warm data contains substantial mismatch too (all four fit videos audited without outer data).

The evidence motivates independently supervised, source-matched current/raw/local cues, unknown-preserving supervision of actual difficult sets, and separation of arbitrary ID binding from visual association. A blind hard geometry gate, zero-dictionary inference patch, stricter memory write rule or improved latentloss is not yet supported as a fullmethod fix. Genuine newproducer/on-policy C training and matched MOTIP/Hungarian controls remain required. The scientific PNG/SVG/PDF plot is exported outsideGit from allnine records, with source/value/hash provenance.

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

## Completed all126ordinary native controls

Allsix videos, frozenlast/FIFO/EMA andtwoencoder modes ×three seeds ×three memories complete. Actualencoder-feature-source/checkpoint/outer-inner exclusions andofficialprediction/GT hashes verified. All pervideo/allofficialmetrics/seeds remain in PHASE_VII_NATIVE_IDENTITY_ALL126_MEASURED.json. Macrovideo mean ±seed sampleSD; no memoryrule orseed selected fromouter values:

| Memory / encoder | HOTA | IDF1 | IDSW |
|---|---:|---:|---:|
| last / frozen | 37.940 | 45.459 | 113.667 |
| last / global | 38.115 ± 0.100 | 45.632 ± 0.196 | 116.889 ± 0.255 |
| last / patch_attention | 38.088 ± 0.101 | 45.605 ± 0.196 | 117.444 ± 0.096 |
| fifo / frozen | 38.784 | 47.426 | 114.500 |
| fifo / global | 38.362 ± 0.004 | 46.343 ± 0.015 | 116.778 ± 0.192 |
| fifo / patch_attention | 38.515 ± 0.001 | 46.661 ± 0.000 | 117.278 ± 0.096 |
| ema / frozen | 38.395 | 46.372 | 115.333 |
| ema / global | 38.220 ± 0.044 | 45.888 ± 0.115 | 117.444 ± 0.509 |
| ema / patch_attention | 38.222 ± 0.019 | 45.853 ± 0.177 | 117.778 ± 0.096 |

FIFO adaptedencoders do notsurpassfrozenFIFO: HOTA38.362global/38.515patch versus38.784frozen, IDF146.343/46.661 versus47.426. Smalllast-memory gains coexist with more switches; no stableencoder/local contributionclaim. These are CacheProtocolA results: originalglobal detectorcache hasunpadfinal3frames. Currentrawalways-pad8 protocol is separatelymeasured andmustnotforceoldcachetail parity.

## Same-coordinate recent-history and dictionary-separated full-set controls

A new owned Ccontrol uses the exactsame ApacheDINO384frozen coordinates onfourfitvideos, innerBT001 andouterSN015, withoutold512GRU ormixingfittedencoders. Fourfitteacher videos (17,681frames) reproduce existingfrozenFIFO predictionbytes before ANY physicalannotation labels are loaded. All sampled actualhigh-stage sets/unknowns are retained; traininglabels require>=3whollyknown same-ID recent observations withinlast4/maxage1500, notfrozenEarlyReference. These verified observationlabels are notunconditional whole-track/biologicalidentity truth. Multipleoldslots withsamephysicalID formpositive sets; duplicatecurrent IDs ignored; unknowns neverhardnegatives orfabricatedNULL.

Natural verifiedno-supplied-match coverage is onlyone query. A separatelyregistered25%requested trainingaugmentation retains a closedknown oldsubset and withholdsoneknowncurrentfish's matchingoldgroup plusallunverified oldslots, keeping>=oneverifieddifferentoldidentity. NULL is logicallynomatchtosuppliedsubset, notbirth/naturalocclusion. Otherdraws retainactualfull unknowninclusive sets. GT-derivedsubset selection istraining-only, neverinference; itsclosed-set distributionlimit isexplicit.

Fourmodes pair/crossTransformer/jointcontent/jointcodes ×threeseeds complete12fixed8epoch/1200step fits (14,400updates) and24wholeexcludednative videos. Everyepochactual model-input/sampling/subset/codes digestmatches acrossmodes perseed. JointcodeEmbedding512×64 andcontentFFN64->256->64 bothhave32,768conditioning parameters andexactequalactivecapacity/standardizedconditioning scale. Maincontent association logits arebitexact invariant toarbitrarycode changes; codes mapcolumns only. CrossTransformer selfattention isreplaced byexactparameter/gradientactiveTokenFFNs. Pairactivecapacity isreported separately, notdeclared equaltofullsets. No officialMOTIP reproduction ornovelty claim.

| Video / model | HOTA | IDF1 | IDSW |
|---|---:|---:|---:|
| BT-001 / frozenFIFO | 35.812 | 45.665 | 99.000 |
| BT-001 / pair | 32.308 ± 1.772 | 35.114 ± 2.272 | 102.333 ± 2.887 |
| BT-001 / cross_transformer | 24.509 ± 1.452 | 26.797 ± 1.818 | 207.000 ± 42.568 |
| BT-001 / joint | 26.859 ± 1.798 | 30.146 ± 1.979 | 169.667 ± 5.508 |
| BT-001 / coded_joint | 26.942 ± 0.858 | 34.570 ± 1.157 | 305.667 ± 27.025 |
| SN-015 / frozenFIFO | 32.225 | 30.445 | 152.000 |
| SN-015 / pair | 30.443 ± 1.282 | 26.435 ± 1.165 | 162.333 ± 11.590 |
| SN-015 / cross_transformer | 28.413 ± 1.463 | 27.471 ± 1.734 | 207.000 ± 52.115 |
| SN-015 / joint | 28.050 ± 0.715 | 25.983 ± 0.882 | 176.000 ± 4.359 |
| SN-015 / coded_joint | 30.850 ± 1.532 | 32.244 ± 3.375 | 294.667 ± 45.654 |

Contentinvariance andcorrectset supervision do notimproveMOT: allnewC HOTAmeans arebelowordinaryFIFO onbothvideos. CodedjointSN015 higherIDF1 accompanies muchmore switches/lowerHOTA. Unknown-inclusive learned-policy histories, NULL/calibration/rawgallery preservation andstrongsource adaptations stillneed controlleddiagnosis; no zerocode/threshold/seed recipe isselected fromouter values. Fullsource/physical-target/support/capacity/officialmetric proof is PHASE_VII_DICTIONARY_JOINT_THREE_SEED_MEASURED.json. Actualjointseed1epoch2exit/freshresume matches continuous8epochs across189tensor/1array/815scalar leaves, alldigests/augmentation/losses/RNG/optimizer; PHASE_VII_DICTIONARY_JOINT_RESTART_MEASURED.json.


### Source-matched TransReID JPM adapter and actual first-fold campaign

Pinned TransReID `dec55046fcdfadee14e2c28e2df89305d8f7557a` JPM is now transplanted to the same licensed DINOv2-S/current crop banks. The original shift5/group2, global/copied shared-local final blocks, five BN necks/classifiers, half-global/half-local loss and after-neck `[global, local1/4, ..., local4/4]` descriptor are retained. DINO162patches yield four40token subsets, with two final shuffled tokens omitted by source floor division. BN statistics fit on BT003/MSK002/PF001/SN001 only; innerBT001/outerSN015 are eval-only. No SIE/camera input, external TransReID weights or official full-method claim. Same-video temporal/same-frame fish supervision replaces original camera-dependent sampling; exact adaptations and MIT notice are in third_party/transreid_jpm/PROVENANCE.md.

Real CPU/DINO/current-fitcrop gradient and delta-restore proof passes: 237 model keys, 76 delta keys, checkpoint21,380,019bytes for the two-identity smoke. This is a smoke vocabulary, not the larger fitting classifier. An actual first-fold JPM seed0 continuous8epoch run and distinct epoch2exit/new-process resume are bitexact: 246 tensor/1array/1286scalar leaves including all BN buffers, optimizer/scaler/scheduler/RNG, all losses and retrieval arrays. All nine distinct global/JPM/CLS-capacity fits (3seeds each) are running under fixed8epoch/1200attempt settings; no outer-based checkpoint or hyperparameter selection. JPM and repeated-CLS control have identical active parameters/branch lengths/supervision. All patch subsets and prefixCLS already globally mix the crop, so JPM does not establish purely local/anatomical evidence. Three-seed gains/native tracking/novelty are unproven until measured.

Current-only deployment with fixed pad128 is separately verified on four actual detector proposals plus one preserved invalid row from frame1 of each excluded video. Current crop pixels equal the original pilot bytes; maximum embedding component differences are1.442e-6/1.057e-6 and cosine differences5.96e-8 against GPU evaluation, numerical agreement rather than descriptor byte equality. Native queues wait for both owned fitting processes to finish and actual GPU1/3 teardown/idle; then produce18 fullvideo feature banks and score54 last/FIFO/EMA controls. Planned feature storage is5.063GiB with40GiBreserve plus1GiBraw margin. This remains first-fold evidence, not allsixouter folds or complete rawonline speed.


All nine unique first-fold source-JPM controls nowcomplete, plus separate exactrestart duplicate. Strictpast detector-gallery gap25 outerSN015 R1: globalBN60.688±1.126%, JPM61.905±.727%, samecapacityCLS61.587±.692%, frozenCLS57.937%; mAP30.755±.315/30.835±.599/30.478±.574 versusfrozen23.857. JPM-minus-capacity R1 perseed is+1/+3/+2 correctqueries outof630, a mean+.317pp, not a robust local novelty result. InnerBT001 R1 global20.566±1.540/JPM20.085±1.137/capacity20.994±.555 versusfrozen19.872, so the JPM contrast doesnotgeneralizeconsistently betweenexcludedvideos. Allseed/sample/support/BN/source/capacity/unknownselection evidence is in PHASE_VII_TRANSREID_JPM_THREE_SEED_MEASURED.json. Mean±sampleSD only; no statisticalsignificance, winningseed or historicalouter-based modelselection claim.

The authoritative training budget is9unique×1200=10,800attempts, plus1200duplicate attempts for the actualrestart=12,000executions; successfulAMPupdates are reportedperfit andmaydiffer. Native queues havepassed prerequisiteprocess/idlechecks andnowexecute all18bank/54memory controls. NativeMOT andallsixfold gains remainunproved. Publication-format PNG/SVG/PDF measuredfigures remain outsideGit at /data3/liuyeqiang/IR-MFT_assets/cfp/phase7/transreid_jpm_three_seed_figures_v1; dots show actualseeds, whiskers sampleSD, deterministicfrozen hasnoSD.

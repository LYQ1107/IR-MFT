# Phase VII failure analysis

Status: IN_PROGRESS. Negative findings register discriminating follow-ups; there is no fixed failure count.

## Authoritative current-state update (2026-10-14)

The earlier pending statements in this running register are historical snapshots. Since then, both registered six-video raw v3 controls (legacy GRU/FIFO and frozen DINO/FIFO) completed, the source-JPM matrix reached 54 full-video controls, and an independently implemented FDTA-style baseline completed 9 offline fits plus 18 causal native FIFO replays. Temporal/quality offline ranking gains do not transfer to online tracking: relative to frozen DINO/FIFO they lose HOTA and IDF1 and add ID switches on both held-out videos. This negative causal result narrows the remaining hypotheses and rules out selecting the adapter as a contribution; it does not terminate the broader goal. See the exact hash-linked reports listed in `PHASE_VII_STATE.json` current_authoritative_snapshot.

The next B-stage audit is now also terminal for its registered scope. All six outer-fold context-V2 models were replayed on their held-out video for seeds 0/1/2 under `full`, ordinary local-temporal and global capacity controls (54 complete rows). Relative to the same-input frozen FIFO, all three modes lose macro HOTA/AssA/IDF1; MOTA is unchanged within rounding and IDSW changes are heterogeneous. The degradation is largest on BT-001 and SN-015, while PF-001 remains byte-equivalent. This separates a causal native failure of the learned adapter from the earlier offline prediction-loss result; it does not justify a project-wide NO_GO or a raw-FPS claim. Exact scalar summaries and full metric-vector hashes are in `PHASE_VII_CONTEXT_IDENTITY_NATIVE_ALLFOLDS_MEASURED.json`.

The six-video actual-state rank decomposition reconstructs native GRU/geometry/confidence row costs within1.2e-7. Pure retrieval and actual association scores differ. On SN-015 native original row Rank@1 is62.2969%, near geometry-only62.3249%, despite pure GRU35.8450% versus FIFO62.3623%. This explains the fixed-row contrast, without proving unchanged Hungarian decisions or endogenous trajectories. SN-015 all-verified-correct histories still have FIFO99.5950% versus GRU52.6510% (2,716queries): contamination alone is insufficient. Appearance-region pool restriction raises GRU to84.5078% versus FIFO94.1110% (7,081different queries); denominator/pool changes are disclosed and are not a deployed improvement.

Original temporal-training producer diagnosis used the exact lifecycle-fixed query manifest referenced by the frozen checkpoint. The initial similarly named pre-fix directory failed the expected manifest hash guard; it was preserved and not accepted. The corrected query source has sha256352d290b0c4a2e228cce46322ff803aa44928c5978db539d5c744e714f27b84a.

On the original PF-001 pool, pure GRU ranks46.0896% on ordinary actual history versus96.8361% on AnchorRecent; clean FIFO93.2634% versus GRU60.3093%. On SN-015 the same comparisons are45.1092% versus28.3419%, and clean FIFO92.8724% versus GRU71.0101%. Fit/calibration exposure and original cutoff/pool differences preclude a causal producer claim. The ordinary GRU objective learned a scene/history-dependent correction, not uniformly better identity retrieval. Targeted next controls preserve raw/global/local identity evidence, test the same history/pool across representation changes, and evaluate joint competition independently of forecasting loss.

The first full-frame warm-source extraction reached a zero-old-track snapshot and the inherited GRU rejected empty packed tensors. The incomplete v1 output and error log are preserved. v2 explicitly uses a0×512 prototype when N=0, preserving that frame instead of discarding it. The new decoder independently tests N=0, M=0, all-masked edges and one-to-one NULL decisions.

Source acquisition TLS failures, bounded selected-source gaps, missing licensing and externally unverified pretrained access remain separate from algorithm failures. Every source registry entry stays semantically incomplete until its required fields are verified. FishMAE's sampler, positional embedding and training-entry defects are source evidence, not a trained baseline failure.

The first registered full-set joint head improves SN-015 no-old-match recall from0.7712% to30.1735% but reduces existing-ID accuracy from83.6619% to64.7523%; total accuracy52.7222% is below baseline54.8239%. The pair control has51.7272%total/74.2500%existing/9.5116%NULL. Bounded edits have no frozen-state accuracy benefit. This does not justify rejection of all joint association or credit the NULL gain as overall tracking improvement. Complete native trajectories have now run and fail as described in JOINT_ASSOCIATION_EXPERIMENTS; the common-cost confound is resolved by the uniform control. Further head optimization must use fitting/inner-calibration data; the exposed outer SN-015 result cannot select new weights/thresholds under a claimed untouched outer evaluation.

Next fitting/calibration diagnostics will measure residual positive/negative distributions, cost-threshold crossings, baseline margin changes, dictionary sensitivity and track-memory identity support. A bounded residual may be unable to express the large no-match decisions needed at native threshold0.97; increasing its amplitude based on an outer score would confound the registered route. Same-input local evidence, ordinary full-set/MOTIP-adapted controls, per-video crossfit and separate motion ablations remain required.

## Paired crop evidence and visual learning follow-up

On identical paired known physical keys/queries/galleries, MSK002 FishReID post-BN R1 falls49.0077→8.2006% from GT to detector crops at gap25. The direction also appears on BT001/BT003/SN001 and is smaller on PF001/SN015. This isolates crop view conditional on verified pair support, not a unique pixel-level cause. The illustrative first two covered pairs in fitting videos show low-resolution silhouettes/box shift; their geometry artifact is external and descriptive.

Four real frozen-ImageNet feature adapters and two real ImageNet-initialized visual-backbone fits finish8epochs. Fitting loss falls but outer detector R1 is43.1746/43.3333/45.0794/44.9206% for global/dualglobal/simple/learned adapters and42.6984/46.6667% for global/learned CNNs, below frozen ImageNet53.6508%. A local CNN advantage over the new global head is not a baseline win. Projection/LN compression, BN domain changes, sampling/budget and crop stability remain competing explanations. Next extract raw fine-tuned2048/pre256/normalized256 on fitting/inner inputs before changing the learning method; do not select from exposed SN015. Real JEPA/motion and improved C competition remain independent unfinished tasks.

No task-level NO_GO is issued. Transport failures remain access failures, source audit completion remains separate from actual pretrained reproduction, and pilot training/runtime does not close full-fold/seed/online validation.

## Fitting/inner feature-stage diagnosis

Frozen stage protocol excludes outerSN015. Existing fixed-last seed0 checkpoints are read without fitting. All fivefit/inner videos use identical per-view query/gallery masks; orthogonal2048→256random projection is label-free, fixedseed73. Detector Rank@1 at gap25:

| Video/role | Frozen raw2048 | Frozen orthogonal256 | Global CNN raw2048 | Global preLN256 | Global LN256 | Local CNN raw2048 | Local preLN256 | Local LN256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BT-003/fit | 17.514 | 14.124 | 63.277 | 76.271 | 76.271 | 41.243 | 61.582 | 61.582 |
| MSK-002/fit | 4.398 | 4.200 | 5.522 | 3.935 | 4.001 | 4.663 | 4.101 | 4.067 |
| PF-001/fit | 70.635 | 79.365 | 92.063 | 92.857 | 92.857 | 93.651 | 90.476 | 90.476 |
| SN-001/fit | 32.308 | 31.346 | 69.231 | 80.000 | 80.000 | 60.769 | 76.346 | 76.346 |
| BT-001/inner | 14.744 | 15.705 | 14.904 | 12.019 | 12.019 | 16.346 | 13.942 | 13.942 |

On innerBT001, local raw backbone16.346% exceeds its head13.942% and frozen14.744%; fixed random projection already reaches15.705%. This is small seed0 inner evidence, not a confirmed improvement. RemovingLayerNorm barely changes either trained head, so LayerNorm alone is not the principal observed loss. Strong fitting improvements onBT003/PF001/SN001 do not transfer uniformly; MSK002 remains hard even in fitting. Preserve raw2048 as a matched new-coordinate control and investigate projection/domain/crop effects before declaring local superiority. No outer checkpoint/hyperparameter is selected.
All gap1/25/100GT/detector R1/R5/mAP and exact checkpoint/input/source proofs are in PHASE_VII_VISUAL_STAGE_DIAGNOSIS_MEASURED.json. Historical model architecture hashes match exactly; the later retrieval-helper empty/unavailable repair has an independent new source snapshot.

## Registered three-seed first-fold visual confirmation

All global/local ImageNet-initialized backbone runs now complete for seeds0/1/2 at the same registered fixed epoch8 and split. Statistics below are mean ± sample SD across training seeds, not query-level independent uncertainty. Crop/query/gallery counts remain the same for each corresponding view.

| Model | Inner BT001 GT | Inner BT001 detector | Outer SN015 GT | Outer SN015 detector |
|---|---:|---:|---:|---:|
| global | 20.000 ± 1.198 | 12.927 ± 1.857 | 45.163 ± 0.364 | 42.804 ± 2.224 |
| learned | 18.995 ± 1.770 | 12.553 ± 1.203 | 44.111 ± 1.516 | 43.757 ± 3.308 |

Frozen ImageNet detector R1 is14.744% oninnerBT001 and53.651% onouterSN015. All new CNNs remain below the outer baseline; the seed0local-vs-global contrast is not stable across seeds. This confirms the negative first recipe, without declaring all local identity learning ineffective or selecting a lucky seed. Only this first fold is confirmed; all six outer/inner folds, stronger pretrained controls, actual localJEPA, repairedC/D and native/rawonline remain unfinished. Complete gap1/25/100R1/R5/mAP, sample denominators, AMP skips, losses, checkpoint/source/data hashes are in PHASE_VII_VISUAL_THREE_SEED_MEASURED.json.

A real globalseed1epoch2early exit and new-process resume onphysicalGPU0 reproduces continuous8epochtraining exactly:818tensor leaves,1NumPyarray and996scalar leaves across model/optimizer/scheduler/AMP/allRNG; training losses/update counts and retrieval are equal. The verification excludes walltime and serialized container bytes and does not certify crosshardware determinism. Stop/resume logs and both complete manifests are hash-linked in PHASE_VII_VISUAL_RESTART_MEASURED.json. Historical seed0source snapshots are preserved; the early-stop CLI added for seed1/2 does not pretend source equality to the previous run.

## Source audit completion: thirty available-source field audits

All30 references now have all11 available source-level semantic fields and exact commit/blob/SHA reading proofs. The final completed set includes MOTRv2, TrackTrack and COVTrack;599listed source/config/license paths have exact provenance and targeted reading records. TFMFT missing model/criterion/sampler and TCEI missing train-entry modules remain explicit coverage exclusions, rather than verified complete training; this is not an exhaustive every-line reading claim or official pretrained reproduction. Weight delivery, unverified derivative/data terms and execution exclusions remain explicit in REPOSITORY_REGISTRY.

GOT-JEPA's registered head5f382645 also contains newerGOT-Edit; the README's paper release84e9324317 was independently materialized from existing local Git objects into a new isolated snapshot. Both actor versions define the same callable twice and the active last body ends with an unassigned tuple expression instead of the earlier loss/statistics return. Target/context feature-extractor parameters are shared; noEMA teacher update is implemented. Its latent target is a256-dimensional tracking filter, not Kfish identity tokens. Static AST and canonical source proofs are external/hash-linked; no foreign code was executed.

V-JEPA2/2.1 ordinary encoders attend bidirectional video clips. AC predictor's frame-causal mask does not establish causal backbone features.2.1Base's published train config selects a model name omitted from the train entry's embed-dimension branch; pretrained hub inference is a different path. CoTracker3's online wrapper initializes query/state separately and advances16frame windows by8; earlier window outputs can use later frames in the same window. Current-frame MOT needs explicit prefix boundaries, delay/emission policy and runtime accounting. Original full-video offline caches are excluded.

TransReID and CLIP-ReID static evaluators remove same-ID/same-camera positives, which would remove within-video fish positives if camera=video. Fish adapters must use physical(video,fish) identities and the established past-only unknown-preserving retrieval masks. Original JPM four branches are hardcoded, and CLIP prompt labels are training-class anchors rather than necessary onlineGT. CLIP stage1 actual contrastive temperature is1.0, not assumed.07; the final cache minibatch requires an empty-batch guard when exactly divisible.

The complete task book explicitly requires first completing environment/all30 source audits before further method development. Earlier executed visual/joint controls remain evidence, with their negative/confounded results preserved. All30available-source audits are now complete with missing-module/license/weight exclusions; this prerequisite gate is satisfied. The next work is controlled strong/local visual training, followed by genuine localJEPA and same-input C/D. Full goal remains IN_PROGRESS; actual localDenseJEPA training, stronger lawful pretrained baselines, repairedsetcompetition, independentmotion and all-fold/three-seed/fullrawonline remain required. See PHASE_VII_SOURCE_AUDIT_MEASURED.json for exact counts and artifact hashes.

## Frozen pretrained diagnostic outcome

DINOv2 official licensed S14 current-crop inference and47,751observation extraction are complete. Same actual/paired pools, query/gap masks and oldbaseline metrics are exactly preserved; paired-detector CLS beatsImageNet on3of6videos, while fixedlocalK4/K8concats are not consistently stronger. This is stronger frozen evidence, not supervisedlocal/JEPA/nativeMOT gain. See REID_COMPARISON table and PHASE_VII_DINOV2_FROZEN_MEASURED.json for allsix R1/R5/mAP/AUC/EER/supports and immutable proofs. Actual raw allframe/runtime and allfold/three-seed trained comparisons remain unfinished.

## DINO visual fitting and reproducibility follow-up

All four initial seed0runs terminal-exited1 when an infinite first-step AMP gradient norm reached strict JSON serialization. Their logs/source snapshots/epoch checkpoints remain preserved outsideGit; manifests mark FAILED and they are excluded from valid fits. The corrected recorder uses FP64 gradient norms and explicit null for actual nonfinite gradients, preserves overflow counts and requires a real finite visual-backbone gradient. Fresh outputs complete; no changed-source checkpoint was resumed. A regression test covers true infinity and large-but-finite FP32 values.

The uncontrolled-workspace DINO real continuous/restart comparison detects203differences across295tensor/1array/1248scalar leaves plus losses/retrieval. Sampling coverage and optimizer updates agree, and initial finite gradients agree, but epoch1/2losses already differ before the stop. Hence this does not isolate a restoration bug. Actual strict deterministic smoke accepts small CUDAcdist backward but rejects the32row CuBLAS path without explicit workspace configuration. A separately declared deterministic launcher sets :4096:8 before CUDA initialization and enables strict deterministic algorithms; actual replay must verify whether this restores equality. Preserve the failed proof and avoid importing the earlier CNN exact-restart claim into DINO.

The completed three-seed visual comparison supports global supervision(+5.450pp above frozen DINO on the first outer actual detector pool), while patch attention has no extra mean gain over global/capacity. This local negative is not a reason to stop domain/local/JEPA research; continue preregistered crossfold/K8, correspondence-aware learning, same-input baselines and actual native trajectories. The current gain is conventional supervised crop retrieval, not native-MOT causation or a novel algorithm.

The separately registered deterministic DINO global seed1 audit now verifies exact continuous versus epoch2exit/fresh-process resume:295tensor/1array/1248scalar leaves, all training losses/updates/coverage and heldout retrieval/feature hashes match. Both actual execution guards record strict algorithms and :4096:8workspace configured beforeCUDA. The failed uncontrolled comparison is retained and is not retroactively certified. Full proof is PHASE_VII_DINOV2_FISH_RESTART_MEASURED.json; this does not certify all uncontrolled allfold executions or crosshardware determinism.

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

The complete firstfoldB three-seed native controls now confirm no JEPA HOTA/IDF1 gain: originalfull31.201±.474/27.727±1.199 andcontextfull30.630±.105/25.679±.692 versus frozenFIFO32.225/30.445. No-ID stillusesGTtemporalgrouping andincreases switches despite smallaggregateHOTA/IDF1 gains. Latentloss andcleanGT-history retrieval areinsufficient nativeevidence; on-policy history contamination, correct targetidentity and truepixelmask controls remain actionable nextsteps. See PHASE_VII_LOCAL_NATIVE_THREE_SEED_MEASURED.json.

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

## Global detector-cache tail differs from declared current padded8 policy

Both wholeSN015 raw campaigns stopafter6,567verifiedframes atframe6,568. CurrentgeneratedIR isexact through6,568; V3proposals firstdifferthere. Originalcache source uses oneglobal27,251frame DataLoader ofbatch8 withunpadfinal3images. Actualcu117V3 probes verifyeachofSN015frames6568–6570: current-only tensor atitsoriginal3batchposition pluszeros for allotherpositions exactlyreconstructs oldcachedboxes/scores, nofuturepixels. Normalrepeat-current8 givesdifferentNMSboxes (orderedcoordinate difference up to13.5pixels) andscores; this isnot globallytoleratedroundoff. Both originalfailedruns/drivers remainpreserved. PHASE_VII_DETECTOR_TAIL_BATCH_MEASURED.json links exactsource/runtime/input/output evidence.

Ownedrawv3 keepsnormal8/currentcrop128/weights/precision/thresholds unchanged. Aftermeasuredoutput only, verifiedpartialtail current-only3validation mayexplain oldcache differences; no validation outputfeeds tracker. Different proposal coordinates do notauthorizefeaturecomparison againstoldcrop references. Normal8before/aftervalidation outputbytes areexact inrealIPC proof PHASE_VII_DETECTOR_TAIL_RUNTIME_MEASURED.json. No prefix tolerance isloosened. Rawv3recordsactualcurrentproposal/feature outputtraces forindependent post-completionCPUreplay andactualofficialscoring; older cachedpredictions maylegitimatelydiffer andarenotraw8truth. Both sixvideo campaigns have restarted innewimmutablev3outputs onactualidleGPU0/2, reusingonlyalreadyverifiedcompleteSN001measurements. FullSN015/newraw6video metrics/replay proofs stillpending; failedpartial timings arenot calledwholeFPS.


### Completed independent observable-overlap and native-motion diagnosis

All 42 full-video controls are complete: seven fixed configurations across six videos, each covering all 27,251 native frames. V3 proposals, RGB features, the legacy GRU seed 0, KF, history writes, lifecycle, and measured output policy are fixed. Disabled controls reproduce the original prediction bytes for every video. The intervention changes only the high-stage appearance weight: candidate quality is one minus its maximum continuous IoU with another current high-confidence detection. Constant-half, eligible-row mean, and three candidate-permutation controls distinguish weight reduction from candidate-specific information. Row means and gate histograms match on a fixed decision input; trajectories can subsequently diverge.

| Control | Macro HOTA | Macro IDF1 |
|---|---:|---:|
| Original | 39.692 | 48.132 |
| Constant half | 38.549 | 46.448 |
| Eligible-row mean | 39.250 | 47.391 |
| Candidate overlap | 39.721 | 47.982 |
| Permutation seed 0 | 39.389 | 47.641 |
| Permutation seed 1 | 39.697 | 47.874 |
| Permutation seed 2 | 39.670 | 47.926 |

Overlap weighting adds 0.029 HOTA points while losing 0.149 IDF1 points. BT-001 loses HOTA and IDF1 relative to the original. SN-001 has identical reported scores for row-mean, overlap, and all permutation controls. These results do not establish a stable independent contribution. Permutation seeds are placebo assignments, not three separately trained methods. All official per-video metrics, alpha arrays, source/input/GT/prediction hashes and threshold crossings are linked in PHASE_VII_OBSERVABLE_OCCLUSION_ALL42_MEASURED.json.

A separate observer records native predicted motion states and reproduces all six prediction files before loading physical labels. Retrospective analysis requires at least three entirely known, consistent recent accepted proposal identities; unknown, mixed, short, missing-reference, and absent/nonunique current GT cases are excluded. Current GT is used only to evaluate center error, normalized by its box diagonal.

| Video | Supported rows | Mean KF error | Mean two-measurement CV error |
|---|---:|---:|---:|
| SN-001 | 4,070 | 0.0552 | 0.0581 |
| SN-015 | 52,658 | 0.0537 | 0.0590 |
| PF-001 | 27,636 | 0.0352 | 0.0362 |
| BT-001 | 15,618 | 0.0653 | 0.0671 |
| BT-003 | 1,249 | 0.0685 | 0.0700 |
| MSK-002 | 37,782 | 0.0619 | 0.0644 |

The four fitting videos contribute 70,737 supported rows, with pooled KF/CV errors 0.05122/0.05313. KF and two-point CV have different history capacity, and the KF state can retain earlier pollution. The latest KF measurement and identity-write timestamps are checked separately; they agree in these supported samples. This comparison does not justify a broad learned-motion replacement or explain failures in excluded histories.

Long accepted-measurement gaps have larger errors: MSK-002 has 302 rows with KF/CV means 0.715/0.720, SN-015 has 322 rows with 0.480/0.962, and inner BT-001 has 45 rows with 0.288/0.274. These are observed gaps, not true occlusion or turn labels. Among known true candidates in this support, exclusion by the original appearance graph occurs in 1/3,912, 79/51,670, 0/27,506, 2/14,908, 0/1,191, and 34/33,536 rows for the six videos in the table order. Missing detections and unclassified histories remain separate limitations. Full support, exclusions, and conditional error distributions are in PHASE_VII_NATIVE_MOTION_ERROR_MEASURED.json.

## Visible-token pretraining probe: storage-safe negative evidence (2026-10-14)

The first completed owned visible-token RGB pretrain (seed0, four fit videos, outer SN-015/inner BT-001 held out) was followed by one frozen-head CE+Triplet identity probe. At SN-015 detector-history gap25, Rank-1 was 57.143% versus the existing frozen CLS reference 57.937% on the same 630/644 supported-query protocol. The learned head selected 50 unknown rows versus 42 for frozen CLS, so its mAP increase (26.513 versus 23.857) is not a clean matched gain. This is useful negative evidence for the current branch, not an identity or MOT contribution. The later storage-safe three-seed masked/visible-target campaign supersedes the single-seed queue note; its first-fold result remains negative/limited and does not justify native or novelty credit.


## Visible-token three-seed campaign (2026-10-15)

The storage-safe campaign completed masked-visible-token and visible-target reconstruction for seeds0/1/2 on the four fit videos, followed by frozen CE+Triplet probes on outer SN-015 with inner BT-001 excluded. Detector-history gap25 Rank-1 was 57.143/57.937/56.825% for masked and 53.016/49.683/51.746% for visible-target, versus frozen CLS57.937% at the same630/644 support. Masked seed1 matches Rank-1 but changes unknown-selected counts; the other masked seeds and all visible-target seeds are lower. This is a reproducible first-fold offline negative/limited result, not a native-MOT or novelty claim. See `PHASE_VII_VISIBLE_TOKEN_MAE_THREE_SEED_MEASURED.json`.

## Final combination gate (2026-10-15)

A measured combination gate now covers the completed context-V2, FDTA-style, source-JPM, dictionary-joint, observable-occlusion, visible-token and pixel/domain controls. Context full/local-temporal/global all lose the same-input frozen FIFO macro identity metrics; FDTA temporal/quality lose held-out HOTA and IDF1; JPM is not above the matched-capacity CLS control consistently; joint routes lose held-out HOTA/IDF1 or add switches; overlap fusion changes HOTA by only +0.029 macro points while lowering IDF1; and the visible-token campaign is offline first-fold evidence without native complementarity. The gate therefore records `DO_NOT_RUN_MECHANICAL_FINAL_STACKING` in `PHASE_VII_FINAL_COMBINATION_GATE_MEASURED.json`. This is a measured decision boundary, not a project-wide NO_GO; a future combination needs a new independent same-input complementary signal.

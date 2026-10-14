# Phase VII identity evidence: first measured decomposition

Status: IN_PROGRESS. Inherited raw/GRU and same-supported retained-cue metrics were recalculated; all six videos exactly match the original scientific metric payloads. This is actual accepted-history detector-feature diagnosis with inherited exposure, not completed GT/detector-crop ReID training or new online MOT evidence.

| Video | Known-positive queries | Raw mean R1 | GRU pure R1 | Geometry row R1 | Real blended row R1 | Cost reconstruction max error |
|---|---:|---:|---:|---:|---:|---:|
| BT-001 | 4223 | 58.5603 | 56.5712 | 60.2652 | 60.2889 | 1.19e-07 |
| BT-003 | 363 | 87.8788 | 87.3278 | 89.2562 | 89.2562 | 5.96e-08 |
| MSK-002 | 8537 | 45.4726 | 42.0288 | 55.9447 | 55.9447 | 1.19e-07 |
| PF-001 | 5742 | 60.5190 | 96.5343 | 60.6583 | 60.6757 | 1.19e-07 |
| SN-001 | 967 | 37.1251 | 73.0093 | 37.1251 | 36.9183 | 5.96e-08 |
| SN-015 | 10710 | 62.3623 | 35.8450 | 62.3249 | 62.2969 | 1.19e-07 |

SN-015 uses the same10,710 known-positive queries for the first pure-ranking contrast. The separate retained-cue support uses8,631 queries: mean76.989920%, GT-selected retained-cue oracle83.489746%. Oracle cue filtering is offline analysis only; unknown history does not prove wrong identity or irrecoverability.

The GRU pure rank is35.845005%, while real geometry/confidence blending gives62.296919%, close to geometry-only62.324930%. Reconstruction reproduces real recorded costs within1.2e-7 on SN-015. Feature norms are within about1.2e-7 of1, and normalized prototype similarities reproduce saved evidence. Thus normalization or a different checkpoint does not explain this contrast; the tracker uses a different score and the geometry prior can largely dominate these fixed-state row decisions. This does not prove that removing GRU preserves full endogenous tracking: Hungarian competition, thresholds, later stages, births and changed histories are outside row retrieval.

PF-001 pure GRU96.534309% versus mean60.519% contradicts a blanket claim that the GRU always discards useful identity evidence. The residual GRU also moves far from original normalized history: SN-015 mean cosine to last0.22285 and to FIFO0.19086. These are measured outputs, not proof of the unique causal source of degradation.

Training source tools/cfp_build_prediction_queries.py uses an earlier ordinary RGB-tracker producer, known unique current identities, positive-present filtering, sampled past cutoff gaps1/5/15/30 and at least two same-frame IDs. The current diagnostic uses historical GRU-produced actual state graphs, retains unknown candidates and evaluates naturally missing histories/candidates separately. SN-015 was calibration for the temporal model and seen by the frozen visual encoder. Candidate label support, generating tracker, requested temporal cutoff and scene exposure differ; their individual contributions remain to be discriminated.

Correct/mixed/unknown retained histories and common native appearance-region support have now been measured. The complete two-crop metadata index is available; the sparse paired-crop visual pilot is completed below; full-frame/new-encoder MOT and all-fold confirmation remain pending. All30available-source audits now completed with explicit exclusions; final method/novelty selection still requires matched trained evidence. The independently implemented first joint/pair probes are explicit task-book controls, with their cost/threshold confound recorded separately. Do not begin large JEPA training from the pure-rank discrepancy alone. PairAUC/EER/mAP/R1/R5 and gap/density/overlap-proxy breakdowns are stored in PHASE_VII_A0_STARTUP_MEASURED.json. Overlap/gap are proxies, not ground-truth occlusion or turning labels.

Original checkpoint-matched temporal training query re-evaluation is complete on all6videos in PHASE_VII_A0_ORIGINAL_PRODUCER_MEASURED.json. On PF-001, GRU46.0896%on ordinary actual histories versus96.8361%on AnchorRecent; on SN-01545.1092%versus28.3419%. Clean FIFO/GRUare93.2634/60.3093%on PF-001and92.8724/71.0101%on SN-015. These original producer/pool/cutoff denominators differ from warm GRU-produced queries, so the change is evidence of scene/history dependence, not a paired causal producer effect or new visual encoder gain.

Further actual-history support and appearance-region diagnosis (v2)

| Video/support | Known-positive queries | FIFO R1 | GRU R1 |
|---|---:|---:|---:|
| BT-001/all_verified_correct | 1069 | 96.3517 | 88.6810 |
| BT-001/some_verified_correct | 2446 | 57.8496 | 56.9501 |
| BT-001/no_verified_correct_unknown_possible | 708 | 3.9548 | 6.7797 |
| BT-003/all_verified_correct | 178 | 99.4382 | 100.0000 |
| BT-003/some_verified_correct | 182 | 76.3736 | 75.2747 |
| BT-003/no_verified_correct_unknown_possible | 3 | 100.0000 | 66.6667 |
| MSK-002/all_verified_correct | 2986 | 79.1360 | 72.6390 |
| MSK-002/some_verified_correct | 4916 | 30.2278 | 28.1530 |
| MSK-002/no_verified_correct_unknown_possible | 635 | 5.1969 | 5.5118 |
| PF-001/all_verified_correct | 2993 | 99.9332 | 99.2315 |
| PF-001/some_verified_correct | 2259 | 20.8942 | 94.5994 |
| PF-001/no_verified_correct_unknown_possible | 490 | 2.4490 | 88.9796 |
| SN-001/all_verified_correct | 180 | 97.2222 | 96.1111 |
| SN-001/some_verified_correct | 709 | 25.6700 | 68.4062 |
| SN-001/no_verified_correct_unknown_possible | 78 | 2.5641 | 61.5385 |
| SN-015/all_verified_correct | 2716 | 99.5950 | 52.6510 |
| SN-015/some_verified_correct | 5915 | 66.6103 | 38.2079 |
| SN-015/no_verified_correct_unknown_possible | 2079 | 1.6354 | 7.1669 |

On SN-015 all2,716 actual histories whose retained proposal labels are verified correct, FIFO99.5950% versus GRU52.6510%. Contamination therefore cannot by itself explain the weak isolated GRU representation on this subset. A restricted native appearance region has7,081 positive-present queries, FIFO94.1110% versus GRU84.5078%. Changing the candidate pool changes both problem difficulty and denominator; these94/84 values must not be directly compared to the10,710-query62/35 values as a method improvement. The remainder is consistent with scene/target/producer distribution shift, but does not uniquely prove one mechanism.

## Completed sparse visual and crop pilot

All six training videos were encoded at preregistered frame residues1/2 modulo20, preserving same-frame sets and temporal galleries:47,751 current GT/detector observations,3.462GiB of features. This is sparse diagnosis, not all-frame runtime. Every selected crop feature is available. Historical FishReID was exposed to all six videos and is a diagnostic reference. Frozen ImageNet and every new fit use the exact ImageNet ResNet50 checkpoint SHA0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a. Matching old detector post-BN features preserves validity, with maximum component difference0.001208 and cosine difference3.756e-6; AMP batch-composition parity is numerical, not byte equality.

Rank@1 percent below uses past gap>=25 frames. Actual-view pools retain unknown detector identities; all representations share each view's queries and masks. GT and detector pools differ, so that table alone cannot isolate crop pixels.

| Video | GT ImageNet | GT FishReID BN | GT local8 visibility | Detector ImageNet | Detector FishReID BN | Detector local8 visibility |
|---|---:|---:|---:|---:|---:|---:|
| BT-001 | 20.952 | 43.016 | 41.587 | 14.744 | 20.513 | 23.237 |
| BT-003 | 12.727 | 97.727 | 87.273 | 17.514 | 40.678 | 42.373 |
| MSK-002 | 13.134 | 49.671 | 37.360 | 4.398 | 5.556 | 6.647 |
| PF-001 | 78.571 | 78.571 | 82.540 | 70.635 | 79.365 | 84.921 |
| SN-001 | 34.192 | 83.677 | 77.320 | 32.308 | 42.500 | 47.692 |
| SN-015 | 54.101 | 68.612 | 70.978 | 53.651 | 66.667 | 68.254 |

The paired comparison joins unique known(video,frame,fish)keys and uses identical keys, queries, galleries and temporal masks for both crop views. All six have zero excluded nonunique pairs. This is conditional on the original detector/GT physical-identity verification rule, not unrestricted perfect identity supervision.

| Video | Matched known keys | Queries | GT FishReID R1 | Detector FishReID R1 |
|---|---:|---:|---:|---:|
| BT-001 | 2229 | 640 | 42.468 | 26.442 |
| BT-003 | 200 | 200 | 97.175 | 53.672 |
| MSK-002 | 5683 | 3136 | 49.008 | 8.201 |
| PF-001 | 2872 | 128 | 86.508 | 81.746 |
| SN-001 | 561 | 560 | 85.385 | 50.000 |
| SN-015 | 5860 | 644 | 75.079 | 71.587 |

MSK002's large crop-view loss survives pool control. Padding, box extent, low resolution, foreground/background mixture and appearance changes are candidate mechanisms, not separately identified causes. No GT crop or physical label is supplied to online tracking.

Six real fits completed at fixed epoch8, seed0, outerSN015/innerBT001, fitting only BT003/MSK002/PF001/SN001. Four stopped-ImageNet adapters compare global, equal-two-projection dualglobal capacity control, simple local pooling and learned weighting. Two fresh ImageNet-initialized ResNet50 models actually update spatial backbone gradients and BN statistics only on the four fitting videos. They share the same supervision: composite physical-ID CE, known temporal positives and same-frame different-fish negatives.

| Representation | Inner BT001 GT | Inner BT001 detector | Outer SN015 GT | Outer SN015 detector | Allocated parameters | Training seconds |
|---|---:|---:|---:|---:|---:|---:|
| Frozen ImageNet raw2048 | 20.952 | 14.744 | 54.101 | 53.651 | 23,508,032 backbone | — |
| local_adapter_global | 17.619 | 13.622 | 46.530 | 43.175 | 1,084,609 | 19.91 |
| local_adapter_dual_global | 17.937 | 13.462 | 46.530 | 43.333 | 1,084,609 | 20.13 |
| local_adapter_simple | 16.825 | 12.500 | 47.319 | 45.079 | 1,084,609 | 19.52 |
| local_adapter_learned | 18.254 | 13.782 | 43.849 | 44.921 | 1,084,609 | 23.75 |
| visual_identity_global | 20.159 | 12.019 | 45.584 | 42.698 | 24,592,641 | 99.23 |
| visual_identity_learned | 20.317 | 13.942 | 43.849 | 46.667 | 24,592,641 | 101.68 |

No first-fold new fit beats frozen ImageNet on outer detector retrieval. A local-versus-global improvement within the trained CNN pair is insufficient to claim a retained baseline gain. This is one fold/seed, not all-fold/three-seed confirmation or new-ReID MOT evidence. Fit loss decreases do not certify generalization. Diagnose raw fine-tuned2048 versus projected/pre-normalization256 and LayerNorm256 on fitting/inner data, then crop stability/domain robustness. Exposed outer scores cannot select the next checkpoint or hyperparameters under an untouched-outer claim.

Full source/data/config/checkpoint hashes and all gap1/25/100 R1/R5/mAP, actual-pool AUC/EER, training updates/AMP skips and masks are retained in `PHASE_VII_VISUAL_PILOT_MEASURED.json`. Atomic model/optimizer/scheduler/AMP/all RNG checkpoints exist outside Git; actual interrupted-resume equivalence remains untested. A later empty-view/unavailable-gallery helper fix preserves historical run snapshots; all pilot features were available, so it does not alter these scores. The256-dimensional new representation cannot use the inherited512-dimensional GRU.

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

Source-level TransReID/CLIP-ReID controls now audited at exact commits. The original crosscamera positive exclusion cannot be reused for same-video fish; JPM four-head scaling/CLS mixing and fit-only prompt dictionaries must remain explicit. One licensed DINOv2S official payload is now acquired/strict-loaded; genuine fish adapters and allfold confirmation remain pending; source completion is not a trained new-ReID result.

## Completed fish-specific source controls

FishMAE and Fish_Re_Identification exact Git trees contain no project LICENSE/COPYING; source execution/transplant and official weights remain excluded pending terms. FishMAE local pretrain loop accepts four arguments (correcting the earlier root-engine comparison), but the sampler does not enforce P×K, ReID omits patch positions, and raw versus nested checkpoint coverage needs verification. The AutoFish metric-learning source uses RGB still-fish retrieval, masked-background GT training crops versus full-rectangle zero-shot crops, variable pretrained versions and normalized-versus-unnormalized embeddings. Both require independently cleared same-input adaptations and causal physical(video,fish) evaluation; neither establishes positive native MOT gain.

## Additional eight source audit constraints

DINOv3 S16 exposes384-D current-crop features and spatial RoPE/Gram training mechanisms under a custom source/weight license; its patch projection drops nonmultiple remainders. No pretrained payload or fish retrieval was acquired. DINO-Tracker optimizes on the entire target video and selects anchors over future frames, so its published per-video method is excluded from strict-online identity comparisons. TFMFT declared transformer/128-D ReID dimensions are not verified implementation because its core modules are absent.

## Official frozen DINOv2 same-crop evidence

Licensed official S14 weight strict175tensor-loaded into the exact audited source; frozen extraction uses the same47,751registered current crops,126x252FP32/162patches, observed padding fractions and unchanged physical pools. Dense/local tokens are real spatial observations; content masks do not remove padding influence from backbone attention or create anatomical/occlusion GT. Fullsource audits30/30complete; no private-pretrain zero-overlap claim. All inherited Fishpost/ImageNet query counts, R1/R5/mAP/AUC/EER reproduce exactly under the new evaluator. The table reports paired known keys with minimum25frame past gap; actual unknown-containing pools and fullgap metrics are in PHASE_VII_DINOV2_FROZEN_MEASURED.json.

| Video | Paired keys | Fishpost detR1 | ImageNet detR1 | DINO CLS GT→detR1 | DINO patchmean detR1 | DINO K4/K8concat detR1 |
|---|---:|---:|---:|---:|---:|---:|
| BT-001 | 2229 | 26.442 | 20.032 | 27.083→25.160 | 23.718 | 25.321/21.635 |
| BT-003 | 200 | 53.672 | 23.729 | 20.339→22.599 | 23.164 | 26.554/25.424 |
| MSK-002 | 5683 | 8.201 | 5.985 | 9.854→5.820 | 6.250 | 6.250/5.721 |
| PF-001 | 2872 | 81.746 | 73.016 | 72.222→76.984 | 78.571 | 66.667/62.698 |
| SN-001 | 561 | 50.000 | 36.923 | 36.154→34.615 | 33.462 | 31.154/28.462 |
| SN-015 | 5860 | 71.587 | 57.143 | 61.746→60.317 | 57.302 | 51.429/50.794 |

DINO CLS improves over pairedImageNet onBT001/PF001/SN015 and falls below it onBT003/MSK002/SN001. Fixed K4/K8concatenation is not consistently better than CLS; PF001/SN015 show substantial degradation. These current-crop measurements support testing identity supervision/domain adaptation and local correspondence with matchedcapacity controls; they do not demonstrate new local learning, stoppedJEPA or full nativeMOT benefit. Frozen descriptors/configs were fixed before this evaluation; do not select recipes on outerSN015. Actual nearest-past positive/current-frame hardnegative and crop stability metrics remain separately available.

Actual DINO fish-supervised firstfold three-seed controls now complete at fixed epoch8: actual detector gap25R1 outerSN015 global63.386±0.716%, simple63.016±0.420%, capacity62.857±0.991%, patch attention62.857±0.572%, versus frozenDINO57.937%. InnerBT001 global22.756±0.424% versus19.872%. All four exclude both heldout videos from visual gradients/classifier labels and preserve384dimensions. The mean patch-attention result does not surpass global/capacity. Paired GT/detector support and AUC/EER are separate; full trained diagnostics and measured hashes are recorded after collection. This does not establish improved native tracking, novelty or external-scene generalization. Allfold/K8/three-seed jobs use the same frozen recipe.

## Same-pair identity discrimination

All twelve completed trained evaluations reproduce the frozen same-input operator and training evaluator exactly, including absent targets/unknowns/query and edge counts. On paired detector keys:

| Video | Frozen R1 | Global three-seed R1 | Frozen AUC / EER | Global three-seed AUC / EER | Patch-attention three-seed R1 |
|---|---:|---:|---:|---:|---:|
| BT001 |25.160|29.274 ±0.648|0.91164 /0.17217|0.94453 ±0.00112 /0.13748 ±0.00220|28.793 ±0.245|
| SN015 |60.317|66.032 ±0.420|0.94340 /0.14131|0.97509 ±0.00157 /0.08057 ±0.00418|65.079 ±0.275|

R1 uses minimum25frame earlier galleries. AUC/EER uses nearest strictly earlier known-positive versus current-frame known-different-fish edges (not the25frame subset): BT001630positive/4362negative, SN015633/5187. Edges repeat fish/frames and are not independent biological samples. Query-level uncertainty and training-seed SD are distinct. Unknowns remain in the actual detector pool but not negative labels; paired keys restrict support explicitly. Current-crop supervised identity improves both ranking and this specific discrimination test; a local-only or native-association causal gain remains unestablished. Full allgap/perview/perproducer/rank5/mAP/unknowns/loss/gradient/checkpoint/source/seed evidence is in PHASE_VII_DINOV2_FISH_THREE_SEED_MEASURED.json.

## All-six-fold K4 supervised confirmation

All72 K4fits/paired diagnostics now complete: six distinct outer videos, inner next in the declared cycle, four fit videos and seeds0/1/2. Source/data/pretrain/sampling/capacity/frozen-prefix/checkpoint and exact diagnostic-support guards passed. Fixed finalepoch8, no outer checkpoint/threshold/architecture selection. Actual detector gap25R1 mean ± sampleSD across training seeds:

| Outer video | Frozen DINO CLS | Global | Simple real parts | CLS-only capacity | Real patch attention |
|---|---:|---:|---:|---:|---:|
| BT-001 | 19.872 | 20.353 ±0.278 | 20.726 ±0.245 | 20.139 ±0.490 | 20.085 ±0.791 |
| BT-003 | 16.949 | 21.846 ±3.405 | 22.222 ±2.844 | 22.222 ±2.609 | 22.222 ±2.139 |
| MSK-002 | 4.431 | 4.850 ±0.191 | 4.916 ±0.106 | 4.905 ±0.125 | 4.894 ±0.152 |
| PF-001 | 74.603 | 75.661 ±1.833 | 75.397 ±1.587 | 76.190 ±1.375 | 74.603 ±0.794 |
| SN-001 | 29.423 | 31.090 ±1.110 | 31.282 ±1.280 | 30.833 ±0.909 | 31.218 ±1.094 |
| SN-015 | 57.937 | 63.386 ±0.716 | 63.016 ±0.420 | 62.857 ±0.991 | 62.857 ±0.572 |

Global mean improves over frozen DINO in6/6 outer videos; PF001 has one improving seed and two equal-baseline seeds. Macro over videos then sampleSD across three seeds: frozen33.869%, global36.198±0.419%, simple36.260±0.377%, capacity36.191±0.241%, patch attention35.980±0.309%. Local spatial attention does not consistently exceed global/capacity; simple pooling has only a small macro difference. No new local novelty, biological-identity disjointness, external-scene or native-MOT claim. K8 continues under the unchanged registered recipe. PHASE_VII_DINOV2_FISH_ALL_FOLDS_K4_MEASURED.json contains full primary actual/paired GT/detector allgapR1/R5/mAP/AUC/EER, perseed/fold support/hash/update/time evidence; all raw/local descriptors remain in linked exact diagnostics.

## Completed K8 confirmation and native producer

All144 supervised DINO fits/diagnostics are complete: K4/K8 × sixouter/inner folds × four controls × three seeds. The unchanged registered K8 recipe gives actual-detector gap25 macroR1 (mean ± sampleSD over training seeds), frozen33.869%:

| K8 control | Rank1 (%) |
|---|---:|
| global | 36.028 ±0.151 |
| simple | 36.061 ±0.204 |
| global_capacity | 36.371 ±0.087 |
| patch_attention | 36.013 ±0.352 |

Patchattention36.013±0.352% does not beat matched active-capacity CLS36.371±0.087%. Full actual/paired GT/detector allgap/R1/R5/mAP/AUC/EER, sampling/source/fit-only/checkpoint proofs are in PHASE_VII_DINOV2_FISH_ALL_FOLDS_K8_MEASURED.json. The previous deterministic restart audit certifies its separately controlled execution, not all144 original uncontrolled fits.

Current-only native RGB/proposal encoding is now complete for frozen and fit-only global seed0 on SN001684frames and SN0156570frames, fixed FP32 current-frame padding128. Exact registered FP16 crop-pixel quantization is retained. Matched512detector-crop samples per producer/video give maximum component difference1.535e-6 and cosine difference2.384e-7 against the training feature operator; this is measured numerical agreement, not byte equality. Allsix/allthree-seed global/patchattention and frozen feature extraction plus last/FIFO/EMA actual endogenous-history fullvideo replay/official scoring are active under immutable sources. Native matched-capacity and stronger same-input source adaptations, newly fitted ordinary GRU coordinates and full raw decode/causalIR/V3 profiling remain required.

## First complete-video current-producer comparisons

Fixed currentRGB/crops/proposals,384coordinate guards, last4 endogenous accepted observations, unchanged tracker lifecycle/geometry and fixed appearanceweight0.3/proximityIoU0.1. Every run covers all684SN001or6570SN015frames, including empty states; labels/GT/teacher/future are excluded online. Metrics use the pinned official TrackEval and audited same GT only after replay. No outer parameter/threshold selection.

| Video / producer / memory | HOTA | AssA | IDF1 | MOTA | IDSW |
|---|---:|---:|---:|---:|---:|
| native_identity_context_full_SN-015_s0_v2 | 30.694 | 13.405 | 25.298 | 91.032 | 146 |
| native_identity_frozen_SN-001_ema_s0_v1 | 43.018 | 30.899 | 60.428 | 78.966 | 26 |
| native_identity_frozen_SN-001_fifo_s0_v1 | 43.023 | 30.907 | 60.443 | 78.996 | 24 |
| native_identity_frozen_SN-001_last_s0_v1 | 42.508 | 30.198 | 58.581 | 78.981 | 25 |
| native_identity_frozen_SN-015_ema_s0_v1 | 30.717 | 13.405 | 26.475 | 91.003 | 153 |
| native_identity_frozen_SN-015_fifo_s0_v1 | 32.225 | 14.786 | 30.445 | 91.019 | 152 |
| native_identity_frozen_SN-015_last_s0_v1 | 30.421 | 13.155 | 26.650 | 91.024 | 152 |
| native_identity_global_SN-001_ema_s0_v1 | 43.018 | 30.899 | 60.428 | 78.966 | 26 |
| native_identity_global_SN-001_fifo_s0_v1 | 43.016 | 30.897 | 60.443 | 78.996 | 24 |
| native_identity_global_SN-001_last_s0_v1 | 42.501 | 30.189 | 58.581 | 78.981 | 25 |
| native_identity_global_SN-015_ema_s0_v1 | 32.252 | 14.797 | 29.313 | 90.992 | 157 |
| native_identity_global_SN-015_fifo_s0_v1 | 32.042 | 14.633 | 29.527 | 90.992 | 156 |
| native_identity_global_SN-015_last_s0_v1 | 32.162 | 14.727 | 29.798 | 91.019 | 157 |
| native_identity_local_global_SN-015_s0_v1 | 31.017 | 13.664 | 26.362 | 91.003 | 145 |

The stronger offline crop retrieval does not yet yield stable native tracking gains: globalFIFO is32.042vsfrozen32.225HOTA onSN015 and43.016vs43.023onSN001. Identical SN001IDF1 is not identical prediction bytes. V2contextfullseed0has30.694HOTA/25.298IDF1/146switches, versusfrozenFIFO32.225/30.445/152; fewer switches does not establish better identity continuity. Original Bglobal native31.017/26.362/145also underperforms FIFO despite strong clean-group retrieval. This distinguishes clean-history ranking from endogenous association and fragmentation; no positive local/JEPA/native contribution is claimed. The three-seed/fullsix campaigns retain all controls. Metrics/report/prediction hashes and all official rawalpha arrays are in PHASE_VII_NATIVE_CURRENT_IDENTITY_METRICS_MEASURED.json and linked external TrackEval reports.

## Registered actual pixel-mask training control

The new owned trainer uses the pinned Apache DINOv2-S backbone, four fitting videos and the original same-frame/temporal sampler, 8 × 150 attempted updates, identical augmentation, last two blocks plus final norm and global384 identity head. Outer SN-015 and inner BT-001 never update fitting or EMA state. Three modes and three seeds are registered in `phase_vii_pixel_masked_identity_v1.json`: supervised-only, visible-target distillation, and true pre-attention pixel masking. All modes run three forward branches. Visible-target and masked modes have the same active predictor and independently stopped EMA target; only masked input differs. The supervised-only zero auxiliary coefficient is an explicit lower active-loss-capacity reference. No anatomy or true occlusion labels are claimed from content fractions.

Half of observable patches are queried using a separate captured RNG, retaining visible content where possible. Official frozen mask embeddings replace selected image embeddings before any backbone attention. Current unmasked teacher targets are training-only; online evaluation calls the student alone. EMA advances only after successful AMP optimizer updates, and the update count must match successful training steps. These are established masked distillation mechanisms, not novelty or official FishMAE reproduction.

Lossless checkpoints omit only bitexact immutable pretrained prefix tensors. The actual licensed model CPU smoke restores all377model keys exactly from87delta keys plus290base keys after a real optimizer/EMA update, with36finite gradient fields and144masked observed patches; the model delta is37,076,321bytes. Student and teacher tails, classifier, predictor, all buffers and optimizer/scheduler/AMP states are saved; all Python/NumPy/CPU/CUDA/sampler/augmentation/mask RNG are captured. The one-update smoke proves serialization/masking/gradient correctness, not an8epoch fit or nativeMOT gain. See PHASE_VII_PIXEL_MASKED_CHECKPOINT_MEASURED.json.

A durable first-fold queue is waiting for the assigned owned GPU1 native producer to fully complete and its process to exit, followed by an actual <=64MiB/zero-utilization check. It registers nine mode/seed fits plus a distinct seed0masked epoch2exit/fresh-process resume versus continuous8epoch fit. Exact model/optimizer/EMA/RNG/loss/retrieval comparison runs before subsequent controls. The initial storage plan reserves42GiB plus5GiB for other jobs and250MiB per fit. Sources/config are hash guarded; no active native/raw source helper was modified. All-six masked folds, genuine MAE pixel-reconstruction/strong source baselines, native pipelines and scientific benefit still require actual experiments.

## Actual RGB reconstruction control, separate from EMA latent targets

An owned masked pixel autoencoding control now predicts588values (14×14×3RGB) per patch directly, with weighted normalized current-pixel targets excluding padding/augmentation-invalid pixels. The official pinned DINO encoder replaces selected patch embeddings before allattention. Two registered modes share the same384->768->588predictor: truepixelmask and visible-pixel-target control. Both use the same four-fit-video sampler/labels/augmentation/8×150attemptedupdates/lasttwoblocks/global384head. There is no EMA teacher; current supervisedidentity and maskedRGBtarget branches are separate. Relative to latent distillation the targetspace, predictorcapacity and forwardcompute differ explicitly. This is an established full-token masked reconstruction control, not official MAE visible-token-only encoder/decoder, FishMAE reproduction, novelmechanism or validatedidentitygain.

Actual licensed DINO/current BT003fit-crop CPU evidence verifies144masked observedpatches,36finitegradient fields, oneoptimizer update, frozenprefix equality, targetpixel perturbation cannot affect maskedpredictions, and all202model-state keys restore exactly from57delta keys plus145immutableofficialbase keys. Modelcheckpoint23,488,085bytes; source/input/checkpoint/gradient/protocol proof in PHASE_VII_PIXEL_RECONSTRUCTION_SMOKE_MEASURED.json. The old first smoke's accidental EMA wording is corrected in the separately re-executed v2 proof; neither smoke establishes8epoch/MOT benefit.

A durable GPU3queue registers sixmode/seed fits plus an actualseed0masked epoch2exit/freshprocessresume duplicate. It waits for the owned nativeproducer to complete, process teardown and actual<=64MiB/zeroGPUutilization, with42GiBreserve/5GiBother-job margin and250MiBperfit. The GPU1EMA-latent-mask queue remains separate and guarded. New fits are pending; all-six/newproducer/native/strongsource comparisons and reliablecontribution remain required.

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


### Source-matched TransReID JPM adapter and actual first-fold campaign

Pinned TransReID `dec55046fcdfadee14e2c28e2df89305d8f7557a` JPM is now transplanted to the same licensed DINOv2-S/current crop banks. The original shift5/group2, global/copied shared-local final blocks, five BN necks/classifiers, half-global/half-local loss and after-neck `[global, local1/4, ..., local4/4]` descriptor are retained. DINO162patches yield four40token subsets, with two final shuffled tokens omitted by source floor division. BN statistics fit on BT003/MSK002/PF001/SN001 only; innerBT001/outerSN015 are eval-only. No SIE/camera input, external TransReID weights or official full-method claim. Same-video temporal/same-frame fish supervision replaces original camera-dependent sampling; exact adaptations and MIT notice are in third_party/transreid_jpm/PROVENANCE.md.

Real CPU/DINO/current-fitcrop gradient and delta-restore proof passes: 237 model keys, 76 delta keys, checkpoint21,380,019bytes for the two-identity smoke. This is a smoke vocabulary, not the larger fitting classifier. An actual first-fold JPM seed0 continuous8epoch run and distinct epoch2exit/new-process resume are bitexact: 246 tensor/1array/1286scalar leaves including all BN buffers, optimizer/scaler/scheduler/RNG, all losses and retrieval arrays. All nine distinct global/JPM/CLS-capacity fits (3seeds each) are running under fixed8epoch/1200attempt settings; no outer-based checkpoint or hyperparameter selection. JPM and repeated-CLS control have identical active parameters/branch lengths/supervision. All patch subsets and prefixCLS already globally mix the crop, so JPM does not establish purely local/anatomical evidence. Three-seed gains/native tracking/novelty are unproven until measured.

Current-only deployment with fixed pad128 is separately verified on four actual detector proposals plus one preserved invalid row from frame1 of each excluded video. Current crop pixels equal the original pilot bytes; maximum embedding component differences are1.442e-6/1.057e-6 and cosine differences5.96e-8 against GPU evaluation, numerical agreement rather than descriptor byte equality. Native queues wait for both owned fitting processes to finish and actual GPU1/3 teardown/idle; then produce18 fullvideo feature banks and score54 last/FIFO/EMA controls. Planned feature storage is5.063GiB with40GiBreserve plus1GiBraw margin. This remains first-fold evidence, not allsixouter folds or complete rawonline speed.


All nine unique first-fold source-JPM controls nowcomplete, plus separate exactrestart duplicate. Strictpast detector-gallery gap25 outerSN015 R1: globalBN60.688±1.126%, JPM61.905±.727%, samecapacityCLS61.587±.692%, frozenCLS57.937%; mAP30.755±.315/30.835±.599/30.478±.574 versusfrozen23.857. JPM-minus-capacity R1 perseed is+1/+3/+2 correctqueries outof630, a mean+.317pp, not a robust local novelty result. InnerBT001 R1 global20.566±1.540/JPM20.085±1.137/capacity20.994±.555 versusfrozen19.872, so the JPM contrast doesnotgeneralizeconsistently betweenexcludedvideos. Allseed/sample/support/BN/source/capacity/unknownselection evidence is in PHASE_VII_TRANSREID_JPM_THREE_SEED_MEASURED.json. Mean±sampleSD only; no statisticalsignificance, winningseed or historicalouter-based modelselection claim.

The authoritative training budget is9unique×1200=10,800attempts, plus1200duplicate attempts for the actualrestart=12,000executions; successfulAMPupdates are reportedperfit andmaydiffer. Native queues havepassed prerequisiteprocess/idlechecks andnowexecute all18bank/54memory controls. NativeMOT andallsixfold gains remainunproved. Publication-format PNG/SVG/PDF measuredfigures remain outsideGit at /data3/liuyeqiang/IR-MFT_assets/cfp/phase7/transreid_jpm_three_seed_figures_v1; dots show actualseeds, whiskers sampleSD, deterministicfrozen hasnoSD.

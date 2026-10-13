# Phase VII failure analysis

Status: IN_PROGRESS. Negative findings register discriminating follow-ups; there is no fixed failure count.

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

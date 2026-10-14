# Phase VII local identity evidence

Status: IN_PROGRESS. No trained local identity improvement is claimed yet.

The full two-crop index covers27,251frames,217,201valid nonambiguous GT observations and258,309actual V3 proposals across98physical(video,fish)identities. It preserves every detector proposal and marks independently unverified physical identities as unknown. GT/current-detector pairs are joined by(video,frame,known fish), with coverage and duplicate counts separately saved. Arrays and frame paths remain outside Git at `/data3/liuyeqiang/IR-MFT_assets/cfp/phase7/identity_two_crop_index_v1`; compact exact hashes/counts are recorded in `PHASE_VII_TWO_CROP_INDEX_MEASURED.json`. The index is all-frame; the subsequent sparse feature/training pilot covers47,751observations and is reported below.

Local token provenance is being audited before training. DINOv3 Small uses patch16/dim384 with four storage tokens; the inherited128×256crop is divisible by16and has128spatial patches. DINOv2 Small uses patch14/dim384; direct source asserts divisibility, so128×256cannot be fed unchanged. A consistent declared padding/resizing route and effective letterbox visibility are required. Neither spatial tokens nor K=4/8 learned slots may be labeled anatomical head/body/tail without annotations.

The next controlled encoder experiments compare global CLS/pooled features, spatial token mean, simple K=4/8aggregation and learned local identity features on identical GT/detector pairs and temporal candidate pools. Known same-video/same-frame different-fish negatives are required. New fish encoder/adapter fitting excludes both distinct outer and inner calibration videos. Existing FishReID/backbone/GRU results remain exposed references. Any new coordinate requires a matched ordinary producer; the old512GRU is not a valid384-dimensional baseline. Actual occlusion/turning labels are absent, so gap/box overlap/shape-change analysis must retain explicit proxy names.

The licensed official DINOv2-S checkpoint has now been acquired within the100MBcap and strict-loaded into exact audited Apache source. Current-crop CLS384/162patches and K4/K8geometric evidence extraction completed on47,751observations; pretrained retrieval diagnostic is recorded below when complete. FishMAE official reproduction and new local supervised/JEPA/native MOT work remain pending. All30available-source audits completed with explicit exclusions.

## Executed local controls

Actual ResNet spatial maps are4×8 at128×256; K4/K8 width strips pool spatial pixels. Equal K8 raw mean equals global spatial mean, providing a genuine simple-aggregation control. Fish strip features use the inherited projection and BN but have no independent part fitting. ImageNet adapters fit only the four permitted videos; dualglobal has equal two projections without local input, separating extra projection capacity from observed spatial information. Letterbox visibility measures the clipped current-image rectangle after crop resize, not fish segmentation or occlusion.

| Representation | Inner BT001 GT | Inner BT001 detector | Outer SN015 GT | Outer SN015 detector | Allocated parameters | Training seconds |
|---|---:|---:|---:|---:|---:|---:|
| Frozen ImageNet raw2048 | 20.952 | 14.744 | 54.101 | 53.651 | 23,508,032 backbone | — |
| local_adapter_global | 17.619 | 13.622 | 46.530 | 43.175 | 1,084,609 | 19.91 |
| local_adapter_dual_global | 17.937 | 13.462 | 46.530 | 43.333 | 1,084,609 | 20.13 |
| local_adapter_simple | 16.825 | 12.500 | 47.319 | 45.079 | 1,084,609 | 19.52 |
| local_adapter_learned | 18.254 | 13.782 | 43.849 | 44.921 | 1,084,609 | 23.75 |
| visual_identity_global | 20.159 | 12.019 | 45.584 | 42.698 | 24,592,641 | 99.23 |
| visual_identity_learned | 20.317 | 13.942 | 43.849 | 46.667 | 24,592,641 | 101.68 |

The two visual-backbone rows have actual spatial gradients and fit-only BN updates; adapter rows stop the input backbone. Simple/learned strips do not yet demonstrate improvement above frozen raw ImageNet. This does not establish universal failure of local evidence: part alignment, detector-crop robustness, pretrained transformer tokens, K4 fitting, all-fold/three-seed and complete native trajectories remain unresolved. Check `PHASE_VII_VISUAL_PILOT_MEASURED.json` for exact parameter/gradient-participation differences and all metrics.

## Registered three-seed first-fold visual confirmation

All global/local ImageNet-initialized backbone runs now complete for seeds0/1/2 at the same registered fixed epoch8 and split. Statistics below are mean ± sample SD across training seeds, not query-level independent uncertainty. Crop/query/gallery counts remain the same for each corresponding view.

| Model | Inner BT001 GT | Inner BT001 detector | Outer SN015 GT | Outer SN015 detector |
|---|---:|---:|---:|---:|
| global | 20.000 ± 1.198 | 12.927 ± 1.857 | 45.163 ± 0.364 | 42.804 ± 2.224 |
| learned | 18.995 ± 1.770 | 12.553 ± 1.203 | 44.111 ± 1.516 | 43.757 ± 3.308 |

Frozen ImageNet detector R1 is14.744% oninnerBT001 and53.651% onouterSN015. All new CNNs remain below the outer baseline; the seed0local-vs-global contrast is not stable across seeds. This confirms the negative first recipe, without declaring all local identity learning ineffective or selecting a lucky seed. Only this first fold is confirmed; all six outer/inner folds, stronger pretrained controls, actual localJEPA, repairedC/D and native/rawonline remain unfinished. Complete gap1/25/100R1/R5/mAP, sample denominators, AMP skips, losses, checkpoint/source/data hashes are in PHASE_VII_VISUAL_THREE_SEED_MEASURED.json.

A real globalseed1epoch2early exit and new-process resume onphysicalGPU0 reproduces continuous8epochtraining exactly:818tensor leaves,1NumPyarray and996scalar leaves across model/optimizer/scheduler/AMP/allRNG; training losses/update counts and retrieval are equal. The verification excludes walltime and serialized container bytes and does not certify crosshardware determinism. Stop/resume logs and both complete manifests are hash-linked in PHASE_VII_VISUAL_RESTART_MEASURED.json. Historical seed0source snapshots are preserved; the early-stop CLI added for seed1/2 does not pretend source equality to the previous run.

The local evidence design now has source-level counterexamples to audit: TransReID JPM parts share globally mixed early-attention/CLS context, CoTracker3 visibility multiplies visibleprobability by confidence, and GOTJEPA spatial relocation leaves many current feature vectors untouched. These are geometric/context/model-confidence mechanisms, not annotatedfishparts or strict masked-token exclusion. DenseJEPA current-target corruption requires actual target masking before leakage-producing aggregation, or past-only context with all current/future target content reserved for training.

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

## Actual supervised DINO first-fold three-seed controls

Fresh official DINOv2S is fitted only on BT003/MSK002/PF001/SN001; innerBT001 and outerSN015 never update visual weights, running state or classifier labels. The first ten blocks/patch/special/position tokens are immutable; final two blocks and final norm receive nonzero finite gradients and change. NoBN modules exist. Current crop alone produces CLS384 and162patch384; geometric K4regions and padding/declared-erasure fractions are not fish anatomy or occlusion labels. Finalepoch8/1200attempted steps is fixed, AdamW backbone1e-5/head3e-4 and the same true same-frame/temporal mixed GT/detector sampler, loss and augmentation apply to all four modes. Matching projection remains384 and begins as identity.

| Mode | InnerBT001 actual detectorR1 | OuterSN015 actual detectorR1 | Trainable backbone | Trainable head |
|---|---:|---:|---:|---:|
| Frozen DINO CLS |19.872|57.937|0|0|
| global |22.756 ±0.424|63.386 ±0.716|3,551,232|175,872|
| simple real-patch pooling |22.543 ±0.370|63.016 ±0.420|3,551,232|323,329|
| same active architecture, CLS content only |22.436 ±0.734|62.857 ±0.991|3,551,232|1,595,714|
| real regional patch attention |22.009 ±0.334|62.857 ±0.572|3,551,232|1,595,714|

Mean ± sampleSD is across seeds0/1/2, with identical fixed crop/query/gallery support. Global improves+5.450pp over frozen DINO on the outer actual detector pool; each seed is above that baseline. Patch attention does not exceed global or the capacity control in the mean. It still has real patch/query gradients and learned observed-region reliability, rather than renamed global tokens. The capacity control repeats current CLS with the same static spatial positions/masks; exact architecture/active-parameter count and patch-perturbation invariance tests distinguish capacity from spatial image content. This is an established supervised A control, not DenseJEPA, joint identity association or novel research evidence.

Paired keys/actual unknown-containing pools are evaluated separately at gaps1/25/100 with R1/R5/mAP/AUC/EER and nearest-past-positive/current-frame-known-negative edge supports. The new evaluator reproduces both the frozen same-input reference and training evaluator metrics exactly. Full local descriptor/paired evidence and checkpoint/source/data hashes are recorded after the measured-stage collector finishes. All six folds/K4/K8/three seeds are preregistered; bounded drivers preserve failures and require40GiBfree/no other-user GPU interference. Native full-frame/raw-online and same-coordinate ordinary history producer remain outstanding.

The original uncontrolled-workspace DINO continuous/restart pair differs already in epoch1, before the restart boundary, despite equal first finite gradients, sampling coverage and update counts. Its failed comparison is retained. A separate deterministic execution recipe fixes cuBLAS workspace before CUDA initialization and enables strict deterministic algorithms; its actual continuous/epoch2exit/fresh-resume audit is reported separately, without retroactive certification of the original twelve fits.

The separately registered deterministic DINO global seed1 audit now verifies exact continuous versus epoch2exit/fresh-process resume:295tensor/1array/1248scalar leaves, all training losses/updates/coverage and heldout retrieval/feature hashes match. Both actual execution guards record strict algorithms and :4096:8workspace configured beforeCUDA. The failed uncontrolled comparison is retained and is not retroactively certified. Full proof is PHASE_VII_DINOV2_FISH_RESTART_MEASURED.json; this does not certify all uncontrolled allfold executions or crosshardware determinism.

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

## Registered actual pixel-mask training control

The new owned trainer uses the pinned Apache DINOv2-S backbone, four fitting videos and the original same-frame/temporal sampler, 8 × 150 attempted updates, identical augmentation, last two blocks plus final norm and global384 identity head. Outer SN-015 and inner BT-001 never update fitting or EMA state. Three modes and three seeds are registered in `phase_vii_pixel_masked_identity_v1.json`: supervised-only, visible-target distillation, and true pre-attention pixel masking. All modes run three forward branches. Visible-target and masked modes have the same active predictor and independently stopped EMA target; only masked input differs. The supervised-only zero auxiliary coefficient is an explicit lower active-loss-capacity reference. No anatomy or true occlusion labels are claimed from content fractions.

Half of observable patches are queried using a separate captured RNG, retaining visible content where possible. Official frozen mask embeddings replace selected image embeddings before any backbone attention. Current unmasked teacher targets are training-only; online evaluation calls the student alone. EMA advances only after successful AMP optimizer updates, and the update count must match successful training steps. These are established masked distillation mechanisms, not novelty or official FishMAE reproduction.

Lossless checkpoints omit only bitexact immutable pretrained prefix tensors. The actual licensed model CPU smoke restores all377model keys exactly from87delta keys plus290base keys after a real optimizer/EMA update, with36finite gradient fields and144masked observed patches; the model delta is37,076,321bytes. Student and teacher tails, classifier, predictor, all buffers and optimizer/scheduler/AMP states are saved; all Python/NumPy/CPU/CUDA/sampler/augmentation/mask RNG are captured. The one-update smoke proves serialization/masking/gradient correctness, not an8epoch fit or nativeMOT gain. See PHASE_VII_PIXEL_MASKED_CHECKPOINT_MEASURED.json.

A durable first-fold queue is waiting for the assigned owned GPU1 native producer to fully complete and its process to exit, followed by an actual <=64MiB/zero-utilization check. It registers nine mode/seed fits plus a distinct seed0masked epoch2exit/fresh-process resume versus continuous8epoch fit. Exact model/optimizer/EMA/RNG/loss/retrieval comparison runs before subsequent controls. The initial storage plan reserves42GiB plus5GiB for other jobs and250MiB per fit. Sources/config are hash guarded; no active native/raw source helper was modified. All-six masked folds, genuine MAE pixel-reconstruction/strong source baselines, native pipelines and scientific benefit still require actual experiments.

## Actual RGB reconstruction control, separate from EMA latent targets

An owned masked pixel autoencoding control now predicts588values (14×14×3RGB) per patch directly, with weighted normalized current-pixel targets excluding padding/augmentation-invalid pixels. The official pinned DINO encoder replaces selected patch embeddings before allattention. Two registered modes share the same384->768->588predictor: truepixelmask and visible-pixel-target control. Both use the same four-fit-video sampler/labels/augmentation/8×150attemptedupdates/lasttwoblocks/global384head. There is no EMA teacher; current supervisedidentity and maskedRGBtarget branches are separate. Relative to latent distillation the targetspace, predictorcapacity and forwardcompute differ explicitly. This is an established full-token masked reconstruction control, not official MAE visible-token-only encoder/decoder, FishMAE reproduction, novelmechanism or validatedidentitygain.

Actual licensed DINO/current BT003fit-crop CPU evidence verifies144masked observedpatches,36finitegradient fields, oneoptimizer update, frozenprefix equality, targetpixel perturbation cannot affect maskedpredictions, and all202model-state keys restore exactly from57delta keys plus145immutableofficialbase keys. Modelcheckpoint23,488,085bytes; source/input/checkpoint/gradient/protocol proof in PHASE_VII_PIXEL_RECONSTRUCTION_SMOKE_MEASURED.json. The old first smoke's accidental EMA wording is corrected in the separately re-executed v2 proof; neither smoke establishes8epoch/MOT benefit.

A durable GPU3queue registers sixmode/seed fits plus an actualseed0masked epoch2exit/freshprocessresume duplicate. It waits for the owned nativeproducer to complete, process teardown and actual<=64MiB/zeroGPUutilization, with42GiBreserve/5GiBother-job margin and250MiBperfit. The GPU1EMA-latent-mask queue remains separate and guarded. New fits are pending; all-six/newproducer/native/strongsource comparisons and reliablecontribution remain required.

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


### Source-matched TransReID JPM adapter and actual first-fold campaign

Pinned TransReID `dec55046fcdfadee14e2c28e2df89305d8f7557a` JPM is now transplanted to the same licensed DINOv2-S/current crop banks. The original shift5/group2, global/copied shared-local final blocks, five BN necks/classifiers, half-global/half-local loss and after-neck `[global, local1/4, ..., local4/4]` descriptor are retained. DINO162patches yield four40token subsets, with two final shuffled tokens omitted by source floor division. BN statistics fit on BT003/MSK002/PF001/SN001 only; innerBT001/outerSN015 are eval-only. No SIE/camera input, external TransReID weights or official full-method claim. Same-video temporal/same-frame fish supervision replaces original camera-dependent sampling; exact adaptations and MIT notice are in third_party/transreid_jpm/PROVENANCE.md.

Real CPU/DINO/current-fitcrop gradient and delta-restore proof passes: 237 model keys, 76 delta keys, checkpoint21,380,019bytes for the two-identity smoke. This is a smoke vocabulary, not the larger fitting classifier. An actual first-fold JPM seed0 continuous8epoch run and distinct epoch2exit/new-process resume are bitexact: 246 tensor/1array/1286scalar leaves including all BN buffers, optimizer/scaler/scheduler/RNG, all losses and retrieval arrays. All nine distinct global/JPM/CLS-capacity fits (3seeds each) are running under fixed8epoch/1200attempt settings; no outer-based checkpoint or hyperparameter selection. JPM and repeated-CLS control have identical active parameters/branch lengths/supervision. All patch subsets and prefixCLS already globally mix the crop, so JPM does not establish purely local/anatomical evidence. Three-seed gains/native tracking/novelty are unproven until measured.

Current-only deployment with fixed pad128 is separately verified on four actual detector proposals plus one preserved invalid row from frame1 of each excluded video. Current crop pixels equal the original pilot bytes; maximum embedding component differences are1.442e-6/1.057e-6 and cosine differences5.96e-8 against GPU evaluation, numerical agreement rather than descriptor byte equality. Native queues wait for both owned fitting processes to finish and actual GPU1/3 teardown/idle; then produce18 fullvideo feature banks and score54 last/FIFO/EMA controls. Planned feature storage is5.063GiB with40GiBreserve plus1GiBraw margin. This remains first-fold evidence, not allsixouter folds or complete rawonline speed.


All nine unique first-fold source-JPM controls nowcomplete, plus separate exactrestart duplicate. Strictpast detector-gallery gap25 outerSN015 R1: globalBN60.688±1.126%, JPM61.905±.727%, samecapacityCLS61.587±.692%, frozenCLS57.937%; mAP30.755±.315/30.835±.599/30.478±.574 versusfrozen23.857. JPM-minus-capacity R1 perseed is+1/+3/+2 correctqueries outof630, a mean+.317pp, not a robust local novelty result. InnerBT001 R1 global20.566±1.540/JPM20.085±1.137/capacity20.994±.555 versusfrozen19.872, so the JPM contrast doesnotgeneralizeconsistently betweenexcludedvideos. Allseed/sample/support/BN/source/capacity/unknownselection evidence is in PHASE_VII_TRANSREID_JPM_THREE_SEED_MEASURED.json. Mean±sampleSD only; no statisticalsignificance, winningseed or historicalouter-based modelselection claim.

The authoritative training budget is9unique×1200=10,800attempts, plus1200duplicate attempts for the actualrestart=12,000executions; successfulAMPupdates are reportedperfit andmaydiffer. Native queues havepassed prerequisiteprocess/idlechecks andnowexecute all18bank/54memory controls. NativeMOT andallsixfold gains remainunproved. Publication-format PNG/SVG/PDF measuredfigures remain outsideGit at /data3/liuyeqiang/IR-MFT_assets/cfp/phase7/transreid_jpm_three_seed_figures_v1; dots show actualseeds, whiskers sampleSD, deterministicfrozen hasnoSD.


### Genuine visible-token near-full encoder reconstruction control

The previous pixel reconstruction controls trained the last two encoder blocks. The new owned MAE control removes queried patch embeddings before attention and trains all 12 blocks, the patch embedding, CLS token, and final norm. Pretrained positions and the unused original mask token remain fixed. A two-layer, 192-dimensional decoder reconstructs normalized current RGB patches; deployment uses the encoder alone. The visible-target control retains query pixels and uses the same learned query marker additively, so active parameter counts match. Encoder token counts and compute differ and must be reported separately.

The unlabeled pool contains 15,217 valid detector crops from four fitting videos: BT-003 324, MSK-002 11,002, PF-001 3,113, and SN-001 778. Sampling never uses identity values or GT crop rows. Current RGB image headers and detector boxes supply letterbox geometry. BT-001 and SN-015 are excluded. These are sparse pilot crops, and inherited detector and foundation exposure remains explicit.

On two real detector crops, the CPU check verifies gradients in every encoder block, patch embedding, and CLS token; deleting 129 queried patches leaves 65 and 130 encoder patches. Perturbing hidden pixels after the registered resize leaves predictions unchanged. All 208 model state entries restore exactly, and decoder calls are excluded from inference. Mixed precision, variable token counts, and active query-marker tests pass.

Six fixed eight-epoch pretrains (150 steps per epoch, two modes, three seeds) and a distinct epoch-two exit/fresh-process resume run are registered. The queue waits for the owned GPU 1 native job to finish and exit, then checks actual idleness. Its storage forecast includes remaining native banks, raw traces, restart states, atomic checkpoint copies, and the 40 GiB reserve. Only explicitly temporary optimizer files created by this new queue can be retired after final encoders are retained; historical assets remain protected.

This is an independently implemented DINO initialization for visible-token RGB reconstruction. Identity CE/Triplet supervision, the ordinary supervised encoder control, all folds, native MOT evaluation, and novelty verification remain required. It establishes neither completed pretraining nor identity gains, and does not reproduce the complete FishMAE method. Evidence: PHASE_VII_VISIBLE_TOKEN_FISH_MAE_CPU_MEASURED.json.


### Actual CUDA validation and separate identity probes

Both visible-token reconstruction modes passed actual FP16 optimizer steps on idle GPU 1, with deterministic algorithms and the declared cuBLAS workspace. Each has 22,639,116 active parameters and gradients in all 12 blocks; the masked mode required one AMP backoff (two attempts for one update), while the visible-target control succeeded in one attempt. These are short executable checks, not completed pretraining or identity gains. PHASE_VII_VISIBLE_TOKEN_MAE_CUDA_MEASURED.json records the measured memory, gradient, runtime, and source evidence.

A separate identity stage is registered for original DINO and each matching-seed masked/visible pretrained encoder. It freezes the encoder and fits a 384-dimensional linear projection and fitting-only classifier, using strict CE, Triplet, or their sum. Unused classifiers are frozen for Triplet-only and their different active counts are explicit. The same four fitting videos, sampling, augmentation, 1,200 attempts, and fixed final epoch apply. The 27-probe queue waits for actual pretraining completion and process exit. This is the frozen encoder/probe comparison; all-block supervised tuning, spatial evidence, native tracking, and full-fold validation remain required.

The first original-DINO CE+Triplet probe completed eight epochs: outer SN-015 detector gap-25 R1 59.683% and mAP 27.698%, versus the unchanged raw encoder 57.937%/23.857%. Inner BT-001 R1 19.231%, so no cross-video or lucky-seed gain claim. Real epoch-two exit and fresh-process resume match the continuous run exactly: 10 tensor leaves, one array, 1,114 scalar leaves, all losses, optimizer/scaler/scheduler/RNG, and canonical retrieval tensor hashes. The full encoder stays byte-identical; only small head/optimizer states are stored. Real retrieval arrays are evaluated in RAM and hashed, preserving space without changing old assets.

The complete source-JPM native matrix now verifies all 54 full-video replays (2 videos × 3 modes × 3 seeds × 3 memories). For FIFO, BT-001 global/JPM/CLS-capacity are HOTA 32.038±0.803/32.511±0.439/32.573±0.483 and IDF1 38.343±1.221/39.163±0.706/39.157±0.701; SN-015 are HOTA 32.206±0.132/32.427±0.199/32.256±0.193 and IDF1 29.805±0.175/30.177±0.335/30.061±0.438. JPM is not above the same-capacity control on BT-001 and has no stable cross-video causal advantage. The exact 54-row manifests, feature/checkpoint hashes and full TrackEval metrics are outside Git at `/data3/liuyeqiang/IR-MFT_assets/cfp/phase7/native_source_jpm_complete_matrix_v1.json` (SHA256 `99a41ed96f3a50d5f2afebdf4e9738fd79a42046709a55f13b4f8cd2781ce492`), with the compact tracked summary in `PHASE_VII_SOURCE_JPM_NATIVE_COMPLETE_MEASURED.json`. No seed or memory is selected from these outer metrics.

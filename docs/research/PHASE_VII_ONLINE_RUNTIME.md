# Phase VII online runtime

Status: IN_PROGRESS. Current measurements time native association/output on frozen V3/RGB caches; no new complete raw online FPS is claimed.

The disabled replay covers684SN-001frames and6,570SN-015frames, with exact historical prediction-byte parity. Its association/output time is4.79s/49.19s; SN-015 p50/p95is7.35/8.96ms. Joint bounded adds new full-set CPU head work and reaches97.54swith p50/p95=14.63/16.58ms; full learned joint reaches111.92swith p50/p95=16.86/19.46ms. The untrained common cost control takes69.62son SN-015with p50/p95=10.49/11.94ms; even its wrapper currently recomputes observable GRU appearance, so it is not a minimum-overhead threshold implementation. These figures are single-run CPU association measurements, not complete-system speed or a statistically confirmed performance comparison.

Training the small heads used two previously idle A100devices and finished in137.80s(joint)/92.25s(pair). PyTorch peak allocated memory was25.14/25.55MB; nvidia-smi showed about540MiBper process including CUDA context. The allocated model has150,533parameters in both modes, with142,084/83,845participating optimizer parameters. This does not include visual encoders, detector, IR construction or raw decode.

The inherited roughly1.2FPS complete-system bottleneck remains a required investigation. A new raw all-frame profiler must include decode, causal IR construction, preprocessing, V3 detector/batch8, RGB crop/encoding/padding128, memory/model work, assignment, writes and output. Report FPS, p50/p95, memory, parameters, countable operations and incremental overhead on the same complete inputs. Every batching/precision/cache optimization first requires input/output/numerical parity; cached replay speed cannot substitute for ProtocolC. All full raw comparisons and new visual-encoder costs remain pending.

## Sparse visual pilot costs

Sparse extraction produced47,751observations and3.462GiBfeatures plus9.38GBnormalized crop cache. Four frozen-ImageNet adapter fits take about19–24seconds each; two genuine backbone fits take99.23seconds(global)/101.68seconds(learned). Precise per-run seconds, allocated/gradient parameters and peak PyTorch allocated memory are in VISUAL_PILOT_MEASURED. AMP overflow skips and effective optimizer updates are explicitly recorded. These costs exclude full raw video decode, causal IR/V3detector and complete crop encoding; they are training/cache measurements, not online FPS. Only idle physical GPU0–3 were used; other users' GPU4–9 jobs were untouched. Free space remains above40GiB.

Global/local backbone controls now complete all three registered seeds at the first fold, with a real epoch2stop/newprocessresume equivalence proof. Runtime remains training/cache-only; no raw onlineFPS claim. Current free space is about148GiB after the crop/features and checkpoint assets; all our GPU0–3 processes have completed, other users' GPU4–9 work is untouched.

Reference auditing used only bounded source text/CPU/network, with no foreign model execution, weight download or GPU allocation. V-JEPA clip recomputation and CoTracker3 chunkdelay/prefix handling must be measured if adopted; no whole-video cached feature/point prediction may be credited as free strict-online work. Current free space remains about147GiB, and GPU4–9 belong to other live jobs.

DINOv2 frozen47,751crop extraction completed with synchronized per-batch cache encoding timings stored outsideGit. Reading input-cache hashes, encoding, serialization and post-write hashes are separate from fullraw ProtocolC; none of these crop timings is reported as complete detector/onlineMOT FPS. New encoder native video integration/fullrawprofiling remains pending.

DINO supervised firstfold seed0 training loops take55.34/63.86/75.05/73.43seconds for global/simple/capacity/patch attention, excluding data verification, final heldout encoding and diagnostics; peak allocated GPU memory is about0.64–0.66GB. All four use3,551,232trainable backbone parameters; heads175,872/323,329/1,595,714/1,595,714. These cache-based fitting costs are not raw online FPS. Complete decode/IR/detection batch8/RGB crop batch128/memory/assignment/output accounting, same-coordinate producer and full native all-frame timing remain pending.

LocalJEPA v1 adapter/control training and diagnostics run CPU-only with two threads perworker, leaving the four registeredGPU0-3visual workers intact. Each first-seed model completes1200actual CPUupdates; all optimizer/EMA/parameter-change and per-epoch walltime fields are numeric/hash-linked. Cached scene forward-time quantiles are not raw RGB/IR/decode/detection/crop/native-assignment/output FPS. Future inference should compare directcontext evidence with training-onlyprediction overhead separately.

## Current RGB native extraction and first context identity replay

Each native frame decodes only its current RGB, uses unchanged causalV3current proposals and fixed padded128 FP32 encoder calls. Normalized crop pixels use the exact registered FP16 quantization. No future minibatching, annotation, GT box, identity label or teacher enters feature production. The complete four runs have synchronized timings:

| Video / encoder | Frames | RGB decode (s) | Crop/mask (s) | Resize/transfer (s) | Encoder (s) | Elapsed (s) | p50 / p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SN-001 / supervised_DINO_global_embedding384 | 684 | 4.75 | 15.93 | 26.38 | 46.33 | 110.24 | 159.75 / 168.88 |
| SN-001 / frozen_DINO_CLS384 | 684 | 4.81 | 16.48 | 25.98 | 45.90 | 110.44 | 159.94 / 170.67 |
| SN-015 / supervised_DINO_global_embedding384 | 6570 | 43.00 | 137.16 | 245.80 | 432.37 | 1015.89 | 153.50 / 161.39 |
| SN-015 / frozen_DINO_CLS384 | 6570 | 43.88 | 142.41 | 243.39 | 424.40 | 1009.72 | 152.71 / 160.68 |

Peak allocated GPU memory is about0.5GB perproducer, with CUDA context additional. Separate CPU cache replay for V2full6570SN015frames takes249.55s; teacher and prediction decoder are excluded. These are shared-machine single-run stage timings, not isolated repeatable latency estimates, and they omit causalIR/V3generation/detection. No complete raw FPS is claimed. An idle-GPU check is performed twice before native allocation; onlyGPU0–3 are used. Latest space/campaign snapshot supersedes earlier historical free-space and finished-GPU paragraphs.

Allsix27251nativeframe linearUKF andfreshKF association times are measured onunchangedproposals/features/GRU/lifecycle, andallpredictionbytes match. Per-video nativeCPUtimes/quantiles/filtercallcounts are hash-linked in PHASE_VII_LINEAR_UNSCENTED_MOTION_MEASURED.json. These shared-machine single-run cache timings are not a controlled repeatability study or whole RGB/IR/V3pipelineFPS. Ccentered/NULLrepair CPUfits complete28800actualupdates andrealrestart; theirhead-onlyruntime andall18native routes remain separately recorded.

## First completed whole raw current-video profile

FrozenDINO+FIFO SN001684frames nowruns fromactualcurrentRGB: decode, fixedcausalprefixEMAresponse/JPEG95, cu117V3FP16batch8repeatCURRENTimage(nofuturebatch), NMS/transfer, RGBcurrentcrop/pad128FP32encoding, causalnativehistory/association/memory andmeasuredoutput. No cachedIR/proposal/feature/GTlabel/teacher input isusedinference. All684generatedIRpixels andproposalrows/scores areexactreference-equal; maxDINOfeaturecomponenterror1.438e-6; finalpredictionbytes equalcachednativeFIFO. Pinnedofficial HOTA43.023/IDF160.443/SW24.

Measuredcompletealgorithmcriticalpath1.1083FPS, p50/p95889.58/993.90ms; observedwallthroughput including post-inference referencevalidation1.0449FPS. Modelstartup/referencevalidation/progressserialization areoutsidecriticalpath andelapsedwithvalidation isseparatelyreported. DetectorIPC RTT includesprep/batch8/NMS/transfer; nestedworkertime isnotdouble-counted. Single shared-machinerun, norepeatability/allsix/finalmethod claim. Decode4.89s, causalIR281.80s, JPEG11.43s, detectorRTT212.37s, currentcrop20.43s, resize/transfer30.65s, DINO45.65s, featuretransfer0.30s, association6.24s, output0.30s. IRgeneration isabout45.7% anddetectorabout34.4% ofthiscriticalpath; association isnotdominant. The response isRGB-derived, not an independentIRsensor.

OriginalRGBbaselinev1 stops atframe681 duringreferencevalidation only: itsold per-video finalcachebatch38 differsnumericallyfromcurrentpad128. Samecurrentpixels/zerootherpositions atbatch38 exactlyreconstruct alloldtailvalues (zeroerror); pad128vsoldtailcosineerror≤1.49e-6. Current-only shapevalidation isseparateandneverfeeds runtimefeatures/tracker. Sourcev1failure ispreserved; ownedv2revalidates withoutaltering thealgorithm/padding/checkpoint/threshold. Allsixrawlegacy/frozenDINOcampaigns areactive, outputs andimage/feature/source/runtimeSHA outsideGit. Finalallvideo/3seed/methodcomparison andisolatedrepeatability remainrequired.

## Both complete raw SN001 pipelines verified

LegacyRGB/GRU ownedv2 nowcompletes all684frames, withallcurrentIR/proposal bytesmatchingreferences, current-only cache-tail shapevalidation recorded, andfinalpredictionbytes equaloriginalbaseline. FrozenDINO/FIFOall684frames also byte-equal itsreference. Allreferences arevalidation-only aftercurrentinference/output; input iscurrentrawRGBplusstrictlypast causalbackground. No GT/teacher/cachedpixels/proposals/features feedonline decisions.

| Pipeline | Completealgorithm FPS | Observed FPS withvalidation | p50 / p95 (ms) | HOTA | IDF1 |
|---|---:|---:|---:|---:|---:|
| legacy_gru | 1.2447 | 1.1710 | 793.60 / 865.10 | 46.238 | 65.251 |
| dino_frozen_fifo | 1.1083 | 1.0449 | 889.58 / 993.90 | 43.023 | 60.443 |

Timing includesactualdecode/causalIR/JPEG/IPC+prep+batch8V3+NMS/currentRGBcrop+pad128encode/association+memory/output. Startup/referencevalidation/progressserialization excludedfromalgorithmcriticalpath, separatelyincludedobservedelapsed. Nestedworkertiming notdouble-counted. Single shared-machine runs, notisolatedrepeatablebenchmark or matchedencoder-training-exposure superiority comparison. Originalroughly1.2FPS is reproduced; IRgeneration anddetector dominate. Allsixrawlegacy/frozen profilescontinue (firstSN015active), finalmethod/allseed/repeatedprofiling stillrequired. Detailedallstages/runtime/input/output/model/sourceSHA andlegacyv1preservedfailure are in PHASE_VII_WHOLE_RAW_BOTH_FIRST_VIDEO_MEASURED.json.

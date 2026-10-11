# Phase I modality identity diagnosis

Status: RGB/causal-IR three-seed training, offline paired diagnosis and canonical dev proposal features complete. Tracking baseline grids and evidence-directed representation repair are in progress. No constructed-IR identity contribution is established.

Predeclared comparisons and interventions are in ARCHITECTURE_DESIGN.md. Local pretrained candidates found: ImageNet `resnet50-0676ba61.pth` and person OSNet `osnet_x1_0_market1501.pth`. They are not DINOv3 or fish-trained weights. Their hashes, source preprocessing and permitted usage must be recorded before extraction. Generic frozen features precede fish fine-tuning.

Paired GT identity indexing and frozen shared ImageNet ResNet50 extraction are complete, including video namespaces, lifespan-uniform sampling and exclusion of four ambiguous SN-013 annotation boxes under protocol v2. 14,528 pairs: 11,967 train / 2,561 dev, 98 train identity labels, 2,048-dimensional normalized features, 128×256 letterboxed crops. Extraction takes 40.018s, batch encoder p50/p95 30.073/32.884ms at batch 64 pairs; this excludes online detector-crop extraction and is not per-frame latency. Feature-to-record alignment and BGR→RGB crop order are checked. Weight and input/feature hashes are recorded in `/data3/liuyeqiang/IR-MFT_assets/cfp/phase1/frozen_resnet50/manifest.json`.

## First diagnostic evidence

Frozen sampler: `configs/cfp/feature_diagnosis_v1.json`, chosen before outcomes. Nearest historical GT-labeled sample must be at least 25 frames before the query. Novel IDs are excluded until they have a gallery positive; 1,260 eligible queries per sequence. The gallery uses oracle identity labels, never interpreted as deployed tracking. Balanced offline pair AUC uses 5,000 temporally separated positive and 5,000 negative pairs per video, identity-uniform sampling, seed 0.

| Video | RGB Recall@1 | IR Recall@1 | Equal fusion Recall@1 | RGB pair AUC | IR pair AUC | Fusion pair AUC |
|---|---:|---:|---:|---:|---:|---:|
| BT-005 | 21.587% | 19.365% | 24.444% | 0.520826 | 0.515682 | 0.521277 |
| SN-013 | 46.746% | 28.175% | 43.175% | 0.585700 | 0.544895 | 0.588943 |

Equal fusion corrects 121/hurts 85 RGB decisions on BT-005; corrects 130/hurts 175 on SN-013. Pair AUC slightly improves on both, while nearest-gallery retrieval harms one video: aggregate pair separability does not ensure association benefit. Wrong-paired IR decreases fusion retrieval to 17.698%/32.698%; normalized noise IR gives 18.016%/40.079%. Zero IR and repeated RGB algebraically reproduce RGB cosine scores and measured results. Corruptions are offline feature interventions. These support further conditional-usefulness diagnosis, not a proven utility estimator. Frozen RGB pair discrimination is weak; a strong fish-trained RGB ReID baseline and matched causal-history input controls are essential before CMUE conclusions.

Artifacts: `.../phase1/frozen_resnet50/dev_diagnosis.json`, `frozen_resnet50_dev_diagnosis.log`. No historical test labels were used. Original legacy IR includes future-derived backgrounds; gallery vectors precede queries, but oracle sampling/labels remain an offline diagnostic. Canonical causal tracking results are reported separately in ABLATION_RESULTS.md.

## Declared fish ReID baseline

`configs/cfp/reid_baseline_v1.json` freezes a ResNet50/512D projection/BN neck, CE with 0.1 label smoothing plus normalized Euclidean batch-hard triplet, margin 0.3. A batch contains eight uniform composite train identities × eight lifespan strata. Triplet positives need a 25-frame gap; negatives must be another identity from the same video. Anchors without both are omitted from triplet while retained for CE, and valid-anchor fraction is reported. All backbone parameters are fine-tuned with AdamW backbone/head LR 3e-5/3e-4, one-epoch warmup, cosine schedule, flip/erasing augmentation, 12 fixed epochs and seeds 0/1/2. The last checkpoint is fixed before outcomes, with no retrieval-based selection; final tracker selection remains dev HOTA/AssA/IDF1.

RGB cache/reader never opens an IR file. Causal IR uses the same budget only after full causal generation; every cache/encoder manifest carries input/split/source/weight hashes. Pre-normalized float16 GT-crop caches are rebuildable acceleration artifacts, not new data or detector boxes. RGB-only training is now complete for all three seeds under commit `a692b300dcdfd2dc549226c1db61eb4a6e8eb0d5`. Stronger encoder/candidate changes must be declared as separate experiments instead of silent baseline weakening.

| Representation | BT-005 oracle Recall@1 | SN-013 oracle Recall@1 |
|---|---:|---:|
| Frozen ImageNet RGB | 21.587% | 46.746% |
| Fish RGB seed 0, fixed epoch 12 | 30.635% | 51.905% |
| Fish RGB seed 1, fixed epoch 12 | 28.492% | 49.127% |
| Fish RGB seed 2, fixed epoch 12 | 28.333% | 49.603% |
| Fish RGB three-seed mean ± sample std | 29.153 ± 1.285% | 50.212 ± 1.485% |

Each video has 1,260 eligible queries, same index and temporal gap as frozen diagnosis. All original development labels remain -1 for classifier fitting. These are current-frame RGB models and oracle-history representation tests, independent of legacy IR. Training/feature export times per seed: 178.03 / 172.64 / 174.61 seconds (check individual manifests for full precision). Inference batch 128 latency is recorded separately and excludes raw decode/detector/preprocessing/tracker. No seed was selected as a winner; downstream seed 0 is the predeclared primary implementation seed, followed by all three-seed comparisons. Crop normalization is rounded to float16 only in the training cache; raw detector-crop inference uses the shared crop function directly, and crop/domain shift remains to quantify.

Report earlier-sample gallery retrieval and offline pair discrimination separately from detector-crop tracking. The diagnostic index uses oracle whole-lifespan uniform sampling and historical GT identity labels; earlier gallery timestamps alone do not make this a deployed causal evaluator. Legacy IR diagnosis cannot validate a strictly online system. Final comparisons require current/past-only IR, matched available RGB history and contrast/background controls. Training IDs are video composite keys, temporal gaps and same-frame hard negatives retained. Correct/zero/repeated-RGB/noisy/wrong-paired IR interventions are required under identical candidates. Establish IR conditional identity value before CMUE claims. No test labels used for fitting or selection.

## Trained causal paired identity diagnosis

All three IR encoders use identical 12-epoch budgets and seeds, with the same composite-ID crop index and only causal constructed inputs. Training/export times: 162.30 / 159.89 / 160.34 seconds. Paired proposal feature banks cover all 51,286 dev detections with zero invalid RGB/IR crops; detector boxes, scores and order are unchanged. Bank hashes, weights and encoding scopes are in `phase1/causal_dev_detection_features_seed*/manifest.json`.

| Seed | BT-005 RGB / IR / fusion Recall@1 | SN-013 RGB / IR / fusion Recall@1 | BT fusion corrections / harms | SN fusion corrections / harms |
|---|---:|---:|---:|---:|
| 0 | 30.635 / 18.413 / 28.651% | 51.905 / 25.079 / 49.444% | 106 / 131 | 138 / 169 |
| 1 | 28.492 / 18.413 / 26.984% | 49.127 / 25.952 / 49.683% | 113 / 132 | 150 / 143 |
| 2 | 28.333 / 17.698 / 28.651% | 49.603 / 24.127 / 46.349% | 139 / 135 | 146 / 187 |

Same 1,260 eligible offline oracle queries per video. Wrong correspondence and normalized noise reduce retrieval; zero IR and repeated same RGB reproduce RGB retrieval exactly. These interventions are offline feature manipulations, not a causal deployed wrong-pair acquisition. Results and pair AUCs are machine-generated in `phase1/fish_paired_causal_seed*/dev_diagnosis.json` and registered. Both beneficial and harmful fusion events occur, while mean fusion remains below RGB on both videos; this motivates conditional diagnosis without proving a utility estimator.

Pre-neck detector features have total per-dimension variance summed near 0.01 and first ten distinct proposal cosines above 0.98 on BT-005. Triplet remains near margin 0.3 while train classifier accuracy reaches about 90%. Default appearance runs across RGB/IR/fusion and seeds produce identical trajectories. `configs/cfp/reid_repair_v1.json` declares normalized post-BN-neck inference using unchanged weights before extracting or seeing its outcomes. A constant identical-vector control isolates geometry/acceptance changes. Original configurations remain preserved; a representation repair is not a scientific contribution.

Matched controls are frozen in `modality_controls_v1.json`: independent RGB seed pairs (0,1)/(1,2)/(2,0), single-frame Lab CLAHE and same causal RGB-background difference. Independent RGB banks are composed from actual measured features. The two derived view datasets are being built; their IR storage column is explicitly marked as an RGB auxiliary control, never a thermal input.

# Phase I modality identity diagnosis

Status: FROZEN_FEATURE_DIAGNOSIS_COMPLETE (legacy exploratory inputs); causal input rebuild and fish ReID training remain pending. No new tracking gain has been measured.

Predeclared comparisons and interventions are in ARCHITECTURE_DESIGN.md. Local pretrained candidates found: ImageNet `resnet50-0676ba61.pth` and person OSNet `osnet_x1_0_market1501.pth`. They are not DINOv3 or fish-trained weights. Their hashes, source preprocessing and permitted usage must be recorded before extraction. Generic frozen features precede fish fine-tuning.

Paired GT identity indexing and frozen shared ImageNet ResNet50 extraction are complete, including video namespaces, lifespan-uniform sampling and exclusion of four ambiguous SN-013 annotation boxes under protocol v2. 14,528 pairs: 11,967 train / 2,561 dev, 98 train identity labels, 2,048-dimensional normalized features, 128×256 letterboxed crops. Extraction takes 40.018s, batch encoder p50/p95 30.073/32.884ms at batch 64 pairs; this excludes online detector-crop extraction and is not per-frame latency. Feature-to-record alignment and BGR→RGB crop order are checked. Weight and input/feature hashes are recorded in `/data3/liuyeqiang/IR-MFT_assets/cfp/phase1/frozen_resnet50/manifest.json`.

## First diagnostic evidence

Frozen sampler: `configs/cfp/feature_diagnosis_v1.json`, chosen before outcomes. Nearest historical GT-labeled sample must be at least 25 frames before the query. Novel IDs are excluded until they have a gallery positive; 1,260 eligible queries per sequence. The gallery uses oracle identity labels, never interpreted as deployed tracking. Balanced offline pair AUC uses 5,000 temporally separated positive and 5,000 negative pairs per video, identity-uniform sampling, seed 0.

| Video | RGB Recall@1 | IR Recall@1 | Equal fusion Recall@1 | RGB pair AUC | IR pair AUC | Fusion pair AUC |
|---|---:|---:|---:|---:|---:|---:|
| BT-005 | 21.587% | 19.365% | 24.444% | 0.520826 | 0.515682 | 0.521277 |
| SN-013 | 46.746% | 28.175% | 43.175% | 0.585700 | 0.544895 | 0.588943 |

Equal fusion corrects 121/hurts 85 RGB decisions on BT-005; corrects 130/hurts 175 on SN-013. Pair AUC slightly improves on both, while nearest-gallery retrieval harms one video: aggregate pair separability does not ensure association benefit. Wrong-paired IR decreases fusion retrieval to 17.698%/32.698%; normalized noise IR gives 18.016%/40.079%. Zero IR and repeated RGB algebraically reproduce RGB cosine scores and measured results. Corruptions are offline feature interventions. These support further conditional-usefulness diagnosis, not a proven utility estimator. Frozen RGB pair discrimination is weak; a strong fish-trained RGB ReID baseline and matched causal-history input controls are essential before CMUE conclusions.

Artifacts: `.../phase1/frozen_resnet50/dev_diagnosis.json`, `frozen_resnet50_dev_diagnosis.log`. No historical test labels were used. Original legacy IR includes future-derived backgrounds; only the gallery schedule is causal here. All tracking HOTA/AssA/IDF1 measurements for new identity modules remain pending.

## Declared fish ReID baseline

`configs/cfp/reid_baseline_v1.json` freezes a ResNet50/512D projection/BN neck, CE with 0.1 label smoothing plus normalized Euclidean batch-hard triplet, margin 0.3. A batch contains eight uniform composite train identities × eight lifespan strata. Triplet positives need a 25-frame gap; negatives must be another identity from the same video. Anchors without both are omitted from triplet while retained for CE, and valid-anchor fraction is reported. All backbone parameters are fine-tuned with AdamW backbone/head LR 3e-5/3e-4, one-epoch warmup, cosine schedule, flip/erasing augmentation, 12 fixed epochs and seeds 0/1/2. The last checkpoint is fixed before outcomes, with no retrieval-based selection; final tracker selection remains dev HOTA/AssA/IDF1.

RGB cache/reader never opens an IR file. Causal IR uses the same budget only after full causal generation; every cache/encoder manifest carries input/split/source/weight hashes. Pre-normalized float16 GT-crop caches are rebuildable acceleration artifacts, not new data or detector boxes. This baseline is implemented and infrastructure-tested; training and its dev outcomes are pending at this commit. Stronger encoder/candidate changes must be declared as separate experiments instead of silent baseline weakening.

Report causal gallery retrieval and offline pair discrimination separately from detector-crop tracking. Legacy whole-sequence median IR is explicitly noncausal; exploratory legacy diagnosis cannot validate the strictly online system. Final comparisons require current/past-only IR, matched available RGB history and contrast/background controls. Training IDs are video composite keys, temporal gaps and same-frame hard negatives retained. Correct/zero/repeated-RGB/noisy/wrong-paired IR interventions are required under identical candidates. Establish IR conditional identity value before CMUE claims. No test labels used for fitting or selection.

# Phase VII local identity evidence

Status: IN_PROGRESS. No trained local identity improvement is claimed yet.

The full two-crop index covers27,251frames,217,201valid nonambiguous GT observations and258,309actual V3 proposals across98physical(video,fish)identities. It preserves every detector proposal and marks independently unverified physical identities as unknown. GT/current-detector pairs are joined by(video,frame,known fish), with coverage and duplicate counts separately saved. Arrays and frame paths remain outside Git at `/data3/liuyeqiang/IR-MFT_assets/cfp/phase7/identity_two_crop_index_v1`; compact exact hashes/counts are recorded in `PHASE_VII_TWO_CROP_INDEX_MEASURED.json`. The index is all-frame; the subsequent sparse feature/training pilot covers47,751observations and is reported below.

Local token provenance is being audited before training. DINOv3 Small uses patch16/dim384 with four storage tokens; the inherited128×256crop is divisible by16and has128spatial patches. DINOv2 Small uses patch14/dim384; direct source asserts divisibility, so128×256cannot be fed unchanged. A consistent declared padding/resizing route and effective letterbox visibility are required. Neither spatial tokens nor K=4/8 learned slots may be labeled anatomical head/body/tail without annotations.

The next controlled encoder experiments compare global CLS/pooled features, spatial token mean, simple K=4/8aggregation and learned local identity features on identical GT/detector pairs and temporal candidate pools. Known same-video/same-frame different-fish negatives are required. New fish encoder/adapter fitting excludes both distinct outer and inner calibration videos. Existing FishReID/backbone/GRU results remain exposed references. Any new coordinate requires a matched ordinary producer; the old512GRU is not a valid384-dimensional baseline. Actual occlusion/turning labels are absent, so gap/box overlap/shape-change analysis must retain explicit proxy names.

One licensed official DINOv2-S backbone acquisition has been attempted with a100MBcap. TLS failed; no DINOv2 pretrained crop result, FishMAE reproduction or pretrained transformer local result is reported; independent ImageNet strip learning has now run. Selected weights, executable forward parity and all30semantic source audits remain pending alongside this foundation work.

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

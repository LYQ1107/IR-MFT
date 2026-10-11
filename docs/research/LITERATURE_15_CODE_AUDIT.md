# Code audit of 15 references

Audit date: 2026-10-12. Sources reside outside this repository at `/data3/liuyeqiang/CFP-FishMOT_references/`. Commits below identify inspected code, not an assertion that all reused repositories were freshly updated. Availability means source inspection, not successfully reproduced training. Exact file/license hashes are in `configs/cfp/reference_sources.json`. No external implementation has been copied into CFP. Published result tables are not comparable measurements on our detections.

## 1. SU-T

**When Trackers Date Fish: A Benchmark and Framework for Underwater Multiple Fish Tracking**, AAAI 2026. [Repository](https://github.com/vranlee/SU-T), commit `479be89a27129f6879c62710a83abac6a8b261ee`. MIT root; bundled FastReID has its own notices. Detector training and tracking entry points are present, fish ReID weights are external.

Inspected `tools/train.py`, `exps/SU-T-ReID.py`, `trackers/SUT_UKF.py`, `SUT_reid_UKF.py`, `association.py`, `UKF.py`, `fast_reid/fast_reid_interfece.py`. YOLOX input B×3×800×1440; one-class xyxy/conf predictions. `SU_T_ReID.update` takes Nx5 boxes/scores and optional NxD embeddings; returns boxes with IDs. UKF, fish box geometry/height-modulated overlap and four-corner velocity drive multi-stage association; appearance adds current/EMA and long-term embeddings. ReID interface crops/clips image boxes, converts color, normalizes output embeddings; configured crop size comes from FastReID. Config is YOLOX-X depth 1.33/width 1.25, SGD detector training, Mosaic/InfiniteSampler and YOLOX IoU/objectness/class losses; FastReID is a separate identity training pipeline, not a detector loss. Frame-level MOT annotations supply detector boxes; track IDs supply crops. Eval consumes MOT files.

Borrow fish-aware hard negatives and geometric cues; do not treat nonrigid box velocity as an identity predictor. Heavy detector and SBS-S101 pretrained fish ReID weights add cost. Overlap with CFP: robust association/long-term memory, but no measured marginal utility of constructed IR and no target-level latent prediction.

## 2. TIDE

**When Fish Look Alike: Tracking Identities with Dual-branch Elasticity**, 2026. [Repository](https://github.com/vranlee/TIDE), commit `fba27aefd1da11bfc7f946240dd66d85ea02a39c`. MIT. Training and inference source present; pretrained backbones/data external.

`src/train.py` uses Adam and JointDataset/DataLoader. `models/tidel.py` dispatches to `models/yolo.py:PoseL`; `tides.py` dispatches to `networks/fastvit.py`. Both take B×3×H×W and return a list of dictionaries of dense heads hm/wh/reg/id (and configured density head). `trains/mot.py:MotLoss` has focal heatmaps or configured MSE, box/offset L1 and CE/focal identity classification. `CMotLoss` additionally couples density-map MSE/SSIM and detection/density consistency with learned uncertainty weights. Imported TripletLoss is not evidence it is active in these inspected forwards. `datasets/dataset/jde.py:JointDataset` reads images with normalized bbox/ID labels and offsets IDs by dataset; CFP instead must key IDs by video. `tracker/matching.py` AGCv1/v2 and `mc_bot_sort.py`/`multitracker.py` perform geometric/embedding association, evaluation uses MOT result files.

Borrow same-frame difficult negatives, nonrigid geometry and branch comparisons. Density head may help crowded detection but changes Protocol A's question. No counterfactual modality supervision; counting consistency is distinct from reliable identity writes. CUDA/DCN and branch-specific checkpoints complicate direct reproduction.

## 3. M4FT

**M4FT: Mamba, Migratory, Mobile and Multiple Fish Tracking**, TCSVT 2026. [Repository](https://github.com/vranlee/M-4-FT), commit `b120e568c07145891e043675892890ed9fe9279d`. MIT. Detector/train/tracker source present, selective-scan compiled dependencies and weights external.

`yolox/models/m4ft.py` defines PatchEmbed2D/PatchMerging, SS2D, ConvSSM1, YOLOPAFPN and YOLOXHead; `yolox/models/yolox.py` combines backbone/head. Input B×3×H×W; internal channel-last state-space maps return to pyramid maps; head produces detector predictions or IoU/objectness/class/L1 losses. `tools/train.py`, `exps/M4FT_exps.py` configure depth .67/width .75, 800×1440, 80 epochs, YOLOX SGD, InfiniteSampler/Mosaic/MixUp and BrackishMOT-M4FT JSON boxes. Tracking is detection-based geometric association (Byte/SORT options), not an identity contrastive model. MOT metrics scripts evaluate outputs.

Borrow efficiency discipline, not a new detector during association diagnosis. Spatial Mamba is not per-track predictive identity memory. Custom `selective_scan_cuda*` dependencies constrain deployment; FLOPs may omit scan kernels unless explicitly supported. Reported paper costs are not our measured costs.

## 4. DM3T

**DM3T: Harmonizing Modalities via Diffusion for Multi-Object Tracking**, CVPR Findings 2026. [Repository](https://github.com/vranlee/DM-3-T), commit `662790ba213b4460f19a1bdbefb9714416eb2b5c`. Apache-2.0. Training/inference source present; RGB-T data/backbone checkpoints/DCN external.

`src/main.py` selects Adam or SGD; `lib/trainer.py:ModleWithLoss` passes visible/IR current images, previous paired images and previous heatmap. `networks/dla_rgbt_main.py:DLA.forward` shares base convolutions and fuses B×C×H×W features before DLA stages. `CrossModalDiffusionFusion` adds random Gaussian noise at each step, concatenates guidance from the other modality and applies residual conv denoisers, returning summed refined features. Noise is also generated in eval (no train-only condition). `DLASeg` has an additional optional DiffusionModule. `tf_fusion_refined.py` contains attention fusion, but existence does not establish it is the active default. GenericLoss sums focal center and configured weighted L1/regression heads; inspected trainer contains no distinct diffusion denoising supervision. Dataset samples paired current/previous frames in COCO-style tracking format. Tracker matches decoded centers/displacements; MOT evaluation consumes predicted trajectories.

Borrow explicit modality/temporal ablations and seed control. This is feature refinement for RGB-T detection, not a model that generates our IR images, and does not justify claiming independent thermal information. Stochastic eval and DCN must be audited in any reproduction. Counterfactual identity utility is distinct from noisy feature fusion.

## 5. TFMFT

**TFMFT: Transformer-based Multiple Fish Tracking**, Computers and Electronics in Agriculture 2024. [Repository](https://github.com/vranlee/TFMFT), commit `d4fbcc0d4df685cc01f5f5d7c9510601a1c9e5b2`. MIT. **Partial source**: only `tf_track.py`, `tracker_multi.py`, `eval_motchallenge.py` Python files. Imported `models`, `engine`, `util` and dataset packages are absent.

`tf_track.py` references DETR-like images/detections and training optimizer groups (SGD or AdamW). Model/criterion/data construction cannot be inspected in this release, so exact backbone shapes, loss and sampler are unavailable; no verified full training command exists. `tracker_multi.py:Multi_Tracker` takes candidate boxes/scores and uses high/mid/low stages with overlap/GIoU and track state. `eval_motchallenge.py` evaluates MOT outputs. MFT22 is the associated dataset; missing training implementation blocks fair full reproduction.

Borrow recovery-stage reasoning, not a presumed released Transformer predictor. Association stages overlap with baseline engineering; they do not implement CMUE or predictive latent identity. Keep unavailable details explicit instead of filling them from DETR conventions.

## 6. PFTrack

**Visible-Thermal Multiple Object Tracking: Large-scale Video Dataset and Progressive Fusion Approach**, 2024. [Repository](https://github.com/wqw123wqw/PFTrack), commit `73d4dc4c5e69e530f7a77d974bff7d2040720728`. MIT. Training/evaluation source present, real VT-MOT data/DLA weights/DCN external.

`src/main.py` uses Adam or SGD. `lib/model/networks/dla_V3_M.py:DLA.forward` embeds current/previous RGB/thermal images and pre_hm; converts feature maps B×C×H×W to spatial tokens, applies temporal fusion per modality then `transformer_fusion_v3.py` cross-modal attention, reshapes/interpolates and residual-adds RGB/T before DLA stages. `base_model.py` returns dense hm/wh/reg/tracking head dictionaries. `lib/trainer.py:GenericLoss` uses focal centers and configured L1/weighted regression including displacement; `generic_dataset_rgbt.py` creates current/prior pairs and GT center/offset targets. Detector/tracker decode centers, displacement and associate online; TrackEval is included in historical assets.

Borrow provenance-aware synchronized pairing and staged fusion controls. Real sensor thermal complementarity cannot be transferred as an assumption to constructed IR. Dense spatiotemporal detection fusion alters Protocol B, while CFP estimates candidate-level identity gains under fixed detections. Multiple frame maps/tokens and DCN are deployment costs.

## 7. MOTIP

**Multiple Object Tracking as ID Prediction**, CVPR 2025. [Repository](https://github.com/MCG-NJU/MOTIP), commit `ffc0e905ac196a603027eca8d18fb0dff48c8bcc` (reused checkout). Apache-2.0. Training/inference present; COCO detector and dataset detector pretraining required.

Inspected `models/motip/id_decoder.py`, `id_criterion.py`, `train.py`, `docs/GET_STARTED.md`. IDDecoder takes trajectory/current tensors B×G×T×N×C with labels/masks/times. One-hot identity vocabulary K+1 is projected and concatenated with features; current detections self-attend and cross-attend to history with relative time positions. Mask `trajectory_times >= unknown_times` blocks present/future historical keys. Output logits target vocabulary IDs; **K+1 denotes newborn**, not two independent NULL/newborn tokens. Runtime maps vocabulary labels to persistent tracks and handles unmatched/lost allocation. IDCriterion removes padded positions and uses CE or focal classification; detection training uses DETR set losses. AdamW with temporal annotated clips and curriculum sampler trains model; validation uses full videos/MOT files.

Borrow causal context masks, complete candidate-label handling and ID-prediction controls on our fixed boxes. Do not rebuild DETR initially or confuse reusable vocabulary labels with stable output IDs. Strong overlap: learned history-to-current identity association; novelty cannot be merely an ID decoder. CFP must demonstrate separate conditional constructed-modality utility and predictive-memory gains.

## 8. TCEI

**Dual-level Adaptation for Multi-Object Tracking: Building Test-Time Calibration from Experience and Intuition**, CVPR 2026. [Repository](https://github.com/1941Zpf/TCEI), commit `145d1b8431398156f8d9f854430e306fdee39eaa`. **No root license found**; no source copied. Inference and checkpoint links present, no released training entry found; do not invent one.

`TCEI/tcei.py` takes logits/embeddings and ID caches, computes softmax entropy, stores low-entropy exemplars up to shot capacity and adjusts logits from positive/negative cache affinities. `submit_and_evaluate.py` integrates MOTIP-like inference, config data and video evaluation. Training losses/optimizer are not exposed as a complete pipeline. Audited edge cases: entropy uses natural log but normalization uses log2(K), K=1 divides by zero; narrow-cache min-max affinity has no epsilon; hardcoded CUDA and unknown-label sentinel 50 may fail small vocabularies. These are code-level risks, not measured tracking failures here. Persistent experiential caches across videos require reset/protocol disclosure.

Borrow observable uncertainty and controlled exemplar writes, reimplement independently with finite-value tests. Experience calibration strongly overlaps safe memory; entropy alone does not establish counterfactual IR benefit. Our data must not inherit cross-video identity caches or test adaptation inconsistent with baselines.

## 9. GOT-JEPA

**GOT-JEPA: Generic Object Tracking with Model Adaptation and Occlusion Handling using Joint-Embedding Predictive Architecture**, TCSVT 2026. [Repository](https://github.com/chenshihfang/GOT). Current inspected commit `5f382645ccb7331280a7bf07f3c5709d1072caf7` now promotes GOT-Edit. Also fetched author-linked paper commit `84e9324317e4afe62c06b2c51a97563f79730a2e` into separate `GOT-JEPA-paper`. Root AGPL-3.0, nested pytracking GPL-3.0; no implementation copied.

`pytracking/ltr/models/tracking/tompnet_JEPA.py:ToMPnet` takes history T×B×3×H×W, target 1×B×3×H×W, history boxes; returns context/expanded/predicted/frozen-target cls/reg filters B×C×1×1. DINOv2 features, masked/relocated target features, frozen target head and two predictors are explicit. `train_settings/tomp/GOT_JEPA_378_pretrain.py` samples LaSOT/GOT10k/TrackingNet/MSCOCO, two context/one target, max gap 200, 432 crops, AdamW and MultiStepLR; finetuning config adds downstream GIoU/LBHinge losses. Actor `ToMPActor_JEPAs1_vicregExpwovc400x_norm2_cexp` uses smooth-L1 prediction and expanded covariance, with duplicate overriding definitions. **Both inspected releases have an incomplete active final __call__**, ending at an unfinished expression rather than returning the earlier loss/stats; source is not verified as a complete runnable training path.

Borrow stop-gradient predictive targets and collapse diagnostics. This is SOT filter prediction with an object template, not MOT candidate identity. Target-frame masked features in training are not a causal MOT history predictor at inference. Significant conceptual overlap requires honest comparisons; heavyweight DINO/search crops/online adaptation are not imported wholesale.

## 10. HybridSORT

**Hybrid-SORT: Weak Cues Matter for Online Multi-Object Tracking**, AAAI 2024. [Repository](https://github.com/ymzis69/HybridSORT), commit `396f8d30db13304c0cbaf1dcf2e16ded93ce1701`. MIT root, FastReID separate license. Detector/ReID/train/tracker source present, dataset weights external.

`trackers/hybrid_sort_tracker/hybrid_sort_reid.py:Hybrid_Sort_ReID.update` accepts Nx5 detections (or YOLOX Nx7), optional aligned NxD embeddings and warp. `KalmanBoxTracker` uses 9-state confidence-aware motion, four-corner velocities, high/low confidence matching and observation-centric recovery. `association.py` combines these cues with appearance and long-term correction. `update_features` L2-normalizes, EMA/adaptive confidence mixes and appends bounded feature deque; no epsilon for zero norm. YOLOX training via `tools/train.py` uses SGD/detection losses/InfiniteSampler; nested FastReID trains crop identities separately with CE/metric losses. MOT evaluation consumes online files.

Use the actual upstream association API with identical frozen detections and aligned fish embeddings, not rename a new cosine tracker as HybridSORT. Baseline memory and confidence adaptation overlap CFP safe writes; prediction must beat them independently. Optional ECC and appearance cost must be reported; feature buffering is not learned future identity prediction.

## 11. BoT-SORT

**BoT-SORT: Robust Associations Multi-Pedestrian Tracking**, 2022. [Repository](https://github.com/NirAharon/BoT-SORT), commit `251985436d6712aaf682aaaf5f71edb4987224bd` (reused). MIT root/FastReID Apache-2.0. Full tracking/YOLOX/FastReID training sources present; pretrained data/weights external.

`tracker/bot_sort.py:BoTSORT.update` takes detector arrays/images. STrack holds box state and normalized current/EMA features; Kalman uses xywh geometry, GMC compensates camera. First matching takes min(score-fused IoU distance, cosine distance/2) with proximity/appearance rejection; low-score stage uses geometry. `fast_reid/fast_reid_interfece.py` batches crop features; `tracker/gmc.py` provides camera transformations. Detector SGD/YOLOX and ReID CE/triplet are separate training with annotated boxes vs crop IDs/PK sampling. Video evaluator produces MOT files.

Strong Protocol A baseline with our same embeddings. Audit GMC setting and preprocessing/candidate thresholds fairly on dev. Do not interpret EMA appearance as PLIM. Camera compensation may be unnecessary in fixed underwater cameras, but removing it is a declared baseline option, not a hidden handicap.

## 12. ByteTrack

**ByteTrack: Multi-Object Tracking by Associating Every Detection Box**, ECCV 2022. [Repository](https://github.com/FoundationVision/ByteTrack), commit `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`. MIT. Detector/tracker training/evaluation present.

`yolox/tracker/byte_tracker.py:BYTETracker.update` takes Nx5 or Nx7 detections and image sizes, rescales boxes if required, separates high/low scores. STrack/Kalman predict geometry; score-fused IoU first assignment, low-score geometric second assignment, unconfirmed/newborn/lost handling produce active tracks. No appearance encoder or identity loss. `tools/train.py`/YOLOX experiment files train detector with SGD, InfiniteSampler/Mosaic/MixUp and IoU/objectness/class/L1 terms; MOTDataset supplies frames/boxes. Evaluation streams frames and writes trajectories.

Essential exact fallback. Historical implementation already lives in IR-MFT; do not overwrite it from newer upstream. Candidate recall, low-confidence rescue and lifecycle are important controls; RGB/IR memory is an added mechanism, not a claim that ByteTrack performs ReID.

## 13. FastReID

**FastReID: A Pytorch Toolbox for General Instance Re-identification**, 2020. [Repository](https://github.com/JDAI-CV/fast-reid), commit `c9bc3ceb2f7a6438b62fb515ea3df6d1e999e95d` (reused). Apache-2.0. Complete training/evaluation framework, dataset/checkpoint downloads external.

`fastreid/modeling/meta_arch/baseline.py:Baseline.forward` takes B×3×H×W image tensors or dict with targets, preprocesses pixel mean/std, runs backbone and configurable embedding/classifier head. Train returns configured losses, eval returns B×D features. Configurable CE, hard-mined triplet, circle and cosface are alternatives, not all compulsory. `tools/train_net.py`, `engine/defaults.py`, `data/samplers`, `solver/build.py` implement training, PK identity batches and Adam(default)/SGD options. Backbone can initialize from ImageNet; checkpoint/person domain must be recorded. Query/gallery retrieval evaluator reports rank/mAP independently of MOT.

Borrow proper identity batches, normalization and strong metric baseline. Crop aspect ratios for pedestrians need fish-domain audit; zero-norm features require guarded normalization. Shared crop encoder does not itself establish constructed-modality utility; parameter and training-data matching are essential.

## 14. V-JEPA 2

**V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning**, 2025. [Repository](https://github.com/facebookresearch/vjepa2), commit `204698b45b3712590f06245fbfba32d3be539812` (current tree also contains 2.1). MIT code; weights have separate distribution terms.

`app/vjepa/train.py` encodes clips B×3×T×H×W with online masked video encoder and frozen EMA target encoder. `src/models/predictor.py:VisionTransformerPredictor` maps context tokens B×Nc×D plus context/target mask indices to B×Nt×D latent targets (default D768/predictor384). Loss averages abs(pred−stopgrad(target))^loss_exp/loss_exp. Target weights momentum-update after online optimizer. `app/vjepa/utils.py` optimizer construction and video dataset/multiblock mask generators configure AdamW, clip duration/fps/frame-step sampling and distributed weighted sampler. `configs/train/vitl16/pretrain-256px-16f.yaml` has 16 nodes×8 GPUs, 256 crops/16 frames and large video mixture; full pretraining is outside project resources. Evaluation probes representation; not a MOT tracker.

Borrow causal target-level prediction, stop-gradient and collapse checks. Masks/video pretraining can include bidirectional temporal context; our online predictor must use strictly historical fish observations. No independent modality utility or identity assignment is supplied.

## 15. DINOv3

**DINOv3**, 2025. [Repository](https://github.com/facebookresearch/dinov3), commit `6876159a11b4df116f30f667f8c9888617df0751`. Custom **DINOv3 License**, not MIT; research acknowledgment and distribution terms are in LICENSE.md. No code/weights redistributed here. Source acquired after TLS retries; pretrained weights require access/download, not yet locally verified.

`dinov3/models/vision_transformer.py:DinoVisionTransformer` takes B×3×H×W with patch masks. `forward_features` returns CLS B×D, storage B×R×D and patch tokens B×(H/16·W/16)×D; ViT-S/16 uses D384. `hub/backbones.py` loads exact architecture/checkpoint. `train/train.py`, `ssl_meta_arch.py` implement AdamW student/EMA teacher training on global/local crops and masked patches: DINO class-token self-distillation, iBOT patch loss, KoLeo regularization and optionally Gram anchoring. `data/samplers.py`, SSL/pretrain configs support large distributed image sampling. Full pretraining, recent PyTorch distributed APIs and gated weights are separate resource/dependency constraints; torch2.0 environment does not guarantee current runtime compatibility.

Use permitted small frozen features or controlled projection fitting, not billion-parameter pretraining. Strong generic features are an essential capacity control; original crop semantics may not resolve similar individual fish. DINOv3 provides representation, not online identity labels/utility/memory.

## Evaluation tool: TrackEval

[Official repository](https://github.com/JonathonLuiten/TrackEval), MIT, commit `12c8791b303e0a0b50f753af204249e622d0281a`. Inspected dataset loader, evaluator aggregation and HOTA/CLEAR/Identity kernels. Kernels are byte-identical to recovered historical fork; dataset/evaluator layout differs. CFP uses standard MOT directory adapter and preserves raw per-alpha values. Dataset loader still uses removed NumPy aliases, handled process-locally. Replaying identical historical files is the metric parity check, distinct from re-running detector inference.

## Executable architecture comparison

| Reference group | Reuse/control in CFP | Required local input | Excluded initial scope |
|---|---|---|---|
| ByteTrack | Exact original baseline replay | Frozen Nx5 boxes/scores, frame sizes | Detector retraining |
| HybridSORT / BoT-SORT / SU-T | Real association API, shared fish features | Same detections + NxD embeddings, declared config | Pretending screenshot metrics are fair |
| FastReID / DINOv3 | Frozen vs fitted crop identity encoders | Train composite IDs, paired crops, hashed pretrained weights | Large-scale pretraining |
| TIDE / M4FT / PFTrack / DM3T | Nonrigid/density/fusion cost and novelty controls | Source inspection first | New detector in Protocol A |
| MOTIP | Candidate decoder/causal masks as controlled option | Track history <t, current candidates, NULL/newborn labels | Rebuild DETR / future keys |
| V-JEPA2 / GOT-JEPA | Small identity predictor, frozen targets, collapse tests | Causal labeled feature sequences | SOT templates / huge video pretraining |
| TCEI | Independently implemented safe uncertainty/cache controls | Online margins, reliability, bounded memory | Unlicensed copying / cross-test-video adaptation |
| TFMFT | Stage recovery discussion; partial source explicitly recorded | Available tracker code | Invented missing model/training |

Novelty burden: CMUE must beat same-evidence/capacity ordinary gates with fixed-candidate counterfactual supervision; PLIM must beat last/EMA/FIFO, ordinary GRU and matched Transformer; reliable writing must have an independent ablation. Combining known gates, predictors and caches without that evidence is insufficient.
